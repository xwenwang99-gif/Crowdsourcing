# -*- coding: utf-8 -*-
"""
main.py  -- crowdsourcing experiment driver.

Cleanups vs. the previous version:
  * every per-method accuracy/F1/bal-acc list is replaced by a single nested
    `metrics` dict:  metrics[method][metric] -> list over runs.
  * each method only has to produce a predicted-label vector `y_pred`; a shared
    `evaluate()` computes accuracy / macro-F1 / balanced-acc the same way for all.
  * `y_true` / `y_pred` naming throughout for ground-truth and predicted labels.
  * results are written to disk (results/run_<timestamp>/) after every run, so an
    idle timeout cannot lose completed runs.  Remember to `git add results && push`
    to keep them if the whole Codespace is later deleted.
"""

import argparse
import sys
from pathlib import Path


def parse_study_args():
    parser = argparse.ArgumentParser(description="LiSER experiment driver. With no arguments, run the original experiment.")
    parser.add_argument("--dataset", choices=["web", "ws-amt", "dog"], help="Select the real-data sensitivity experiment.")
    parser.add_argument("--mode", choices=["init", "lq", "both"], default="both",
                        help="init: one data draw, varying starts. lq: varying data, one fixed start. both: starts within draws.")
    parser.add_argument("--lq-ratios", type=int, nargs="+", help="Injected workers per original worker. Defaults: WEB 0, WS-AMT 2, DOG 5.")
    parser.add_argument("--draws", type=int, help="Number of independent worker draws (default 10 when varying contamination).")
    parser.add_argument("--starts", type=int, help="Number of starts per draw (default 10 for init, 3 for both, 1 for lq).")
    parser.add_argument("--data-seed", type=int, default=999)
    parser.add_argument("--fit-seed", type=int, default=10000)
    parser.add_argument("--loader", help="Optional dataset loader as module:function, e.g. src.CF:get_CF.")
    parser.add_argument("--output", type=Path, default=Path("results/real_variability"))
    parser.add_argument("--liser-only", action="store_true", help="Skip the Table 3 comparator (DS on WEB, CBCC elsewhere).")
    args = parser.parse_args()
    if any(n is not None and n < 1 for n in [args.draws, args.starts]):
        parser.error("--draws and --starts must be positive.")
    if args.lq_ratios is not None and any(r < 0 for r in args.lq_ratios):
        parser.error("--lq-ratios must be nonnegative integers.")
    if args.mode == "init" and args.draws not in (1, None):
        parser.error("--mode init holds the data fixed; use --draws 1.")
    if args.mode == "lq" and args.starts not in (1, None):
        parser.error("--mode lq holds the fit seed fixed; use --starts 1.")
    if min(args.fit_seed, args.data_seed) < 0 or max(args.fit_seed, args.data_seed) >= 2**32:
        parser.error("Use seed values in [0, 2**32).")
    if len(sys.argv) > 1 and args.dataset is None:
        parser.error("Select --dataset for a real-data experiment.")
    return args


if __name__ == "__main__":
    CLI = parse_study_args()
else:
    CLI = None

import hashlib
import importlib
import os
import json
import time
import warnings
import random
import torch
import numpy as np
import pandas as pd
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    recall_score,
)
from src.lfgp_withoutO_new import LFGP
from src.CBCC import cbcc
from src.Spectral_with_mean_agreement import _hq_and_label_infer
from src.Diagnosis import diagnose
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #
N_RUNS        = 10
MAXITER       = 200
N_TASK        = 200
N_WORKER      = 400
N_TASK_GROUPS = 5

LAMBDA1 = 1
LAMBDA2_0 = 1
LAMBDA2_1 = 1
HQ_RATIO = 1/30
DATASET = "synthetic"
REAL_DATA = DATASET != "synthetic"   # real data has no true worker tiers / latent factors
LQ_RATIO =2
# which methods to run (replaces the eigen_ex / DS_ex / ... flags)

DRAW_SPECTRAL = 0
DRAW_HQ_VOTES = 0
DRAW_ADAPTIVE_CUTOFFS = True  # save sorted spectral-score cutoff plots for first run
REMOVE_GLOBAL_LQ = False
SAVE_RESULTS = True # False: no results/run_<timestamp>/ folder, nothing written to disk


ENABLE = {
    "Eigen_L2":   1,   # LFGP fit + spectral worker tiering
    "Likelihood": 1,   # same LFGP fit, labels via _mc_infer_by_task (no spectral step)
    "Mean_Agreement": 1,   # same Stage 1 + simple mean-agreement HQ selection
    "Latent_Norm": 1,      # same Stage 1 + top worker-factor norm selection
    "Spectral_Gap": 1,     # adaptive HQ count: largest spectral-score gap
    "Spectral_Elbow": 1,   # adaptive HQ count: geometric elbow
    "Eigen_Oracle": 0,   # spectral tiering + label infer on the TRUE task grouping
    "DS":       1,
    "MV_HQ":    0,
    "MV":       0,
    "GLAD":     0,
    "GTIC":     0,
    "LFGP":     0,
    "CBCC":     1,
}

METHODS = [m for m, on in ENABLE.items() if on]
FIT_SEEDS = {"ours": 10000,"GLAD": 20000,"MultiSPA": 30000,"GTIC": 40000,"LFGP": 50000,"CBCC": 60000,}

DATA_KW = dict(                       # getdata_biased arguments, kept in one place
    n_task=N_TASK, n_worker=N_WORKER, n_task_groups=N_TASK_GROUPS,
    k=3, sigma=1, obs_prob=1, hq_ratio=1/30, bias_ratio=0,
    delta=1, n_classes=N_TASK_GROUPS,rho = 0.5, random = False, draw_latent_heatmap = False
)

# --------------------------------------------------------------------------- #
#  output directory (persists on disk across idle timeout)
# --------------------------------------------------------------------------- #
RUN_ID  = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", f"run_{RUN_ID}")
if SAVE_RESULTS and not (CLI and CLI.dataset):
    os.makedirs(OUT_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_fit_seed(method, run):
    if REAL_DATA:
        return FIT_SEEDS[method]       # fixed for real data
    else:
        return FIT_SEEDS[method] + run # varies for synthetic data

def out_path(name):
    return os.path.join(OUT_DIR, name) if SAVE_RESULTS else None


def _to_native(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def save_json(path, obj):
    if path is None:
        return
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_native)



def evaluate(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=-1).astype(int)

    acc, macro_f1 = diagnose(y_true, y_pred, n_classes=n_classes)
    bal = balanced_accuracy_score(y_true, y_pred)

    secondary_confusion_by_class = []

    for c in range(n_classes):
        mask = y_true == c

        if np.any(mask):
            secondary_class = (c + 1) % n_classes

            secondary_confusion_by_class.append(
                np.mean(y_pred[mask] == secondary_class)
            )

    secondary_confusion = float(
        np.mean(secondary_confusion_by_class)
    )

    label_nmi = normalized_mutual_info_score(
        y_true, y_pred, average_method="arithmetic"
    )

    recalls = recall_score(
        y_true, y_pred,
        labels=np.arange(n_classes),
        average=None,
        zero_division=0,
    )

    scores = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "bal_acc": bal,
        "label_nmi": label_nmi,
        "secondary_confusion": secondary_confusion,
    }

    for k, recall in enumerate(recalls):
        scores[f"recall_{k + 1}"] = recall
    return scores


def build_summary(metrics):
    rows = {}

    for name in METHODS:
        if len(metrics[name]["accuracy"]) == 0:
            continue

        a = summarize_runs(metrics[name]["accuracy"])
        f = summarize_runs(metrics[name]["macro_f1"])
        b = summarize_runs(metrics[name]["bal_acc"])
        lnmi = summarize_runs(metrics[name]["label_nmi"])
        sec = summarize_runs(metrics[name]["secondary_confusion"])

        rows[name] = {
            "Accuracy": a["mean"],
            "Acc sd": a["sd"],
            "Acc CI": [round(a["ci_low"], 4), round(a["ci_high"], 4)],

            "Macro F1": f["mean"],
            "F1 sd": f["sd"],
            "F1 CI": [round(f["ci_low"], 4), round(f["ci_high"], 4)],

            "Balanced Acc": b["mean"],
            "Bal sd": b["sd"],
            "Bal CI": [round(b["ci_low"], 4), round(b["ci_high"], 4)],

            "Label NMI": lnmi["mean"],
            "Label NMI sd": lnmi["sd"],

            "Secondary Confusion": sec["mean"],
            "Secondary Confusion sd": sec["sd"],

            "n_runs": a["n"],
        }

        for k in range(N_TASK_GROUPS):
            key = f"recall_{k + 1}"
            r = summarize_runs(metrics[name][key])
            rows[name][f"R{k + 1}"] = r["mean"]
            rows[name][f"R{k + 1} sd"] = r["sd"]

        if metrics[name]["cluster_acc"]:
            c = summarize_runs(metrics[name]["cluster_acc"])
            rows[name]["Cluster Acc"] = c["mean"]
            rows[name]["Cluster sd"] = c["sd"]

        if metrics[name]["cluster_ari"]:
            ari = summarize_runs(metrics[name]["cluster_ari"])
            rows[name]["Cluster ARI"] = ari["mean"]
            rows[name]["ARI sd"] = ari["sd"]

        if metrics[name]["cluster_nmi"]:
            cnmi = summarize_runs(metrics[name]["cluster_nmi"])
            rows[name]["Cluster NMI"] = cnmi["mean"]
            rows[name]["Cluster NMI sd"] = cnmi["sd"]

    return pd.DataFrame.from_dict(rows, orient="index")
def spectral_to_V(hq_workers_pred, biased_workers_pred, n_worker, n_groups):
    """Convert per-group HQ/biased index lists into an LFGP-style V matrix
    (n_worker, n_groups), 0=LQ default, in PREDICTED-group coordinates."""
    V_spec = np.zeros((n_worker, n_groups), dtype=int)
    for g in range(n_groups):
        if hq_workers_pred[g] is not None and len(hq_workers_pred[g]):
            V_spec[np.asarray(hq_workers_pred[g], dtype=int), g] = 1
        if biased_workers_pred[g] is not None and len(biased_workers_pred[g]):
            V_spec[np.asarray(biased_workers_pred[g], dtype=int), g] = 2
    return V_spec

# --------------------------------------------------------------------------- #
#  metric store:  metrics[method][metric] -> list over runs
# --------------------------------------------------------------------------- #
# accuracy    -- plain label accuracy, no permutation.
# cluster_acc -- best accuracy over label permutations.  The LFGP-based methods
#                record it themselves from their task grouping U; every other
#                method gets it from its predicted labels in evaluate().


# Real-data sensitivity study. The legacy simulation driver continues below.
def fit_study_methods(data, fit_seed, args):
    rating, y_true, task_group, R_obs, n_tasks, n_workers, n_classes = data
    rating = rating.copy()
    set_seed(fit_seed)
    model = LFGP(lf_dim=n_classes, n_worker_group=n_classes, lambda1=LAMBDA1,
                 lambda2_0=LAMBDA2_0, lambda2_1=LAMBDA2_1)
    model._prescreen(rating)
    _, B, U, _, _ = model._mc_fit(rating, key=task_group, scheme="ds",
                                  epsilon=1e-5, maxiter=MAXITER, verbose=1)
    _, labels, _, _, _ = _hq_and_label_infer(
        U.astype(int), R_obs, y_true, None, n_tasks, n_workers, n_classes,
        hq_ratio=0.1, LABEL_MODE="task", WORKER_SELECTION="spectral",
        verbose=False, MIN_COVERAGE=0, return_spectral=True, B=B,
    )
    predictions = {"LiSER": labels}
    if not args.liser_only:
        if args.dataset == "web":
            set_seed(fit_seed)
            predictions["DS"] = model._init_task_member_ds(rating)[:, 1]
        else:
            set_seed(fit_seed)
            predictions["CBCC"] = cbcc(task=rating[:, 0], worker=rating[:, 1],
                                       label=rating[:, 2], n_classes=n_classes)["pred"]
    return {name: evaluate(y_true, labels, n_classes)["accuracy"]
            for name, labels in predictions.items()}


def annotation_digest(rating):
    # Sort triples so a loader may return the same annotations in a different order.
    triples = np.asarray(rating[:, :3], dtype=np.int64)
    order = np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0]))
    return hashlib.sha256(triples[order].tobytes()).hexdigest()


def write_study_csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def summarize_study(frame):
    summaries = []
    for method, group in frame.groupby("method", sort=False):
        by_draw = group.groupby("data_seed")["accuracy"]
        means = by_draw.mean()
        summaries.append({
            "method": method, "draws": len(means), "fits": len(group),
            "mean_accuracy": means.mean(),
            "initialization_sd": np.sqrt(by_draw.var(ddof=1).mean()),
            "sd_of_draw_means": means.std(ddof=1),
        })
    return pd.DataFrame(summaries)


def run_ratio(args, loader, loader_name, clean, ratio, destination):
    n_draws = 1 if ratio == 0 or args.mode == "init" else (args.draws or 10)
    n_starts = args.starts or (1 if args.mode == "lq" else
                              10 if args.mode == "init" or ratio == 0 else 3)
    if ratio == 0 and args.draws and args.draws > 1 and args.lq_ratios in ([0], None):
        raise ValueError("r=0 has no injected workers to redraw. Use --draws 1 or omit --draws.")
    if args.data_seed + n_draws > 2**32:
        raise ValueError("The selected data seeds exceed the 32-bit seed range.")
    clean_rating, clean_truth, clean_groups, _, clean_tasks, clean_workers, classes = clean
    human_ids = np.unique(clean_rating[:, 1])
    human_digest = annotation_digest(clean_rating)
    seen_draws = {}
    config = dict(dataset=args.dataset, loader=loader_name, mode=args.mode, ratio=ratio,
                  draws=n_draws, starts=n_starts, data_seed=args.data_seed,
                  fit_seed=args.fit_seed, liser_only=args.liser_only, gamma=0.1,
                  maxiter=MAXITER, penalties=[LAMBDA1, LAMBDA2_0, LAMBDA2_1],
                  human_annotations=human_digest,
                  gold_labels=np.asarray(clean_truth, dtype=int).ravel().tolist())
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = destination / f"r{ratio}_{args.mode}_{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Results: {folder} ({n_draws} draw(s) x {n_starts} start(s))", flush=True)
    # SQLite releases the lock if a fit is interrupted, including in Codespaces.
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(folder / "study.lock", timeout=5)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        save_json(folder / "config.json", config)
        result_path = folder / "accuracy_by_fit.csv"
        records = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
        completed = {(r["data_seed"], r["start"]): r["annotation_hash"] for r in records}
        for draw in range(n_draws):
            data_seed = args.data_seed + draw
            set_seed(data_seed)
            data = clean if ratio == 0 else loader(r=ratio, seed=data_seed)
            rating, truth, groups, _, n_tasks, workers, n_classes = data
            if ((n_tasks, n_classes) != (clean_tasks, classes)
                    or not np.array_equal(truth, clean_truth) or not np.array_equal(groups, clean_groups)
                    or workers != clean_workers * (1 + ratio)):
                raise ValueError("The loader changed the benchmark tasks, labels, or expected worker count.")
            if ratio and annotation_digest(rating[np.isin(rating[:, 1], human_ids)]) != human_digest:
                raise ValueError("The augmented loader changed original annotations or worker IDs. Keep them fixed across draws.")
            digest = annotation_digest(rating)
            if digest in seen_draws:
                raise ValueError(f"Data seeds {seen_draws[digest]} and {data_seed} produced identical annotations. Check the loader's seed.")
            seen_draws[digest] = data_seed
            for start in range(n_starts):
                key = (data_seed, start + 1)
                if key in completed:
                    if completed[key] != digest:
                        raise ValueError("Regenerated annotations differ from the saved draw. Use a separate output folder.")
                    continue
                # The LQ-only study uses exactly the same fitting seed for every draw.
                fit_seed = args.fit_seed
                if args.mode != "lq" and (draw or start):
                    fit_seed = int(np.random.SeedSequence([args.fit_seed, draw, start]).generate_state(1)[0])
                print(f"data_seed={data_seed} start={start+1}/{n_starts} fit_seed={fit_seed}", flush=True)
                scores = fit_study_methods(data, fit_seed, args)
                for method, accuracy in scores.items():
                    records.append(dict(dataset=args.dataset, lq_ratio=ratio,
                                        injected_workers=workers-clean_workers, data_seed=data_seed,
                                        start=start+1, fit_seed=fit_seed, annotation_hash=digest,
                                        method=method, accuracy=accuracy))
                write_study_csv(pd.DataFrame(records), result_path)
        frame = pd.DataFrame(records)
        summary = summarize_study(frame)
        write_study_csv(summary, folder / "summary.csv")
        print(summary.to_string(index=False), flush=True)
        if not args.liser_only:
            comparator = "DS" if args.dataset == "web" else "CBCC"
            pairs = frame.pivot(index=["data_seed", "start"], columns="method", values="accuracy")
            diff = (pairs["LiSER"] - pairs[comparator]).groupby("data_seed").mean()
            write_study_csv(diff.rename("liser_minus_baseline").reset_index(), folder / "paired_by_draw.csv")
            print(f"Mean paired LiSER - {comparator}: {100 * diff.mean():.2f} percentage points", flush=True)


def run_real_study(args):
    defaults = {"web": ("src.WEB:get_WEB", 0, (1000, 158, 5)),
                "dog": ("src.DOG:get_DOG", 5, (807, 109, 4)),
                "ws-amt": ("src.CF:get_CF", 2, (300, 110, 5))}
    default_loader, default_ratio, expected = defaults[args.dataset]
    loader_name = args.loader or default_loader
    module, function = loader_name.split(":")
    loader = getattr(importlib.import_module(module), function)
    set_seed(args.data_seed)
    clean = loader(r=0, seed=args.data_seed)
    if tuple(clean[4:7]) != expected:
        raise ValueError(f"{loader_name} returned clean (tasks, workers, classes)={tuple(clean[4:7])}; "
                         f"{args.dataset} in Table 3 has {expected}. Select its loader with --loader module:function.")
    for ratio in dict.fromkeys(args.lq_ratios or [default_ratio]):
        run_ratio(args, loader, loader_name, clean, ratio, args.output / args.dataset)


if CLI is not None and CLI.dataset:
    run_real_study(CLI)
    raise SystemExit(0)


from src.DOG import get_DOG
from src.FACE import get_FACE
from src.BIRD import get_BIRD
from src.WEB import get_WEB
from src.lfgp_paper import LFGP_PAPER
from src.GTIC import gtic
from src.CF import get_CF
from src.NetEaseCrowd import get_NETEASE
from src.getdata_biased import getdata_biased
from src.peera import peerA
from src.Spectral_with_mean_agreement import _hq_and_label_infer, tier_centers_in_lf_space, infer_labels_from_spectral_hq, plot_spectral_cutoffs
from src.Diagnosis import (
    diagnose, 
    summarize_runs,
    build_tier_vectors, 
    worker_diagnose, 
    worker_diagnose_runs, 
    plot_tier_confusion,
    build_worker_summary, 
    task_group_mapping,
    plot_spectral_diagnostics)
from src.hq_vote_diagnostic import hq_vote_report
from src.hq_vote_diagnostic import hq_vote_report, plot_worker_lf_pca,true_tier_centers, plot_loss_trajectory


start = time.perf_counter()
removed_worker_records = []
adaptive_gamma_records = []

# per-method worker-tier vectors:  tier_lists[method]["true"/"pred"] -> list over runs
tier_lists = {name: {"true": [], "pred": []}
              for name in ("Eigen_L2", "Mean_Agreement", "Latent_Norm", "Spectral_Gap", "Spectral_Elbow", "Likelihood", "Eigen_Oracle")}
if DATASET != "synthetic":
    loader = {"dog": get_DOG, "face": get_FACE, "bird": get_BIRD, "netease": get_NETEASE, "web": get_WEB, "cf": get_CF,}[DATASET]
    rating, y_true, task_group, R_obs, N_TASK, N_WORKER, N_TASK_GROUPS = loader(r=LQ_RATIO, seed = 999)
    worker_label = task_lf = worker_lf = None   # no ground-truth worker tiers

metrics = {m: {"accuracy": [], "macro_f1": [], "bal_acc": [], "label_nmi": [], "secondary_confusion": [], **{f"recall_{k + 1}": [] for k in range(N_TASK_GROUPS)},"cluster_acc": [], "cluster_ari": [],"cluster_nmi": [],}
           for m in METHODS}
for i in range(N_RUNS):
    if DATASET == "synthetic":
        np.random.seed(i)
        rating, y_true, task_group, worker_label, R_obs, task_lf, worker_lf = getdata_biased(**DATA_KW)
 
    produced = {}   # method name -> predicted label vector for this run

    # LFGP model is needed by Eigen_L2, Likelihood, and DS
    model = None
    if ENABLE["Eigen_L2"] or ENABLE["Mean_Agreement"] or ENABLE["Latent_Norm"] or ENABLE["Spectral_Gap"] or ENABLE["Spectral_Elbow"] or ENABLE["Likelihood"] or ENABLE["DS"]:
        model = LFGP(lf_dim=N_TASK_GROUPS, n_worker_group=N_TASK_GROUPS,
                     lambda1=LAMBDA1, lambda2_0=LAMBDA2_0, lambda2_1=LAMBDA2_1)
        model._prescreen(rating)

    # one LFGP fit shared by the likelihood-only and spectral methods
    if ENABLE["Eigen_L2"] or ENABLE["Likelihood"] or ENABLE["Mean_Agreement"] or ENABLE["Latent_Norm"] or ENABLE["Spectral_Gap"] or ENABLE["Spectral_Elbow"]:
        set_seed(get_fit_seed("ours", i))
        A, B, U, V, clusters_np = model._mc_fit(
            rating, key=task_group, scheme="ds", epsilon=1e-5,
            maxiter=MAXITER, verbose=1,
        )
        pred_group = U.astype(int)
        cluster_acc = model.task_acc(pred_group, task_group)
        cluster_ari = adjusted_rand_score(task_group, pred_group)
        cluster_nmi = normalized_mutual_info_score(
            task_group, pred_group, average_method="arithmetic"
        )

    if ENABLE["Likelihood"] and not REAL_DATA:
        # likelihood step only: majority vote of the fit's own HQ workers (V) 
        hq_lik = [np.where(V[:, g] == 1)[0] for g in range(N_TASK_GROUPS)]
        biased_lik = [np.where(V[:, g] == 2)[0] for g in range(N_TASK_GROUPS)]
        yt_tier, yp_tier = build_tier_vectors(
            worker_label, hq_lik, biased_lik,
            pred_group, y_true, N_TASK_GROUPS)
        tier_lists["Likelihood"]["true"].append(yt_tier)
        tier_lists["Likelihood"]["pred"].append(yp_tier)

    if ENABLE["Likelihood"]:    
        #y_pred = model._mc_infer(rating )
        y_pred = model._mc_infer_by_task(rating)
        metrics["Likelihood"]["cluster_acc"].append(cluster_acc)
        metrics["Likelihood"]["cluster_ari"].append(cluster_ari)
        metrics["Likelihood"]["cluster_nmi"].append(cluster_nmi)

        produced["Likelihood"] = y_pred.astype(int)
    if ENABLE["Eigen_L2"] or ENABLE["Mean_Agreement"] or ENABLE["Latent_Norm"] or ENABLE["Spectral_Gap"] or ENABLE["Spectral_Elbow"]:
        _, y_pred_spec, hq_spec, biased_spec, spectral = _hq_and_label_infer(
            pred_group, R_obs, y_true, worker_label,
            N_TASK, N_WORKER, N_TASK_GROUPS, hq_ratio=HQ_RATIO,
            LABEL_MODE="task", WORKER_SELECTION="spectral",
            verbose=False, MIN_COVERAGE=0, return_spectral=True, B=B,
        )

        # Spectral LiSER result.
        if ENABLE["Eigen_L2"]:
            if DRAW_SPECTRAL:
                plot_spectral_diagnostics(
                    spectral,
                    save_dir=OUT_DIR,
                    run=i,
                    draw=DRAW_SPECTRAL,
                )

            if not REAL_DATA:
                yt_tier, yp_tier = build_tier_vectors(
                    worker_label, hq_spec, biased_spec,
                    pred_group, task_group, N_TASK_GROUPS,
                )
                tier_lists["Eigen_L2"]["true"].append(yt_tier)
                tier_lists["Eigen_L2"]["pred"].append(yp_tier)

            hq_vote_report(
                rating, pred_group, hq_spec, N_TASK_GROUPS,
                OUT_DIR, f"Eigen_L2_run{i}",
                y_true=y_true, draw=bool(DRAW_HQ_VOTES) and SAVE_RESULTS,
            )

            metrics["Eigen_L2"]["cluster_acc"].append(cluster_acc)
            metrics["Eigen_L2"]["cluster_ari"].append(cluster_ari)
            metrics["Eigen_L2"]["cluster_nmi"].append(cluster_nmi)
            produced["Eigen_L2"] = y_pred_spec

        # Simple mean-agreement worker-selection result.
        # Reuses the SAME 'spectral' object, so Stage 1, task groups,
        # agreement matrices, tested workers, and hq_ratio are identical.
        if ENABLE["Mean_Agreement"]:
            _, y_pred_mean, hq_mean, biased_mean = infer_labels_from_spectral_hq(
                pred_group=pred_group,
                R_obs=R_obs,
                spectral=spectral,
                n_task=N_TASK,
                n_task_groups=N_TASK_GROUPS,
                label=y_true,
                label_mode="task",
                worker_selection="mean_agreement",
            )

            if not REAL_DATA:
                yt_tier, yp_tier = build_tier_vectors(
                    worker_label, hq_mean, biased_mean,
                    pred_group, task_group, N_TASK_GROUPS,
                )
                tier_lists["Mean_Agreement"]["true"].append(yt_tier)
                tier_lists["Mean_Agreement"]["pred"].append(yp_tier)

            hq_vote_report(
                rating, pred_group, hq_mean, N_TASK_GROUPS,
                OUT_DIR, f"Mean_Agreement_run{i}",
                y_true=y_true, draw=bool(DRAW_HQ_VOTES) and SAVE_RESULTS,
            )

            metrics["Mean_Agreement"]["cluster_acc"].append(cluster_acc)
            metrics["Mean_Agreement"]["cluster_ari"].append(cluster_ari)
            metrics["Mean_Agreement"]["cluster_nmi"].append(cluster_nmi)
            produced["Mean_Agreement"] = y_pred_mean

        # Latent-factor norm baseline: rank workers by ||B[j, g, :]||_2.
        # Reuses the exact same Stage-1 B, task partition, eligible workers,
        # HQ count, and HQ-only label voting rule.
        if ENABLE["Latent_Norm"]:
            _, y_pred_lf, hq_lf, biased_lf = infer_labels_from_spectral_hq(
                pred_group=pred_group, R_obs=R_obs, spectral=spectral,
                n_task=N_TASK, n_task_groups=N_TASK_GROUPS,
                label=y_true, label_mode="task", worker_selection="latent_norm",
            )

            if not REAL_DATA:
                yt_tier, yp_tier = build_tier_vectors(
                    worker_label, hq_lf, biased_lf,
                    pred_group, task_group, N_TASK_GROUPS,
                )
                tier_lists["Latent_Norm"]["true"].append(yt_tier)
                tier_lists["Latent_Norm"]["pred"].append(yp_tier)

            hq_vote_report(
                rating, pred_group, hq_lf, N_TASK_GROUPS,
                OUT_DIR, f"Latent_Norm_run{i}",
                y_true=y_true, draw=bool(DRAW_HQ_VOTES) and SAVE_RESULTS,
            )

            metrics["Latent_Norm"]["cluster_acc"].append(cluster_acc)
            metrics["Latent_Norm"]["cluster_ari"].append(cluster_ari)
            metrics["Latent_Norm"]["cluster_nmi"].append(cluster_nmi)
            produced["Latent_Norm"] = y_pred_lf

        # Adaptive spectral selection: NO true HQ counts and NO fixed gamma.
        # All methods share the same fitted task groups and spectral embedding.
        for method_name, selection_name, k_key, gamma_key in (
            ("Spectral_Gap", "largest_gap", "k_hat_largest_gap", "gamma_hat_largest_gap"),
            ("Spectral_Elbow", "geometric_elbow", "k_hat_geometric_elbow", "gamma_hat_geometric_elbow"),
        ):
            if not ENABLE[method_name]:
                continue
            _, y_pred_adapt, hq_adapt, biased_adapt = infer_labels_from_spectral_hq(
                pred_group=pred_group, R_obs=R_obs, spectral=spectral,
                n_task=N_TASK, n_task_groups=N_TASK_GROUPS,
                label=y_true, label_mode="task", worker_selection=selection_name,
            )
            produced[method_name] = y_pred_adapt
            metrics[method_name]["cluster_acc"].append(cluster_acc)
            metrics[method_name]["cluster_ari"].append(cluster_ari)
            metrics[method_name]["cluster_nmi"].append(cluster_nmi)

            if not REAL_DATA:
                yt_tier, yp_tier = build_tier_vectors(
                    worker_label, hq_adapt, biased_adapt,
                    pred_group, task_group, N_TASK_GROUPS,
                )
                tier_lists[method_name]["true"].append(yt_tier)
                tier_lists[method_name]["pred"].append(yp_tier)

            for g in range(N_TASK_GROUPS):
                adaptive_gamma_records.append({
                    "run": i + 1, "group": g, "method": method_name,
                    "k_hat": int(spectral[k_key][g]),
                    "gamma_hat": float(spectral[gamma_key][g]),
                    "n_eligible": int(spectral["tested_mask"][:, g].sum()),
                })

        if DRAW_ADAPTIVE_CUTOFFS and i == 0:
            plot_spectral_cutoffs(spectral, save_dir=OUT_DIR if SAVE_RESULTS else None,
                                  run=i + 1, show=not SAVE_RESULTS)

    if ENABLE["Eigen_Oracle"]:
        # Upper bound on the spectral step: feed the ground-truth task grouping
        # so that any tiering/label error is attributable to the eigen-decomposition
        # alone, not to clustering error propagated from the LFGP fit.
        oracle_group = np.asarray(task_group, dtype=int)

        _, y_pred_or, hq_or, biased_or = _hq_and_label_infer(
            oracle_group, R_obs, y_true, worker_label,
            N_TASK, N_WORKER, N_TASK_GROUPS, hq_ratio=HQ_RATIO,
            LABEL_MODE="task", verbose=False,
            MIN_COVERAGE=0, return_spectral=False,
        )
        if not REAL_DATA:
            yt_tier, yp_tier = build_tier_vectors(
                worker_label, hq_or, biased_or,
                oracle_group, y_true, N_TASK_GROUPS)
            tier_lists["Eigen_Oracle"]["true"].append(yt_tier)
            tier_lists["Eigen_Oracle"]["pred"].append(yp_tier)

        hq_vote_report(rating, oracle_group, hq_or, N_TASK_GROUPS,
                       OUT_DIR, f"Eigen_Oracle_run{i}",
                       y_true=y_true, draw=bool(DRAW_HQ_VOTES) and SAVE_RESULTS)

        metrics["Eigen_Oracle"]["cluster_acc"].append(1.0)   # oracle grouping
        produced["Eigen_Oracle"] = y_pred_or

    if ENABLE["DS"]:
        y_pred = model._init_task_member_ds(rating)[:, 1]
        #cluster_acc = model.task_acc(y_pred, task_group)
        #metrics["DS"]["cluster_acc"].append(cluster_acc)
        produced["DS"] = y_pred        

    if ENABLE["MV_HQ"]:
        y_pred = np.full(N_TASK, -1, dtype=int)

        for t in range(N_TASK):
            g = task_group[t]
            hq = np.where(worker_label[:, g, 1] == 1)[0]

            labs = rating[
                np.isin(rating[:, 1], hq) &
                (rating[:, 0] == t)
            ][:, 2]

            if len(labs):
                y_pred[t] = mode(labs, axis=None).mode.item()

        produced["MV_HQ"] = y_pred
    if ENABLE["MV"]:
        y_pred = np.full(N_TASK, -1, dtype=int)
        for t in range(N_TASK):
            labs = rating[rating[:, 0] == t][:, 2]
            if len(labs):
                y_pred[t] = mode(labs, axis=None).mode.item()
        produced["MV"] = y_pred

    if ENABLE["GLAD"]:
        set_seed(get_fit_seed("GLAD", i))
        y_pred, _ = peerA(rating, N_TASK_GROUPS, N_WORKER)._GLAD()
        produced["GLAD"] = y_pred

    if ENABLE["GTIC"]:
        set_seed(get_fit_seed("GTIC", i))
        model_lfgp = LFGP_PAPER(
            lf_dim=N_TASK_GROUPS,
            n_worker_group=N_TASK_GROUPS,
            lambda1=1,
            lambda2=1
        )

        gtic_result = gtic(
            rating, n=N_TASK, m=N_WORKER,
            K=N_TASK_GROUPS, missing_val=-1
        )

        # Raw K-means task clusters
        gtic_group = gtic_result.cluster_id

        cluster_acc = model_lfgp.task_acc(gtic_group, task_group)
        cluster_ari = adjusted_rand_score(task_group, gtic_group)
        cluster_nmi = normalized_mutual_info_score(
            task_group, gtic_group, average_method="arithmetic"
        )

        metrics["GTIC"]["cluster_acc"].append(cluster_acc)
        metrics["GTIC"]["cluster_ari"].append(cluster_ari)
        metrics["GTIC"]["cluster_nmi"].append(cluster_nmi)

        produced["GTIC"] = gtic_result.y_hat
    if ENABLE["LFGP"]:
        set_seed(get_fit_seed("LFGP", i))
        model_lfgp = LFGP_PAPER(
            lf_dim=N_TASK_GROUPS,
            n_worker_group=N_TASK_GROUPS,
            lambda1=1,
            lambda2=1
        )
        model_lfgp._prescreen(rating)
        model_lfgp._mc_fit(rating, scheme="ds", epsilon=1e-2, maxiter=MAXITER, verbose=0)

        # Raw task-group assignment from LFGP
        lfgp_group = model_lfgp.U.astype(int)

        cluster_acc = model_lfgp.task_acc(lfgp_group, task_group)
        cluster_ari = adjusted_rand_score(task_group, lfgp_group)
        cluster_nmi = normalized_mutual_info_score(
            task_group, lfgp_group, average_method="arithmetic"
        )

        metrics["LFGP"]["cluster_acc"].append(cluster_acc)
        metrics["LFGP"]["cluster_ari"].append(cluster_ari)
        metrics["LFGP"]["cluster_nmi"].append(cluster_nmi)

        y_pred = model_lfgp._mc_infer(rating)[:, 1].astype(int)
        produced["LFGP"] = y_pred
        
    if ENABLE["CBCC"]:
        set_seed(get_fit_seed("CBCC", i))
        res = cbcc(task = rating[:,0], worker= rating[:,1], label=rating[:,2], n_classes = N_TASK_GROUPS)
        produced["CBCC"] = res["pred"]
        
    # ---- evaluate everything the same way and record ----
    for name, y_pred in produced.items():
        scores = evaluate(y_true, y_pred, N_TASK_GROUPS)
        for key, val in scores.items():
            metrics[name][key].append(val)

    # ---- persist after every run so a timeout can't lose finished runs ----
    if SAVE_RESULTS:
        save_json(out_path("metrics_raw.json"), metrics)
        if adaptive_gamma_records:
            pd.DataFrame(adaptive_gamma_records).to_csv(out_path("gamma_adaptive_by_group.csv"), index=False)
        pd.DataFrame(removed_worker_records).to_csv(
            out_path("removed_workers.csv"), index=False)
        print(f"[run {i + 1}/{N_RUNS}] done; results saved to {OUT_DIR}")
    else:
        print(f"[run {i + 1}/{N_RUNS}] done (SAVE_RESULTS off, nothing written)")

# --------------------------------------------------------------------------- #
#  summarize, print, and save
# --------------------------------------------------------------------------- #
summary_df = build_summary(metrics)
pd.set_option("display.width", 200, "display.max_columns", None)
print("\n==== SUMMARY ====")
print(summary_df)

# --------------------------------------------------------------------------- #
# Individual-run accuracies
# --------------------------------------------------------------------------- #
run_acc_df = pd.DataFrame({
    name: pd.Series(metrics[name]["accuracy"])
    for name in METHODS
})

run_acc_df.index = np.arange(1, len(run_acc_df) + 1)
run_acc_df.index.name = "Run"

print("\n==== ACCURACY BY RUN ====")
print(run_acc_df.round(4))

if SAVE_RESULTS:
    run_acc_df.to_csv(out_path("accuracy_by_run.csv"))

if SAVE_RESULTS:
    summary_df.to_csv(out_path("summary.csv"))
save_json(out_path("summary.json"), summary_df.to_dict(orient="index"))
save_json(out_path("config.json"),
          {"run_id": RUN_ID, "n_runs": N_RUNS, "maxiter": MAXITER,
           "enable": ENABLE, 
           "remove_global_lq": REMOVE_GLOBAL_LQ,
           "data_kw": DATA_KW})

for _name in ("Eigen_L2", "Mean_Agreement", "Latent_Norm", "Spectral_Gap", "Spectral_Elbow", "Likelihood", "Eigen_Oracle"):
    if not SAVE_RESULTS:
        break
    if not ENABLE[_name] or not tier_lists[_name]["true"]:
        continue
    worker_agg = worker_diagnose_runs(
        tier_lists[_name]["true"], tier_lists[_name]["pred"])
    plot_tier_confusion(                           # the heatmap (one file per method)
        worker_agg["confusion_rownorm"], annot_counts=worker_agg["confusion_sum"],
        path=os.path.join(OUT_DIR, f"worker_confusion_{_name}.png"),
        title=f"Worker-tier recovery ({_name})")

    build_worker_summary(worker_agg).to_csv(
        os.path.join(OUT_DIR, f"summary_worker_{_name}.csv"), index=False)



runtime = time.perf_counter() - start
save_json(out_path("runtime.json"),
          {"runtime_sec": round(runtime, 1), "n_runs": N_RUNS,
           "sec_per_run": round(runtime / N_RUNS, 1)})
print(f"\nRuntime: {runtime:.1f}s   |   "
      + (f"all outputs in: {OUT_DIR}" if SAVE_RESULTS else "no outputs written"))