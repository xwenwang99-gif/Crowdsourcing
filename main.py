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
)
from src.DOG import get_DOG
from src.FACE import get_FACE
from src.BIRD import get_BIRD
from src.lfgp_withoutO_new import LFGP
from src.lfgp_paper import LFGP_PAPER
from src.GTIC import gtic
from src.CBCC import cbcc
from src.NetEaseCrowd import get_NETEASE
from src.multispa import multispa_fit_predict
from src.getdata_biased import getdata_biased
from src.getdata_new import getdata_new
from src.eigenInfer import _hq_and_label_infer, tier_centers_in_lf_space
from src.Diagnosis import (
    diagnose, 
    summarize_runs,
    build_tier_vectors, 
    worker_diagnose, 
    worker_diagnose_runs, 
    plot_tier_confusion,
    build_worker_summary, 
    print_spectral_worker_comparison,
    task_group_mapping,
    plot_spectral_diagnostics)
from src.hq_vote_diagnostic import hq_vote_report
from src.peera import peerA
from src.hq_vote_diagnostic import hq_vote_report, plot_worker_lf_pca,true_tier_centers, plot_loss_trajectory
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #
N_RUNS        = 1
MAXITER       = 100
N_TASK        = 200
N_WORKER      = 400
N_TASK_GROUPS = 5

LAMBDA1 = 1
LAMBDA2_0 = 1
LAMBDA2_1 = 1
# which methods to run (replaces the eigen_ex / DS_ex / ... flags)

DRAW_SPECTRAL = 0
DRAW_HQ_VOTES = 0
REMOVE_GLOBAL_LQ = False
SAVE_RESULTS = True  # False: no results/run_<timestamp>/ folder, nothing written to disk
HQ_RATIO = 1/30
DATASET = "netease"
REAL_DATA = DATASET != "synthetic"   # real data has no true worker tiers / latent factors
LQ_RATIO = 5

ENABLE = {
    "Eigen_L2":   1,   # LFGP fit + spectral worker tiering
    "Likelihood": 1,   # same LFGP fit, labels via _mc_infer_by_task (no spectral step)
    "Likelihood2":  0,   # warm-restarted likelihood, init from spectral tiers
    "Eigen_L2_v2":  0,
    "Eigen_Oracle": 0,   # spectral tiering + label infer on the TRUE task grouping
    "DS":       1,
    "MV_HQ":    0,
    "MV":       1,
    "GLAD":     0,
    "MultiSPA": 0,
    "GTIC":     0,
    "LFGP":     0,
    "CBCC":     0,
}

METHODS = [m for m, on in ENABLE.items() if on]

DATA_KW = dict(                       # getdata_biased arguments, kept in one place
    n_task=N_TASK, n_worker=N_WORKER, n_task_groups=N_TASK_GROUPS,
    k=3, sigma=1, obs_prob=1, hq_ratio=1/6, bias_ratio=0,
    delta=1, n_classes=N_TASK_GROUPS,rho = 1/2
)

# --------------------------------------------------------------------------- #
#  output directory (persists on disk across idle timeout)
# --------------------------------------------------------------------------- #
RUN_ID  = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", f"run_{RUN_ID}")
if SAVE_RESULTS:
    os.makedirs(OUT_DIR, exist_ok=True)


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


# --------------------------------------------------------------------------- #
#  shared evaluation: every method just hands us a predicted-label vector
# --------------------------------------------------------------------------- #
def cluster_accuracy(y_true, y_pred, n_classes):
    """Best accuracy over all label permutations (Hungarian matching).
    Predictions outside [0, n_classes) (e.g. -1 for 'no label') count as wrong."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=-1).astype(int)
    valid = (y_pred >= 0) & (y_pred < n_classes)
    counts = confusion_matrix(y_true[valid], y_pred[valid], labels=np.arange(n_classes))
    row_ind, col_ind = linear_sum_assignment(-counts)
    return counts[row_ind, col_ind].sum() / len(y_true)


def evaluate(y_true, y_pred, n_classes):
    acc, macro_f1 = diagnose(y_true, y_pred, n_classes=n_classes)
    bal = balanced_accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": macro_f1, "bal_acc": bal,
            "cluster_acc": cluster_accuracy(y_true, y_pred, n_classes)}


def build_summary(metrics):
    rows = {}
    for name in METHODS:
        if len(metrics[name]["accuracy"]) == 0:
            continue
        a = summarize_runs(metrics[name]["accuracy"])
        f = summarize_runs(metrics[name]["macro_f1"])
        b = summarize_runs(metrics[name]["bal_acc"])
        rows[name] = {
            "Accuracy":     a["mean"], "Acc sd": a["sd"],
            "Acc CI":       [round(a["ci_low"], 4), round(a["ci_high"], 4)],
            "Macro F1":     f["mean"], "F1 sd":  f["sd"],
            "F1 CI":        [round(f["ci_low"], 4), round(f["ci_high"], 4)],
            "Balanced Acc": b["mean"], "Bal sd": b["sd"],
            "Bal CI":       [round(b["ci_low"], 4), round(b["ci_high"], 4)],
            "n_runs":       a["n"],
        }
        if metrics[name].get("cluster_acc"):
            c = summarize_runs(metrics[name]["cluster_acc"])
            rows[name].update({
                "Cluster Acc": c["mean"], "Cluster sd": c["sd"],
                "Cluster CI":  [round(c["ci_low"], 4), round(c["ci_high"], 4)],
            })

        if metrics[name].get("cluster_ari"):
            ari = summarize_runs(metrics[name]["cluster_ari"])
            rows[name].update({
                "Cluster ARI": ari["mean"],
                "ARI sd": ari["sd"],
                "ARI CI": [round(ari["ci_low"], 4), round(ari["ci_high"], 4)],
            })
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
metrics = {m: {"accuracy": [], "macro_f1": [], "bal_acc": [], "cluster_acc": [], "cluster_ari": [],}
           for m in METHODS}

start = time.perf_counter()
removed_worker_records = []

# per-method worker-tier vectors:  tier_lists[method]["true"/"pred"] -> list over runs
tier_lists = {name: {"true": [], "pred": []}
              for name in ("Eigen_L2", "Likelihood", "Likelihood2", "Eigen_L2_v2", "Eigen_Oracle",)}
for i in range(N_RUNS):
    np.random.seed(i)
    if DATASET == "synthetic":
        rating, y_true, task_group, worker_label, R_obs, task_lf, worker_lf = getdata_biased(**DATA_KW)
    else:
        loader = {"dog": get_DOG, "face": get_FACE, "bird": get_BIRD, "netease": get_NETEASE}[DATASET]
        rating, y_true, task_group, R_obs, N_TASK, N_WORKER, N_TASK_GROUPS = loader(r=LQ_RATIO, seed = i)
        worker_label = task_lf = worker_lf = None   # no ground-truth worker tiers
    '''
    # -----------------------
    # Same fitting seed
    # -----------------------
    fit_seed = 999

    random.seed(fit_seed)
    np.random.seed(fit_seed)    
    torch.manual_seed(fit_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fit_seed)

    '''
    produced = {}   # method name -> predicted label vector for this run

    # LFGP model is needed by Eigen_L2, Likelihood, and DS
    model = None
    if ENABLE["Eigen_L2"] or ENABLE["Likelihood"] or ENABLE["DS"]:
        model = LFGP(lf_dim=N_TASK_GROUPS, n_worker_group=N_TASK_GROUPS,
                     lambda1=LAMBDA1, lambda2_0=LAMBDA2_0, lambda2_1=LAMBDA2_1)
        model._prescreen(rating)

    # one LFGP fit shared by the likelihood-only and spectral methods
    if ENABLE["Eigen_L2"] or ENABLE["Likelihood"]:
        A, B, U, V, clusters_np = model._mc_fit(
            rating, key=task_group, scheme="ds", U_init=task_group, epsilon=1e-5,
            maxiter=MAXITER, verbose=1,
        )
        pred_group = U.astype(int)
        cluster_acc = model.task_acc(pred_group, task_group)
        cluster_ari = adjusted_rand_score(task_group, pred_group)

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
        produced["Likelihood"] = y_pred.astype(int)
        metrics["Likelihood"]["cluster_ari"].append(cluster_ari)

    if ENABLE["Eigen_L2"]:        
        _, y_pred, hq_workers_pred, biased_workers_pred, spectral = _hq_and_label_infer(
            pred_group, R_obs, y_true, worker_label,
            N_TASK, N_WORKER, N_TASK_GROUPS,hq_ratio = HQ_RATIO,
            LABEL_MODE="task", verbose=False,
            MIN_COVERAGE=0, return_spectral=True,
        )
        if DRAW_SPECTRAL:
            plot_spectral_diagnostics(
                spectral,
                save_dir=OUT_DIR,
                run=i,
                draw=DRAW_SPECTRAL
            )
                
        if not REAL_DATA:
            print_spectral_worker_comparison(
                spectral,
                worker_label,
                pred_group,
                task_group,
            )

            yt_tier, yp_tier = build_tier_vectors(
                worker_label, hq_workers_pred, biased_workers_pred,
                pred_group, task_group, N_TASK_GROUPS)
            tier_lists["Eigen_L2"]["true"].append(yt_tier)
            tier_lists["Eigen_L2"]["pred"].append(yp_tier)

        hq_vote_report(rating, pred_group, hq_workers_pred, N_TASK_GROUPS,
               OUT_DIR, f"Eigen_L2_run{i}",
               y_true=y_true, draw=bool(DRAW_HQ_VOTES) and SAVE_RESULTS)

        metrics["Eigen_L2"]["cluster_acc"].append(cluster_acc)
        produced["Eigen_L2"] = y_pred
        if ENABLE["Likelihood2"]:
            V_spec = spectral_to_V(hq_workers_pred, biased_workers_pred,
                                   N_WORKER, N_TASK_GROUPS)            
            
            clusters_spec = tier_centers_in_lf_space(B, V_spec, )
            if REMOVE_GLOBAL_LQ:
                warm_worker_mask = spectral["warm_worker_mask"]
            else:
                warm_worker_mask = np.ones(N_WORKER, dtype=bool)
            
            removed_workers = np.where(~warm_worker_mask)[0]
            n_removed = len(removed_workers)
            
            removed_worker_records.append({
                "run": i,
                "remove_global_lq": REMOVE_GLOBAL_LQ,
                "n_removed": n_removed,
                "n_kept": N_WORKER - n_removed,
                "removed_ids": ",".join(map(str, removed_workers)),
            })
            
            print(
                f"Run {i}: global LQ removal={REMOVE_GLOBAL_LQ}, "
                f"removed {n_removed}/{N_WORKER} workers "
                f"({n_removed / N_WORKER:.1%})"
            )

            
            model2 = LFGP(lf_dim=N_TASK_GROUPS, n_worker_group=N_TASK_GROUPS,
                          lambda1=LAMBDA1, lambda2_0=LAMBDA2_0, lambda2_1=LAMBDA2_1)
            model2._prescreen(rating)
            A2, B2, U2, V2, clusters2 = model2._mc_fit(
                rating, key=task_group, scheme="warm", epsilon=1e-5,
                maxiter=MAXITER, verbose=0,
                A_init=A, B_init=B,         # the first fit's grouping
                U_init=pred_group,
                V_init=V_spec,              # the SPECTRAL tiering
                clusters_init=clusters_spec,
                worker_active_mask=warm_worker_mask
            )
    
            cluster_acc2 = model2.task_acc(U2.astype(int), task_group)
            y_pred2 = model2._mc_infer_by_task(rating)

            pred_group2 = U2.astype(int)

            # Keep globally filtered LQ workers out of the second spectral pass.
            R_obs_warm = R_obs.copy()
            R_obs_warm[:, ~warm_worker_mask] = np.nan
            
            _, y_pred_e2, hq_workers_pred2, biased_workers_pred2, spectral = _hq_and_label_infer(
                pred_group2, R_obs_warm, y_true, worker_label,
                N_TASK, N_WORKER, N_TASK_GROUPS,
                LABEL_MODE="task", verbose=False,
                MIN_COVERAGE=0, return_spectral=True,
            )

                
            if not REAL_DATA:
                yt_tier, yp_tier = build_tier_vectors(
                    worker_label, hq_workers_pred2, biased_workers_pred2,
                    pred_group2, y_true, N_TASK_GROUPS)
            
                hq2 = [np.where(V2[:, g] == 1)[0] for g in range(N_TASK_GROUPS)]
                bi2 = [np.where(V2[:, g] == 2)[0] for g in range(N_TASK_GROUPS)]
                yt_lik2, yp_lik2 = build_tier_vectors(
                    worker_label, hq2, bi2, pred_group2, y_true, N_TASK_GROUPS)
                tier_lists["Likelihood2"]["true"].append(yt_lik2)
                tier_lists["Likelihood2"]["pred"].append(yp_lik2)
                
                # Eigen_L2_v2 keeps the spectral-pass vectors:
                tier_lists["Eigen_L2_v2"]["true"].append(yt_tier)
                tier_lists["Eigen_L2_v2"]["pred"].append(yp_tier)
        
            hq_vote_report(rating, pred_group2, hq_workers_pred2, N_TASK_GROUPS,
                   OUT_DIR, f"Eigen_L2_v2_run{i}",
                   y_true=y_true, draw=bool(DRAW_HQ_VOTES) and SAVE_RESULTS)
    
            metrics["Eigen_L2_v2"]["cluster_acc"].append(cluster_acc2)
            produced["Eigen_L2_v2"] = y_pred_e2
    
            metrics["Likelihood2"]["cluster_acc"].append(cluster_acc2)
            produced["Likelihood2"] = np.nan_to_num(y_pred2, nan=-1).astype(int)
            
            
            if not REAL_DATA:
                #B_true = np.transpose(worker_lf, (1, 0, 2))
                clusters_true = true_tier_centers(worker_lf, np.argmax(worker_label, axis=2))
                plot_worker_lf_pca(
                    [worker_lf, B, B2],
                    worker_tier_true=np.argmax(worker_label, axis=2),
                    clusters_list=[clusters_true, clusters_np, clusters2],
                    titles=("ground truth", "fit 1 (cold)", "fit 2 (warm)"),
                    path=out_path(f"worker_lf_pca_run{i}.png"),
                    draw=bool(DRAW_HQ_VOTES),
                )
            
            plot_loss_trajectory(
                model.loss_history, model2.loss_history,
                acc_cold=model.acc_history, acc_warm=model2.acc_history,
                path=out_path(f"loss_run{i}.png"),
                draw=bool(DRAW_HQ_VOTES),
                title=f"Objective trajectory (run {i})",
            )
    if ENABLE["Eigen_Oracle"]:
        # Upper bound on the spectral step: feed the ground-truth task grouping
        # so that any tiering/label error is attributable to the eigen-decomposition
        # alone, not to clustering error propagated from the LFGP fit.
        oracle_group = np.asarray(task_group, dtype=int)

        _, y_pred_or, hq_or, biased_or = _hq_and_label_infer(
            oracle_group, R_obs, y_true, worker_label,
            N_TASK, N_WORKER, N_TASK_GROUPS,
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
        y_pred, _ = peerA(rating, N_TASK_GROUPS, N_WORKER)._GLAD()
        produced["GLAD"] = y_pred

    if ENABLE["MultiSPA"]:
        y_pred = multispa_fit_predict(
            rating, K=N_TASK_GROUPS, assume_triplets=True).y_hat
        produced["MultiSPA"] = y_pred

    if ENABLE["GTIC"]:
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

        metrics["GTIC"]["cluster_acc"].append(cluster_acc)
        metrics["GTIC"]["cluster_ari"].append(cluster_ari)

        # Final GTIC class prediction
        produced["GTIC"] = gtic_result.y_hat
    if ENABLE["LFGP"]:
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

        metrics["LFGP"]["cluster_acc"].append(cluster_acc)
        metrics["LFGP"]["cluster_ari"].append(cluster_ari)

        # Final inferred labels
        y_pred = model_lfgp._mc_infer(rating)[:, 1].astype(int)
        produced["LFGP"] = y_pred
        
    if ENABLE["CBCC"]:
        res = cbcc(task = rating[:,0], worker= rating[:,1], label=rating[:,2], n_classes = N_TASK_GROUPS)
        produced["CBCC"] = res["pred"]
        
    # ---- evaluate everything the same way and record ----
    for name, y_pred in produced.items():
        scores = evaluate(y_true, y_pred, N_TASK_GROUPS)
        if len(metrics[name]["cluster_acc"]) > i:   # already recorded from U
            scores.pop("cluster_acc")
        for key, val in scores.items():
            metrics[name][key].append(val)

    # ---- persist after every run so a timeout can't lose finished runs ----
    if SAVE_RESULTS:
        save_json(out_path("metrics_raw.json"), metrics)
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

for _name in ("Eigen_L2", "Likelihood", "Likelihood2", "Eigen_L2_v2", "Eigen_Oracle"):
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