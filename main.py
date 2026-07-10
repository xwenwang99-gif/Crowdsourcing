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

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import balanced_accuracy_score

from src.lfgp_withoutO_biased import LFGP
from src.GTIC import gtic
from src.multispa import multispa_fit_predict
from src.getdata_biased import getdata_biased
from src.eigenInfer import _hq_and_label_infer
from src.Diagnosis import diagnose, summarize_runs,build_tier_vectors, worker_diagnose, worker_diagnose_runs, plot_tier_confusion,build_worker_summary
from src.peera import peerA


warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #
N_RUNS        = 1
MAXITER       = 100
N_TASK        = 200
N_WORKER      = 200
N_TASK_GROUPS = 10

# which methods to run (replaces the eigen_ex / DS_ex / ... flags)
ENABLE = {
    "Eigen_L2":   1,   # LFGP fit + spectral worker tiering
    "Likelihood": 1,   # same LFGP fit, labels via _mc_infer_by_task (no spectral step)
    "DS":       1,
    "MV_HQ":    0,
    "MV":       0,
    "GLAD":     0,
    "MultiSPA": 0,
    "GTIC":     0,
}

# biased-center scheme for the LFGP worker KMeans / penalty (toggle — one
# LFGP fit with the selected scheme). Both identify worker groups by
# center-norm magnitude (smallest=LQ, largest=HQ, middle=biased):
#   "free2" — two free centers + one center fixed at the origin (LQ)
#   "free3" — all three centers free
BIAS_SCHEME = "free2"

METHODS = [m for m, on in ENABLE.items() if on]

DATA_KW = dict(                       # getdata_biased arguments, kept in one place
    n_task=N_TASK, n_worker=N_WORKER, n_task_groups=N_TASK_GROUPS,
    k=3, sigma=1.0, obs_prob=1, hq_ratio=1/3, bias_ratio=1/2,
    delta=1, n_classes=N_TASK_GROUPS,
)

# --------------------------------------------------------------------------- #
#  output directory (persists on disk across idle timeout)
# --------------------------------------------------------------------------- #
RUN_ID  = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("results", f"run_{RUN_ID}")
os.makedirs(OUT_DIR, exist_ok=True)


def _to_native(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_native)


# --------------------------------------------------------------------------- #
#  shared evaluation: every method just hands us a predicted-label vector
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred, n_classes):
    acc, macro_f1 = diagnose(y_true, y_pred, n_classes=n_classes)
    bal = balanced_accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": macro_f1, "bal_acc": bal}


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
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
#  metric store:  metrics[method][metric] -> list over runs
# --------------------------------------------------------------------------- #
metrics = {m: {"accuracy": [], "macro_f1": [], "bal_acc": []} for m in METHODS}
# the LFGP-based methods additionally report the task-grouping (cluster) accuracy
for _name in ("Eigen_L2", "Likelihood"):
    if _name in metrics:
        metrics[_name]["cluster_acc"] = []

start = time.perf_counter()

# worker-tier vectors for Eigen_L2:  tier_lists["true"/"pred"] -> list over runs
tier_lists = {"true": [], "pred": []}
for i in range(N_RUNS):
    np.random.seed(i)

    rating, y_true, worker_label, R_obs, task_lf, worker_lf = getdata_biased(**DATA_KW)

    produced = {}   # method name -> predicted label vector for this run

    # LFGP model is needed by Eigen_L2, Likelihood, and DS
    model = None
    if ENABLE["Eigen_L2"] or ENABLE["Likelihood"] or ENABLE["DS"]:
        model = LFGP(lf_dim=N_TASK_GROUPS, n_worker_group=N_TASK_GROUPS,
                     lambda1=1, lambda2_0=1, lambda2_1=2)
        model._prescreen(rating)

    # one LFGP fit shared by the likelihood-only and spectral methods
    if ENABLE["Eigen_L2"] or ENABLE["Likelihood"]:
        A, B, U, V, clusters_np = model._mc_fit(
            rating, key=y_true, scheme="ds", epsilon=1e-5,
            maxiter=MAXITER, verbose=1, A=task_lf, B=worker_lf,
            bias_scheme=BIAS_SCHEME,
        )
        pred_group = U.astype(int)

        cluster_acc = model.task_acc(pred_group, y_true)

    if ENABLE["Likelihood"]:
        # likelihood step only: majority vote of the fit's own HQ workers (V)
        metrics["Likelihood"]["cluster_acc"].append(cluster_acc)
        produced["Likelihood"] = model._mc_infer(rating)

    if ENABLE["Eigen_L2"]:
        _, y_pred, hq_workers_pred, biased_workers_pred = _hq_and_label_infer(
            pred_group, R_obs, y_true, worker_label,
            N_TASK, N_WORKER, N_TASK_GROUPS,
            LABEL_MODE="group", verbose=False,
            MIN_COVERAGE=5, return_spectral=False,
        )

        yt_tier, yp_tier = build_tier_vectors(
            worker_label, hq_workers_pred, biased_workers_pred,
            pred_group, y_true, N_TASK_GROUPS)
        tier_lists["true"].append(yt_tier)
        tier_lists["pred"].append(yp_tier)

        metrics["Eigen_L2"]["cluster_acc"].append(cluster_acc)
        produced["Eigen_L2"] = y_pred
    if ENABLE["DS"]:
        y_pred = model._init_task_member_ds(rating)[:, 1]
        produced["DS"] = y_pred

    if ENABLE["MV_HQ"]:
        y_pred = np.full(N_TASK, -1)
        for t in range(N_TASK):
            hq = np.where(worker_label[:, y_true[t]] == 1)[0]
            labs = rating[np.isin(rating[:, 1], hq) & (rating[:, 0] == t)][:, 2]
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
        y_pred = gtic(
            rating, n=N_TASK, m=N_WORKER, K=N_TASK_GROUPS, missing_val=-1).y_hat
        produced["GTIC"] = y_pred

    # ---- evaluate everything the same way and record ----
    for name, y_pred in produced.items():
        scores = evaluate(y_true, y_pred, N_TASK_GROUPS)
        for key, val in scores.items():
            metrics[name][key].append(val)

    # ---- persist after every run so a timeout can't lose finished runs ----
    save_json(os.path.join(OUT_DIR, "metrics_raw.json"), metrics)
    print(f"[run {i + 1}/{N_RUNS}] done; results saved to {OUT_DIR}")

# --------------------------------------------------------------------------- #
#  summarize, print, and save
# --------------------------------------------------------------------------- #
summary_df = build_summary(metrics)
pd.set_option("display.width", 200, "display.max_columns", None)
print("\n==== SUMMARY ====")
print(summary_df)

summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"))
save_json(os.path.join(OUT_DIR, "summary.json"), summary_df.to_dict(orient="index"))
save_json(os.path.join(OUT_DIR, "config.json"),
          {"run_id": RUN_ID, "n_runs": N_RUNS, "maxiter": MAXITER,
           "enable": ENABLE, "bias_scheme": BIAS_SCHEME, "data_kw": DATA_KW})

if ENABLE["Eigen_L2"]:
    worker_agg = worker_diagnose_runs(tier_lists["true"], tier_lists["pred"])
    plot_tier_confusion(                           # the heatmap
        worker_agg["confusion_rownorm"], annot_counts=worker_agg["confusion_sum"],
        path=os.path.join(OUT_DIR, "worker_confusion_Eigen_L2.png"),
        title=f"Worker-tier recovery (Eigen_L2, {BIAS_SCHEME})")

    build_worker_summary(worker_agg).to_csv(
        os.path.join(OUT_DIR, "summary_worker_Eigen_L2.csv"), index=False)



runtime = time.perf_counter() - start
print(f"\nRuntime: {runtime:.1f}s   |   all outputs in: {OUT_DIR}")