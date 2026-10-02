# -*- coding: utf-8 -*-
"""
diagnosis.py
============
Evaluation metrics for (a) recovered task LABELS and (b) recovered WORKER TIERS.

Label metrics
-------------
    accuracy, macro_f1 = diagnose(y_true, y_pred)
    per_class_f1(...) , diagnose_runs(...)

Worker-tier metrics (HQ / biased / LQ)
--------------------------------------
    res = worker_diagnose(y_true_tier, y_pred_tier)      # one run
    worker_diagnose_runs(true_list, pred_list)           # aggregate over runs
    build_tier_vectors(...)                              # build the tier vectors
                                                         #   from pipeline outputs
    plot_tier_confusion(cm, path=...)                    # confusion-matrix heatmap

Tier coding (matches eigenInfer.py): 0 = LQ, 1 = HQ, 2 = biased.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, confusion_matrix, precision_recall_fscore_support,
    adjusted_rand_score,
    accuracy_score,
)
from scipy.optimize import linear_sum_assignment

TIER_NAMES = ("LQ", "HQ", "Biased")   # index = tier code


# ========================================================================== #
#  PART A — task-label metrics
# ========================================================================== #
def diagnose(y_true, y_pred, n_classes=None, count_unlabeled_as_wrong=True):
    """Task-label accuracy and macro-F1 for a single run. Returns (acc, macro_f1)."""
    y_true = np.asarray(y_true).ravel().astype(float)
    y_pred = np.asarray(y_pred).ravel().astype(float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    y_pred = np.where(np.isnan(y_pred), -1.0, y_pred)
    C = int(n_classes) if n_classes is not None else int(np.nanmax(y_true)) + 1
    labels = list(range(C))
    valid = ~np.isnan(y_true)
    if not count_unlabeled_as_wrong:
        valid &= (y_pred >= 0)
    yt, yp = y_true[valid].astype(int), y_pred[valid].astype(int)
    if yt.size == 0:
        return float("nan"), float("nan")
    acc = float(np.mean(yp == yt))
    mf1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    return acc, mf1


def per_class_f1(y_true, y_pred, n_classes=None):
    """Per-class F1 array (length C)."""
    y_true = np.asarray(y_true).ravel().astype(float)
    y_pred = np.asarray(y_pred).ravel().astype(float)
    y_pred = np.where(np.isnan(y_pred), -1.0, y_pred)
    C = int(n_classes) if n_classes is not None else int(np.nanmax(y_true)) + 1
    valid = ~np.isnan(y_true)
    return f1_score(y_true[valid].astype(int), y_pred[valid].astype(int),
                    labels=list(range(C)), average=None, zero_division=0)


def summarize_runs(values, ci=95):
    """Mean, sd, and percentile CI over a list of per-run values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    lo = (100 - ci) / 2.0
    return {"mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "ci_low": float(np.percentile(v, lo)),
            "ci_high": float(np.percentile(v, 100 - lo)),
            "n": int(v.size)}


def diagnose_runs(y_true_list, y_pred_list, n_classes=None,
                  count_unlabeled_as_wrong=True, ci=95):
    """Run diagnose() over many runs; summarize accuracy and macro-F1."""
    accs, f1s = [], []
    for yt, yp in zip(y_true_list, y_pred_list):
        a, f = diagnose(yt, yp, n_classes=n_classes,
                        count_unlabeled_as_wrong=count_unlabeled_as_wrong)
        accs.append(a); f1s.append(f)
    return {"accuracy": summarize_runs(accs, ci=ci),
            "macro_f1": summarize_runs(f1s, ci=ci),
            "per_run": {"accuracy": np.array(accs), "macro_f1": np.array(f1s)}}


# ========================================================================== #
#  PART B — worker-tier classification metrics (HQ / biased / LQ)
# ========================================================================== #
def build_tier_vectors(
    worker_label,
    hq_workers_pred,
    biased_workers_pred,
    pred_group,
    label,
    n_task_groups,
    truth_argmax_to_tier=None,
):
    n_worker = worker_label.shape[0]

    V_true = np.argmax(
        worker_label,
        axis=2,
    )

    if truth_argmax_to_tier is not None:
        V_true = np.vectorize(
            truth_argmax_to_tier.get
        )(V_true)

    # predicted-group -> true-group one-to-one mapping
    group_map = task_group_mapping(
        pred_group,
        label,
        n_task_groups,
    )

    V_pred = np.zeros(
        (n_worker, n_task_groups),
        dtype=int,
    )

    for g_pred in range(n_task_groups):

        g_true = group_map[g_pred]

        # Skip unmatched group defensively
        if g_true < 0:
            continue

        if hq_workers_pred[g_pred] is not None:
            V_pred[
                hq_workers_pred[g_pred],
                g_true,
            ] = 1

        if biased_workers_pred[g_pred] is not None:
            V_pred[
                biased_workers_pred[g_pred],
                g_true,
            ] = 2

    return (
        V_true.ravel().astype(int),
        V_pred.ravel().astype(int),
    )


def worker_diagnose(y_true_tier, y_pred_tier, tier_names=TIER_NAMES):
    """
    Tier-classification metrics for a single run.

    Returns a dict with:
      accuracy, macro_f1, ari
      precision/recall/f1  : dict keyed by tier name
      biased_as_hq         : P(predicted HQ | true biased)   <-- the critical error
      hq_as_biased         : P(predicted biased | true HQ)
      confusion            : 3x3 count matrix (rows=true, cols=pred), labels 0,1,2
    """
    labels = list(range(len(tier_names)))
    yt = np.asarray(y_true_tier).ravel().astype(int)
    yp = np.asarray(y_pred_tier).ravel().astype(int)

    cm = confusion_matrix(yt, yp, labels=labels)
    acc = float(np.mean(yt == yp))
    macro_f1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0)
    ari = float(adjusted_rand_score(yt, yp))

    def rate(true_c, pred_c):
        denom = cm[true_c, :].sum()
        return float(cm[true_c, pred_c] / denom) if denom > 0 else float("nan")

    HQ, BIAS = 1, 2
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ari": ari,
        "precision": dict(zip(tier_names, p.astype(float))),
        "recall":    dict(zip(tier_names, r.astype(float))),
        "f1":        dict(zip(tier_names, f.astype(float))),
        "biased_as_hq": rate(BIAS, HQ),
        "hq_as_biased": rate(HQ, BIAS),
        "confusion": cm,
    }


def worker_diagnose_runs(true_list, pred_list, tier_names=TIER_NAMES, ci=95):
    """
    Aggregate worker_diagnose over runs. Returns scalar summaries (mean/sd/CI),
    per-tier P/R/F1 summaries, and the summed + row-normalized confusion matrix
    (the latter ready for plot_tier_confusion).
    """
    scalar_keys = ["accuracy", "macro_f1", "ari", "biased_as_hq", "hq_as_biased"]
    scalars = {k: [] for k in scalar_keys}
    perf = {n: {"precision": [], "recall": [], "f1": []} for n in tier_names}
    cms = []
    for yt, yp in zip(true_list, pred_list):
        d = worker_diagnose(yt, yp, tier_names=tier_names)
        cms.append(d["confusion"])
        for k in scalar_keys:
            scalars[k].append(d[k])
        for n in tier_names:
            for m in ("precision", "recall", "f1"):
                perf[n][m].append(d[m][n])

    cm_sum = np.sum(cms, axis=0).astype(float)
    row = cm_sum.sum(axis=1, keepdims=True)
    cm_rownorm = np.divide(cm_sum, row, out=np.zeros_like(cm_sum), where=row > 0)

    return {
        "summary":   {k: summarize_runs(scalars[k], ci=ci) for k in scalar_keys},
        "per_tier":  {n: {m: summarize_runs(perf[n][m], ci=ci)
                          for m in ("precision", "recall", "f1")} for n in tier_names},
        "confusion_sum": cm_sum,
        "confusion_rownorm": cm_rownorm,
    }


def plot_tier_confusion(cm, tier_names=TIER_NAMES, normalize="true",
                        path="worker_confusion.png", title=None, annot_counts=None):
    """
    Save a confusion-matrix heatmap. `cm` may be counts or already row-normalized.

    normalize : 'true' row-normalizes counts to recall (each row sums to 1);
                None plots cm as-is.
    annot_counts : optional 3x3 count matrix to annotate alongside the rate
                   (e.g. pass the summed counts when cm is row-normalized).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=float)
    if normalize == "true":
        row = cm.sum(axis=1, keepdims=True)
        cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        vmax = 1.0
    else:
        cmn = cm
        vmax = None

    if annot_counts is not None:
        annot = np.empty(cmn.shape, dtype=object)
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                annot[i, j] = f"{cmn[i, j]:.2f}\n({int(annot_counts[i, j])})"
        fmt = ""
    else:
        annot, fmt = True, ".2f"

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cmn, annot=annot, fmt=fmt, cmap="Blues", vmin=0, vmax=vmax,
                xticklabels=tier_names, yticklabels=tier_names, cbar=True, ax=ax)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("True tier")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path

def build_worker_summary(agg):
    """Tidy numeric summary from worker_diagnose_runs(...), ready for CSV."""
    rows = []
    def add(section, name, metric, s):
        rows.append({"section": section, "name": name, "metric": metric,
                    "mean": round(s["mean"], 4), "sd": round(s["sd"], 4),
                    "ci_low": round(s["ci_low"], 4), "ci_high": round(s["ci_high"], 4),
                    "n": s["n"]})
    for tier, md in agg["per_tier"].items():           # per-tier P/R/F1
        add("worker_tier", tier, "Precision", md["precision"])
        add("worker_tier", tier, "Recall",    md["recall"])
        add("worker_tier", tier, "F1",        md["f1"])
    for metric, s in agg["summary"].items():           # accuracy, macro_f1, ari, biased_as_hq, hq_as_biased
        add("worker_overall", "Overall", metric, s)
    return pd.DataFrame(rows, columns=["section","name","metric","mean","sd","ci_low","ci_high","n"])

def print_spectral_worker_comparison(spectral, worker_label, pred_group, y_true):
    """
    Compare direct KMeans, two-stage Agg->Agg, and hybrid Agg->KMeans
    worker-tier recovery on the same spectral embedding.
    """

    V_true = np.argmax(worker_label, axis=2).astype(int)
    V_km = spectral["tier_kmeans"].astype(int)
    V_agg = spectral["tier_agg"].astype(int)
    V_hybrid = spectral["tier_hybrid"].astype(int)
    tested = spectral["tested_mask"]

    n_groups = V_true.shape[1]
    group_map = task_group_mapping(pred_group, y_true, n_groups)

    print("\n" + "=" * 125)
    print("SPECTRAL WORKER GROUPING: KMEANS vs AGG->AGG vs AGG->KMEANS")
    print("=" * 125)

    print(
        f"{'Pred->True':<10}"
        f"{'KM Acc':>10}"
        f"{'AGG Acc':>10}"
        f"{'HYB Acc':>10}"
        f"{'KM F1':>10}"
        f"{'AGG F1':>10}"
        f"{'HYB F1':>10}"
        f"{'KM ARI':>10}"
        f"{'AGG ARI':>10}"
        f"{'HYB ARI':>10}"
        f"{'N':>8}"
    )
    print("-" * 125)

    all_true = []
    km_all_pred = []
    agg_all_pred = []
    hyb_all_pred = []

    for g_pred in range(n_groups):
        g_true = group_map[g_pred]
        mask = tested[:, g_pred]

        if mask.sum() == 0:
            continue

        yt = V_true[mask, g_true]
        ykm = V_km[mask, g_pred]
        yagg = V_agg[mask, g_pred]
        yhyb = V_hybrid[mask, g_pred]

        km_acc = accuracy_score(yt, ykm)
        agg_acc = accuracy_score(yt, yagg)
        hyb_acc = accuracy_score(yt, yhyb)

        km_f1 = f1_score(yt, ykm, labels=[0, 1, 2], average="macro", zero_division=0)
        agg_f1 = f1_score(yt, yagg, labels=[0, 1, 2], average="macro", zero_division=0)
        hyb_f1 = f1_score(yt, yhyb, labels=[0, 1, 2], average="macro", zero_division=0)

        km_ari = adjusted_rand_score(yt, ykm)
        agg_ari = adjusted_rand_score(yt, yagg)
        hyb_ari = adjusted_rand_score(yt, yhyb)

        print(
            f"{g_pred}->{g_true:<7}"
            f"{km_acc:>10.3f}"
            f"{agg_acc:>10.3f}"
            f"{hyb_acc:>10.3f}"
            f"{km_f1:>10.3f}"
            f"{agg_f1:>10.3f}"
            f"{hyb_f1:>10.3f}"
            f"{km_ari:>10.3f}"
            f"{agg_ari:>10.3f}"
            f"{hyb_ari:>10.3f}"
            f"{mask.sum():>8d}"
        )

        all_true.append(yt)
        km_all_pred.append(ykm)
        agg_all_pred.append(yagg)
        hyb_all_pred.append(yhyb)

    if not all_true:
        print("-" * 125)
        print("No tested worker-group pairs.")
        print("=" * 125)
        return

    yt = np.concatenate(all_true)
    ykm = np.concatenate(km_all_pred)
    yagg = np.concatenate(agg_all_pred)
    yhyb = np.concatenate(hyb_all_pred)

    km_acc = accuracy_score(yt, ykm)
    agg_acc = accuracy_score(yt, yagg)
    hyb_acc = accuracy_score(yt, yhyb)

    km_f1 = f1_score(yt, ykm, labels=[0, 1, 2], average="macro", zero_division=0)
    agg_f1 = f1_score(yt, yagg, labels=[0, 1, 2], average="macro", zero_division=0)
    hyb_f1 = f1_score(yt, yhyb, labels=[0, 1, 2], average="macro", zero_division=0)

    km_ari = adjusted_rand_score(yt, ykm)
    agg_ari = adjusted_rand_score(yt, yagg)
    hyb_ari = adjusted_rand_score(yt, yhyb)

    print("-" * 125)
    print(
        f"{'Overall':<10}"
        f"{km_acc:>10.3f}"
        f"{agg_acc:>10.3f}"
        f"{hyb_acc:>10.3f}"
        f"{km_f1:>10.3f}"
        f"{agg_f1:>10.3f}"
        f"{hyb_f1:>10.3f}"
        f"{km_ari:>10.3f}"
        f"{agg_ari:>10.3f}"
        f"{hyb_ari:>10.3f}"
        f"{len(yt):>8d}"
    )
    print("=" * 125)
    
def task_group_mapping(pred_group, y_true, n_groups):
    pred_group = np.asarray(pred_group).astype(int)
    y_true = np.asarray(y_true).astype(int)

    counts = np.zeros(
        (n_groups, n_groups),
        dtype=int,
    )

    # rows = predicted group
    # cols = true group
    np.add.at(
        counts,
        (pred_group, y_true),
        1,
    )

    row_ind, col_ind = linear_sum_assignment(-counts)

    mapping = np.full(
        n_groups,
        -1,
        dtype=int,
    )

    mapping[row_ind] = col_ind

    return mapping
