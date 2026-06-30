# -*- coding: utf-8 -*-
"""
diagnosis.py
============
Evaluation metrics for recovered task labels.

Primary entry point
-------------------
    accuracy, macro_f1 = diagnose(y_true, y_pred)

`y_pred` is the `task_label_pred` array returned by `_hq_and_label_infer`
(majority vote over the selected high-quality workers); `y_true` is the
ground-truth task label array.

Helpers
-------
    per_class_f1(...)      per-class F1 (exposes labels the biased cohort attacks)
    summarize_runs(...)    mean / sd / 95% CI over independent simulation runs
    diagnose_runs(...)     run diagnose() over many runs and summarize both metrics
"""

import numpy as np
from sklearn.metrics import f1_score


# --------------------------------------------------------------------------- #
#  core single-run metric
# --------------------------------------------------------------------------- #
def diagnose(y_true, y_pred, n_classes=None, count_unlabeled_as_wrong=True):
    """
    Task-label accuracy and macro-F1 for a single run.

    Parameters
    ----------
    y_true : array-like, shape (n_tasks,)
        Ground-truth task labels in {0, ..., C-1}.
    y_pred : array-like, shape (n_tasks,)
        Predicted task labels. Unassigned tasks may be marked -1 or NaN.
    n_classes : int, optional
        Number of label classes C. If None, inferred as max(y_true)+1.
    count_unlabeled_as_wrong : bool, default True
        If True, tasks whose prediction is -1/NaN are counted as incorrect,
        matching the pipeline's own `task_accuracy = mean(pred == label)`.
        If False, those tasks are dropped before scoring.

    Returns
    -------
    accuracy : float
    macro_f1 : float
    """
    y_true = np.asarray(y_true).ravel().astype(float)
    y_pred = np.asarray(y_pred).ravel().astype(float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    # treat NaN predictions as "unassigned" (-1)
    y_pred = np.where(np.isnan(y_pred), -1.0, y_pred)

    C = int(n_classes) if n_classes is not None else int(np.nanmax(y_true)) + 1
    labels = list(range(C))

    valid = ~np.isnan(y_true)                      # drop tasks with no ground truth
    if not count_unlabeled_as_wrong:
        valid &= (y_pred >= 0)                     # also drop unassigned predictions
    yt = y_true[valid].astype(int)
    yp = y_pred[valid].astype(int)

    if yt.size == 0:
        return float("nan"), float("nan")

    accuracy = float(np.mean(yp == yt))
    macro_f1 = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
    return accuracy, macro_f1


# --------------------------------------------------------------------------- #
#  optional helpers
# --------------------------------------------------------------------------- #
def per_class_f1(y_true, y_pred, n_classes=None):
    """Return an array of per-class F1 scores (length C)."""
    y_true = np.asarray(y_true).ravel().astype(float)
    y_pred = np.asarray(y_pred).ravel().astype(float)
    y_pred = np.where(np.isnan(y_pred), -1.0, y_pred)
    C = int(n_classes) if n_classes is not None else int(np.nanmax(y_true)) + 1
    valid = ~np.isnan(y_true)
    yt = y_true[valid].astype(int)
    yp = y_pred[valid].astype(int)
    return f1_score(yt, yp, labels=list(range(C)), average=None, zero_division=0)


def summarize_runs(values, ci=95):
    """
    Mean, sd, and bootstrap-style percentile CI over a list of per-run values.

    Returns a dict: {'mean', 'sd', 'ci_low', 'ci_high', 'n'}.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    lo = (100 - ci) / 2.0
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "ci_low": float(np.percentile(v, lo)),
        "ci_high": float(np.percentile(v, 100 - lo)),
        "n": int(v.size),
    }


def diagnose_runs(y_true_list, y_pred_list, n_classes=None,
                  count_unlabeled_as_wrong=True, ci=95):
    """
    Run diagnose() over many independent runs and summarize both metrics.

    Parameters
    ----------
    y_true_list, y_pred_list : list of array-like
        One (y_true, y_pred) pair per simulation run.

    Returns
    -------
    dict with keys 'accuracy' and 'macro_f1', each mapping to the
    summarize_runs() dict, plus 'per_run' holding the raw arrays.
    """
    if len(y_true_list) != len(y_pred_list):
        raise ValueError("y_true_list and y_pred_list must have the same length")
    accs, f1s = [], []
    for yt, yp in zip(y_true_list, y_pred_list):
        a, f = diagnose(yt, yp, n_classes=n_classes,
                        count_unlabeled_as_wrong=count_unlabeled_as_wrong)
        accs.append(a)
        f1s.append(f)
    return {
        "accuracy": summarize_runs(accs, ci=ci),
        "macro_f1": summarize_runs(f1s, ci=ci),
        "per_run": {"accuracy": np.array(accs), "macro_f1": np.array(f1s)},
    }


# --------------------------------------------------------------------------- #
#  demo / self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    C = 5

    # ---- single run ----
    y_true = rng.integers(0, C, size=200)
    y_pred = y_true.copy()
    flip = rng.random(200) < 0.15            # corrupt 15% of labels
    y_pred[flip] = rng.integers(0, C, size=flip.sum())
    y_pred[rng.random(200) < 0.05] = -1      # 5% left unassigned

    acc, mf1 = diagnose(y_true, y_pred, n_classes=C)
    print(f"single run  ->  accuracy = {acc:.3f},  macro-F1 = {mf1:.3f}")
    print("per-class F1:", np.round(per_class_f1(y_true, y_pred, C), 3))

    # ---- many runs (how you'd fill the paper table) ----
    yt_list, yp_list = [], []
    for _ in range(100):
        yt = rng.integers(0, C, size=200)
        yp = yt.copy()
        f = rng.random(200) < 0.15
        yp[f] = rng.integers(0, C, size=f.sum())
        yt_list.append(yt); yp_list.append(yp)

    summary = diagnose_runs(yt_list, yp_list, n_classes=C)
    a, m = summary["accuracy"], summary["macro_f1"]
    print(f"\nover {a['n']} runs:")
    print(f"  accuracy  = {a['mean']:.3f} +/- {a['sd']:.3f}  "
          f"95% CI [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
    print(f"  macro-F1  = {m['mean']:.3f} +/- {m['sd']:.3f}  "
          f"95% CI [{m['ci_low']:.3f}, {m['ci_high']:.3f}]")