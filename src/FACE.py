# -*- coding: utf-8 -*-
"""
FACE.py -- loader for the Face Sentiment Identification dataset
(Mozafari et al. 2014; distributed with Zheng et al., VLDB 2017).

584 face images, 27 workers, 5,242 labels, 4 classes (balanced, 146 each).

Expected files (relative to the project root):
    data/face/face_answer.csv   columns: question, worker, answer
    data/face/face_truth.csv    columns: question, truth

Returns the same tuple as get_DOG():
    rating        (n_label, 3) int array: [task_idx, worker_idx, label], all 0-based
    y_true        (n_task,)    int array of ground-truth labels
    R_obs         (n_task, n_worker) float array, NaN where unobserved
    n_task, n_worker, n_classes
"""

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DIR = os.path.join(_HERE, "..", "data", "face")


def load_crowd_csv(answer_csv, truth_csv):
    """Generic loader for (question, worker, answer) + (question, truth) CSVs.
    Shared by get_FACE and get_BIRD."""
    ans = pd.read_csv(answer_csv)
    truth = pd.read_csv(truth_csv)

    # keep only tasks that have ground truth, drop duplicate (task, worker) pairs
    ans = ans[ans["question"].isin(truth["question"])]
    ans = ans.drop_duplicates(subset=["question", "worker"], keep="first")

    # map raw ids -> contiguous 0-based indices
    task_ids = np.sort(truth["question"].unique())
    task_map = {q: i for i, q in enumerate(task_ids)}
    worker_ids = np.sort(ans["worker"].astype(str).unique())   # worker ids are strings
    worker_map = {w: j for j, w in enumerate(worker_ids)}

    # labels -> contiguous 0..K-1 (Face is already 0..3; this is just a safeguard)
    classes = np.sort(pd.unique(pd.concat([ans["answer"], truth["truth"]])))
    label_map = {c: k for k, c in enumerate(classes)}

    n_task, n_worker, n_classes = len(task_ids), len(worker_ids), len(classes)

    rating = np.column_stack([
        ans["question"].map(task_map).to_numpy(),
        ans["worker"].astype(str).map(worker_map).to_numpy(),
        ans["answer"].map(label_map).to_numpy(),
    ]).astype(int)

    y_true = np.empty(n_task, dtype=int)
    y_true[truth["question"].map(task_map).to_numpy()] = truth["truth"].map(label_map).to_numpy()

    R_obs = np.full((n_task, n_worker), np.nan)
    R_obs[rating[:, 0], rating[:, 1]] = rating[:, 2]

    return rating, y_true, R_obs, n_task, n_worker, n_classes


def get_FACE(data_dir=_DEFAULT_DIR):
    return load_crowd_csv(os.path.join(data_dir, "face_answer.csv"),
                          os.path.join(data_dir, "face_truth.csv"))


if __name__ == "__main__":
    rating, y_true, R_obs, n_task, n_worker, K = get_FACE()
    print(f"tasks={n_task} workers={n_worker} labels={len(rating)} classes={K}")
    print("class counts:", np.bincount(y_true, minlength=K))
    obs = (~np.isnan(R_obs)).astype(int)
    ov = obs.T @ obs
    iu = np.triu_indices(n_worker, 1)
    print(f"pairwise overlap: median={np.median(ov[iu]):.0f}, mean={ov[iu].mean():.1f}")
