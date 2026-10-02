# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DIR = os.path.join(_HERE, "dataset")


def load_crowd_csv(answer_csv, truth_csv):
    """Generic loader for (question, worker, answer) + (question, truth) CSVs."""

    ans = pd.read_csv(answer_csv)
    truth = pd.read_csv(truth_csv)

    # Keep only tasks that have ground truth
    ans = ans[ans["question"].isin(truth["question"])]
    ans = ans.drop_duplicates(subset=["question", "worker"], keep="first")

    # Map raw IDs -> contiguous 0-based indices
    task_ids = np.sort(truth["question"].unique())
    task_map = {q: i for i, q in enumerate(task_ids)}

    worker_ids = np.sort(ans["worker"].astype(str).unique())
    worker_map = {w: j for j, w in enumerate(worker_ids)}

    # Labels -> contiguous 0,...,K-1
    classes = np.sort(ans["answer"].unique())
    label_map = {c: k for k, c in enumerate(classes)}

    n_task = len(task_ids)
    n_worker = len(worker_ids)
    n_classes = len(classes)

    rating = np.column_stack([
        ans["question"].map(task_map).to_numpy(),
        ans["worker"].astype(str).map(worker_map).to_numpy(),
        ans["answer"].map(label_map).to_numpy(),
    ]).astype(int)

    y_true = np.empty(n_task, dtype=int)
    y_true[truth["question"].map(task_map).to_numpy()] = (
        truth["truth"].map(label_map).to_numpy()
    )

    R_obs = np.full((n_task, n_worker), np.nan)
    R_obs[rating[:, 0], rating[:, 1]] = rating[:, 2]

    return rating, y_true, R_obs, n_task, n_worker, n_classes


def get_FACE(data_dir=_DEFAULT_DIR, r=0.0, seed=None):
    rating, y_true, R_obs, n_task, n_worker, n_classes = load_crowd_csv(
        os.path.join(data_dir, "face_answer.csv"),
        os.path.join(data_dir, "face_truth.csv")
    )

    # ----------------------------------------
    # Add global random LQ workers
    # ----------------------------------------
    if r > 0:
        rng = np.random.default_rng(seed)
        n_lq = int(round(r * n_worker))

        # One random label for every task × synthetic worker
        lq_labels = rng.integers(0, n_classes, size=(n_task, n_lq))

        # Construct long-format rows: [task, worker, label]
        tasks = np.repeat(np.arange(n_task), n_lq)
        workers = np.tile(np.arange(n_worker, n_worker + n_lq), n_task)
        labels = lq_labels.ravel()

        lq_rating = np.column_stack([tasks, workers, labels])
        rating = np.vstack([rating, lq_rating])

        # Append actual labels to R_obs
        R_obs = np.hstack([R_obs, lq_labels.astype(float)])

        n_worker += n_lq

    return rating, y_true, R_obs, n_task, n_worker, n_classes