# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DIR = os.path.join(_HERE, "dataset")


def get_CF(data_dir=_DEFAULT_DIR, r=0.0, seed=None):
    # ----------------------------------------
    # Read CF_amt.csv
    # Columns:
    # worker, task, observed label, truth, time
    # ----------------------------------------
    path = os.path.join(data_dir, "CF_amt.csv")

    df = pd.read_csv(
        path,
        header=None,
        names=["worker", "task", "answer", "truth", "time"]
    )

    # One response per worker-task pair
    df = df.drop_duplicates(
        subset=["task", "worker"],
        keep="first"
    )

    # ----------------------------------------
    # Remap task and worker IDs to 0,1,2,...
    # ----------------------------------------
    task_ids = np.sort(df["task"].unique())
    worker_ids = np.sort(df["worker"].astype(str).unique())

    task_map = {
        task: i
        for i, task in enumerate(task_ids)
    }

    worker_map = {
        worker: j
        for j, worker in enumerate(worker_ids)
    }

    # ----------------------------------------
    # rating = [task, worker, observed label]
    # ----------------------------------------
    rating = np.column_stack([
        df["task"].map(task_map).to_numpy(),
        df["worker"].astype(str).map(worker_map).to_numpy(),
        df["answer"].to_numpy(),
    ]).astype(int)

    # ----------------------------------------
    # Ground truth
    # ----------------------------------------
    truth_df = (
        df[["task", "truth"]]
        .drop_duplicates(subset=["task"])
        .set_index("task")
    )

    y_true = np.array([
        int(truth_df.loc[task, "truth"])
        for task in task_ids
    ], dtype=int)

    n_task = len(task_ids)
    n_worker = len(worker_ids)

    n_classes = int(max(
        rating[:, 2].max(),
        y_true.max()
    )) + 1

    # ----------------------------------------
    # Observation matrix
    # ----------------------------------------
    R_obs = np.full(
        (n_task, n_worker),
        np.nan
    )

    R_obs[
        rating[:, 0],
        rating[:, 1]
    ] = rating[:, 2]

    # ----------------------------------------
    # Add global random LQ workers
    # ----------------------------------------
    if r > 0:
        rng = np.random.default_rng(seed)

        n_worker_original = n_worker
        n_lq = int(round(r * n_worker_original))

        # Each synthetic LQ worker labels every task
        # uniformly at random over the 5 classes
        lq_labels = rng.integers(
            0,
            n_classes,
            size=(n_task, n_lq)
        )

        tasks = np.repeat(
            np.arange(n_task),
            n_lq
        )

        workers = np.tile(
            np.arange(
                n_worker_original,
                n_worker_original + n_lq
            ),
            n_task
        )

        labels = lq_labels.ravel()

        lq_rating = np.column_stack([
            tasks,
            workers,
            labels
        ])

        rating = np.vstack([
            rating,
            lq_rating
        ])

        R_obs = np.hstack([
            R_obs,
            lq_labels.astype(float)
        ])

        n_worker += n_lq

    # ----------------------------------------
    # Diagnostics
    # ----------------------------------------
    print(
        f"CF: {n_task} tasks, "
        f"{n_worker} workers, "
        f"{len(rating)} annotations, "
        f"{n_classes} classes"
    )

    return (
        rating,
        y_true,
        y_true,
        R_obs,
        n_task,
        n_worker,
        n_classes
    )