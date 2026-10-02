# -*- coding: utf-8 -*-
"""
BIRD.py -- Bluebirds loader with optional global LQ workers.

r controls the number of synthetic random workers relative to the
number of original workers.

Examples:
    r = 0.0 -> 0 added workers
    r = 0.5 -> ~20 added workers
    r = 1.0 -> 39 added workers
    r = 2.0 -> 78 added workers

Each synthetic LQ worker labels every task uniformly at random.
"""

import os
import numpy as np
from src.FACE import load_crowd_csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DIR = os.path.join(_HERE, "dataset")


def get_BIRD(data_dir=_DEFAULT_DIR, r=0.0, seed=None):
    rating, y_true, R_obs, n_task, n_worker, n_classes = load_crowd_csv(
        os.path.join(data_dir, "bird_answer.csv"),
        os.path.join(data_dir, "bird_truth.csv")
    )

    if r <= 0:
        return rating, y_true, R_obs, n_task, n_worker, n_classes

    rng = np.random.default_rng(seed)

    # Number of synthetic global-LQ workers
    n_lq = int(round(r * n_worker))

    # Original worker IDs are assumed to be 0, ..., n_worker-1
    # Synthetic workers therefore start at n_worker
    lq_rows = []

    for worker_id in range(n_worker, n_worker + n_lq):
        # Random label for every task
        labels = rng.integers(0, n_classes, size=n_task)

        for task_id in range(n_task):
            lq_rows.append([task_id, worker_id, labels[task_id]])

    lq_rows = np.asarray(lq_rows, dtype=rating.dtype)

    # rating: [task_id, worker_id, label]
    rating = np.vstack([rating, lq_rows])

    # R_obs: n_task x n_worker
    # Synthetic workers label every task, so append columns of 1s
    lq_obs = np.ones((n_task, n_lq), dtype=R_obs.dtype)
    R_obs = np.hstack([R_obs, lq_obs])

    n_worker_new = n_worker + n_lq


    return rating, y_true, R_obs, n_task, n_worker_new, n_classes