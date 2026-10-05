import numpy as np
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def get_DOG(r=0.0, seed=None):
    raw_rating = []

    with open(os.path.join(_HERE, "dataset", "dogs-1.merged.label.tsv"), "r") as f:
        for line in f:
            item, worker, label = line.split()
            raw_rating.append((item, worker, int(label)))

    # ----------------------------------------
    # Read ground truth
    # ----------------------------------------
    truth_dict = {}

    with open(os.path.join(_HERE, "dataset", "dogs-1.merged.truth.tsv"), "r") as f:
        for line in f:
            item, truth = line.split()
            truth_dict[item] = int(truth)

    # ----------------------------------------
    # Remap item and worker IDs to 0,1,2,...
    # ----------------------------------------
    item_ids = sorted(set(x[0] for x in raw_rating))
    worker_ids = sorted(set(x[1] for x in raw_rating))

    item_map = {item: i for i, item in enumerate(item_ids)}
    worker_map = {worker: j for j, worker in enumerate(worker_ids)}

    # ----------------------------------------
    # rating = [task, worker, observed label]
    # ----------------------------------------
    rating = np.array([
        [item_map[item], worker_map[worker], label]
        for item, worker, label in raw_rating
    ], dtype=int)

    # ----------------------------------------
    # y_true[i] = true label for task i
    # ----------------------------------------
    y_true = np.array([
        truth_dict[item]
        for item in item_ids
    ], dtype=int)

    n_task = len(item_ids)
    n_worker = len(worker_ids)
    n_classes = int(rating[:, 2].max()) + 1

    R_obs = np.full((n_task, n_worker), np.nan)
    R_obs[rating[:, 0], rating[:, 1]] = rating[:, 2]

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

    return rating, y_true, y_true, R_obs, n_task, n_worker, n_classes