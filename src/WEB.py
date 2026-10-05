import numpy as np
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def get_WEB(r=0.0, seed=None):
    raw_rating = []

    # ----------------------------------------
    # Read crowd labels
    # ----------------------------------------
    with open(os.path.join(_HERE, "dataset", "web_crowd.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item, worker, label = line.split()
            raw_rating.append((item, worker, int(label)-1))

    # ----------------------------------------
    # Read ground truth
    # ----------------------------------------
    truth_dict = {}

    with open(os.path.join(_HERE, "dataset", "web_truth.txt"), "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item, truth = line.split()
            truth_dict[item] = int(truth) - 1

    raw_rating, truth_dict = select_web_subset(
        raw_rating,
        truth_dict,
        tasks_per_class=200,
        seed=123
    )

    # ----------------------------------------
    # Keep only tasks with ground truth
    # ----------------------------------------
    raw_rating = [
        (item, worker, label)
        for item, worker, label in raw_rating
        if item in truth_dict
    ]

    # ----------------------------------------
    # Remap item and worker IDs to 0,1,2,...
    # ----------------------------------------
    item_ids = sorted(set(item for item, _, _ in raw_rating))
    worker_ids = sorted(set(worker for _, worker, _ in raw_rating))

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

    n_classes = int(max(
        rating[:, 2].max(),
        y_true.max()
    )) + 1

    # ----------------------------------------
    # Observation matrix
    # ----------------------------------------
    R_obs = np.full((n_task, n_worker), np.nan)
    R_obs[rating[:, 0], rating[:, 1]] = rating[:, 2]

    # ----------------------------------------
    # Add global random LQ workers
    # ----------------------------------------
    if r > 0:
        rng = np.random.default_rng(seed)

        n_worker_original = n_worker
        n_lq = int(round(r * n_worker_original))

        # Each synthetic LQ worker labels every task
        # uniformly at random over all classes
        lq_labels = rng.integers(
            0,
            n_classes,
            size=(n_task, n_lq)
        )

        tasks = np.repeat(np.arange(n_task), n_lq)

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
        f"WEB: {n_task} tasks, "
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

def select_web_subset(raw_rating, truth_dict, tasks_per_class=200, seed=123):
    rng = np.random.default_rng(seed)

    # Number of annotations for each task
    task_counts = {}
    for item, worker, label in raw_rating:
        task_counts[item] = task_counts.get(item, 0) + 1

    selected_items = []

    classes = sorted(set(truth_dict.values()))

    for c in classes:
        items_c = [
            item for item, truth in truth_dict.items()
            if truth == c and item in task_counts
        ]

        # Random tie-breaking
        random_order = {
            item: rng.random()
            for item in items_c
        }

        # Prefer better-covered tasks
        items_c = sorted(
            items_c,
            key=lambda item: (
                -task_counts[item],
                random_order[item]
            )
        )

        n = min(tasks_per_class, len(items_c))
        selected_items.extend(items_c[:n])

        print(
            f"Class {c}: selected {n} / {len(items_c)} tasks"
        )

    selected_items = set(selected_items)

    raw_rating_subset = [
        row for row in raw_rating
        if row[0] in selected_items
    ]

    truth_subset = {
        item: truth
        for item, truth in truth_dict.items()
        if item in selected_items
    }

    return raw_rating_subset, truth_subset