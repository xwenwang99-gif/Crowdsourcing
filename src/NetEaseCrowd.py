import numpy as np
import pandas as pd
from datasets import load_dataset


def get_NETEASE(
    r=0,
    tasks_per_cap=300,
    min_ann_per_worker=30,
    min_workers_per_task=5,
    min_final_ann_per_worker=25,
    seed=123
):

    # Independent RNG streams for this replication
    ss = np.random.SeedSequence(seed)
    sample_ss, lq_ss = ss.spawn(2)

    sample_rng = np.random.default_rng(sample_ss)
    lq_rng = np.random.default_rng(lq_ss)

    # --------------------------------------------------
    # Load raw NetEaseCrowd data
    # --------------------------------------------------
    ds = load_dataset("liuhyuu/NetEaseCrowd", split="train")
    df = ds.to_pandas()

    subset = make_netease_subset(
        df,
        tasks_per_cap=tasks_per_cap,
        min_ann_per_worker=min_ann_per_worker,
        min_workers_per_task=min_workers_per_task,
        min_final_ann_per_worker=min_final_ann_per_worker,
        rng=sample_rng
    )

    # --------------------------------------------------
    # Remap task and worker IDs to 0, 1, 2, ...
    # --------------------------------------------------
    task_ids = sorted(subset["taskId"].unique())
    worker_ids = sorted(subset["workerId"].unique())

    task_map = {task: i for i, task in enumerate(task_ids)}
    worker_map = {worker: j for j, worker in enumerate(worker_ids)}

    # --------------------------------------------------
    # rating = [task, worker, observed label]
    # --------------------------------------------------
    rating = np.array([
        [task_map[task], worker_map[worker], label]
        for task, worker, label in subset[
            ["taskId", "workerId", "answer"]
        ].itertuples(index=False, name=None)
    ], dtype=int)

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------
    truth_map = (
        subset[["taskId", "truth"]]
        .drop_duplicates("taskId")
        .set_index("taskId")["truth"]
        .to_dict()
    )

    y_true = np.array([truth_map[task] for task in task_ids], dtype=int)

    # --------------------------------------------------
    # Dimensions
    # --------------------------------------------------
    n_task = len(task_ids)
    n_worker_original = len(worker_ids)

    n_classes = int(max(
        rating[:, 2].max(),
        y_true.max()
    )) + 1

    # --------------------------------------------------
    # Inject synthetic low-quality workers
    # r = number of LQ workers relative to original workers
    # --------------------------------------------------
    rng = np.random.default_rng(seed)

    n_lq = int(round(r * n_worker_original))

    if n_lq > 0:
        # Each synthetic LQ worker labels every task
        # uniformly at random over the available classes
        lq_labels = rng.integers(
            0,
            n_classes,
            size=(n_task, n_lq)
        )

        lq_worker_ids = np.arange(
            n_worker_original,
            n_worker_original + n_lq
        )

        lq_rating = np.column_stack((
            np.repeat(np.arange(n_task), n_lq),
            np.tile(lq_worker_ids, n_task),
            lq_labels.reshape(-1)
        ))

        rating = np.vstack((rating, lq_rating))

    n_worker = n_worker_original + n_lq

    # --------------------------------------------------
    # Observation matrix
    # --------------------------------------------------
    R_obs = np.full((n_task, n_worker), np.nan)

    R_obs[
        rating[:, 0].astype(int),
        rating[:, 1].astype(int)
    ] = rating[:, 2]

    # --------------------------------------------------
    # Diagnostics
    # --------------------------------------------------
    labels_per_task = np.bincount(
        rating[:, 0].astype(int),
        minlength=n_task
    )

    labels_per_worker = np.bincount(
        rating[:, 1].astype(int),
        minlength=n_worker
    )

    print(
        f"NetEase subset: "
        f"{n_task} tasks, "
        f"{n_worker} workers, "
        f"{len(rating)} annotations, "
        f"{n_classes} classes"
    )

    print(
        "Task labels:   min/median/mean/max =",
        labels_per_task.min(),
        np.median(labels_per_task),
        labels_per_task.mean(),
        labels_per_task.max()
    )

    print(
        "Worker labels: min/median/mean/max =",
        labels_per_worker.min(),
        np.median(labels_per_worker),
        labels_per_worker.mean(),
        labels_per_worker.max()
    )

    # Keep this for compatibility with your current pipeline for now
    return rating, y_true, y_true, R_obs, n_task, n_worker, n_classes


def make_netease_subset(
    df,
    tasks_per_cap=300,
    min_ann_per_worker=30,
    min_workers_per_task=5,
    min_final_ann_per_worker=25,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    # --------------------------------------------------
    # 1. Initial worker filtering on FULL dataset
    # --------------------------------------------------
    worker_stats = df.groupby("workerId").agg(
        n_annotations=("taskId", "size"),
    )

    good_workers = worker_stats[
        (worker_stats["n_annotations"] >= min_ann_per_worker)
    ].index

    d = df[df["workerId"].isin(good_workers)].copy()

    # --------------------------------------------------
    # 2. Keep tasks with enough retained workers
    # --------------------------------------------------
    task_counts = d.groupby("taskId")["workerId"].nunique()

    good_tasks = task_counts[
        task_counts >= min_workers_per_task
    ].index

    d = d[d["taskId"].isin(good_tasks)].copy()

    # --------------------------------------------------
    # 3. Sample equal number of tasks per capability
    # --------------------------------------------------
    selected_tasks = []

    for cap, g in d.groupby("capability"):
        task_coverage = (
            g.groupby("taskId")["workerId"]
            .nunique()
            .rename("coverage")
            .reset_index()
        )

         # Randomize among tasks with similar coverage
        task_coverage["random_order"] = rng.random(len(task_coverage))

        # Coverage is still the primary criterion
        task_coverage = task_coverage.sort_values(
            ["coverage", "random_order"],
            ascending=[False, True]
        )

        n = min(tasks_per_cap, len(task_coverage))
        chosen_df = task_coverage.head(n)
        chosen = chosen_df["taskId"].to_numpy()

        selected_tasks.extend(chosen)

        print(
            f"Capability {cap}: "
            f"selected {n} / {len(task_coverage)} tasks, "
            f"coverage range = "
            f"{chosen_df['coverage'].min()}--"
            f"{chosen_df['coverage'].max()}"
        )
    subset = d[d["taskId"].isin(selected_tasks)].copy()

    # --------------------------------------------------
    # 4. IMPORTANT: enforce worker/task conditions
    #    AFTER task sampling
    # --------------------------------------------------
    subset = prune_final_subset(
        subset,
        min_final_ann_per_worker=min_final_ann_per_worker,
        min_workers_per_task=min_workers_per_task
    )

    return subset


def prune_final_subset(
    df,
    min_final_ann_per_worker=30,
    min_workers_per_task=3
):
    """
    Iteratively enforce constraints on the FINAL sampled subset.

    Every retained worker has >= min_final_ann_per_worker labels.
    Every retained task has >= min_workers_per_task workers.
    """
    df = df.copy()

    while True:
        old_rows = len(df)
        old_workers = df["workerId"].nunique()
        old_tasks = df["taskId"].nunique()

        # Remove workers with too few final annotations
        worker_counts = df.groupby("workerId").size()

        good_workers = worker_counts[
            worker_counts >= min_final_ann_per_worker
        ].index

        df = df[df["workerId"].isin(good_workers)].copy()

        # Removing workers may make some tasks too sparse
        task_counts = df.groupby("taskId")["workerId"].nunique()

        good_tasks = task_counts[
            task_counts >= min_workers_per_task
        ].index

        df = df[df["taskId"].isin(good_tasks)].copy()

        # Removing tasks may push workers below the threshold,
        # so repeat until stable.
        if (
            len(df) == old_rows
            and df["workerId"].nunique() == old_workers
            and df["taskId"].nunique() == old_tasks
        ):
            break

    if len(df) == 0:
        raise ValueError(
            "Final pruning removed all observations. "
            "Try lowering min_final_ann_per_worker or increasing tasks_per_cap."
        )

    return df