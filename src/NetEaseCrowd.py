import numpy as np
from datasets import load_dataset


def get_NETEASE(
        r = 0,
    tasks_per_cap=100,
    min_cap_per_worker=3,
    min_ann_per_worker=30,
    min_workers_per_task=3,
    seed=123
):
    # --------------------------------------------------
    # Load raw NetEaseCrowd data
    # --------------------------------------------------
    ds = load_dataset("liuhyuu/NetEaseCrowd", split="train")
    df = ds.to_pandas()

    subset = make_netease_subset(
        df,
        tasks_per_cap=tasks_per_cap,
        min_cap_per_worker=min_cap_per_worker,
        min_ann_per_worker=min_ann_per_worker,
        min_workers_per_task=min_workers_per_task,
        seed=seed
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
    # Ground truth for selected tasks
    # --------------------------------------------------
    truth_map = (
        subset[["taskId", "truth"]]
        .drop_duplicates("taskId")
        .set_index("taskId")["truth"]
        .to_dict()
    )

    y_true = np.array([
        truth_map[task]
        for task in task_ids
    ], dtype=int)

    # --------------------------------------------------
    # Dataset dimensions
    # --------------------------------------------------
    n_task = len(task_ids)
    n_worker = len(worker_ids)

    n_classes = int(max(
        rating[:, 2].max(),
        y_true.max()
    )) + 1

    # --------------------------------------------------
    # R_obs stores actual labels, NaN = missing
    # --------------------------------------------------
    R_obs = np.full((n_task, n_worker), np.nan)

    R_obs[
        rating[:, 0],
        rating[:, 1]
    ] = rating[:, 2]

    print(
        f"NetEase subset: "
        f"{n_task} tasks, "
        f"{n_worker} workers, "
        f"{len(rating)} annotations, "
        f"{n_classes} classes"
    )

    return rating, y_true, R_obs, n_task, n_worker, n_classes


def make_netease_subset(
    df,
    tasks_per_cap=100,
    min_cap_per_worker=3,
    min_ann_per_worker=30,
    min_workers_per_task=3,
    seed=123
):
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # 1. Keep workers active in several capabilities
    # --------------------------------------------------
    worker_stats = df.groupby("workerId").agg(
        n_annotations=("taskId", "size"),
        n_capabilities=("capability", "nunique")
    )

    good_workers = worker_stats[
        (worker_stats["n_capabilities"] >= min_cap_per_worker) &
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
        tasks = g["taskId"].unique()

        n = min(tasks_per_cap, len(tasks))
        chosen = rng.choice(tasks, size=n, replace=False)

        selected_tasks.extend(chosen)

        print(
            f"Capability {cap}: "
            f"selected {n} / {len(tasks)} tasks"
        )

    subset = d[d["taskId"].isin(selected_tasks)].copy()

    return subset