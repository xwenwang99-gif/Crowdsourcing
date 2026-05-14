# -*- coding: utf-8 -*-
"""
Refactored data generator for the latent-factor crowdsourcing model.

Key change: `noise_ratio` (float in [0, 1)) replaces the hard-coded
`noise_group` switch.  Within each task group g, a fraction `noise_ratio`
of workers are *low-quality* (uniform random latent factors) and the
remaining fraction `1 - noise_ratio` are *high-quality* (concentrated
around the correct label direction for group g).

Any number of task/label classes (n_classes) is supported, though the
simulation below defaults to 5 to match the original experiments.
"""

import numpy as np


def getdata(
    n_task: int,
    n_worker: int,
    n_task_groups: int,
    k: float = 3.0,
    sigma: float = 1.0,
    obs_prob: float = 1.0,
    noise_ratio: float = 0.0,   # fraction of low-quality workers per group
    n_classes: int = 5,         # number of label classes / latent dimensions
    seed: int = None,
):
    """
    Parameters
    ----------
    n_task        : total number of tasks (must be divisible by n_task_groups)
    n_worker      : total number of workers (must be divisible by n_task_groups)
    n_task_groups : number of task groups (= number of distinct true labels)
    k             : signal strength for high-quality workers / task centroids
    sigma         : noise std for the multivariate-normal draws
    obs_prob      : probability that a worker observes a given task
    noise_ratio   : fraction of workers that are low-quality in each group.
                    0.0 → all workers are high-quality (original noise_group=0
                    setting with 1/n_task_groups high-quality ratio).
                    Increasing noise_ratio raises the proportion of random
                    (low-quality) workers.  Must be in [0, 1).
    n_classes     : dimensionality of the latent factor space (= number of
                    label classes)
    seed          : optional random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    assert 0.0 <= noise_ratio < 1.0, "noise_ratio must be in [0, 1)"
    assert n_task % n_task_groups == 0, "n_task must be divisible by n_task_groups"
    assert n_worker % n_task_groups == 0, "n_worker must be divisible by n_task_groups"

    tasks_per_group = n_task // n_task_groups

    # ------------------------------------------------------------------
    # 1. Task latent factors A  (n_task × n_classes)
    #    Group g is concentrated around the g-th standard basis vector * k.
    # ------------------------------------------------------------------
    A = np.zeros((n_task, n_classes))
    for g in range(n_task_groups):
        centroid = np.zeros(n_classes)
        centroid[g % n_classes] = k          # wrap if n_task_groups > n_classes
        start = g * tasks_per_group
        end   = start + tasks_per_group
        A[start:end] = np.random.multivariate_normal(
            centroid, sigma * np.eye(n_classes), tasks_per_group
        )

    # ------------------------------------------------------------------
    # 2. Worker latent factors B_g  (n_worker × n_classes) for each group g
    #
    #    Within group g the high-quality workers occupy a contiguous block
    #    of size `n_hq` starting at index `g * workers_per_group`.
    #    All other workers are low-quality (uniform random in [-k, k]).
    #
    #    The `noise_ratio` parameter controls what fraction of the
    #    n_worker // n_task_groups "slots" are high-quality:
    #        n_hq = round((1 - noise_ratio) * workers_per_group)
    #    so noise_ratio=0 gives the original 1/n_task_groups high-quality
    #    fraction, and noise_ratio → 1 makes almost everyone low-quality.
    # ------------------------------------------------------------------
    workers_per_group = n_worker // n_task_groups
    n_hq = max(1, round((1.0 - noise_ratio) * workers_per_group))

    # B[g] is the latent factor matrix for group g
    B = np.zeros((n_task_groups, n_worker, n_classes))

    worker_label = np.zeros((n_worker, n_task_groups), dtype=int)

    for g in range(n_task_groups):
        hq_start = g * workers_per_group          # start of high-quality block
        hq_end   = hq_start + n_hq               # end   of high-quality block

        centroid = np.zeros(n_classes)
        centroid[g % n_classes] = k

        # high-quality block: concentrated around the correct direction
        B[g, hq_start:hq_end] = np.random.multivariate_normal(
            centroid, sigma * np.eye(n_classes), n_hq
        )

        # everyone else: low-quality (uniform random)
        lq_indices = list(range(0, hq_start)) + list(range(hq_end, n_worker))
        if lq_indices:
            B[g, lq_indices] = (
                np.random.random((len(lq_indices), n_classes)) * 2 * k - k
            )

        # record which workers are high-quality for this group
        worker_label[hq_start:hq_end, g] = 1

    # ------------------------------------------------------------------
    # 3. Compute response logits and sample labels
    #    R_tsr[i, j, c] = A[i] · B[g][j]  for each class-response c
    #    (here c indexes label classes, not task groups)
    # ------------------------------------------------------------------
    # R_tsr shape: (n_task, n_worker, n_classes)
    R_tsr = np.zeros((n_task, n_worker, n_classes))
    for g in range(n_task_groups):
        start = g * tasks_per_group
        end   = start + tasks_per_group
        # A[start:end] @ B[g].T  → (tasks_per_group, n_worker)
        scores = A[start:end].dot(B[g].T)          # (tasks_per_group, n_worker)
        for c in range(n_classes):
            R_tsr[start:end, :, c] = scores        # same score vector per class

        # More faithful to original: use B[g] per-group score for that group
        # The original code indexed R_tsr[:, :, g] = A @ B_g.T which assigns
        # the g-th worker factor score to the g-th *class* slot.  We replicate
        # that exact logic here for full backwards compatibility:
        R_tsr[:, :, g] = A.dot(B[g].T)

    # Softmax over label classes to get probabilities
    exp_logits = np.exp(R_tsr)
    probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)   # (n_task, n_worker, n_classes)

    # Sample observed labels
    R = np.array([
        [np.random.choice(n_classes, p=probs[i, j]) for j in range(n_worker)]
        for i in range(n_task)
    ])

    # ------------------------------------------------------------------
    # 4. Subsample observations according to obs_prob
    # ------------------------------------------------------------------
    records = []
    for i in range(n_task):
        sub_n = max(1, int(n_worker * obs_prob))
        sub_workers = np.sort(
            np.random.choice(n_worker, size=sub_n, replace=False)
        )
        tmp = np.column_stack([
            np.full(sub_n, i),
            sub_workers,
            R[i, sub_workers],
        ])
        records.append(tmp)

    rating = np.concatenate(records, axis=0)

    R_obs = np.full((n_task, n_worker), np.nan)
    task_ids   = rating[:, 0].astype(int)
    worker_ids = rating[:, 1].astype(int)
    labels_obs = rating[:, 2].astype(int)
    R_obs[task_ids, worker_ids] = labels_obs

    # Ground-truth task labels (0-indexed group membership)
    label = np.repeat(np.arange(n_task_groups), tasks_per_group)

    # B reshaped to (n_worker, n_task_groups, n_classes) for downstream use,
    # transposed to match the original stacking convention: (n_worker, n_task_groups, n_classes)
    B_out = B.transpose(1, 0, 2)   # (n_worker, n_task_groups, n_classes)

    return rating, label, worker_label, R_obs, A, B_out


# -----------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    for noise in [0.0, 2/7, 0.5]:
        rating, label, wl, R_obs, A, B = getdata(
            n_task=500,
            n_worker=100,
            n_task_groups=5,
            k=3.0,
            sigma=1.0,
            obs_prob=1.0,
            noise_ratio=noise,
            seed=42,
        )
        hq_counts = wl.sum(axis=0)
        print(
            f"noise_ratio={noise:.3f} | "
            f"rating shape={rating.shape} | "
            f"high-quality workers per group: {hq_counts}"
        )