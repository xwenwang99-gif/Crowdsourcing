# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 13:03:40 2026

@author: wangl
"""


# -*- coding: utf-8 -*-

import numpy as np


def getdata(
    n_task: int,
    n_worker: int,
    n_task_groups: int,
    n_worker_groups: int,
    k: float = 3.0,
    sigma: float = 1.0,
    obs_prob: float = 1.0,
    noise_proportion: float = 0.0,
):
    """
    Generate synthetic crowdsourcing data with latent task/worker factors.

    Parameters
    ----------
    n_task : int
        Total number of tasks. Must be divisible by n_task_groups.
    n_worker : int
        Total number of workers.
    n_task_groups : int
        Number of task groups (currently fixed at 5).
    n_worker_groups : int
        Number of HQ worker groups (currently fixed at 5, one per task group).
    k : float
        Signal strength for latent factors. Higher = more separated groups.
    sigma : float
        Noise standard deviation on latent factor draws.
    obs_prob : float
        Fraction of workers that observe each task (in [0, 1]).
    noise_proportion : float
        Fraction of workers that are pure noise (not HQ for any task group).
        Must be in [0, 1). The remaining (1 - noise_proportion) workers are
        split equally across the n_worker_groups HQ groups.

    Returns
    -------
    rating : np.ndarray, shape (n_obs, 3)
        Observed ratings as rows of [task_id, worker_id, label].
    label : np.ndarray, shape (n_task,)
        True task group labels (0 to n_task_groups - 1).
    worker_label : np.ndarray, shape (n_worker, n_task_groups)
        Binary matrix indicating which workers are HQ for each task group.
        Noise workers have all-zero rows.
    R_obs : np.ndarray, shape (n_task, n_worker)
        Observed rating matrix with np.nan for unobserved entries.
    """
    assert 0.0 <= noise_proportion < 1.0, "noise_proportion must be in [0, 1)"
    assert n_task % n_task_groups == 0, "n_task must be divisible by n_task_groups"

    n_dim = 5  # latent dimension (fixed at 5)

    # -----------------------------------------------------------------------
    # 1. Compute worker group sizes
    # -----------------------------------------------------------------------
    n_noise_workers = int(round(n_worker * noise_proportion))
    n_hq_workers = n_worker - n_noise_workers

    # Each HQ group gets an equal share of the non-noise workers
    hq_group_size = n_hq_workers // n_worker_groups
    # Any remainder goes to noise (keeps indexing clean)
    n_noise_workers += n_hq_workers - hq_group_size * n_worker_groups
    n_hq_workers = hq_group_size * n_worker_groups

    # Worker index layout:
    #   [0 .. hq_group_size-1]             → HQ group 0 (expert on task group 0)
    #   [hq_group_size .. 2*hq_group_size-1] → HQ group 1
    #   ...
    #   [n_hq_workers .. n_worker-1]       → noise workers

    # -----------------------------------------------------------------------
    # 2. Task latent factors  A  (n_task x n_dim)
    #    Tasks in group g are drawn from a mean vector with k in position g.
    # -----------------------------------------------------------------------
    tasks_per_group = n_task // n_task_groups

    A = np.zeros((n_task, n_dim))
    for g in range(n_task_groups):
        mean = np.zeros(n_dim)
        mean[g] = k
        start = g * tasks_per_group
        end = start + tasks_per_group
        A[start:end, :] = np.random.multivariate_normal(mean, sigma * np.eye(n_dim), tasks_per_group)

    # -----------------------------------------------------------------------
    # 3. Worker latent factors  B_c  for each label class c  (n_worker x n_dim)
    #
    #    For HQ group g:  mean vector for class c = k * e_c  if c == g, else 0
    #      → worker in HQ group g has a strong response in dimension g for class g
    #    For noise workers: uniform random in [-k, k] for every class
    # -----------------------------------------------------------------------
    # B[c] is the factor matrix used to score label class c
    B = np.zeros((n_dim, n_worker, n_dim))   # B[c, worker, dim]

    for g in range(n_worker_groups):
        start_w = g * hq_group_size
        end_w = start_w + hq_group_size

        for c in range(n_dim):
            mean = np.zeros(n_dim)
            if c == g:
                mean[c] = k          # HQ signal: strong alignment with task group g
            # else mean stays zero → neutral / random behaviour on other classes
            B[c, start_w:end_w, :] = np.random.multivariate_normal(
                mean, sigma * np.eye(n_dim), hq_group_size
            )

    # Noise workers: random uniform in [-k, k] for all classes
    if n_noise_workers > 0:
        start_noise = n_hq_workers
        for c in range(n_dim):
            B[c, start_noise:, :] = np.random.uniform(-k, k, size=(n_noise_workers, n_dim))

    # -----------------------------------------------------------------------
    # 4. Compute rating probabilities via softmax over label classes
    #    R_tsr[i, j, c] = A[i] · B[c, j]
    # -----------------------------------------------------------------------
    # Shape: (n_task, n_worker, n_dim)
    R_tsr = np.einsum("id,cjd->ijc", A, B)

    # Numerically stable softmax along class axis
    R_tsr -= R_tsr.max(axis=2, keepdims=True)
    exp_logits = np.exp(R_tsr)
    probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)  # (n_task, n_worker, n_dim)

    # -----------------------------------------------------------------------
    # 5. Sample observed labels
    # -----------------------------------------------------------------------
    # Vectorised multinomial draw: for each (task, worker) pair pick a class
    probs_flat = probs.reshape(-1, n_dim)                        # (n_task*n_worker, n_dim)
    cumprobs = probs_flat.cumsum(axis=1)
    u = np.random.uniform(size=(probs_flat.shape[0], 1))
    R_full = (u < cumprobs).argmax(axis=1).reshape(n_task, n_worker)   # (n_task, n_worker)

    # -----------------------------------------------------------------------
    # 6. Apply observation mask (obs_prob)
    # -----------------------------------------------------------------------
    records = []
    for i in range(n_task):
        sub_n_worker = max(1, int(n_worker * obs_prob))
        sub_workers = np.sort(
            np.random.choice(np.arange(n_worker), size=sub_n_worker, replace=False)
        )
        tmp = np.column_stack([
            np.full(sub_n_worker, i),
            sub_workers,
            R_full[i, sub_workers],
        ])
        records.append(tmp)

    rating = np.concatenate(records, axis=0).astype(int)

    R_obs = np.full((n_task, n_worker), np.nan)
    R_obs[rating[:, 0], rating[:, 1]] = rating[:, 2]

    # -----------------------------------------------------------------------
    # 7. Ground-truth labels
    # -----------------------------------------------------------------------
    label = np.repeat(np.arange(n_task_groups), tasks_per_group)

    # worker_label[j, g] = 1  iff worker j is HQ for task group g
    worker_label = np.zeros((n_worker, n_task_groups), dtype=int)
    for g in range(n_task_groups):
        start_w = g * hq_group_size
        end_w = start_w + hq_group_size
        worker_label[start_w:end_w, g] = 1

    return rating, label, worker_label, R_obs
    
