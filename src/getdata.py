# -*- coding: utf-8 -*-

import numpy as np


def getdata(
    n_task: int,
    n_worker: int,
    n_task_groups: int,
    k: float = 3.0,
    sigma: float = 1.0,
    obs_prob: float = 1.0,
    noise_proportion: float = 0.0,
):
    """
    Generate synthetic crowdsourcing data with latent task/worker factors.

    The number of label classes, latent dimension, task groups, and HQ worker
    groups all equal n_task_groups, making the setup fully flexible.

    Parameters
    ----------
    n_task : int
        Total number of tasks. Must be divisible by n_task_groups.
    n_worker : int
        Total number of workers.
    n_task_groups : int
        Number of task groups, label classes, and HQ worker groups.
        Also sets the latent factor dimension. Can be any integer >= 2.
    k : float
        Signal strength for latent factors. Higher = more separated groups.
    sigma : float
        Noise std on latent factor draws. Higher = more overlap between groups.
    obs_prob : float
        Fraction of workers that label each task. 1.0 = fully observed.
    noise_proportion : float
        Fraction of workers that are pure noise (not HQ for any task group).
        Must be in [0, 1). The remaining workers are split equally across the
        n_task_groups HQ groups.

    Returns
    -------
    rating : np.ndarray, shape (n_obs, 3)
        Observed ratings as rows of [task_id, worker_id, label].
    label : np.ndarray, shape (n_task,)
        True task group labels (0 to n_task_groups - 1).
    worker_label : np.ndarray, shape (n_worker, n_task_groups)
        Binary indicator: worker_label[j, g] = 1 iff worker j is HQ for
        task group g. Noise workers are all-zero rows.
    R_obs : np.ndarray, shape (n_task, n_worker)
        Observed rating matrix; np.nan where unobserved.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    assert n_task_groups >= 2, "n_task_groups must be at least 2"
    assert n_task % n_task_groups == 0, (
        f"n_task ({n_task}) must be divisible by n_task_groups ({n_task_groups})"
    )
    assert 0.0 <= noise_proportion < 1.0, "noise_proportion must be in [0, 1)"
    assert 0.0 < obs_prob <= 1.0, "obs_prob must be in (0, 1]"

    # n_dim drives the latent space; it equals n_task_groups so each group
    # has its own dedicated latent dimension
    n_dim = n_task_groups
    n_classes = n_task_groups   # label classes equal task groups

    # ------------------------------------------------------------------
    # 1. Worker group sizes
    #
    # Layout (by worker index):
    #   [0 .. hq_size)              -> HQ group 0  (expert on task group 0)
    #   [hq_size .. 2*hq_size)      -> HQ group 1
    #   ...
    #   [n_hq .. n_worker)          -> noise workers
    #
    # Any remainder from integer division is absorbed into noise so that
    # all HQ groups are exactly the same size.
    # ------------------------------------------------------------------
    n_noise = int(round(n_worker * noise_proportion))
    n_hq = n_worker - n_noise
    hq_group_size = n_hq // n_task_groups       # floor division
    n_noise += n_hq - hq_group_size * n_task_groups   # absorb remainder
    n_hq = hq_group_size * n_task_groups

    assert hq_group_size >= 1, (
        f"HQ group size is 0. Increase n_worker or decrease noise_proportion. "
        f"Need n_worker * (1 - noise_proportion) >= n_task_groups ({n_task_groups})."
    )

    # ------------------------------------------------------------------
    # 2. Task latent factors  A  (n_task x n_dim)
    #
    #    Task group g is drawn from a Gaussian with mean k*e_g,
    #    where e_g is the g-th standard basis vector.
    #    This places each group along its own axis in latent space.
    # ------------------------------------------------------------------
    tasks_per_group = n_task // n_task_groups
    A = np.zeros((n_task, n_dim))

    for g in range(n_task_groups):
        mean = np.zeros(n_dim)
        mean[g] = k
        sl = slice(g * tasks_per_group, (g + 1) * tasks_per_group)
        A[sl] = np.random.multivariate_normal(mean, sigma * np.eye(n_dim), tasks_per_group)

    # ------------------------------------------------------------------
    # 3. Worker latent factors  B  (n_classes x n_worker x n_dim)
    #
    #    B[c, j, :] is worker j's factor vector for label class c.
    #    The inner product A[i] @ B[c, j] gives the logit that worker j
    #    assigns label c to task i.
    #
    #    HQ group g:
    #      - For class c == g: mean = k * e_g  (strong correct signal)
    #      - For class c != g: mean = 0        (neutral on off-classes)
    #
    #    Noise workers: uniform in [-k, k] for all classes -> random labels
    # ------------------------------------------------------------------
    B = np.zeros((n_classes, n_worker, n_dim))

    for g in range(n_task_groups):
        sl_w = slice(g * hq_group_size, (g + 1) * hq_group_size)
        for c in range(n_classes):
            mean = np.zeros(n_dim)
            if c == g:
                mean[g] = k
            B[c, sl_w] = np.random.multivariate_normal(
                mean, sigma * np.eye(n_dim), hq_group_size
            )

    if n_noise > 0:
        for c in range(n_classes):
            B[c, n_hq:] = np.random.uniform(-k, k, size=(n_noise, n_dim))

    # ------------------------------------------------------------------
    # 4. Logits and softmax probabilities
    #
    #    logit[i, j, c] = A[i] . B[c, j]
    #    prob[i, j, c]  = softmax over c
    # ------------------------------------------------------------------
    logits = np.einsum("id,cjd->ijc", A, B)      # (n_task, n_worker, n_classes)

    logits -= logits.max(axis=2, keepdims=True)   # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)

    # ------------------------------------------------------------------
    # 5. Sample labels - vectorised inverse-CDF multinomial
    # ------------------------------------------------------------------
    probs_flat = probs.reshape(-1, n_classes)               # (n_task*n_worker, n_classes)
    u = np.random.uniform(size=(probs_flat.shape[0], 1))
    R_full = (u < probs_flat.cumsum(axis=1)).argmax(axis=1) \
               .reshape(n_task, n_worker)                   # (n_task, n_worker)

    # ------------------------------------------------------------------
    # 6. Observation mask (obs_prob)
    # ------------------------------------------------------------------
    records = []
    for i in range(n_task):
        sub_n = max(1, int(n_worker * obs_prob))
        sub_workers = np.sort(
            np.random.choice(n_worker, size=sub_n, replace=False)
        )
        records.append(np.column_stack([
            np.full(sub_n, i),
            sub_workers,
            R_full[i, sub_workers],
        ]))

    rating = np.concatenate(records, axis=0).astype(int)

    R_obs = np.full((n_task, n_worker), np.nan)
    R_obs[rating[:, 0], rating[:, 1]] = rating[:, 2]

    # ------------------------------------------------------------------
    # 7. Ground-truth labels and worker quality indicators
    # ------------------------------------------------------------------
    label = np.repeat(np.arange(n_task_groups), tasks_per_group)

    worker_label = np.zeros((n_worker, n_task_groups), dtype=int)
    for g in range(n_task_groups):
        sl_w = slice(g * hq_group_size, (g + 1) * hq_group_size)
        worker_label[sl_w, g] = 1

    return rating, label, worker_label, R_obs