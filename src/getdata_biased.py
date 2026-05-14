# -*- coding: utf-8 -*-
"""
getdata_biased.py
-----------------
Latent-factor crowdsourcing data generator with three worker archetypes:

  1. High-quality (HQ)   — votes reliably for the correct label.
  2. Biased (delta=1)    — votes systematically for label (g+delta) mod G.
  3. Low-quality (LQ)    — near-random votes.

Latent factor model (following original code exactly):
  logit(task i, worker j, class c) = A[i] · B[c][j]
  where A[i] is the task factor and B[c][j] is worker j's factor in
  the "class-c slot".  Softmax over c gives the label probability.

Key construction insight
------------------------
Task group g has centroid A[i] ≈ k*e_g (the g-th standard basis vector).
For class-c logit to be large we need B[c][j] to point in the e_g direction:
  logit(c) = A[i] · B[c][j] ≈ k*e_g · B[c][j] = k * B[c][j][g-th coord]

  HQ worker (group g):
    B[g][j] ≈ k*e_g   →  logit(c=g) ≈ k²        (correct class wins)

  Biased worker (group g, delta=1, g_wrong = g+1 mod G):
    B[g][j]       ≈ 0  (no signal in correct slot)
    B[g_wrong][j] ≈ k*e_g  (task direction in the wrong-class slot)
    → logit(c=g)       ≈ 0
    → logit(c=g_wrong) ≈ k²   (wrong class wins)

  LQ worker:
    B[c][j] ≈ Uniform[-k, k]^d  for all c  →  near-uniform votes

Parameters
----------
n_task        : int   — total tasks; must be divisible by n_task_groups
n_worker      : int   — total workers; must be divisible by n_task_groups
n_task_groups : int   — number of task/label groups (default 5)
k             : float — signal strength; centroids are k * e_g
sigma         : float — std of the Gaussian draws around each centroid
obs_prob      : float — probability a worker observes a given task
hq_ratio      : float — fraction of workers per group that are HQ
bias_ratio    : float — fraction of workers per group that are biased
                        lq_ratio = 1 - hq_ratio - bias_ratio (implicit)
delta         : int   — label offset for biased group (default 1)
n_classes     : int   — latent-space dimension (= number of label classes)
seed          : int   — optional RNG seed

Returns
-------
rating      : (N_obs, 3) int array   — [task_id, worker_id, label]
label       : (n_task,) int array    — ground-truth task labels (0-indexed)
worker_type : (n_worker, n_task_groups, 3) int array
              worker_type[j, g, t]=1 iff worker j is archetype t for group g
              t=0: HQ,  t=1: biased,  t=2: LQ
R_obs       : (n_task, n_worker) float array — labels (NaN=unobserved)
A           : (n_task, n_classes) float array  — task latent factors
B           : (n_worker, n_task_groups, n_classes) float array
              B[j, g, :] = worker j's latent factor in class-g slot
"""

import numpy as np


def getdata_biased(
    n_task: int,
    n_worker: int,
    n_task_groups: int = 5,
    k: float = 3.0,
    sigma: float = 1.0,
    obs_prob: float = 1.0,
    hq_ratio: float = 1 / 3,
    bias_ratio: float = 1 / 3,
    delta: int = 1,
    n_classes: int = 5,
    seed: int = None,
):
    # ------------------------------------------------------------------ #
    # 0. Validation                                                        #
    # ------------------------------------------------------------------ #
    if seed is not None:
        np.random.seed(seed)

    assert n_task % n_task_groups == 0, \
        "n_task must be divisible by n_task_groups"
    assert n_worker % n_task_groups == 0, \
        "n_worker must be divisible by n_task_groups"
    assert hq_ratio > 0 and bias_ratio > 0, \
        "hq_ratio and bias_ratio must both be positive"
    assert 1 <= delta < n_task_groups, \
        "delta must be in [1, n_task_groups)"

    tasks_per_group   = n_task   // n_task_groups
    workers_per_group = n_worker // n_task_groups

    n_hq   = max(1, round(hq_ratio   * workers_per_group))
    n_bias = max(1, round(bias_ratio * workers_per_group))
    n_lq   = workers_per_group - n_hq - n_bias

    # ------------------------------------------------------------------ #
    # 1. Task latent factors  A  (n_task × n_classes)                     #
    #    Group g centroid = k * e_{g mod n_classes}                       #
    # ------------------------------------------------------------------ #
    A = np.zeros((n_task, n_classes))
    for g in range(n_task_groups):
        centroid = np.zeros(n_classes)
        centroid[g % n_classes] = k
        ts, te = g * tasks_per_group, (g + 1) * tasks_per_group
        A[ts:te] = np.random.multivariate_normal(
            centroid, sigma * np.eye(n_classes), tasks_per_group
        )

    # ------------------------------------------------------------------ #
    # 2. Worker latent factors  B  (n_task_groups × n_worker × n_classes) #
    #                                                                      #
    # B[c][j] is worker j's factor for the class-c logit.                 #
    # Initialise all slots as LQ (uniform random).                        #
    # Then overwrite structured HQ and biased blocks.                     #
    #                                                                      #
    # Worker layout (contiguous blocks of size workers_per_group):        #
    #   group g occupies indices [g*wpg, (g+1)*wpg)                      #
    #     HQ     : [g*wpg,               g*wpg + n_hq)                   #
    #     Biased : [g*wpg + n_hq,        g*wpg + n_hq + n_bias)          #
    #     LQ     : [g*wpg + n_hq+n_bias, (g+1)*wpg)                      #
    #                                                                      #
    # HQ worker of group g:                                               #
    #   B[g][j] ≈ k*e_g  (correct-class slot points at task direction)   #
    #   All other B[c][j] remain random (LQ initialisation)               #
    #                                                                      #
    # Biased worker of group g (g_wrong = (g+delta) mod G):              #
    #   B[g][j]       ≈ 0  (zero out the correct-class slot)             #
    #   B[g_wrong][j] ≈ k*e_g  (wrong-class slot also points at task)   #
    #   → logit(c=g_wrong) = A[i]·B[g_wrong][j] ≈ k²  >> logit(c=g)≈0  #
    #   All other B[c][j] remain random (small contribution)              #
    # ------------------------------------------------------------------ #
    B = np.random.random((n_task_groups, n_worker, n_classes)) * 2 * k - k

    worker_type_idx = np.full((n_worker, n_task_groups), 2, dtype=int)  # default LQ

    for g in range(n_task_groups):
        ws = g * workers_per_group   # block start

        hq_start   = ws
        hq_end     = ws + n_hq
        bias_start = hq_end
        bias_end   = bias_start + n_bias

        g_correct = g % n_classes
        g_wrong   = (g + delta) % n_classes

        # Centroid in the task direction (e_g) — used by both HQ and biased
        task_centroid = np.zeros(n_classes)
        task_centroid[g_correct] = k
        
        # Reset all slots for this worker block to random before writing
        for c in range(n_classes):
            B[c, hq_start:hq_start+workers_per_group] = (
                np.random.random((workers_per_group, n_classes)) * 2 * k - k
            )
        # HQ: correct-class slot B[g] → k*e_g
        B[g, hq_start:hq_end] = np.random.multivariate_normal(
            task_centroid, sigma * np.eye(n_classes), n_hq
        )
        worker_type_idx[hq_start:hq_end, g] = 0

        # Biased:
        #   correct slot B[g][j] ≈ 0  (near-zero so logit(c=g) ≈ 0)
        B[g, bias_start:bias_end] = np.random.multivariate_normal(
            np.zeros(n_classes), sigma * np.eye(n_classes), n_bias
        )
        #   wrong slot B[g_wrong][j] ≈ k*e_g  (so logit(c=g_wrong) ≈ k²)
        B[g_wrong, bias_start:bias_end] = np.random.multivariate_normal(
            task_centroid, (sigma+1) * np.eye(n_classes), n_bias
        )
        worker_type_idx[bias_start:bias_end, g] = 1

        # LQ: already random from initialisation, no overwrite needed

    # One-hot worker_type tensor
    worker_type = np.zeros((n_worker, n_task_groups, 3), dtype=int)
    for t in range(3):
        worker_type[:, :, t] = (worker_type_idx == t).astype(int)

    # ------------------------------------------------------------------ #
    # 3. Response tensor and label sampling                               #
    #                                                                      #
    # logit(task i, worker j, class c) = A[i] · B[c][j]                  #
    # R_tsr[:, :, c] = A @ B[c].T       (exact original formula)         #
    # ------------------------------------------------------------------ #
    R_tsr = np.zeros((n_task, n_worker, n_classes))
    for c in range(n_classes):
        R_tsr[:, :, c] = A.dot(B[c].T)

    # Numerically stable softmax
    R_tsr -= R_tsr.max(axis=2, keepdims=True)
    exp_logits = np.exp(R_tsr)
    probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)

    # Sample labels
    R = np.array([
        [np.random.choice(n_classes, p=probs[i, j]) for j in range(n_worker)]
        for i in range(n_task)
    ])

    # ------------------------------------------------------------------ #
    # 4. Subsample observations                                           #
    # ------------------------------------------------------------------ #
    records = []
    for i in range(n_task):
        sub_n       = max(1, int(n_worker * obs_prob))
        sub_workers = np.sort(
            np.random.choice(n_worker, size=sub_n, replace=False)
        )
        records.append(np.column_stack([
            np.full(sub_n, i), sub_workers, R[i, sub_workers]
        ]))

    rating = np.concatenate(records, axis=0)

    R_obs = np.full((n_task, n_worker), np.nan)
    R_obs[rating[:, 0].astype(int), rating[:, 1].astype(int)] = rating[:, 2]

    label = np.repeat(np.arange(n_task_groups), tasks_per_group)

    # Return B as (n_worker, n_task_groups, n_classes)
    B_out = B.transpose(1, 0, 2)

    return rating, label, worker_type, R_obs, A, B_out


# ------------------------------------------------------------------ #
# Smoke test + sanity checks                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    G = 5
    print("=== Worker count checks ===")
    for cfg in [
        dict(hq_ratio=1/3, bias_ratio=1/3),
        dict(hq_ratio=0.5, bias_ratio=0.25),
        dict(hq_ratio=0.2, bias_ratio=0.4),
    ]:
        _, _, wt, _, _, _ = getdata_biased(
            n_task=500, n_worker=150, n_task_groups=G, k=3.0, sigma=1.0, seed=42, **cfg
        )
        print(
            f"  hq={cfg['hq_ratio']:.2f} bias={cfg['bias_ratio']:.2f} | "
            f"HQ={wt[:,:,0].sum(0)} | Biased={wt[:,:,1].sum(0)} | LQ={wt[:,:,2].sum(0)}"
        )

    print("\n=== Bias sanity check (group-0 tasks, k=5, sigma=0.1) ===")
    rating, label, worker_type, R_obs, A, B = getdata_biased(
        n_task=1000, n_worker=150, n_task_groups=G,
        k=5.0, sigma=0.1, obs_prob=1.0,
        hq_ratio=1/3, bias_ratio=1/3, seed=0,
    )
    wpg    = 150 // G
    n_hq   = max(1, round(1/3 * wpg))
    n_bias = max(1, round(1/3 * wpg))

    g0_tasks     = np.where(label == 0)[0]
    hq_workers   = np.arange(0, n_hq)
    bias_workers = np.arange(n_hq, n_hq + n_bias)
    lq_workers   = np.arange(n_hq + n_bias, wpg)

    def vote_dist(tasks, workers, R_obs, n_classes=5):
        votes = R_obs[np.ix_(tasks, workers)].flatten()
        votes = votes[~np.isnan(votes)].astype(int)
        c = np.bincount(votes, minlength=n_classes)
        return c / c.sum()

    hq_d   = vote_dist(g0_tasks, hq_workers,   R_obs)
    bias_d = vote_dist(g0_tasks, bias_workers, R_obs)
    lq_d   = vote_dist(g0_tasks, lq_workers,   R_obs)

    print(f"  HQ    vote dist (true=0):                {np.round(hq_d,   3)}")
    print(f"  Biased vote dist (true=0, bias→label 1): {np.round(bias_d, 3)}")
    print(f"  LQ    vote dist (true=0):                {np.round(lq_d,   3)}")
    print()
    print(f"  HQ    P(correct=0): {hq_d[0]:.3f}")
    print(f"  Biased P(correct=0): {bias_d[0]:.3f}  |  P(wrong=1): {bias_d[1]:.3f}")
    print(f"  LQ    P(correct=0): {lq_d[0]:.3f}  (expect ~0.2 for 5 classes)")