# -*- coding: utf-8 -*-
"""
Spectral worker-feature extraction aligned with the likelihood/spectral fusion method.

The spectral step is a feature/evidence extractor, not only a hard HQ/biased/LQ
classifier. For each predicted task group g and worker j it returns

    x_spectral[g, j, :] = Lambda_g^{1/2} V_g[j, :]

CHANGES FROM THE PREVIOUS VERSION
---------------------------------
(i)   Centering constant is now 1 / n_categories, not p0 = ||qhat||^2.

      Rationale: with u_ij the chance-centered one-hot of worker j's label on
      task i, and v_j = pi_j - 1/C its mean, the exact identity is

          E[agreement_{jl}] = <pi_j, pi_l>,   <v_j, v_l> = <pi_j, pi_l> - 1/C.

      Subtracting p0 = ||qhat||^2 >= 1/C over-subtracts by delta = p0 - 1/C,
      injecting a rank-one -delta * 11^T contaminant. delta grows as the group's
      pooled marginal concentrates on the true label, i.e. it is worst exactly
      when the task clustering is good. Its effect is to scramble the *radial*
      coordinate of the embedding while leaving pairwise distances intact, so
      the k-means partition survives but the "order clusters by center norm"
      labeling rule can invert (HQ and biased swap). Under the correct 1/C
      centering, an LQ worker's population embedding is exactly the origin.

(ii)  K defaults to min(2, n_categories - 1).

      Rationale: the population signal matrix has rank C - 1 capped at 2. For
      binary labels (C = 2) the biased direction is the negation of the HQ
      direction, so rank(M) = 1 and there is no second signal eigenvector; the
      old K = 2 grabbed a bulk eigenvector and folded pure noise into the
      embedding.

(iii) Tiering is a single 3-center KMeans on the K-dimensional embedding.

REMAINING KNOWN ISSUES (not addressed here)
-------------------------------------------
- Workers with coverage < min_coverage are excluded from the tier KMeans but
  still contribute rows to Atilde, where their high-variance agreement estimates
  tilt the leading eigenvectors. A cleaner design builds Atilde on the covered
  workers only and scores the rest by projection onto U_g.
- Entries of `agreement` have coverage-dependent variance; all are weighted
  equally in the eigendecomposition.
- tier_spectral = 0 conflates "low coverage / no information" with "LQ / no
  signal". Use the returned `tested_mask` to distinguish them downstream.
"""

import warnings

import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans

LQ, HQ, BIASED = 0, 1, 2


def _safe_mode_int(x, default=-1):
    """Return the integer mode of a 1D array after dropping NaN values."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return default
    return int(mode(x, axis=None, keepdims=False).mode)


def _pairwise_agreement(R_g, n_worker, n_categories, fill):
    """
    Worker-worker agreement-rate matrix.

    agreement[j, l] = fraction of commonly-labelled tasks on which workers j and
    l gave the same label. Pairs with no overlap are imputed at `fill`, the
    chance-agreement level, so that Atilde = agreement - fill is 0 for them
    (i.e. "no information", not "no agreement").

    Self-agreement is 1 by definition; it is overwritten downstream when the
    matrix is hollowed.

    Vectorised: C matrix products of shape (T, n) instead of an O(n^2 T) loop.
    """
    obs = ~np.isnan(R_g)                                   # (T, n)
    overlap = obs.astype(float).T @ obs.astype(float)      # (n, n)

    match = np.zeros((n_worker, n_worker), dtype=float)
    for c in range(n_categories):
        Xc = (R_g == c).astype(float)                      # NaN == c -> False
        match += Xc.T @ Xc

    with np.errstate(invalid="ignore", divide="ignore"):
        agreement = np.where(overlap > 0, match / np.maximum(overlap, 1.0), fill)

    np.fill_diagonal(agreement, 1.0)
    coverage = obs.sum(axis=0).astype(int)
    return agreement, coverage, overlap


def _within_cluster_agreement(agreement, idx):
    """Mean off-diagonal raw agreement among the workers in `idx`."""
    if idx.size < 2:
        return np.nan
    sub = agreement[np.ix_(idx, idx)]
    off = ~np.eye(idx.size, dtype=bool)
    return float(sub[off].mean())


def spectral_worker_features_one_group(
    R_g,
    n_worker=None,
    n_categories=None,
    K=None,
    min_coverage=5,
    eps=1e-9,
    n_init=10,
    random_state=0,
    return_matrices=False,
    check_tier_order=True,
):
    """
    Extract spectral worker features for ONE predicted task group.

    Parameters
    ----------
    R_g : array, shape (n_tasks_in_group, n_worker)
        Observed labels, integers 0..C-1, with np.nan for missing.
    n_worker, n_categories : int, optional
        Inferred from R_g if omitted. Pass n_categories explicitly when calling
        per-group, so that the centering constant 1/C is the same across groups.
    K : int, optional
        Number of leading spectral directions. Defaults to min(2, C - 1) and is
        clamped to that value if a larger K is passed.
    min_coverage : int
        Workers with fewer observed labels in this group are excluded from the
        tier KMeans and reported as tier 0 with tested_mask False.
    check_tier_order : bool
        Cross-check the norm-ordering of the KMeans centers against within-cluster
        raw agreement. Both encode Assumption 3 (a > b) but read it off different
        quantities; disagreement is a red flag and raises a warning.

    Returns
    -------
    dict with keys:
        spectral_embedding : (n_worker, K_eff)   x^S_{jg} = Lambda^{1/2} V[j, :]
        tier_spectral      : (n_worker,) int     0=LQ/background, 1=HQ, 2=biased
        hq_idx, biased_idx, lq_idx, tested_idx, tested_mask
        eigvals, eigvecs, positive_eig_mask, cluster_centers, coverage
        qhat, p0, center_const, within_agreement, tier_order_ok
    """
    R_g = np.asarray(R_g, dtype=float)
    if R_g.ndim != 2:
        raise ValueError("R_g must be a 2D array: tasks in group x workers.")

    if n_worker is None:
        n_worker = R_g.shape[1]
    if n_worker != R_g.shape[1]:
        raise ValueError("n_worker does not match R_g.shape[1].")

    if n_categories is None:
        if np.all(np.isnan(R_g)):
            raise ValueError("Cannot infer n_categories: R_g has no observed labels.")
        n_categories = int(np.nanmax(R_g)) + 1
        warnings.warn(
            "n_categories inferred from this group only; pass it explicitly so "
            "the centering constant 1/C is shared across groups.",
            RuntimeWarning,
        )
    if n_categories < 2:
        raise ValueError("n_categories must be at least 2.")

    # (ii) rank of the population signal matrix is (C - 1) capped at 2.
    K_max = min(2, n_categories - 1, n_worker)
    if K is None:
        K = K_max
    elif K > K_max:
        warnings.warn(
            f"K={K} exceeds the population signal rank min(2, C-1)={K_max}; "
            f"clamping to {K_max}. The extra directions would be pure noise.",
            RuntimeWarning,
        )
        K = K_max
    K_eff = int(max(K, 1))

    # ------------------------------------------------------------------
    # 1. Agreement matrix and (i) the correct centering constant.
    # ------------------------------------------------------------------
    center_const = 1.0 / n_categories

    agreement, coverage, overlap = _pairwise_agreement(
        R_g, n_worker, n_categories, fill=center_const
    )

    # Pooled marginal and p0 are retained as diagnostics only. They are NOT the
    # centering constant: p0 >= 1/C, and the excess injects a rank-one 11^T term.
    pooled_counts = np.zeros(n_categories, dtype=float)
    for c in range(n_categories):
        pooled_counts[c] = np.count_nonzero(R_g == c)
    qhat = (
        pooled_counts / pooled_counts.sum()
        if pooled_counts.sum() > 0
        else np.full(n_categories, center_const)
    )
    p0 = float(np.sum(qhat ** 2))

    Atilde = agreement - center_const
    

    # ------------------------------------------------------------------
    # 2. Eigendecomposition and spectral embedding.
    # ------------------------------------------------------------------
    eigvals_all, eigvecs_all = np.linalg.eigh(Atilde)
    order = np.argsort(eigvals_all)[::-1]
    eigvals_all = eigvals_all[order]
    eigvecs_all = eigvecs_all[:, order]

    lam_raw = eigvals_all[:K_eff].copy()
    V_raw = eigvecs_all[:, :K_eff].copy()

    pos_mask = lam_raw > eps
    lam_pos = np.where(pos_mask, lam_raw, 0.0)

    spectral_embedding = V_raw * np.sqrt(lam_pos)[None, :]

    # ------------------------------------------------------------------
    # 3. (iii) Tiers: one 3-center KMeans on the embedding.
    # ------------------------------------------------------------------
    tested_mask = coverage >= min_coverage
    tested = np.where(tested_mask)[0]

    tier_spectral = np.full(n_worker, LQ, dtype=int)
    hq_idx = np.array([], dtype=int)
    biased_idx = np.array([], dtype=int)
    cluster_centers = None
    within_agreement = {"hq": np.nan, "biased": np.nan}
    tier_order_ok = None

    if tested.size >= 3 and np.linalg.norm(spectral_embedding[tested]) > eps:
        X = spectral_embedding[tested, :]
        n_unique = np.unique(np.round(X / eps), axis=0).shape[0]
        n_clusters = int(min(3, tested.size, n_unique))

        if n_clusters >= 2:
            km = KMeans(
                n_clusters=n_clusters, n_init=n_init, random_state=random_state
            ).fit(X)
            cluster_centers = km.cluster_centers_
            norms = np.linalg.norm(cluster_centers, axis=1)
            rank = np.argsort(norms)[::-1]          # descending: HQ, biased, LQ

            hq_idx = tested[km.labels_ == rank[0]]
            tier_spectral[hq_idx] = HQ
            if n_clusters == 3:
                biased_idx = tested[km.labels_ == rank[1]]
                tier_spectral[biased_idx] = BIASED
            else:
                warnings.warn(
                    f"Only {n_clusters} distinct clusters found; the biased tier "
                    "is empty and those workers fall into tier 0 (LQ).",
                    RuntimeWarning,
                )

            # Cross-check the ordering. HQ should have higher within-cluster raw
            # agreement than biased (a^2*kappa + 1/C  vs  b^2*kappa + 1/C).
            if check_tier_order and biased_idx.size >= 2 and hq_idx.size >= 2:
                within_agreement["hq"] = _within_cluster_agreement(agreement, hq_idx)
                within_agreement["biased"] = _within_cluster_agreement(agreement, biased_idx)
                tier_order_ok = bool(
                    within_agreement["hq"] >= within_agreement["biased"]
                )
                if not tier_order_ok:
                    warnings.warn(
                        "Center-norm ordering says HQ, but within-cluster raw "
                        f"agreement disagrees (HQ {within_agreement['hq']:.4f} < "
                        f"biased {within_agreement['biased']:.4f}). Assumption 3 "
                        "(a > b) may be violated, or the two tiers are swapped.",
                        RuntimeWarning,
                    )

    lq_idx = np.where(tier_spectral == LQ)[0]

    out = {
        "spectral_embedding": spectral_embedding,
        "tier_spectral": tier_spectral,
        "hq_idx": hq_idx,
        "biased_idx": biased_idx,
        "lq_idx": lq_idx,
        "tested_idx": tested,
        "tested_mask": tested_mask,
        "eigvals": lam_raw,
        "eigvecs": V_raw,
        "eigvals_all": eigvals_all,
        "positive_eig_mask": pos_mask,
        "cluster_centers": cluster_centers,
        "coverage": coverage,
        "qhat": qhat,
        "p0": p0,
        "center_const": center_const,
        "K_eff": K_eff,
        "within_agreement": within_agreement,
        "tier_order_ok": tier_order_ok,
    }
    if return_matrices:
        out["agreement"] = agreement
        out["Atilde"] = Atilde
        out["overlap"] = overlap
    return out


def extract_spectral_worker_features(
    pred_group,
    R_obs,
    n_task_groups,
    n_worker=None,
    n_categories=None,
    K=None,
    min_coverage=5,
    n_init=10,
    random_state=0,
    return_matrices=False,
):
    """
    Extract spectral worker features for all predicted task groups.

    Returns
    -------
    dict with
        embedding     : (n_worker, n_task_groups, K_max)
        coverage      : (n_worker, n_task_groups)
        tested_mask   : (n_worker, n_task_groups) bool
        tier_spectral : (n_worker, n_task_groups)   0=LQ, 1=HQ, 2=biased
    """
    R_obs = np.asarray(R_obs, dtype=float)
    pred_group = np.asarray(pred_group)

    if n_worker is None:
        n_worker = R_obs.shape[1]
    if n_categories is None:
        n_categories = int(np.nanmax(R_obs)) + 1

    K_max = min(2, n_categories - 1, n_worker)
    if K is None:
        K = K_max

    embedding = np.full((n_worker, n_task_groups, K_max), np.nan)
    coverage = np.zeros((n_worker, n_task_groups), dtype=int)
    tested_mask = np.zeros((n_worker, n_task_groups), dtype=bool)
    tier_spectral = np.zeros((n_worker, n_task_groups), dtype=int)

    group_outputs = []
    for g in range(n_task_groups):
        tasks_g = np.where(pred_group == g)[0]
        if tasks_g.size == 0:
            group_outputs.append({"tasks": tasks_g, "empty": True})
            continue

        out_g = spectral_worker_features_one_group(
            R_obs[tasks_g, :],
            n_worker=n_worker,
            n_categories=n_categories,
            K=K,
            min_coverage=min_coverage,
            n_init=n_init,
            random_state=random_state,
            return_matrices=return_matrices,
        )
        out_g["tasks"] = tasks_g
        out_g["empty"] = False
        group_outputs.append(out_g)

        k = out_g["spectral_embedding"].shape[1]
        embedding[:, g, :k] = out_g["spectral_embedding"]
        coverage[:, g] = out_g["coverage"]
        tested_mask[:, g] = out_g["tested_mask"]
        tier_spectral[:, g] = out_g["tier_spectral"]

    return {
        "embedding": embedding,
        "coverage": coverage,
        "tested_mask": tested_mask,
        "tier_spectral": tier_spectral,
        "group_outputs": group_outputs,
        "tier_encoding": {"LQ": LQ, "HQ": HQ, "Biased": BIASED},
        "K": K_max,
        "center_const": 1.0 / n_categories,
    }


def infer_labels_from_spectral_hq(
    pred_group,
    R_obs,
    spectral,
    n_task,
    n_task_groups,
    label=None,
    label_mode="task",
):
    """
    Spectral-only baseline: majority vote over the HQ tier within each group.
    In the fused method this is replaced by q_{jgH}-weighted voting.
    """
    R_obs = np.asarray(R_obs, dtype=float)
    pred_group = np.asarray(pred_group)
    task_label_pred = np.full(n_task, -1, dtype=int)
    group_label_pred = np.full(n_task_groups, -1, dtype=int)

    hq_workers_pred, biased_workers_pred = [], []

    for g in range(n_task_groups):
        tier_g = spectral["tier_spectral"][:, g]
        hq_g = np.where(tier_g == HQ)[0]
        biased_g = np.where(tier_g == BIASED)[0]
        hq_workers_pred.append(hq_g)
        biased_workers_pred.append(biased_g)

        tasks_g = np.where(pred_group == g)[0]
        if tasks_g.size == 0 or hq_g.size == 0:
            continue

        if label_mode == "group":
            group_label_pred[g] = _safe_mode_int(R_obs[np.ix_(tasks_g, hq_g)].ravel())
            if group_label_pred[g] != -1:
                task_label_pred[tasks_g] = group_label_pred[g]
        elif label_mode == "task":
            for i in tasks_g:
                task_label_pred[i] = _safe_mode_int(R_obs[i, hq_g])
        else:
            raise ValueError("label_mode must be 'task' or 'group'.")

    task_accuracy = None
    if label is not None:
        task_accuracy = float(np.mean(task_label_pred == np.asarray(label)))
    return task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred


def _hq_and_label_infer(
    pred_group,
    R_obs,
    label,
    worker_label,
    n_task,
    n_worker,
    n_task_groups,
    LABEL_MODE="task",
    verbose=False,
    MIN_COVERAGE=5,
    return_spectral=True,
    **legacy_kwargs,
):
    """
    Backward-compatible wrapper. `TIER_METHOD` is accepted and ignored: tiering
    is now always a 3-center KMeans on the embedding.
    """
    if "TIER_METHOD" in legacy_kwargs:
        warnings.warn(
            "TIER_METHOD is deprecated and ignored; tiering is a 3-center KMeans "
            "on the spectral embedding.",
            DeprecationWarning,
        )
    unknown = set(legacy_kwargs) - {"TIER_METHOD"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")

    spectral = extract_spectral_worker_features(
        pred_group=pred_group,
        R_obs=R_obs,
        n_task_groups=n_task_groups,
        n_worker=n_worker,
        n_categories=int(np.nanmax(R_obs)) + 1,
        K=None,
        min_coverage=MIN_COVERAGE,
    )

    task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred = (
        infer_labels_from_spectral_hq(
            pred_group=pred_group,
            R_obs=R_obs,
            spectral=spectral,
            n_task=n_task,
            n_task_groups=n_task_groups,
            label=label,
            label_mode=LABEL_MODE,
        )
    )

    if verbose:
        print("embedding shape:", spectral["embedding"].shape, " K =", spectral["K"])
        print("centering constant 1/C =", spectral["center_const"])

    if return_spectral:
        return task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred, spectral
    return task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred