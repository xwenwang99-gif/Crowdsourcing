# -*- coding: utf-8 -*-
"""
Spectral worker-feature extraction aligned with the likelihood/spectral fusion method.

Main change from the older spectral step:
    The spectral step is treated as a feature/evidence extractor, not only as a
    hard HQ/biased/LQ classifier.

For each predicted task group g and worker j, it returns:
    x_spectral[g, j, :]  = Lambda_g^{1/2} V_g[j, :]
    score_sq[g, j]       = ||x_spectral[g, j, :]||_2^2
    score[g, j]          = sqrt(score_sq[g, j])

The later likelihood/spectral fusion script can use score_sq[g, j] or score[g, j]
as the spectral-side evidence for W_{jg}.
"""

import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans


def _safe_mode_int(x, default=-1):
    """Return the integer mode of a 1D array after dropping NaN values."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return default
    return int(mode(x, axis=None, keepdims=False).mode)


def _bh_reject(pvals, tested_idx, fdr_q):
    """
    Benjamini-Hochberg rejection mask over a subset of tested indices.
    Untested workers are not rejected.
    """
    reject = np.zeros_like(pvals, dtype=bool)
    tested_idx = np.asarray(tested_idx, dtype=int)
    if tested_idx.size == 0:
        return reject

    p_tested = pvals[tested_idx]
    order = np.argsort(p_tested)
    p_sorted = p_tested[order]
    m = p_sorted.size
    thresh = fdr_q * (np.arange(1, m + 1) / m)
    below = p_sorted <= thresh

    if below.any():
        k_star = np.max(np.where(below)[0])
        crit = p_sorted[k_star]
        reject[tested_idx] = p_tested <= crit

    return reject


def spectral_worker_features_one_group(
    R_g,
    n_worker=None,
    n_categories=None,
    K=2,
    B_perm=2000,
    fdr_q=0.10,
    min_coverage=5,
    rng=None,
    eps=1e-9,
    return_matrices=False,
):
    """
    Extract spectral worker features for ONE predicted task group.

    Parameters
    ----------
    R_g : array, shape (n_tasks_in_group, n_worker)
        Observed labels for tasks in one predicted task group. Missing labels
        should be np.nan. Labels are assumed to be integers 0, ..., C-1.

    n_worker : int, optional
        Number of workers. If None, inferred from R_g.shape[1].

    n_categories : int, optional
        Number of label categories. If None, inferred from max observed label + 1.

    K : int, default=2
        Number of leading positive spectral directions to keep. For the proposed
        three-tier structure, K=2 corresponds to HQ and biased signal directions.

    B_perm : int, default=2000
        Number of null resamples used for the worker-level no-skill p-values.

    fdr_q : float, default=0.10
        BH-FDR level for optional spectral-only signal-worker detection.

    min_coverage : int, default=5
        Workers with fewer than this many observed labels inside the group are
        not tested and are treated as low-information for the spectral-only tier.

    return_matrices : bool, default=False
        If True, include agreement and centered agreement matrices in output.
        These can be large, so False is safer for large simulations.

    Returns
    -------
    out : dict
        Main fields:
            spectral_embedding : array, shape (n_worker, K)
                x^S_{jg} = Lambda_g^{1/2} V_g[j, :].
            spectral_score_sq : array, shape (n_worker,)
                s^S_{jg} = ||x^S_{jg}||_2^2 = sum_k lambda_k v_k[j]^2.
                This is the recommended scalar score for later fusion.
            spectral_score : array, shape (n_worker,)
                sqrt(spectral_score_sq). Kept for compatibility with your old code.
            pvals : array, shape (n_worker,)
                No-skill p-values using the fixed-subspace null.
            signal_mask : bool array, shape (n_worker,)
                BH-FDR rejection mask. True means the spectral view finds signal.
            tier_spectral : int array, shape (n_worker,)
                Optional spectral-only hard tier: 0=LQ/background, 1=HQ, 2=biased.
                This is diagnostic only; the new method should combine scores later.
            hq_idx, biased_idx, lq_idx : int arrays
                Indices implied by tier_spectral.
    """
    if rng is None:
        rng = np.random.default_rng()

    R_g = np.asarray(R_g, dtype=float)
    if R_g.ndim != 2:
        raise ValueError("R_g must be a 2D array: tasks in group × workers.")

    if n_worker is None:
        n_worker = R_g.shape[1]
    if n_worker != R_g.shape[1]:
        raise ValueError("n_worker does not match R_g.shape[1].")

    observed = R_g[~np.isnan(R_g)]
    if n_categories is None:
        if observed.size == 0:
            raise ValueError("Cannot infer n_categories because R_g has no observed labels.")
        n_categories = int(np.nanmax(R_g)) + 1

    # ------------------------------------------------------------------
    # 1. Pooled within-group marginal and scalar chance baseline.
    # ------------------------------------------------------------------
    pooled_counts = np.zeros(n_categories, dtype=float)
    coverage = np.zeros(n_worker, dtype=int)

    for w in range(n_worker):
        lab_w = R_g[:, w]
        lab_w = lab_w[~np.isnan(lab_w)].astype(int)
        coverage[w] = lab_w.size
        if lab_w.size > 0:
            pooled_counts += np.bincount(lab_w, minlength=n_categories)

    if pooled_counts.sum() > 0:
        qhat = pooled_counts / pooled_counts.sum()
    else:
        qhat = np.full(n_categories, 1.0 / n_categories)

    p0 = float(np.sum(qhat ** 2))

    # ------------------------------------------------------------------
    # 2. Worker-worker agreement matrix, with no-overlap pairs imputed at p0.
    # ------------------------------------------------------------------
    agreement = np.full((n_worker, n_worker), p0, dtype=float)

    for w1 in range(n_worker):
        obs1 = ~np.isnan(R_g[:, w1])
        for w2 in range(w1, n_worker):
            valid = obs1 & ~np.isnan(R_g[:, w2])
            if valid.sum() > 0:
                val = np.mean(R_g[valid, w1] == R_g[valid, w2])
                agreement[w1, w2] = val
                agreement[w2, w1] = val

    # Centering removes the common chance-agreement block p0 * 11^T.
    Atilde = agreement - p0
    Atilde = 0.5 * (Atilde + Atilde.T)

    # ------------------------------------------------------------------
    # 3. Eigendecomposition and spectral embedding.
    # ------------------------------------------------------------------
    eigvals_all, eigvecs_all = np.linalg.eigh(Atilde)
    order = np.argsort(eigvals_all)[::-1]
    eigvals_all = eigvals_all[order]
    eigvecs_all = eigvecs_all[:, order]

    K_eff = min(K, n_worker)
    lam_raw = eigvals_all[:K_eff].copy()
    V_raw = eigvecs_all[:, :K_eff].copy()

    # Only positive eigenvalues contribute to the affinity embedding.
    pos_mask = lam_raw > eps
    lam_pos = np.where(pos_mask, lam_raw, 0.0)

    spectral_embedding = V_raw * np.sqrt(lam_pos)[None, :]
    spectral_score_sq = np.sum(spectral_embedding ** 2, axis=1)
    spectral_score = np.sqrt(np.maximum(spectral_score_sq, 0.0))

    # ------------------------------------------------------------------
    # 4. Fixed-subspace no-skill null for p-values.
    #    The p-values are optional diagnostics for the spectral-only tier.
    #    The later fusion method can ignore them or use them as extra features.
    # ------------------------------------------------------------------
    tested = np.where(coverage >= min_coverage)[0]
    pvals = np.ones(n_worker, dtype=float)

    # For the projection identity, ignore nonpositive eigenvalues.
    lam_safe = np.where(pos_mask, lam_raw, 1.0)

    for j in tested:
        idx_j = np.where(~np.isnan(R_g[:, j]))[0]
        n_j = idx_j.size
        if n_j == 0:
            continue

        Wsub = R_g[idx_j, :]
        valid_jw = ~np.isnan(Wsub)
        counts_w = valid_jw.sum(axis=0)
        counts_w_safe = np.where(counts_w > 0, counts_w, 1)

        # Mc[c][r, w] = 1 if worker w labeled task r as c and the label exists.
        Mc = [(valid_jw & (Wsub == c)).astype(float) for c in range(n_categories)]

        # Null labels for worker j are drawn iid from pooled background qhat.
        P = rng.choice(n_categories, size=(B_perm, n_j), p=qhat)

        agree_cnt = np.zeros((B_perm, n_worker), dtype=float)
        for c in range(n_categories):
            agree_cnt += (P == c).astype(float) @ Mc[c]

        agree_rate = agree_cnt / counts_w_safe
        agree_rate[:, counts_w == 0] = p0

        col = agree_rate - p0
        col[:, j] = 1.0 - p0  # self-agreement under the same centered convention

        proj = col @ V_raw
        null_score_sq = np.sum(
            np.where(pos_mask, (proj ** 2) / lam_safe, 0.0),
            axis=1,
        )
        null_score = np.sqrt(np.maximum(null_score_sq, 0.0))

        pvals[j] = (1.0 + np.sum(null_score >= spectral_score[j])) / (B_perm + 1.0)

    signal_mask = _bh_reject(pvals, tested, fdr_q=fdr_q)
    sig_workers = np.where(signal_mask)[0]

    # ------------------------------------------------------------------
    # 5. Optional spectral-only hard labels.
    #    This is NOT the final new-method W_jg. It is only a spectral diagnostic.
    #    Final W_jg should come from combining likelihood and spectral evidence.
    # ------------------------------------------------------------------
    tier_spectral = np.zeros(n_worker, dtype=int)  # 0 = LQ/background
    hq_idx = np.array([], dtype=int)
    biased_idx = np.array([], dtype=int)

    if sig_workers.size == 1:
        hq_idx = sig_workers
        tier_spectral[hq_idx] = 1
    elif sig_workers.size >= 2:
        s_sig = spectral_score_sq[sig_workers].reshape(-1, 1)
        if np.ptp(s_sig) < eps:
            hq_idx = sig_workers
            tier_spectral[hq_idx] = 1
        else:
            km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(s_sig)
            centers = km.cluster_centers_.ravel()
            hq_cluster = int(np.argmax(centers))
            biased_cluster = int(np.argmin(centers))

            hq_idx = sig_workers[km.labels_ == hq_cluster]
            biased_idx = sig_workers[km.labels_ == biased_cluster]
            tier_spectral[hq_idx] = 1
            tier_spectral[biased_idx] = 2

    lq_idx = np.where(tier_spectral == 0)[0]

    out = {
        "spectral_embedding": spectral_embedding,          # x^S_{jg}
        "spectral_score_sq": spectral_score_sq,            # recommended s^S_{jg}
        "spectral_score": spectral_score,                  # sqrt version
        "pvals": pvals,
        "signal_mask": signal_mask,
        "tier_spectral": tier_spectral,                    # diagnostic only
        "hq_idx": hq_idx,
        "biased_idx": biased_idx,
        "lq_idx": lq_idx,
        "eigvals": lam_raw,
        "eigvecs": V_raw,
        "positive_eig_mask": pos_mask,
        "coverage": coverage,
        "qhat": qhat,
        "p0": p0,
        "tested_idx": tested,
    }

    if return_matrices:
        out["agreement"] = agreement
        out["Atilde"] = Atilde

    return out


def extract_spectral_worker_features(
    pred_group,
    R_obs,
    n_task_groups,
    n_worker=None,
    n_categories=None,
    K=2,
    B_perm=2000,
    fdr_q=0.10,
    min_coverage=5,
    random_state=None,
    return_matrices=False,
):
    """
    Extract spectral worker features for all predicted task groups.

    Returns
    -------
    spectral : dict
        spectral["score_sq"] has shape (n_worker, n_task_groups).
        spectral["embedding"] has shape (n_worker, n_task_groups, K).
        spectral["tier_spectral"] has shape (n_worker, n_task_groups), with
        0=LQ/background, 1=HQ, 2=biased as a spectral-only diagnostic.

    This is the object you should save and pass to the later fusion script.
    """
    R_obs = np.asarray(R_obs, dtype=float)
    pred_group = np.asarray(pred_group)

    if n_worker is None:
        n_worker = R_obs.shape[1]
    if n_categories is None:
        n_categories = int(np.nanmax(R_obs)) + 1

    rng = np.random.default_rng(random_state)

    score_sq = np.full((n_worker, n_task_groups), np.nan, dtype=float)
    score = np.full((n_worker, n_task_groups), np.nan, dtype=float)
    embedding = np.full((n_worker, n_task_groups, K), np.nan, dtype=float)
    pvals = np.full((n_worker, n_task_groups), np.nan, dtype=float)
    coverage = np.zeros((n_worker, n_task_groups), dtype=int)
    tier_spectral = np.zeros((n_worker, n_task_groups), dtype=int)
    signal_mask = np.zeros((n_worker, n_task_groups), dtype=bool)

    group_outputs = []

    for g in range(n_task_groups):
        tasks_g = np.where(pred_group == g)[0]

        if tasks_g.size == 0:
            group_outputs.append({"tasks": tasks_g, "empty": True})
            continue

        R_g = R_obs[tasks_g, :]
        out_g = spectral_worker_features_one_group(
            R_g,
            n_worker=n_worker,
            n_categories=n_categories,
            K=K,
            B_perm=B_perm,
            fdr_q=fdr_q,
            min_coverage=min_coverage,
            rng=rng,
            return_matrices=return_matrices,
        )
        out_g["tasks"] = tasks_g
        out_g["empty"] = False
        group_outputs.append(out_g)

        K_eff = out_g["spectral_embedding"].shape[1]
        score_sq[:, g] = out_g["spectral_score_sq"]
        score[:, g] = out_g["spectral_score"]
        embedding[:, g, :K_eff] = out_g["spectral_embedding"]
        pvals[:, g] = out_g["pvals"]
        coverage[:, g] = out_g["coverage"]
        tier_spectral[:, g] = out_g["tier_spectral"]
        signal_mask[:, g] = out_g["signal_mask"]

    return {
        "score_sq": score_sq,
        "score": score,
        "embedding": embedding,
        "pvals": pvals,
        "coverage": coverage,
        "tier_spectral": tier_spectral,
        "signal_mask": signal_mask,
        "group_outputs": group_outputs,
        "tier_encoding": {"LQ": 0, "HQ": 1, "Biased": 2},
        "score_definition": "score_sq[j,g] = sum_k lambda_gk * v_gk[j]^2 = ||Lambda_g^{1/2} V_g[j,:]||_2^2",
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
    Optional compatibility helper: use spectral-only HQ workers to infer labels.

    In the new combined method, you will likely replace this by q_{jgH}-weighted
    voting after combining likelihood and spectral evidence. This function is
    kept only so the spectral-only baseline still works.
    """
    R_obs = np.asarray(R_obs, dtype=float)
    pred_group = np.asarray(pred_group)
    task_label_pred = np.full(n_task, -1, dtype=int)
    group_label_pred = np.full(n_task_groups, -1, dtype=int)

    hq_workers_pred = []
    biased_workers_pred = []

    for g in range(n_task_groups):
        tier_g = spectral["tier_spectral"][:, g]
        hq_g = np.where(tier_g == 1)[0]
        biased_g = np.where(tier_g == 2)[0]
        hq_workers_pred.append(hq_g)
        biased_workers_pred.append(biased_g)

        tasks_g = np.where(pred_group == g)[0]
        if tasks_g.size == 0 or hq_g.size == 0:
            continue

        if label_mode == "group":
            all_labels = R_obs[np.ix_(tasks_g, hq_g)].ravel()
            group_label_pred[g] = _safe_mode_int(all_labels, default=-1)
            if group_label_pred[g] != -1:
                task_label_pred[tasks_g] = group_label_pred[g]

        elif label_mode == "task":
            for i in tasks_g:
                task_label_pred[i] = _safe_mode_int(R_obs[i, hq_g], default=-1)

        else:
            raise ValueError("label_mode must be 'task' or 'group'.")

    task_accuracy = None
    if label is not None:
        label = np.asarray(label)
        task_accuracy = np.mean(task_label_pred == label)

    return task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred


# --------------------------------------------------------------------------
# Backward-compatible wrapper for your old call pattern.
# --------------------------------------------------------------------------
def _hq_and_label_infer(
    pred_group,
    R_obs,
    label,
    worker_label,
    n_task,
    n_worker,
    n_task_groups,
    n_worker_groups=None,
    LABEL_MODE="task",
    verbose=False,
    B_PERM=2000,
    FDR_Q=0.10,
    MIN_COVERAGE=5,
    RANDOM_STATE=None,
    return_spectral=True,
):
    """
    Backward-compatible version of your old function.

    Difference from the old return value:
        If return_spectral=True, the function returns
            task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred, spectral
        where spectral contains score_sq, score, embedding, pvals, coverage, etc.

    If you need the exact old four outputs, call with return_spectral=False.
    """
    spectral = extract_spectral_worker_features(
        pred_group=pred_group,
        R_obs=R_obs,
        n_task_groups=n_task_groups,
        n_worker=n_worker,
        n_categories=int(np.nanmax(R_obs)) + 1,
        K=2,
        B_perm=B_PERM,
        fdr_q=FDR_Q,
        min_coverage=MIN_COVERAGE,
        random_state=RANDOM_STATE,
        return_matrices=False,
    )

    task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred = \
        infer_labels_from_spectral_hq(
            pred_group=pred_group,
            R_obs=R_obs,
            spectral=spectral,
            n_task=n_task,
            n_task_groups=n_task_groups,
            label=label,
            label_mode=LABEL_MODE,
        )

    if verbose:
        print("Spectral score matrix shape:", spectral["score_sq"].shape)
        print("Use spectral['score_sq'][j, g] as the scalar spectral score s^S_{jg}.")
        print("Use spectral['embedding'][j, g, :] as x^S_{jg}.")

    if return_spectral:
        return task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred, spectral
    return task_accuracy, task_label_pred, hq_workers_pred, biased_workers_pred
