# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 19:18:19 2026

@author: wangl
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union, Any


@dataclass
class MultiSPAResult:
    A: np.ndarray                  # (M, K, K) confusion matrices A_m
    d: np.ndarray                  # (K,) class prior
    y_hat: np.ndarray              # (N,) predicted labels in {0..K-1}
    post: np.ndarray               # (N, K) posterior (unnormalized then normalized)
    Rhat: Dict[Tuple[int, int], np.ndarray]  # pairwise co-occurrence matrices


def build_label_matrix(
    ratings: np.ndarray,
    n_tasks: Optional[int] = None,
    n_workers: Optional[int] = None,
    missing_val: int = -1,
    assume_zero_indexed: bool = True,
) -> np.ndarray:
    """
    Convert triplets ratings[:, (task, worker, label)] into dense L (N, M).
    Assumes task/worker/label are integers. Labels must be 0..K-1 (or will be shifted outside).

    ratings: shape (E, 3): (task_id, worker_id, label)
    """
    ratings = np.asarray(ratings)
    if ratings.ndim != 2 or ratings.shape[1] < 3:
        raise ValueError("ratings must be (E,3) array: columns (task, worker, label)")

    t = ratings[:, 0].astype(int)
    w = ratings[:, 1].astype(int)
    lab = ratings[:, 2].astype(int)

    if not assume_zero_indexed:
        t = t - 1
        w = w - 1
        lab = lab - 1

    if n_tasks is None:
        n_tasks = int(t.max()) + 1
    if n_workers is None:
        n_workers = int(w.max()) + 1

    L = np.full((n_tasks, n_workers), missing_val, dtype=int)
    L[t, w] = lab
    return L


def pairwise_cooccurrence(
    L: np.ndarray,
    K: int,
    missing_val: int = -1,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Estimate \hat R_{m,l} where R_{m,l}(i,j) = P(X_m=i, X_l=j)
    via sample averaging on co-labeled tasks (paper Eq.(4) and discussion). :contentReference[oaicite:1]{index=1}

    Returns dict keyed by (m,l) with KxK matrices.
    """
    N, M = L.shape
    Rhat: Dict[Tuple[int, int], np.ndarray] = {}

    for m in range(M):
        for l in range(M):
            if m == l:
                continue
            mask = (L[:, m] != missing_val) & (L[:, l] != missing_val)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            xm = L[idx, m].astype(int)
            xl = L[idx, l].astype(int)

            # joint counts
            mat = np.zeros((K, K), dtype=float)
            for a, b in zip(xm, xl):
                if 0 <= a < K and 0 <= b < K:
                    mat[a, b] += 1.0

            mat /= idx.size  # convert to empirical joint PMF
            Rhat[(m, l)] = mat

    return Rhat


def normalize_columns_l1(Z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    colsum = Z.sum(axis=0, keepdims=True)
    colsum = np.maximum(colsum, eps)
    return Z / colsum


def spa_successive_projection(X: np.ndarray, K: int) -> np.ndarray:
    """
    Successive Projection Algorithm (SPA) selection indices.
    Implements the greedy rule (paper Eq.(10)). :contentReference[oaicite:2]{index=2}

    X: (K, L) matrix whose columns live in convex cone, want K 'anchor' columns.
    Returns indices of selected columns (length K).
    """
    # We work with Euclidean projection using QR updates (Gram-Schmidt-like).
    Kdim, L = X.shape
    selected = []
    Q = np.zeros((Kdim, 0), dtype=float)

    for _ in range(K):
        if Q.shape[1] == 0:
            # choose max norm column
            norms = np.sum(X * X, axis=0)
            q = int(np.argmax(norms))
            selected.append(q)
            # add normalized column to Q
            v = X[:, q]
            v = v / (np.linalg.norm(v) + 1e-12)
            Q = np.concatenate([Q, v[:, None]], axis=1)
            continue

        # project columns onto orthogonal complement of span(Q)
        proj = Q @ (Q.T @ X)          # (Kdim, L)
        resid = X - proj
        norms = np.sum(resid * resid, axis=0)
        q = int(np.argmax(norms))
        selected.append(q)

        # orthonormalize new direction
        v = resid[:, q]
        nv = np.linalg.norm(v) + 1e-12
        v = v / nv
        Q = np.concatenate([Q, v[:, None]], axis=1)

    return np.array(selected, dtype=int)


def hungarian_assignment(cost: np.ndarray) -> np.ndarray:
    """
    Minimal Hungarian for small K using scipy if available; fallback brute-force if not.
    Returns perm p where column j in target aligns to column p[j] in reference.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        # r is [0..K-1] rows, c is assigned columns
        perm = np.zeros(cost.shape[0], dtype=int)
        perm[r] = c
        return perm
    except Exception:
        # brute force for small K
        import itertools
        K = cost.shape[0]
        best = None
        best_val = float("inf")
        for perm in itertools.permutations(range(K)):
            val = sum(cost[i, perm[i]] for i in range(K))
            if val < best_val:
                best_val = val
                best = np.array(perm, dtype=int)
        return best


def align_confusions(A: np.ndarray, ref: int = 0) -> np.ndarray:
    """
    Align column permutations across A_m's.
    DS model identifiable up to common column permutation; we fix by matching to A_ref.

    We match by maximizing diagonal mass similarity (equivalently minimizing a cost).
    """
    M, K, _ = A.shape
    A_aligned = A.copy()
    Aref = A_aligned[ref]

    for m in range(M):
        if m == ref:
            continue

        # Build cost between columns: want columns that "look similar".
        # Use L2 distance between columns as cost.
        cost = np.zeros((K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                cost[i, j] = np.linalg.norm(Aref[:, i] - A_aligned[m][:, j])

        perm = hungarian_assignment(cost)
        A_aligned[m] = A_aligned[m][:, perm]  # permute columns
    return A_aligned


def estimate_D_from_pair(A_m: np.ndarray, A_l: np.ndarray, R_ml: np.ndarray) -> Optional[np.ndarray]:
    """
    From R_{m,l} = A_m D A_l^T => D = A_m^{-1} R_{m,l} (A_l^T)^{-1}.
    Returns D (KxK) or None if inversion fails.
    """
    try:
        invAm = np.linalg.inv(A_m)
        invAltT = np.linalg.inv(A_l.T)
        D = invAm @ R_ml @ invAltT
        return D
    except np.linalg.LinAlgError:
        return None


def multispa_train(
    L: np.ndarray,
    K: int,
    missing_val: int = -1,
    min_cocolabeled: int = 5,
    ref_worker: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """
    Implements MultiSPA (Algorithm 1, algebraic part). :contentReference[oaicite:3]{index=3}
    Returns (A, d, Rhat).
    """
    N, M = L.shape
    Rhat = pairwise_cooccurrence(L, K=K, missing_val=missing_val)

    # Estimate each A_m by building Z_m = [R_{m,m1}, ..., R_{m,mT}] and running SPA
    A = np.zeros((M, K, K), dtype=float)

    for m in range(M):
        # pick neighbors who co-labeled at least min_cocolabeled tasks
        neighbors = []
        blocks = []
        for l in range(M):
            if l == m:
                continue
            key = (m, l)
            if key not in Rhat:
                continue
            # check co-labeled count
            mask = (L[:, m] != missing_val) & (L[:, l] != missing_val)
            if mask.sum() < min_cocolabeled:
                continue
            neighbors.append(l)
            blocks.append(Rhat[key])

        if len(blocks) == 0:
            # If no neighbors, fall back to "nearly uniform" confusion
            A[m] = np.eye(K) * 0.7 + (np.ones((K, K)) - np.eye(K)) * (0.3 / (K - 1))
            continue

        Zm = np.concatenate(blocks, axis=1)  # (K, K*T(m))
        Zm = normalize_columns_l1(Zm)

        idx = spa_successive_projection(Zm, K=K)
        A[m] = Zm[:, idx]

        # ensure columns sum to 1 (stochastic)
        A[m] = normalize_columns_l1(A[m])

    # Fix permutation mismatch across A_m's
    A = align_confusions(A, ref=ref_worker)

    # Estimate D by averaging over available pairs with ref_worker
    Ds = []
    for m in range(M):
        if m == ref_worker:
            continue
        key = (m, ref_worker)
        if key in Rhat:
            Dm = estimate_D_from_pair(A[m], A[ref_worker], Rhat[key])
            if Dm is not None:
                Ds.append(Dm)

    if len(Ds) == 0:
        # fallback to uniform prior if cannot estimate D
        d = np.ones(K) / K
        return A, d, Rhat

    Davg = np.mean(Ds, axis=0)
    d = np.clip(np.diag(Davg), 1e-12, None)
    d = d / d.sum()

    return A, d, Rhat


def map_predict_labels(
    L: np.ndarray,
    A: np.ndarray,
    d: np.ndarray,
    missing_val: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAP prediction under DS model:
      p(y=k | observed) ∝ d(k) * Π_m A_m[x_m, k]
    (paper notes: after identifying parameters, construct ML/MAP estimators). :contentReference[oaicite:4]{index=4}
    """
    N, M = L.shape
    K = d.shape[0]
    logd = np.log(d + 1e-12)

    post_log = np.zeros((N, K), dtype=float)
    for i in range(N):
        post_log[i] = logd
        for m in range(M):
            lab = L[i, m]
            if lab == missing_val:
                continue
            lab = int(lab)
            if 0 <= lab < K:
                post_log[i] += np.log(A[m][lab, :] + 1e-12)

    # normalize
    post = np.exp(post_log - post_log.max(axis=1, keepdims=True))
    post = post / post.sum(axis=1, keepdims=True)

    y_hat = post.argmax(axis=1).astype(int)
    return y_hat, post




def multispa_fit_predict(
    L_or_ratings: Union[np.ndarray, Any],
    K: int,
    y_true: Optional[np.ndarray] = None,
    missing_val: int = -1,
    assume_triplets: bool = False,
    assume_zero_indexed: bool = True,
    min_cocolabeled: int = 5,
    ref_worker: int = 0,
) -> MultiSPAResult:
    """
    Convenience wrapper:
    - If assume_triplets=True, L_or_ratings is (E,3) triplets.
    - Else L_or_ratings is dense (N,M) label matrix.

    Returns fitted parameters + predicted labels.
    """
    if assume_triplets:
        L = build_label_matrix(
            L_or_ratings,
            missing_val=missing_val,
            assume_zero_indexed=assume_zero_indexed,
        )
    else:
        L = np.asarray(L_or_ratings).copy()
        if L.dtype != int:
            L = L.astype(int)

    A, d, Rhat = multispa_train(
        L=L, K=K, missing_val=missing_val,
        min_cocolabeled=min_cocolabeled, ref_worker=ref_worker
    )
    y_hat, post = map_predict_labels(L=L, A=A, d=d, missing_val=missing_val)

    return MultiSPAResult(A=A, d=d, y_hat=y_hat, post=post, Rhat=Rhat)


def accuracy(y_true: np.ndarray, y_hat: np.ndarray, ignore_val: Optional[int] = None) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    if ignore_val is not None:
        mask = (y_true != ignore_val)
        y_true = y_true[mask]
        y_hat = y_hat[mask]
    return float(np.mean(y_true == y_hat))

