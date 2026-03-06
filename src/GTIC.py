# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 17:22:55 2026

@author: wangl
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class GTICResult:
    y_hat: np.ndarray                 # (N,) inferred labels in {0,...,K-1}
    cluster_id: np.ndarray            # (N,) cluster assignment in {0,...,K-1}
    cluster_to_class: Dict[int, int]  # cluster -> class mapping
    features: np.ndarray              # (N, K+1) conceptual features used for clustering
    centroids: np.ndarray             # (K, K+1) final kmeans centroids


    def _compute_u_features(
        L: np.ndarray,
        K: int,
        alpha: Optional[np.ndarray] = None,
        missing_val: int = -1,
    ) -> np.ndarray:
        N, J = L.shape
        if alpha is None:
            alpha = np.ones(K, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (K,):
            raise ValueError(f"alpha must have shape (K,), got {alpha.shape}")
    
        # ── Step 1: mask out missing values (entire matrix at once) ──
        mask = (L != missing_val)          # (N, J) boolean
    
        # ── Step 2: count votes per class for every task at once ──
        counts = np.array([
            np.bincount(L[i][mask[i]].astype(int), minlength=K)
            if mask[i].any()
            else np.zeros(K)
            for i in range(N)
        ], dtype=float)                    # (N, K)
    
        # ── Step 3: Dirichlet-MAP for all tasks in one shot ──
        N_i = counts.sum(axis=1, keepdims=True)          # (N, 1)  — votes per task
        denom = N_i + alpha.sum() - K                    # (N, 1)  — broadcast denominator
        U = (counts + alpha - 1.0) / denom               # (N, K)  — all tasks at once
    
        # ── Step 4: handle tasks with no labels (fallback to uniform) ──
        no_labels = ~mask.any(axis=1)                    # (N,) bool
        U[no_labels] = 1.0 / K
    
        # ── Step 5: clip and renormalize (vectorized) ──
        U = np.clip(U, 0.0, 1.0)                        # (N, K)
        row_sums = U.sum(axis=1, keepdims=True)          # (N, 1)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)  # avoid divide by zero
        U = U / row_sums                                 # (N, K) normalized
    
        # ── Step 6: compute uz feature for all tasks at once ──
        uz = (U[:, 1:] - U[:, :-1]).sum(axis=1) / K     # (N,)  telescoping sum
        feats = np.concatenate([U, uz[:, None]], axis=1) # (N, K+1)
    
        return feats


def _init_centroids_from_u(feats: np.ndarray, K: int) -> np.ndarray:
    """
    Paper's centroid initialization:
    For each class k, pick the example with largest u_k (breaking ties by next best),
    ensuring all centroids are distinct examples.
    """
    N = feats.shape[0]
    U = feats[:, :K]  # only u1..uK
    chosen = set()
    centroids = np.zeros((K, feats.shape[1]), dtype=float)

    for k in range(K):
        # sort indices by descending U[:,k]
        order = np.argsort(-U[:, k])
        picked = None
        for idx in order:
            if idx not in chosen:
                picked = idx
                break
        if picked is None:
            # fallback (should be rare)
            picked = (k % N)
        chosen.add(picked)
        centroids[k] = feats[picked]
    return centroids


def _kmeans(
    X: np.ndarray,
    K: int,
    init_centroids: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple K-means (Euclidean) with given initialization.
    Returns (cluster_id, centroids).
    """
    N, d = X.shape
    C = init_centroids.copy()

    for _ in range(max_iter):
        # Assign
        # distances: (N, K)
        distances = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        z = distances.argmin(axis=1)

        # Update
        C_new = C.copy()
        for k in range(K):
            members = X[z == k]
            if members.size == 0:
                # empty cluster: keep old centroid (or re-seed)
                continue
            C_new[k] = members.mean(axis=0)

        shift = np.linalg.norm(C_new - C)
        C = C_new
        if shift < tol:
            break

    return z, C


def _map_clusters_to_classes_one_to_one(
    feats: np.ndarray,
    cluster_id: np.ndarray,
    K: int,
) -> Dict[int, int]:
    """
    Paper step 4-5:
    For each cluster s, compute t_s[k] = sum_{i in cluster s} u_i[k]
    Then map each cluster to one and only one class.

    The paper suggests doing it from largest cluster to smallest (greedy).
    We implement that greedy one-to-one assignment.
    """
    U = feats[:, :K]
    cluster_to_class: Dict[int, int] = {}
    remaining_classes = set(range(K))

    # cluster sizes
    sizes = np.array([(cluster_id == s).sum() for s in range(K)])
    clusters_by_size = np.argsort(-sizes)

    # precompute t_s
    t = np.zeros((K, K), dtype=float)
    for s in range(K):
        members = U[cluster_id == s]
        if members.size == 0:
            continue
        t[s] = members.sum(axis=0)

    for s in clusters_by_size:
        if len(remaining_classes) == 0:
            break
        # pick remaining class with max t[s,k]
        rem = np.array(sorted(list(remaining_classes)))
        k_best = rem[np.argmax(t[s, rem])]
        cluster_to_class[int(s)] = int(k_best)
        remaining_classes.remove(int(k_best))

    # If any cluster didn't get assigned (e.g., empty clusters), assign arbitrarily
    for s in range(K):
        if s not in cluster_to_class:
            if remaining_classes:
                cluster_to_class[s] = int(min(remaining_classes))
                remaining_classes.remove(cluster_to_class[s])
            else:
                cluster_to_class[s] = 0
    return cluster_to_class

def transform_data(rating, n, m):
    L = np.zeros((n,m))
    L = L - 1
    for l in range(len(rating)):
        L[int(rating[l,0]), int(rating[l,1])] = rating[l, 2]
    return L

def gtic(
    rating,
    n: int,
    m: int,
    K: int,
    alpha: Optional[np.ndarray] = None,
    missing_val: int = -1,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> GTICResult:
    """
    GTIC main entry.

    Parameters
    ----------
    L : array (N, J)
        Noisy labels. Each entry in {0,...,K-1} or missing_val if unlabeled by that worker.
    K : int
        Number of classes.
    alpha : array (K,), optional
        Dirichlet prior parameters. Default all ones (uninformative).
    missing_val : int
        Marker for missing labels.
    """
    L = transform_data(rating, n, m)
    
    feats = _compute_u_features(L, K=K, alpha=alpha, missing_val=missing_val)
    initC = _init_centroids_from_u(feats, K=K)
    z, C = _kmeans(feats, K=K, init_centroids=initC, max_iter=max_iter, tol=tol)
    cluster_to_class = _map_clusters_to_classes_one_to_one(feats, z, K=K)

    y_hat = np.array([cluster_to_class[int(s)] for s in z], dtype=int)
    
    

    return GTICResult(
        y_hat=y_hat,
        cluster_id=z.astype(int),
        cluster_to_class=cluster_to_class,
        features=feats,
        centroids=C,
    )



