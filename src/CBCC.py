"""CBCC: Community-Based Bayesian Classifier Combination (Venanzi et al., WWW'14).

Single-file Python port of the CommunityModel in the authors' C# / Infer.NET code.
Inference is variational message passing with the Blei-Lafferty softmax bound
(as in Infer.NET's SoftmaxOp_BL06). Requires numpy and scipy.

Usage
-----
    from cbcc import cbcc, cbcc_from_matrix

    # labels as three parallel arrays (0-based indices);
    # M is chosen automatically by model evidence over M = 1..10
    res = cbcc(task_idx, worker_idx, labels, n_classes=5, seed=0)

    # or from a (n_tasks, n_workers) matrix with -1 for missing labels
    res = cbcc_from_matrix(L, seed=0)

    # fix M yourself, or change the search
    res = cbcc_from_matrix(L, n_communities=4, seed=0)
    res = cbcc_from_matrix(L, m_range=range(1, 7), n_restarts=5, verbose=True)

    res["pred"]          # (I,)      predicted label per task
    res["posterior"]     # (I, C)    posterior over true labels
    res["n_communities"] # selected M
    res["evidence_by_m"] # {M: log evidence} (automatic mode only)
    res["community"]     # (K, M)    community membership probabilities
    res["worker_cm"]     # (K, C, C) worker confusion matrices (row = true label)
    res["community_cm"]  # (M, C, C) community confusion matrices
    res["log_evidence"]  # variational lower bound on log p(labels)

Model
-----
    p      ~ Dir(1)                         t_i   ~ Cat(p)
    h      ~ Dir(community_pseudo_count)    m_k   ~ Cat(h)
    s^m_c  ~ N(mu0_c, diag(var0_c))         s^k_c ~ N(s^{m_k}_c, I / noise_precision)
    pi^k_c = softmax(s^k_c)                 y_ik  ~ Cat(pi^k_{t_i})
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.special import digamma, gammaln, logsumexp, polygamma, softmax

__all__ = ["cbcc", "cbcc_from_matrix"]

_LOG2PI = np.log(2.0 * np.pi)


# ---------------------------------------------------------------- helpers
def _dir_elog(a):
    return digamma(a) - digamma(a.sum(-1, keepdims=True))


def _dir_kl(q, p):
    q0, p0 = q.sum(-1), p.sum(-1)
    return float(np.sum(gammaln(q0) - gammaln(q).sum(-1) - gammaln(p0) + gammaln(p).sum(-1)
                        + ((q - p) * _dir_elog(q)).sum(-1)))


def _normalise_log(x):
    return np.exp(x - logsumexp(x, axis=-1, keepdims=True))


def _neg_entropy(p):
    return float(np.sum(p * np.log(np.where(p > 0, p, 1.0))))


def _expected_counts(task, worker, label, q_t, K, C):
    """N[k, c, j] = sum over worker k's labels equal to j of q(t_i = c)."""
    N = np.zeros((K, C, C))
    for c in range(C):
        np.add.at(N[:, c, :], (worker, label), q_t[task, c])
    return N


def _task_messages(task, worker, label, elog_pi, I, C):
    msg = np.zeros((I, C))
    np.add.at(msg, task, elog_pi[worker, :, label])
    return msg


def _update_worker_scores(mu, var, prior_mean, N, v, newton_steps):
    """Optimise each worker score row under the Blei-Lafferty bound:
    fixed point for the (diagonal) variances, Newton steps for the means."""
    C = mu.shape[-1]
    eye = np.eye(C)
    Nt = N.sum(-1)[..., None]
    for _ in range(newton_steps):
        w = softmax(mu + 0.5 * var, axis=-1)
        var = 1.0 / (v + Nt * w)
        w = softmax(mu + 0.5 * var, axis=-1)
        grad = N - Nt * w - v * (mu - prior_mean)
        hess = Nt[..., None] * (w[..., :, None] * eye - w[..., :, None] * w[..., None, :]) + v * eye
        mu = mu + np.linalg.solve(hess, grad[..., None])[..., 0]
    var = 1.0 / (v + Nt * softmax(mu + 0.5 * var, axis=-1))
    return mu, var


def _elog_pi(mu, var):
    return mu - logsumexp(mu + 0.5 * var, axis=-1, keepdims=True)


# ---------------------------------------------------------------- main entry
def cbcc(task, worker, label, n_classes=None, n_tasks=None, n_workers=None,
         n_communities="auto", m_range=range(1, 11), n_restarts=3, verbose=False,
         noise_precision=5.0, community_pseudo_count=10.0,
         initial_worker_belief=0.5, n_iter=35, tol=None, init="kmeans",
         newton_steps=2, seed=None):
    """Run CBCC on a set of crowd labels.

    Parameters
    ----------
    task, worker, label : int arrays of equal length, 0-based indices; one entry per label.
    n_classes, n_tasks, n_workers : sizes (inferred from the data if None).
    n_communities : number of worker communities M, or "auto" (default) to select M by
        model evidence: every M in `m_range` is fitted `n_restarts` times and the single
        run with the highest log evidence is returned (Venanzi et al., 2014).
    m_range : candidate values of M when n_communities="auto" (default 1..10).
    n_restarts : random restarts per M when n_communities="auto"; the best one is kept.
    verbose : print the evidence of each M during the search.
    noise_precision : precision of worker scores around their community (C#: NoisePrecision = 5).
    community_pseudo_count : symmetric Dirichlet prior on community proportions (C#: 10).
    initial_worker_belief : b; prior confusion rows ~ Dir(1, ..., b/(1-b)*(C-1), ..., 1) (C#: 0.5).
    n_iter : VMP iterations (C#: 35).
    tol : stop early when the ELBO changes by less than tol (None = always run n_iter).
    init : "kmeans" (k-means++ on warmed-up worker scores; more robust) or
           "random" (random community per worker, as in the C# code).
    seed : int or np.random.Generator.

    Returns
    -------
    dict with pred, posterior, community, worker_cm, community_cm, community_prob,
    log_evidence, elbo_trace, n_communities, n_effective_communities, and, when M was
    selected automatically, evidence_by_m = {M: best log evidence over restarts}.
    """
    task, worker, label = (np.asarray(a, dtype=int).ravel() for a in (task, worker, label))
    if not (len(task) == len(worker) == len(label)):
        raise ValueError("task, worker and label must have the same length")
    sizes = dict(
        n_classes=int(n_classes if n_classes is not None else label.max() + 1),
        n_tasks=int(n_tasks if n_tasks is not None else task.max() + 1),
        n_workers=int(n_workers if n_workers is not None else worker.max() + 1),
    )
    settings = dict(noise_precision=noise_precision, community_pseudo_count=community_pseudo_count,
                    initial_worker_belief=initial_worker_belief, n_iter=n_iter, tol=tol,
                    init=init, newton_steps=newton_steps)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    if not (isinstance(n_communities, str) and n_communities == "auto"):
        return _cbcc_fixed_m(task, worker, label, n_communities=int(n_communities),
                             rng=rng, **sizes, **settings)

    # ---- select M by model evidence
    candidates = sorted({int(m) for m in m_range if 1 <= int(m) <= sizes["n_workers"]})
    if not candidates:
        raise ValueError("m_range contains no valid community counts")
    best, evidence_by_m = None, {}
    for M in candidates:
        restarts = 1 if M == 1 else n_restarts      # M = 1 has no initialisation randomness
        best_m = None
        for _ in range(restarts):
            res = _cbcc_fixed_m(task, worker, label, n_communities=M, rng=rng, **sizes, **settings)
            if best_m is None or res["log_evidence"] > best_m["log_evidence"]:
                best_m = res
        evidence_by_m[M] = best_m["log_evidence"]
        if verbose:
            print(f"M={M:2d}  log evidence = {best_m['log_evidence']:.2f}  "
                  f"(effective communities: {best_m['n_effective_communities']})")
        if best is None or best_m["log_evidence"] > best["log_evidence"]:
            best = best_m
    best["evidence_by_m"] = evidence_by_m
    if verbose:
        print(f"selected M = {best['n_communities']}")
    return best


def _cbcc_fixed_m(task, worker, label, n_classes, n_tasks, n_workers, n_communities,
                  noise_precision, community_pseudo_count, initial_worker_belief,
                  n_iter, tol, init, newton_steps, rng):
    """One CBCC fit with a fixed number of communities."""
    C, I, K, M = n_classes, n_tasks, n_workers, n_communities
    v = float(noise_precision)

    # Community score prior: Gaussian whose softmax approximates
    # Dir(1, ..., b/(1-b)*(C-1), ..., 1) per row (log-Gamma moment matching).
    a = np.ones((C, C))
    np.fill_diagonal(a, initial_worker_belief / (1.0 - initial_worker_belief) * (C - 1))
    mu0 = np.broadcast_to(digamma(a), (M, C, C)).copy()
    lam0 = np.broadcast_to(1.0 / polygamma(1, a), (M, C, C)).copy()
    beta0 = np.ones(C)
    h0 = np.full(M, float(community_pseudo_count))

    # ---- initialisation
    # q(t) starts from a soft majority vote. (The C# starts it uniform; with C = 2 the
    # score prior is then symmetric and inference never leaves the 50/50 fixed point.)
    votes = np.zeros((I, C))
    np.add.at(votes, (task, label), 1.0)
    q_t = (votes + 1.0) / (votes + 1.0).sum(1, keepdims=True)
    var_k = np.full((K, C, C), 1.0 / v)
    mu_m, var_m = mu0.copy(), 1.0 / lam0
    q_m = np.zeros((K, M))
    if init == "kmeans" and M > 1:
        mu_k = np.broadcast_to(mu0[0], (K, C, C)).copy()
        for _ in range(10):        # warm-up with the shared prior, ignoring communities
            N = _expected_counts(task, worker, label, q_t, K, C)
            mu_k, var_k = _update_worker_scores(mu_k, var_k, mu0[0][None], N, v, newton_steps)
            q_t = _normalise_log(_task_messages(task, worker, label, _elog_pi(mu_k, var_k), I, C))
        _, assign = kmeans2(mu_k.reshape(K, -1), M, minit="++", seed=rng)
        q_m[np.arange(K), assign] = 1.0
        prec_m = lam0 + v * q_m.sum(0)[:, None, None]
        mu_m = (lam0 * mu0 + v * np.einsum("km,kcj->mcj", q_m, mu_k)) / prec_m
        var_m = 1.0 / prec_m
    elif init in ("random", "kmeans"):
        q_m[np.arange(K), rng.integers(M, size=K)] = 1.0
        mu_k = np.einsum("km,mcj->kcj", q_m, mu_m)
    else:
        raise ValueError("init must be 'kmeans' or 'random'")
    h = h0 + q_m.sum(0)
    beta = beta0 + q_t.sum(0)

    # ---- variational message passing
    trace = []
    for _ in range(n_iter):
        # worker score matrices
        N = _expected_counts(task, worker, label, q_t, K, C)
        mu_k, var_k = _update_worker_scores(mu_k, var_k, np.einsum("km,mcj->kcj", q_m, mu_m),
                                            N, v, newton_steps)
        # community score matrices
        prec_m = lam0 + v * q_m.sum(0)[:, None, None]
        mu_m = (lam0 * mu0 + v * np.einsum("km,kcj->mcj", q_m, mu_k)) / prec_m
        var_m = 1.0 / prec_m
        # community memberships and proportions
        sq = ((mu_k[:, None] - mu_m[None]) ** 2).sum((-1, -2))
        q_m = _normalise_log(_dir_elog(h)[None] - 0.5 * v * (sq + var_m.sum((-1, -2))[None]))
        h = h0 + q_m.sum(0)
        # true labels and class proportions
        elog_pi = _elog_pi(mu_k, var_k)
        q_t = _normalise_log(_dir_elog(beta)[None] + _task_messages(task, worker, label, elog_pi, I, C))
        beta = beta0 + q_t.sum(0)

        # evidence lower bound
        sq = ((mu_k[:, None] - mu_m[None]) ** 2).sum((-1, -2))
        e = float(np.sum(q_t[task] * elog_pi[worker, :, label]))
        e += float(np.sum(q_t * _dir_elog(beta))) - _neg_entropy(q_t) - _dir_kl(beta, beta0)
        e += float(np.sum(q_m * _dir_elog(h))) - _neg_entropy(q_m) - _dir_kl(h, h0)
        e += float(np.sum(q_m * (0.5 * C * C * (np.log(v) - _LOG2PI)
                                 - 0.5 * v * (sq + var_k.sum((1, 2))[:, None] + var_m.sum((1, 2))[None]))))
        e += 0.5 * float(np.sum(_LOG2PI + 1.0 + np.log(var_k)))
        e += float(np.sum(0.5 * (np.log(lam0) - _LOG2PI) - 0.5 * lam0 * ((mu_m - mu0) ** 2 + var_m)))
        e += 0.5 * float(np.sum(_LOG2PI + 1.0 + np.log(var_m)))
        trace.append(e)
        if tol is not None and len(trace) > 1 and abs(trace[-1] - trace[-2]) < tol:
            break

    return {
        "pred": q_t.argmax(1),
        "posterior": q_t,
        "community": q_m,
        "worker_cm": softmax(mu_k, axis=-1),
        "community_cm": softmax(mu_m, axis=-1),
        "community_prob": h / h.sum(),
        "log_evidence": trace[-1],
        "elbo_trace": np.array(trace),
        "n_communities": M,
        # communities holding at least half a worker's worth of membership
        "n_effective_communities": int((q_m.sum(0) >= 0.5).sum()),
    }


def cbcc_from_matrix(L, missing=-1, n_classes=None, **kwargs):
    """CBCC on a (n_tasks, n_workers) label matrix with `missing` for absent labels.
    Labels must be 0-based class indices. Extra keyword arguments go to `cbcc`."""
    L = np.asarray(L)
    task, worker = np.nonzero(L != missing)
    return cbcc(task, worker, L[task, worker], n_classes=n_classes,
                n_tasks=L.shape[0], n_workers=L.shape[1], **kwargs)


if __name__ == "__main__":
    # quick self-test on synthetic data with 4 worker communities
    rng = np.random.default_rng(0)
    I, K, C = 300, 60, 5
    eye = np.eye(C)
    cms = [0.85 * eye + 0.15 / C, np.full((C, C), 1 / C),
           0.45 * eye + 0.45 * eye[[0] * C] + 0.1 / C, 0.6 * eye + 0.3 * np.roll(eye, 1, 1) + 0.1 / C]
    comm = rng.integers(4, size=K)
    truth = rng.integers(C, size=I)
    L = -np.ones((I, K), dtype=int)
    for i in range(I):
        for k in rng.choice(K, 10, replace=False):
            L[i, k] = rng.choice(C, p=cms[comm[k]][truth[i]])
    import time
    t0 = time.time()
    res = cbcc_from_matrix(L, seed=0, verbose=True)          # automatic M
    mv = np.array([np.bincount(r[r >= 0], minlength=C).argmax() for r in L])
    print(f"majority vote acc = {(mv == truth).mean():.3f}")
    print(f"CBCC acc          = {(res['pred'] == truth).mean():.3f}   "
          f"selected M = {res['n_communities']}   ({time.time() - t0:.1f}s)")
    print("community sizes   =", np.round(res["community"].sum(0)).astype(int), " true:", np.bincount(comm))