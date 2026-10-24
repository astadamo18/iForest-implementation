"""ORCA (Bay & Schwabacher, KDD 2003): nested-loop distance-based outliers with pruning.

Setting used for all datasets: attributes scaled to [0,1] (dprep -s01), no shuffling (dprep -norand, original file
order), k = 5, N = n/8 reported outliers, squared Euclidean distance, score = mean distance to the k nearest neighbours,
blocks of 1000 test cases, initial cutoff 0. As in the ORCA log, pruned points get score 0 and every surviving point
keeps its exact score, so all points can be ranked for the AUC. Reported time excludes the scaling (= dprep).
"""
import time
import numpy as np


def run_orca(X, k=5, n_frac=8, block=1000, chunk=2000):
    """Returns (scores, time in seconds)."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    mn, mx = X.min(0), X.max(0)
    X = (X - mn) / np.where(mx > mn, mx - mn, 1.0)
    sq = (X ** 2).sum(1)
    N = n // n_frac

    t0 = time.perf_counter()
    scores = np.zeros(n)
    top = np.empty(0)                                        # the N best scores so far
    cutoff = 0.0
    for start in range(0, n, block):
        B = np.arange(start, min(start + block, n))
        best = np.full((len(B), k), np.inf)
        alive = np.ones(len(B), dtype=bool)
        for s in range(0, n, chunk):
            if not alive.any():
                break
            a = np.flatnonzero(alive)
            bi = B[a]
            e = min(s + chunk, n)
            d2 = sq[bi][:, None] + sq[s:e][None, :] - 2.0 * X[bi] @ X[s:e].T
            d2 = np.maximum(d2, 0.0)
            d2[d2 < 1e-10] = 0.0                             # duplicates: exactly zero distance
            d2[bi[:, None] == np.arange(s, e)[None, :]] = np.inf   # a point is not its own neighbour
            best[a] = np.partition(np.concatenate([best[a], d2], axis=1), k - 1, axis=1)[:, :k]
            sc = best[a].mean(1)
            alive[a[np.isfinite(sc) & (sc <= cutoff)]] = False   # pruning: score <= cutoff
        sc_b = best.mean(1)
        scores[B[alive]] = sc_b[alive]
        top = np.sort(np.concatenate([top, sc_b[alive]]))[::-1][:N]
        if len(top) >= N and top[-1] > cutoff:
            cutoff = top[-1]                                 # new cutoff = N-th best score
    return scores, time.perf_counter() - t0
