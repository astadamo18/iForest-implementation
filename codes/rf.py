"""Unsupervised Random Forest outlier detection (Shi & Horvath, 2006).

Real data = class 1, synthetic data (each attribute uniform between its minimum and maximum, same size) = class 2
(Addcl2); t = 100 trees, other parameters default. Proximity = share of trees in which two points fall in the same
leaf; outlier score = n / sum_k prox(i, k)^2.
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def run_rf(X, y, runs=10, t=100):
    """Returns (list of AUCs, mean time per run in seconds)."""
    X = np.asarray(X, dtype=float); n, d = X.shape
    aucs, times = [], []
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        Xs = rng.uniform(X.min(0), X.max(0), size=(n, d))
        rf = RandomForestClassifier(n_estimators=t, random_state=seed, n_jobs=1)
        rf.fit(np.vstack([X, Xs]), np.r_[np.ones(n), np.zeros(n)])
        leaf = rf.apply(X)
        P = np.zeros((n, n))
        for j in range(t):
            P += leaf[:, [j]] == leaf[:, j][None, :]
        P /= t; np.fill_diagonal(P, 0)
        score = n / np.maximum((P ** 2).sum(1), 1e-12)
        times.append(time.perf_counter() - t0)
        aucs.append(roc_auc_score(y, score))
    return aucs, float(np.mean(times))
