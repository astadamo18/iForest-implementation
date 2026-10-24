"""LOF (Breunig et al., 2000), k = 10, raw (unscaled) data.

The k-distance neighbourhood includes all ties (Definitions 3-4 of the LOF paper). Duplicates are not removed;
eps is added to the reachability sum so that lrd stays finite for points with k identical copies.
"""
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors


def run_lof(X, k=10, eps=1e-10):
    """Returns (LOF scores, time in seconds)."""
    X = np.asarray(X, dtype=float)
    t0 = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(X)
    kd = nn.kneighbors(X)[0][:, k]                                   # k-distance (column 0 is the point itself)
    dist, ind = nn.radius_neighbors(X, radius=kd * (1 + 1e-12))      # neighbourhood incl. ties
    keep = [i != p for p, i in enumerate(ind)]
    ind = [i[m] for i, m in zip(ind, keep)]; dist = [d[m] for d, m in zip(dist, keep)]   # remove the point itself
    size = np.array([len(i) for i in ind])
    reach_sum = np.array([np.maximum(kd[i], d).sum() for i, d in zip(ind, dist)]) + eps
    lrd = size / reach_sum
    lof = np.array([lrd[i].mean() for i in ind]) / lrd
    return lof, time.perf_counter() - t0
