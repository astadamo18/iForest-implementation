"""Isolation Forest (Liu, Ting & Zhou, 2008), from scratch.

Algorithm 1 iForest(X, t, psi) | Algorithm 2 iTree(X, e, l) | Algorithm 3 PathLength
Eq. (1) c(n) = 2H(n-1) - 2(n-1)/n,  H(i) ~ ln(i) + Euler's constant
Eq. (2) s(x, n) = 2^(-E(h(x)) / c(n))
"""
import math, time
import numpy as np
from sklearn.metrics import roc_auc_score

EULER = 0.5772156649015329


def c_factor(n):
    """Average path length of an unsuccessful search in a binary search tree with n nodes (Eq. 1)."""
    if n <= 1: return 0.0
    if n == 2: return 1.0
    return 2.0 * (math.log(n - 1.0) + EULER) - 2.0 * (n - 1.0) / n


class ITree:
    def __init__(self, X, height_limit, rng):
        self.feat, self.thr, self.left, self.right, self.val = [], [], [], [], []
        self._build(X, 0, height_limit, rng)
        self.feat, self.left, self.right = (np.asarray(a, dtype=np.int64) for a in (self.feat, self.left, self.right))
        self.thr, self.val = np.asarray(self.thr), np.asarray(self.val)

    def _new_node(self):
        for a, v in ((self.feat, -1), (self.thr, 0.0), (self.left, -1), (self.right, -1), (self.val, 0.0)): a.append(v)
        return len(self.feat) - 1

    def _build(self, X, e, l, rng):
        node, size = self._new_node(), X.shape[0]
        if e >= l or size <= 1:                          # external node (height limit or one instance)
            self.val[node] = e + c_factor(size); return node
        mn, mx = X.min(0), X.max(0)
        cand = np.flatnonzero(mx > mn)
        if cand.size == 0:                               # all instances identical
            self.val[node] = e + c_factor(size); return node
        q = int(rng.choice(cand)); p = rng.uniform(mn[q], mx[q])   # random attribute and split value
        mask = X[:, q] < p
        if mask.all() or not mask.any():
            p = (mn[q] + mx[q]) / 2.0; mask = X[:, q] < p
        self.feat[node], self.thr[node] = q, p
        self.left[node] = self._build(X[mask], e + 1, l, rng)
        self.right[node] = self._build(X[~mask], e + 1, l, rng)
        return node

    def path_length(self, X, height_limit):
        rows = np.arange(len(X)); node = np.zeros(len(X), dtype=np.int64)
        for _ in range(height_limit):
            f = self.feat[node]; internal = f >= 0
            if not internal.any(): break
            xv = X[rows, np.where(internal, f, 0)]
            node = np.where(internal, np.where(xv < self.thr[node], self.left[node], self.right[node]), node)
        return self.val[node]                            # e + c(Size) at the external node


def run_iforest(X, y, runs=10, t=100, psi=256):
    """Paper setting: t=100 trees, psi=256, height limit ceil(log2 psi), results averaged over 10 runs.
    Returns (list of AUCs, mean training CPU time in s, mean evaluation CPU time in s)."""
    X = np.ascontiguousarray(X, dtype=np.float64); n = len(X)
    ps = min(psi, n); hl = int(math.ceil(math.log2(ps)))
    aucs, tr, ev = [], [], []
    for seed in range(runs):
        rng = np.random.default_rng(seed)
        t0 = time.process_time()
        trees = [ITree(X[rng.choice(n, ps, replace=False)], hl, rng) for _ in range(t)]   # sub-sample without replacement
        t1 = time.process_time()
        h = sum(tree.path_length(X, hl) for tree in trees) / t
        s = 2.0 ** (-h / c_factor(ps))
        t2 = time.process_time()
        aucs.append(roc_auc_score(y, s)); tr.append(t1 - t0); ev.append(t2 - t1)
    return aucs, float(np.mean(tr)), float(np.mean(ev))
