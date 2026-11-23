"""Intrinsic dimension estimators. TwoNN: Facco et al., Sci. Rep. 7, 12140 (2017)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _mu(X):
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 1e-12
    return r2[ok] / r1[ok]


def twonn(X, discard=0.1):
    """Linear fit of -log(1-F) vs log(mu) through the origin; slope = dimension. Top 10% of mu discarded."""
    mu = np.sort(_mu(X))
    n = len(mu)
    m = int(n * (1 - discard))
    F = np.arange(1, n + 1) / n
    x, y = np.log(mu[:m]), -np.log(1 - F[:m])
    return float((x * y).sum() / (x * x).sum())


def twonn_mle(X):
    """Maximum-likelihood version of the same estimator."""
    mu = _mu(X)
    return float(len(mu) / np.log(mu).sum())
