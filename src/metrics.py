"""Evaluation metrics shared by every method (all computed on held-out test data)."""
import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def recon_stats(X, Xh):
    """Variance explained (1 - SSE/SST, X standardised so mean ~ 0) and RMSE in standardised units."""
    err = X - Xh
    return dict(ve=1 - (err ** 2).sum() / (X ** 2).sum(), rmse=float(np.sqrt((err ** 2).mean())))


def per_feature_rmse(X, Xh):
    return np.sqrt(((X - Xh) ** 2).mean(axis=0))


def trust(X, Z, k=10):
    """Trustworthiness: are neighbours in the embedding true neighbours in the original space? (1 = perfect)"""
    return float(trustworthiness(X, Z, n_neighbors=k))


def knn_overlap(X, Z, k=10):
    """Mean fraction of the k nearest neighbours that are the same before and after reduction."""
    def nn(A):
        return NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A, return_distance=False)[:, 1:]
    a, b = nn(X), nn(Z)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def rp_reconstruct(W, X):
    """Random projection is not invertible: project, then least-squares reconstruct with the pseudo-inverse."""
    Z = X @ W.T
    return Z, Z @ np.linalg.pinv(W.T)
