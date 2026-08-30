import numpy as np
import scipy.sparse as sp

EPS = 1e-8


def compatibility_kernel(discrepancy, scale, adjacency, gamma=2.0, eps=EPS):
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    n = adjacency.shape[0]
    src = np.repeat(np.arange(n), np.diff(adjacency.indptr))
    dst = adjacency.indices

    w = np.exp(-discrepancy ** 2 / (gamma ** 2 * scale[src] * scale[dst] + eps))
    kernel = sp.csr_matrix((w, (src, dst)), shape=(n, n))

    # Symmetric in exact arithmetic; averaged as a safeguard.
    kernel = (kernel + kernel.T) * 0.5
    kernel.sort_indices()
    return kernel


def row_stochastic(kernel):
    degree = np.asarray(kernel.sum(1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("kernel has an empty row; the operator is undefined")
    return (sp.diags(1.0 / degree) @ kernel).tocsr()


def calibrate(embedding, operator, eta=0.3, steps=10):
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must lie in (0, 1), got {eta}")

    h = np.ascontiguousarray(embedding, dtype=np.float64).copy()
    for _ in range(steps):
        h = (1 - eta) * h + eta * (operator @ h)
    return h
