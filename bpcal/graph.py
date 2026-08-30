import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

MAD_TO_SIGMA = 1.4826


def build_adjacency(coords, k=15, algorithm="auto"):
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < {n}, got {k}")

    dist, nbr = NearestNeighbors(
        n_neighbors=min(k + 40, n),
        algorithm=algorithm).fit(coords).kneighbors(coords)

    # Push self to the end, then order ties by index so the graph does not
    # depend on which k-NN backend scikit-learn selects.
    is_self = nbr == np.arange(n)[:, None]
    order = np.lexsort((nbr, np.where(is_self, np.inf, dist)), axis=1)
    keep = np.take_along_axis(nbr, order, axis=1)[:, :k]

    a = sp.csr_matrix((np.ones(keep.size, np.float32),
                       (np.repeat(np.arange(n), k), keep.ravel())), shape=(n, n))
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    return a


def edge_discrepancy(embedding, adjacency):
    embedding = np.ascontiguousarray(embedding, dtype=np.float64)
    src = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    return np.linalg.norm(embedding[src] - embedding[adjacency.indices], axis=1)


def _row_quantile(rows_sorted, degree, q):
    pos = q * (degree - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, degree - 1)
    frac = pos - lo
    idx = np.arange(len(degree))
    return rows_sorted[idx, lo] * (1 - frac) + rows_sorted[idx, hi] * frac


def local_scale(discrepancy, adjacency):
    n = adjacency.shape[0]
    degree = np.diff(adjacency.indptr)

    # Ragged neighbourhoods padded to a dense array so the per-spot medians
    # and quantiles can be taken by sorting instead of looping.
    padded = np.full((n, degree.max()), np.nan)
    src = np.repeat(np.arange(n), degree)
    padded[src, np.arange(len(discrepancy)) - np.repeat(adjacency.indptr[:-1], degree)] = discrepancy

    median = _row_quantile(np.sort(padded, axis=1), degree, 0.5)
    lower_quartile = _row_quantile(np.sort(padded, axis=1), degree, 0.25)
    mad = _row_quantile(np.sort(np.abs(padded - median[:, None]), axis=1), degree, 0.5)
    return np.maximum(MAD_TO_SIGMA * mad, lower_quartile)
