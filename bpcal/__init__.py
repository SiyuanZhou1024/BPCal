"""BPCal: boundary-aware prototype calibration for spatial domain identification."""

from .cluster import cluster, evaluate
from .core import calibrate, compatibility_kernel, row_stochastic
from .data import load_h5ad, load_loom, load_npz, load_visium, normalize, preprocess
from .graph import build_adjacency, edge_discrepancy, local_scale

__all__ = [
    "run",
    "normalize", "preprocess",
    "load_npz", "load_loom", "load_h5ad", "load_visium",
    "build_adjacency", "edge_discrepancy", "local_scale",
    "compatibility_kernel", "row_stochastic", "calibrate",
    "cluster", "evaluate",
]

__version__ = "1.0.0"


def run(counts, coords, n_clusters, k=15, gamma=2.0, eta=0.3, steps=10, n_pcs=30):
    embedding = preprocess(counts, n_pcs)
    adjacency = build_adjacency(coords, k)
    discrepancy = edge_discrepancy(embedding, adjacency)
    scale = local_scale(discrepancy, adjacency)
    kernel = compatibility_kernel(discrepancy, scale, adjacency, gamma)
    calibrated = calibrate(embedding, row_stochastic(kernel), eta, steps)
    return cluster(calibrated, n_clusters)
