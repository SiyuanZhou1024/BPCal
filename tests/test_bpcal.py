import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from bpcal import (build_adjacency, calibrate, cluster, compatibility_kernel,
                   edge_discrepancy, evaluate, local_scale, normalize,
                   preprocess, row_stochastic, run)

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def toy():
    n = 60
    coords = np.vstack([RNG.uniform([0, 0], [10, 10], (n, 2)),
                        RNG.uniform([10, 0], [20, 10], (n, 2))])
    embedding = np.vstack([RNG.normal(0.0, 1.0, (n, 5)),
                           RNG.normal(6.0, 1.0, (n, 5))])
    return coords, embedding


@pytest.fixture(scope="module")
def chain(toy):
    coords, embedding = toy
    adjacency = build_adjacency(coords, 6)
    discrepancy = edge_discrepancy(embedding, adjacency)
    scale = local_scale(discrepancy, adjacency)
    kernel = compatibility_kernel(discrepancy, scale, adjacency, 2.0)
    return adjacency, discrepancy, scale, kernel


def test_normalize_scales_each_row_to_target():
    counts = RNG.poisson(5.0, (50, 8)).astype(float)
    np.testing.assert_allclose(np.expm1(normalize(counts)).sum(1), 1e4, rtol=1e-9)


def test_normalize_tolerates_empty_rows():
    assert np.all(normalize(np.zeros((3, 4))) == 0.0)


def test_adjacency_is_symmetric_and_hollow(toy):
    coords, _ = toy
    adjacency = build_adjacency(coords, 6)
    assert (adjacency != adjacency.T).nnz == 0
    assert adjacency.diagonal().sum() == 0


def test_adjacency_matches_brute_force_or_rule(toy):
    coords, _ = toy
    k, n = 6, len(coords)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)

    expected = np.zeros((n, n), dtype=bool)
    for i in range(n):
        expected[i, idx[i, 1:]] = True
    expected |= expected.T
    np.fill_diagonal(expected, False)

    assert np.array_equal(build_adjacency(coords, k).toarray() > 0, expected)


def test_symmetrisation_can_raise_degree_above_k(toy):
    coords, _ = toy
    degree = np.diff(build_adjacency(coords, 6).indptr)
    assert degree.min() >= 6
    assert degree.max() > 6


@pytest.mark.parametrize("backend", ["ball_tree", "kd_tree", "brute"])
def test_adjacency_is_backend_independent(toy, backend):
    coords, _ = toy
    reference = build_adjacency(coords, 6)
    assert (build_adjacency(coords, 6, algorithm=backend) != reference).nnz == 0


def test_rejects_out_of_range_k(toy):
    coords, _ = toy
    with pytest.raises(ValueError):
        build_adjacency(coords, 0)
    with pytest.raises(ValueError):
        build_adjacency(coords, len(coords))


def test_discrepancy_matches_pairwise_norms(toy, chain):
    _, embedding = toy
    adjacency, discrepancy, _, _ = chain
    src = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    expected = np.linalg.norm(embedding[src] - embedding[adjacency.indices], axis=1)
    np.testing.assert_allclose(discrepancy, expected, atol=1e-12)


def test_local_scale_matches_loop(chain):
    adjacency, discrepancy, scale, _ = chain
    for i in range(adjacency.shape[0]):
        row = discrepancy[adjacency.indptr[i]:adjacency.indptr[i + 1]]
        mad = np.median(np.abs(row - np.median(row)))
        assert scale[i] == pytest.approx(
            max(1.4826 * mad, np.percentile(row, 25)), abs=1e-12)


def test_lower_quartile_floor_engages_when_mad_vanishes():
    coords = np.array([[x, y] for x in range(6) for y in range(6)], float)
    embedding = np.zeros((36, 3))
    embedding[:, 0] = np.arange(36) * 1e-9

    adjacency = build_adjacency(coords, 4)
    discrepancy = edge_discrepancy(embedding, adjacency)
    scale = local_scale(discrepancy, adjacency)

    mad_only = np.array([
        1.4826 * np.median(np.abs(row - np.median(row)))
        for row in (discrepancy[adjacency.indptr[i]:adjacency.indptr[i + 1]]
                    for i in range(36))])

    assert np.any(scale > mad_only + 1e-18)
    assert np.all(scale > 0)


def test_kernel_matches_closed_form(chain):
    adjacency, discrepancy, scale, kernel = chain
    src = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr))
    dst = adjacency.indices
    expected = np.exp(-discrepancy ** 2 / (4.0 * scale[src] * scale[dst] + 1e-8))
    np.testing.assert_allclose(kernel.toarray()[src, dst], expected, rtol=1e-10)


def test_kernel_is_symmetric_and_confined_to_edges(chain):
    adjacency, _, _, kernel = chain
    assert abs(kernel - kernel.T).max() < 1e-14
    assert np.all(kernel.toarray()[adjacency.toarray() == 0] == 0)


def test_kernel_rejects_non_positive_gamma(chain):
    adjacency, discrepancy, scale, _ = chain
    with pytest.raises(ValueError):
        compatibility_kernel(discrepancy, scale, adjacency, 0.0)


def test_operator_rows_sum_to_one(chain):
    _, _, _, kernel = chain
    operator = row_stochastic(kernel)
    np.testing.assert_allclose(np.asarray(operator.sum(1)).ravel(), 1.0, atol=1e-12)
    assert operator.min() >= 0.0


def test_calibration_equals_matrix_power(toy, chain):
    _, embedding = toy
    _, _, _, kernel = chain
    operator = row_stochastic(kernel)

    eta, steps = 0.3, 10
    dense = (1 - eta) * np.eye(len(embedding)) + eta * operator.toarray()
    expected = np.linalg.matrix_power(dense, steps) @ embedding

    np.testing.assert_allclose(
        calibrate(embedding, operator, eta, steps), expected, atol=1e-9)


def test_zero_steps_is_the_identity(toy, chain):
    _, embedding = toy
    _, _, _, kernel = chain
    np.testing.assert_allclose(
        calibrate(embedding, row_stochastic(kernel), 0.3, 0), embedding)


def test_calibration_stays_inside_the_initial_hull(toy, chain):
    _, embedding = toy
    _, _, _, kernel = chain
    operator = row_stochastic(kernel)
    low, high = embedding.min(0), embedding.max(0)

    for steps in (1, 5, 25):
        calibrated = calibrate(embedding, operator, 0.3, steps)
        assert np.all(calibrated >= low - 1e-9)
        assert np.all(calibrated <= high + 1e-9)


def test_calibration_rejects_eta_outside_unit_interval(toy, chain):
    _, embedding = toy
    _, _, _, kernel = chain
    operator = row_stochastic(kernel)
    for eta in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            calibrate(embedding, operator, eta, 5)


def test_kernel_weight_decreases_with_discrepancy():
    delta = np.linspace(0.01, 5.0, 50)
    assert np.all(np.diff(np.exp(-delta ** 2 / (4.0 + 1e-8))) < 0)


def test_kernel_weight_increases_with_scale():
    product = np.linspace(0.1, 5.0, 50)
    assert np.all(np.diff(np.exp(-1.0 / (4.0 * product + 1e-8))) > 0)


def test_operations_stay_sparse(chain):
    adjacency, _, _, kernel = chain
    assert sp.issparse(kernel)
    assert sp.issparse(row_stochastic(kernel))
    assert kernel.nnz <= adjacency.nnz


def test_preprocess_returns_requested_dimension():
    counts = RNG.poisson(4.0, (80, 40)).astype(float)
    assert preprocess(counts, 12).shape == (80, 12)


def test_clustering_is_reproducible(toy):
    _, embedding = toy
    np.testing.assert_array_equal(cluster(embedding, 2), cluster(embedding, 2))


def test_evaluate_reports_perfect_agreement():
    labels = np.repeat([0, 1, 2], 10)
    scores = evaluate(labels, labels)
    assert scores["ARI"] == pytest.approx(100.0)
    assert scores["NMI"] == pytest.approx(100.0)


def test_evaluate_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        evaluate(np.zeros(5), np.zeros(6))


def test_run_recovers_two_separated_domains(toy):
    coords, _ = toy
    n = len(coords) // 2
    counts = np.vstack([RNG.poisson(2.0, (n, 40)), RNG.poisson(9.0, (n, 40))])
    labels = run(counts, coords, 2, k=6, n_pcs=5, steps=5)
    assert evaluate(labels, np.repeat([0, 1], n))["ARI"] > 90.0


def test_run_is_deterministic(toy):
    coords, _ = toy
    n = len(coords) // 2
    counts = np.vstack([RNG.poisson(2.0, (n, 40)), RNG.poisson(9.0, (n, 40))])
    a = run(counts, coords, 2, k=6, n_pcs=5, steps=5)
    b = run(counts, coords, 2, k=6, n_pcs=5, steps=5)
    np.testing.assert_array_equal(a, b)
