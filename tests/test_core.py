"""Tests for the numpy-only XYOnion core (no scikit-learn needed)."""

import numpy as np
import pytest

from clustered_xyonion._core import (
    _max_pairwise_distance,
    _whiten,
    apportion_test_counts,
    round_half_up,
    xyonion_group_split,
)


def make_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = X @ [1.0, -2.0] + 0.1 * rng.normal(size=n)
    return X, y


def test_round_half_up():
    assert [round_half_up(v) for v in (0.5, 1.5, 2.5, 2.49)] == [1, 2, 3, 2]


def test_apportion_is_exact():
    counts = apportion_test_counts([7, 7, 7], 0.25)  # quotas 1.75 each, total 5
    assert counts.sum() == round_half_up(0.25 * 21)
    assert set(counts) <= {1, 2}


def test_apportion_largest_remainder():
    # quotas 2.4, 0.9, 0.6 -> floors 2, 0, 0; total round(3.9) = 4 -> +1 to the
    # two largest remainders (0.9, 0.6).
    np.testing.assert_array_equal(apportion_test_counts([8, 3, 2], 0.3), [2, 1, 1])


def test_max_pairwise_distance_matches_brute_force():
    rng = np.random.RandomState(0)
    Z = rng.normal(size=(50, 3))
    brute = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)).max()
    assert np.isclose(_max_pairwise_distance(Z, chunk_size=7), brute)


def test_whiten_gives_mahalanobis_distance():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 3)) @ np.array([[2.0, 0.5, 0], [0, 1.0, 0.3], [0, 0, 0.2]])
    Z = _whiten(X)
    S_inv = np.linalg.pinv(np.cov(X, rowvar=False))
    d = X[0] - X[1]
    assert np.isclose(np.linalg.norm(Z[0] - Z[1]), np.sqrt(d @ S_inv @ d))


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis"])
def test_group_split_counts_and_first_layer(metric):
    X, y = make_data(100)
    is_test, layer = xyonion_group_split(X, y, 25, 3, metric, np.random.RandomState(0))
    assert is_test.sum() == 25
    assert layer.max() >= 1
    # The outermost layer always goes to train (avoids extrapolation in test).
    assert not is_test[layer == 0].any()
    for k in range(layer.max() + 1):
        members = is_test[layer == k]
        assert members.size == 3 and (members.all() or not members.any())
    # Train reaches 75 in steps of 3 and test reaches 24, so 1 sample is left
    # for the final random split (and must be test to make 25).
    assert (layer == -1).sum() == 1
    assert is_test[layer == -1].all()


def test_layers_keep_running_ratio():
    X, y = make_data(400)
    is_test, layer = xyonion_group_split(X, y, 100, 10, "euclidean", np.random.RandomState(0))
    order = [bool(is_test[layer == k][0]) for k in range(layer.max() + 1)]
    # 1:3 ratio -> roughly one test layer for every three train layers.
    assert order[0] is False
    for k in range(4, len(order) + 1, 4):
        assert sum(order[:k]) == k // 4


def test_group_split_euclidean_outermost_first():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    _, layer = xyonion_group_split(X, y, 1, 1, "euclidean", np.random.RandomState(0))
    assert layer[4] == 0  # far outlier is peeled first


def test_mahalanobis_centroid_is_fixed_euclidean_is_recomputed():
    # Overall mean is 3.5, so 20 is peeled first by both metrics. Then:
    # - fixed centroid (3.5): -3 is 6.5 away, 4 is 0.5 away -> -3 is next;
    # - recomputed centroid (mean of rest = 0.2): -3 is 3.2, 4 is 3.8 -> 4 next.
    X = np.array([[-3.0], [0.0], [0.0], [0.0], [4.0], [20.0]])
    y = np.zeros(6)
    _, lay_e = xyonion_group_split(X, y, 1, 1, "euclidean", np.random.RandomState(0))
    _, lay_m = xyonion_group_split(X, y, 1, 1, "mahalanobis", np.random.RandomState(0))
    assert lay_e[5] == 0 and lay_m[5] == 0
    assert lay_e[4] == 1
    assert lay_m[0] == 1


def test_y_contributes_to_distance():
    # X identical for all samples: only y can decide who is outermost.
    X = np.zeros((5, 2))
    y = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
    _, layer = xyonion_group_split(X, y, 1, 1, "euclidean", np.random.RandomState(0))
    assert layer[4] == 0


def test_constant_y_and_singular_cov_ok():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=40), np.zeros(40)])  # rank-deficient
    y = np.ones(40)
    for metric in ("euclidean", "mahalanobis"):
        is_test, _ = xyonion_group_split(X, y, 10, 1, metric, rng)
        assert is_test.sum() == 10


def test_random_core_is_seeded():
    X, y = make_data(50)
    a, _ = xyonion_group_split(X, y, 13, 6, "euclidean", np.random.RandomState(3))
    b, _ = xyonion_group_split(X, y, 13, 6, "euclidean", np.random.RandomState(3))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_groups(n):
    X, y = make_data(n)
    for n_test in range(n + 1):
        is_test, _ = xyonion_group_split(X, y, n_test, 1, "mahalanobis", np.random.RandomState(0))
        assert is_test.sum() == n_test


def test_group_split_rejects_bad_args():
    X, y = make_data(10)
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError):
        xyonion_group_split(X, y, 11, 1, "euclidean", rng)
    with pytest.raises(ValueError):
        xyonion_group_split(X, y, 2, 0, "euclidean", rng)
    with pytest.raises(ValueError):
        xyonion_group_split(X, y, 2, 1, "cosine", rng)
