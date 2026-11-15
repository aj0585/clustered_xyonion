"""Tests for the numpy-only XYOnion core (no scikit-learn needed)."""

import numpy as np
import pytest

from clustered_xyonion._core import (
    TEST,
    TRAIN,
    VALIDATION,
    _max_pairwise_distance,
    _whiten,
    apportion_counts,
    round_half_up,
    xyonion_group_split,
)


def make_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = X @ [1.0, -2.0] + 0.1 * rng.normal(size=n)
    return X, y


def split(X, y, n_test, layer_size, metric="euclidean", seed=0, n_validation=0):
    return xyonion_group_split(
        X, y, n_test, layer_size, metric, np.random.RandomState(seed),
        n_validation=n_validation,
    )


def test_round_half_up():
    assert [round_half_up(v) for v in (0.5, 1.5, 2.5, 2.49)] == [1, 2, 3, 2]


# --- apportion_counts ---------------------------------------------------


def test_apportion_is_exact():
    counts = apportion_counts([7, 7, 7], 0.25)  # quotas 1.75 each, total 5
    assert counts.sum() == round_half_up(0.25 * 21)
    assert set(counts) <= {1, 2}


def test_apportion_largest_remainder():
    # quotas 2.4, 0.9, 0.6 -> floors 2, 0, 0; total round(3.9) = 4 -> +1 to the
    # two largest remainders (0.9, 0.6).
    np.testing.assert_array_equal(apportion_counts([8, 3, 2], 0.3), [2, 1, 1])


def test_apportion_zero_fraction():
    np.testing.assert_array_equal(apportion_counts([5, 9], 0.0), [0, 0])


def test_apportion_respects_capacity():
    # Six groups of 1 with 30% test and 30% validation: test takes 2 groups,
    # validation must use 2 *other* groups because each group has room for 1.
    sizes = np.ones(6, dtype=int)
    test = apportion_counts(sizes, 0.3)
    val = apportion_counts(sizes, 0.3, capacity=sizes - test)
    assert test.sum() == 2 and val.sum() == 2
    assert (test + val <= sizes).all()


# --- distance helpers ---------------------------------------------------


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


def test_whiten_drops_zero_variance_direction():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=30), np.full(30, 4.0)])
    assert _whiten(X).shape == (30, 1)


# --- two subsets (train/test) -------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis"])
def test_group_split_counts_and_first_layer(metric):
    X, y = make_data(100)
    subset, layer = split(X, y, 25, 3, metric)
    assert (subset == TEST).sum() == 25
    assert not (subset == VALIDATION).any()
    assert layer.max() >= 1
    # The outermost layer always goes to train (avoids extrapolation in test).
    assert (subset[layer == 0] == TRAIN).all()
    for k in range(layer.max() + 1):
        members = subset[layer == k]
        assert members.size == 3 and len(set(members)) == 1
    # Train reaches 75 in steps of 3 and test reaches 24, so 1 sample is left
    # for the final random split (and must be test to make 25).
    assert (layer == -1).sum() == 1
    assert (subset[layer == -1] == TEST).all()


def test_layers_keep_running_ratio():
    X, y = make_data(400)
    subset, layer = split(X, y, 100, 10)
    order = [int(subset[layer == k][0]) for k in range(layer.max() + 1)]
    # 1:3 ratio -> one test layer for every three train layers.
    assert order[0] == TRAIN
    for k in range(4, len(order) + 1, 4):
        assert order[:k].count(TEST) == k // 4


def test_group_split_euclidean_outermost_first():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    _, layer = split(X, y, 1, 1)
    assert layer[4] == 0  # far outlier is peeled first


def test_mahalanobis_centroid_is_fixed_euclidean_is_recomputed():
    # Overall mean is 3.5, so 20 is peeled first by both metrics. Then:
    # - fixed centroid (3.5): -3 is 6.5 away, 4 is 0.5 away -> -3 is next;
    # - recomputed centroid (mean of rest = 0.2): -3 is 3.2, 4 is 3.8 -> 4 next.
    X = np.array([[-3.0], [0.0], [0.0], [0.0], [4.0], [20.0]])
    y = np.zeros(6)
    _, lay_e = split(X, y, 1, 1, "euclidean")
    _, lay_m = split(X, y, 1, 1, "mahalanobis")
    assert lay_e[5] == 0 and lay_m[5] == 0
    assert lay_e[4] == 1
    assert lay_m[0] == 1


def test_y_contributes_to_distance():
    # X identical for all samples: only y can decide who is outermost.
    X = np.zeros((5, 2))
    y = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
    _, layer = split(X, y, 1, 1)
    assert layer[4] == 0


def test_constant_y_and_singular_cov_ok():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=40), np.zeros(40)])  # rank-deficient
    y = np.ones(40)
    for metric in ("euclidean", "mahalanobis"):
        subset, _ = split(X, y, 10, 1, metric)
        assert (subset == TEST).sum() == 10


def test_random_core_is_seeded():
    X, y = make_data(50)
    a, _ = split(X, y, 13, 6, seed=3)
    b, _ = split(X, y, 13, 6, seed=3)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_groups(n):
    X, y = make_data(n)
    for n_test in range(n + 1):
        subset, _ = split(X, y, n_test, 1, "mahalanobis")
        assert (subset == TEST).sum() == n_test


def test_group_split_rejects_bad_args():
    X, y = make_data(10)
    with pytest.raises(ValueError):
        split(X, y, 11, 1)
    with pytest.raises(ValueError):
        split(X, y, 6, 1, n_validation=5)
    with pytest.raises(ValueError):
        split(X, y, 2, 0)
    with pytest.raises(ValueError):
        split(X, y, 2, 1, "cosine")


# --- three subsets (train/validation/test) ------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis"])
def test_three_way_exact_counts_and_whole_layers(metric):
    X, y = make_data(200)
    subset, layer = split(X, y, 30, 4, metric, n_validation=30)
    assert np.bincount(subset, minlength=3).tolist() == [140, 30, 30]
    assert (subset[layer == 0] == TRAIN).all()
    for k in range(layer.max() + 1):
        assert len(set(subset[layer == k])) == 1


def test_three_way_validation_and_test_share_depths():
    # With equal validation and test fractions, both should get layers from
    # the outside, the middle and the inside, alternating with each other.
    X, y = make_data(400)
    subset, layer = split(X, y, 60, 6, n_validation=60)
    held_out = [int(subset[layer == k][0]) for k in range(layer.max() + 1)
                if subset[layer == k][0] != TRAIN]
    assert held_out.count(VALIDATION) == held_out.count(TEST)
    assert all(a != b for a, b in zip(held_out, held_out[1:]))


def test_two_way_unchanged_when_validation_is_zero():
    X, y = make_data(120)
    a, la = split(X, y, 30, 3, seed=5)
    b, lb = split(X, y, 30, 3, seed=5, n_validation=0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(la, lb)
