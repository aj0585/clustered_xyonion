"""Tests for the numpy-only XYOnion core (no scikit-learn needed)."""

import numpy as np
import pytest

from clustered_xyonion._core import (
    TEST,
    TRAIN,
    VALIDATION,
    _stdsslct,
    _whiten,
    apportion_counts,
    distslct,
    round_half_up,
    xyonion_group_split,
)


def make_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = X @ [1.0, -2.0] + 0.1 * rng.normal(size=n)
    return X, y


def split(X, y, counts, fractions=None, n_onion=3, loop_fraction=0.1, seed=0):
    n = X.shape[0]
    if fractions is None:
        fractions = tuple(c / n for c in counts)
    return xyonion_group_split(X, y, counts, fractions, n_onion, loop_fraction,
                               np.random.RandomState(seed))


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
    sizes = np.ones(6, dtype=int)
    test = apportion_counts(sizes, 0.3)
    val = apportion_counts(sizes, 0.3, capacity=sizes - test)
    assert test.sum() == 2 and val.sum() == 2
    assert (test + val <= sizes).all()


# --- whitening ----------------------------------------------------------


def test_whiten_gives_mahalanobis_distance():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 3)) @ np.array([[2.0, 0.5, 0], [0, 1.0, 0.3], [0, 0, 0.2]])
    Z = _whiten(X)
    S_inv = np.linalg.pinv(np.cov(X, rowvar=False))
    d = X[0] - X[1]
    assert np.isclose(np.linalg.norm(Z[0] - Z[1]), np.sqrt(d @ S_inv @ d))


def test_whiten_matches_reference_zca_distances():
    # The reference uses V diag(1/sqrt(lam)) V^T; ours omits the final V^T
    # rotation, which leaves every distance unchanged.
    rng = np.random.RandomState(1)
    X = rng.normal(size=(40, 3)) @ rng.normal(size=(3, 3))
    lam, V = np.linalg.eigh(np.cov(X, rowvar=False))
    Z_ref = X @ (V @ np.diag(1 / np.sqrt(lam)) @ V.T)
    Z = _whiten(X)
    d_ref = np.linalg.norm(Z_ref[:, None] - Z_ref[None], axis=2)
    d = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    np.testing.assert_allclose(d, d_ref, atol=1e-9)


def test_whiten_drops_zero_variance_direction():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=30), np.full(30, 4.0)])
    assert _whiten(X).shape == (30, 1)


# --- DISTSLCT -----------------------------------------------------------


def test_distslct_hand_worked_example():
    # One feature, constant y (so only X counts). Centred x is
    # [-3.2, -2.2, -1.2, -0.2, 6.8]:
    # 1st pick (largest x^2): index 4 (x=10).
    # 2nd pick: largest distance to x=10 -> index 0 (x=0).
    # 3rd pick: summed distances to {10, 0} are 10 for x=1, 2 and 3 -> tie,
    #           first wins -> index 1.
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
    y = np.zeros(5)
    assert distslct(X, y, 3).tolist() == [4, 0, 1]


def test_stdsslct_orthogonalises():
    # Picks the longest vector, then the one most orthogonal to it.
    X = np.array([[5.0, 0.0], [4.0, 0.5], [0.0, 3.0], [-1.0, -1.0]])
    y = np.zeros(4)
    assert _stdsslct(X, y, 2).tolist() == [0, 2]


def test_distslct_y_breaks_ties_in_x():
    X = np.zeros((5, 2))
    y = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
    assert distslct(X, y, 1).tolist() == [4]


def test_distslct_unique_and_bounded():
    X, y = make_data(30)
    picks = distslct(X, y, 25)
    assert len(picks) == len(set(picks)) == 25
    assert distslct(X, y, 100).size == 30
    assert distslct(X, y, 0).size == 0


def test_distslct_recentres_and_balances():
    # Shifting X or y, or rescaling y, must not change the picks: X and y are
    # re-centred on the samples given, and each block is divided by its max.
    X, y = make_data(80)
    base = distslct(X, y, 12)
    np.testing.assert_array_equal(distslct(X + 7.0, y - 3.0, 12), base)
    np.testing.assert_array_equal(distslct(X, 1000.0 * y, 12), base)


def test_distslct_picks_the_outside():
    X, y = make_data(400)
    picks = distslct(X, np.zeros(400), 20)
    radius = np.linalg.norm(X - X.mean(axis=0), axis=1)
    assert np.median(radius[picks]) > np.quantile(radius, 0.8)


# --- xyonion_group_split ------------------------------------------------


def test_group_split_layer_sizes_and_order():
    X, y = make_data(1000)
    subset, layer = split(X, y, (600, 200, 200))
    sizes = [int((layer == k).sum()) for k in range(layer.max() + 1)]
    kinds = [int(subset[layer == k][0]) for k in range(layer.max() + 1)]
    # m0 = 1000, 900, 810: train 0.6, test 0.2, validation 0.2 of 10% of m0.
    assert sizes == [60, 20, 20, 54, 18, 18, 49, 16, 16]
    assert kinds == [TRAIN, TEST, VALIDATION] * 3
    assert np.bincount(subset, minlength=3).tolist() == [600, 200, 200]


def test_group_split_outermost_layer_is_train():
    X, y = make_data(300)
    subset, layer = split(X, y, (225, 0, 75))
    assert (subset[layer == 0] == TRAIN).all()


def test_group_split_zero_rounds_is_random():
    X, y = make_data(100)
    subset, layer = split(X, y, (75, 0, 25), n_onion=0)
    assert (layer == -1).all()
    assert (subset == TEST).sum() == 25


def test_group_split_never_overshoots_counts():
    # Fractions used for layer sizes ask for more test than the exact count
    # allows; the layer is capped and the counts stay exact.
    X, y = make_data(100)
    subset, _ = split(X, y, (90, 0, 10), fractions=(0.1, 0.0, 0.9), loop_fraction=1.0)
    assert np.bincount(subset, minlength=3).tolist() == [90, 0, 10]


def test_random_core_is_seeded():
    X, y = make_data(50)
    a, _ = split(X, y, (37, 0, 13), seed=3)
    b, _ = split(X, y, (37, 0, 13), seed=3)
    np.testing.assert_array_equal(a, b)


def test_constant_y_and_singular_x_ok():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.normal(size=40), np.zeros(40)])
    y = np.ones(40)
    subset, _ = split(X, y, (30, 0, 10))
    assert (subset == TEST).sum() == 10


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_groups(n):
    X, y = make_data(n)
    for n_test in range(n + 1):
        subset, _ = split(X, y, (n - n_test, 0, n_test), fractions=(0.75, 0.0, 0.25))
        assert (subset == TEST).sum() == n_test


def test_group_split_rejects_bad_args():
    X, y = make_data(10)
    with pytest.raises(ValueError):
        split(X, y, (5, 0, 4))      # doesn't add up to 10
    with pytest.raises(ValueError):
        split(X, y, (11, 0, -1))
    with pytest.raises(ValueError):
        split(X, y, (8, 0, 2), n_onion=-1)
