"""Tests for the scikit-learn-facing splitter (needs scikit-learn 1.0.2)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_validate

from clustered_xyonion import (
    HDBSCAN_KNN,
    XYOnionSplit,
    xyonion_train_test_split,
    xyonion_train_val_test_split,
)
from clustered_xyonion._core import TEST, TRAIN, VALIDATION, round_half_up


def make_data(n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 2))
    y = X @ [1.0, -2.0] + 0.1 * rng.normal(size=n)
    return X, y


def make_strings(n_per=150, seed=0):
    """Two long, thin, parallel string-like clusters."""
    rng = np.random.RandomState(seed)
    t = rng.uniform(0, 10, size=2 * n_per)
    offset = np.repeat([0.0, 5.0], n_per)
    X = np.column_stack([t, offset + 0.1 * rng.normal(size=2 * n_per)])
    y = t + offset
    return X, y


KMEANS2 = {"clustering": "kmeans", "clustering_params": {"n_clusters": 2}}


# --- splitter -----------------------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis"])
@pytest.mark.parametrize("clustering", [None, "kmeans"])
def test_split_exact_test_fraction(metric, clustering):
    X, y = make_strings()
    params = {"n_clusters": 2} if clustering == "kmeans" else None
    cv = XYOnionSplit(test_fraction=0.3, metric=metric, clustering=clustering,
                      clustering_params=params, random_state=0)
    (train, test), = list(cv.split(X, y))
    assert len(test) == round_half_up(0.3 * len(y))
    assert len(train) + len(test) == len(y)
    assert np.intersect1d(train, test).size == 0


def test_get_n_splits_and_repr():
    cv = XYOnionSplit(test_fraction=0.2, random_state=1)
    assert cv.get_n_splits() == 1
    assert "XYOnionSplit(" in repr(cv) and "test_fraction=0.2" in repr(cv)


def test_split_does_not_store_state():
    X, y = make_data(40)
    cv = XYOnionSplit(random_state=0)
    list(cv.split(X, y))
    assert vars(cv) == vars(XYOnionSplit(random_state=0))


def test_clustered_split_is_stratified_per_cluster():
    X, y = make_strings(n_per=150)
    info = XYOnionSplit(test_fraction=0.2, random_state=0, **KMEANS2).split_info(X, y)
    for g in np.unique(info.cluster_labels):
        in_g = info.cluster_labels == g
        assert (info.subset[in_g] == TEST).sum() == round_half_up(0.2 * in_g.sum())
        # Each cluster's own outermost layer is in train.
        assert (info.subset[in_g & (info.layer == 0)] == TRAIN).all()


def layer_sizes_and_subsets(info):
    ids = range(info.layer.max() + 1)
    sizes = [int((info.layer == k).sum()) for k in ids]
    kinds = [int(info.subset[info.layer == k][0]) for k in ids]
    return sizes, kinds


def test_layer_sizes_follow_reference():
    # Each round: train layer round(0.1 * m0 * 0.75), then test layer
    # round(0.1 * m0 * 0.25), m0 = samples left at the start of the round.
    X, y = make_data(400)
    info = XYOnionSplit(test_fraction=0.25, random_state=0).split_info(X, y)
    sizes, kinds = layer_sizes_and_subsets(info)
    assert sizes == [30, 10, 27, 9, 24, 8]  # m0 = 400, 360, 324
    assert kinds == [TRAIN, TEST] * 3
    assert (info.subset == TEST).sum() == 100  # random core makes it exact


def test_n_onion_sets_number_of_rounds():
    X, y = make_data(400)
    for n_onion in (0, 1, 5):
        info = XYOnionSplit(n_onion=n_onion, random_state=0).split_info(X, y)
        assert len(layer_sizes_and_subsets(info)[0]) == 2 * n_onion
        assert (info.subset == TEST).sum() == 100


def test_tiny_sample_skips_empty_layers():
    # 12 samples: train layers round(0.1 * m0 * 0.75) = 1 each round, test
    # layers round(0.1 * m0 * 0.25) = 0 so they're skipped; test comes from
    # the random core.
    X, y = make_data(12)
    info = XYOnionSplit(test_fraction=0.25, random_state=0).split_info(X, y)
    sizes, kinds = layer_sizes_and_subsets(info)
    assert sizes == [1, 1, 1] and kinds == [TRAIN] * 3
    assert (info.subset == TEST).sum() == 3


def test_random_state_reproducible_and_passed_to_kmeans():
    X, y = make_strings()
    kw = dict(clustering="kmeans", clustering_params={"n_clusters": 3, "n_init": 1})
    a = XYOnionSplit(random_state=1, **kw).split_info(X, y)
    b = XYOnionSplit(random_state=1, **kw).split_info(X, y)
    np.testing.assert_array_equal(a.subset, b.subset)
    assert a.clusterer.random_state == 1


def test_explicit_clusterer_random_state_is_respected():
    X, y = make_strings()
    cv = XYOnionSplit(clustering=KMeans(n_clusters=2, random_state=7), random_state=1)
    assert cv.split_info(X, y).clusterer.random_state == 7


def test_clustering_params_override_instance_without_mutating_it():
    X, y = make_strings()
    km = KMeans(n_clusters=2)
    cv = XYOnionSplit(clustering=km, clustering_params={"n_clusters": 3})
    assert cv.split_info(X, y).clusterer.n_clusters == 3
    assert km.n_clusters == 2


# --- HDBSCAN_KNN placeholder --------------------------------------------


def test_hdbscan_params_including_n_neighbors_are_passed_down():
    cv = XYOnionSplit(
        clustering="hdbscan",
        clustering_params={"min_cluster_size": 20, "n_neighbors": 7},
    )
    clusterer = cv._make_clusterer()
    assert isinstance(clusterer, HDBSCAN_KNN)
    assert clusterer.min_cluster_size == 20
    assert clusterer.n_neighbors == 7
    assert "random_state" not in clusterer.get_params()


def test_hdbscan_is_a_stub_for_now():
    X, y = make_strings()
    with pytest.raises(NotImplementedError, match="placeholder"):
        list(XYOnionSplit(clustering="hdbscan").split(X, y))


# --- validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad_y, match",
    [(None, "requires y"), ([], "non-empty"), (np.zeros((20, 2)), "single response")],
)
def test_y_validation(bad_y, match):
    X, _ = make_data(20)
    with pytest.raises(ValueError, match=match):
        list(XYOnionSplit().split(X, bad_y))


def test_column_vector_y_accepted():
    X, y = make_data(40)
    a = XYOnionSplit(random_state=0).split_info(X, y)
    b = XYOnionSplit(random_state=0).split_info(X, y.reshape(-1, 1))
    np.testing.assert_array_equal(a.subset, b.subset)


@pytest.mark.parametrize(
    "params",
    [
        {"test_fraction": 0.0},
        {"test_fraction": 1.0},
        {"test_fraction": 10},
        {"loop_fraction": 0.0},
        {"metric": "cosine"},
        {"clustering": "dbscan"},
        {"clustering": object()},
        {"clustering_params": [("n_clusters", 2)]},
    ],
)
def test_invalid_params(params):
    X, y = make_data(40)
    with pytest.raises(ValueError):
        list(XYOnionSplit(**params).split(X, y))


def test_too_few_samples_for_both_sets():
    X, y = make_data(2)
    with pytest.raises(ValueError, match="non-empty"):
        list(XYOnionSplit(test_fraction=0.1).split(X, y))


# --- train_test_split-style helper --------------------------------------


def test_train_test_split_numpy():
    X, y = make_data(60)
    w = np.arange(60)
    X_tr, X_te, y_tr, y_te, w_tr, w_te = xyonion_train_test_split(
        X, y, w, test_fraction=0.2, random_state=0
    )
    assert X_te.shape == (12, 2) and y_tr.shape == (48,)
    np.testing.assert_array_equal(X[w_te], X_te)  # extra array split the same way


def test_train_test_split_pandas_keeps_types():
    X, y = make_data(60)
    Xdf, ys = pd.DataFrame(X, columns=["a", "b"]), pd.Series(y)
    X_tr, X_te, y_tr, y_te = xyonion_train_test_split(Xdf, ys, random_state=0, **KMEANS2)
    assert isinstance(X_tr, pd.DataFrame) and isinstance(y_te, pd.Series)
    assert list(X_tr.index) == list(y_tr.index)


def test_train_test_split_matches_splitter():
    X, y = make_strings()
    _, X_te, _, _ = xyonion_train_test_split(X, y, random_state=0, **KMEANS2)
    (_, test), = list(XYOnionSplit(random_state=0, **KMEANS2).split(X, y))
    np.testing.assert_array_equal(X_te, X[test])


def test_inconsistent_lengths():
    X, y = make_data(20)
    with pytest.raises(ValueError):
        xyonion_train_test_split(X, y[:10])


# --- use as cv -----------------------------------------------------------


def test_usable_as_cv():
    X, y = make_data(80)
    cv = XYOnionSplit(random_state=0)
    search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=cv).fit(X, y)
    assert search.n_splits_ == 1
    res = cross_validate(Ridge(), X, y, cv=cv)
    assert len(res["test_score"]) == 1


# --- train / validation / test -------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis"])
@pytest.mark.parametrize("clustering", [None, "kmeans"])
def test_three_way_exact_counts(metric, clustering):
    X, y = make_strings()
    params = {"n_clusters": 2} if clustering == "kmeans" else None
    info = XYOnionSplit(test_fraction=0.2, validation_fraction=0.1, metric=metric,
                        clustering=clustering, clustering_params=params,
                        random_state=0).split_info(X, y)
    n = len(y)
    assert len(info.test_indices) == round_half_up(0.2 * n)
    assert len(info.validation_indices) == round_half_up(0.1 * n)
    assert len(info.train_indices) == n - round_half_up(0.2 * n) - round_half_up(0.1 * n)
    all_idx = np.concatenate([info.train_indices, info.validation_indices, info.test_indices])
    assert np.array_equal(np.sort(all_idx), np.arange(n))


def test_three_way_is_stratified_per_cluster():
    X, y = make_strings(n_per=150)
    info = XYOnionSplit(test_fraction=0.2, validation_fraction=0.2,
                        random_state=0, **KMEANS2).split_info(X, y)
    for g in np.unique(info.cluster_labels):
        in_g = info.cluster_labels == g
        counts = np.bincount(info.subset[in_g], minlength=3)
        assert counts[VALIDATION] == round_half_up(0.2 * in_g.sum())
        assert counts[TEST] == round_half_up(0.2 * in_g.sum())
        assert (info.subset[in_g & (info.layer == 0)] == TRAIN).all()


def test_three_way_layers_go_train_test_validation():
    # train 0.65, test 0.25, validation 0.1; loop_fraction 0.5 so m0 halves.
    X, y = make_data(400)
    info = XYOnionSplit(test_fraction=0.25, validation_fraction=0.1,
                        loop_fraction=0.5, random_state=0).split_info(X, y)
    sizes, kinds = layer_sizes_and_subsets(info)
    assert sizes == [130, 50, 20, 65, 25, 10, 33, 13, 5]  # m0 = 400, 200, 100
    assert kinds == [TRAIN, TEST, VALIDATION] * 3
    assert np.bincount(info.subset, minlength=3).tolist() == [260, 40, 100]


def test_mahalanobis_ignores_feature_scaling_euclidean_does_not():
    # Stretching one feature changes which samples look "outer" in Euclidean
    # distance, but whitening undoes the stretch, so Mahalanobis layers match.
    X, y = make_data(300)
    X_stretched = X * [1.0, 50.0]
    for metric, same in (("mahalanobis", True), ("euclidean", False)):
        a = XYOnionSplit(metric=metric, random_state=0).split_info(X, y)
        b = XYOnionSplit(metric=metric, random_state=0).split_info(X_stretched, y)
        assert np.array_equal(a.layer, b.layer) == same


def test_validation_zero_matches_two_way():
    X, y = make_strings()
    a = XYOnionSplit(random_state=0, **KMEANS2).split_info(X, y)
    b = XYOnionSplit(validation_fraction=0.0, random_state=0, **KMEANS2).split_info(X, y)
    np.testing.assert_array_equal(a.subset, b.subset)
    assert a.validation_indices.size == 0


def test_split_refuses_three_way():
    X, y = make_data(40)
    with pytest.raises(ValueError, match="validation_fraction=0"):
        list(XYOnionSplit(validation_fraction=0.2).split(X, y))


@pytest.mark.parametrize(
    "params",
    [
        {"validation_fraction": -0.1},
        {"validation_fraction": 1.0},
        {"validation_fraction": True},
        {"test_fraction": 0.6, "validation_fraction": 0.4},
    ],
)
def test_invalid_validation_fraction(params):
    X, y = make_data(40)
    with pytest.raises(ValueError):
        XYOnionSplit(**params).split_info(X, y)


def test_validation_set_must_be_non_empty():
    X, y = make_data(10)
    with pytest.raises(ValueError, match="non-empty"):
        XYOnionSplit(validation_fraction=0.01).split_info(X, y)


def test_train_val_test_split_helper():
    X, y = make_data(100)
    w = np.arange(100)
    out = xyonion_train_val_test_split(X, y, w, test_fraction=0.2,
                                       validation_fraction=0.1, random_state=0)
    X_tr, X_va, X_te, y_tr, y_va, y_te, w_tr, w_va, w_te = out
    assert (len(y_tr), len(y_va), len(y_te)) == (70, 10, 20)
    np.testing.assert_array_equal(X[w_va], X_va)
    with pytest.raises(ValueError):
        xyonion_train_val_test_split(X, y, validation_fraction=0.0)


def test_train_val_test_split_pandas_keeps_types():
    X, y = make_data(60)
    Xdf, ys = pd.DataFrame(X, columns=["a", "b"]), pd.Series(y)
    X_tr, X_va, X_te, y_tr, y_va, y_te = xyonion_train_val_test_split(
        Xdf, ys, random_state=0, **KMEANS2)
    assert isinstance(X_va, pd.DataFrame) and isinstance(y_va, pd.Series)
    assert list(X_va.index) == list(y_va.index)
