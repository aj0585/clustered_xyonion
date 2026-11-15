import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator

from clustered_xyonion import XYOnion, xyonion_train_test_split
from clustered_xyonion._core import round_half_up


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


# --- estimator ----------------------------------------------------------


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis"])
@pytest.mark.parametrize("clustering", [None, "kmeans", "hdbscan"])
def test_fit_exact_test_fraction(metric, clustering):
    X, y = make_strings()
    params = {"kmeans": {"n_clusters": 2}, "hdbscan": {"min_cluster_size": 20}}
    est = XYOnion(
        test_size=0.3,
        metric=metric,
        clustering=clustering,
        clustering_params=params.get(clustering),
        random_state=0,
    ).fit(X, y)
    assert est.test_mask_.sum() == round_half_up(0.3 * len(y))
    assert len(est.train_indices_) + len(est.test_indices_) == len(y)
    assert np.intersect1d(est.train_indices_, est.test_indices_).size == 0


def test_clustered_split_is_stratified_per_cluster():
    X, y = make_strings(n_per=150)
    est = XYOnion(
        test_size=0.2, clustering="kmeans", clustering_params={"n_clusters": 2},
        random_state=0,
    ).fit(X, y)
    for g in np.unique(est.cluster_labels_):
        in_g = est.cluster_labels_ == g
        assert est.test_mask_[in_g].sum() == round_half_up(0.2 * in_g.sum())
        # Each cluster's own outermost layer is in train.
        assert not est.test_mask_[in_g & (est.layer_ == 0)].any()


def test_layer_size_follows_loop_fraction():
    X, y = make_data(400)
    est = XYOnion(test_size=0.25, loop_fraction=0.1, random_state=0).fit(X, y)
    sizes = np.bincount(est.layer_[est.layer_ >= 0])
    assert (sizes == 10).all()  # round(0.1 * 0.25 * 400)


def test_tiny_sample_uses_layers_of_one():
    X, y = make_data(12)
    est = XYOnion(test_size=0.25, loop_fraction=0.1, random_state=0).fit(X, y)
    sizes = np.bincount(est.layer_[est.layer_ >= 0])
    assert (sizes == 1).all()
    assert est.test_mask_.sum() == 3


def test_random_state_reproducible_and_passed_to_kmeans():
    X, y = make_strings()
    kw = dict(clustering="kmeans", clustering_params={"n_clusters": 3, "n_init": 1})
    a = XYOnion(random_state=1, **kw).fit(X, y)
    b = XYOnion(random_state=1, **kw).fit(X, y)
    np.testing.assert_array_equal(a.test_mask_, b.test_mask_)
    assert a.clusterer_.random_state == 1


def test_explicit_clusterer_random_state_is_respected():
    X, y = make_strings()
    est = XYOnion(clustering=KMeans(n_clusters=2, random_state=7), random_state=1)
    est.fit(X, y)
    assert est.clusterer_.random_state == 7


def test_nested_clusterer_params_via_set_params():
    est = XYOnion(clustering=KMeans(n_clusters=2))
    est.set_params(clustering__n_clusters=4)
    assert est.clustering.n_clusters == 4
    X, y = make_strings()
    est.fit(X, y)
    assert len(np.unique(est.cluster_labels_)) == 4


def test_clustering_params_override_instance():
    X, y = make_strings()
    est = XYOnion(clustering=KMeans(n_clusters=2), clustering_params={"n_clusters": 3})
    est.fit(X, y)
    assert est.clusterer_.n_clusters == 3
    assert est.clustering.n_clusters == 2  # not mutated


@pytest.mark.parametrize(
    "bad_y, match",
    [
        (None, "requires y"),
        (np.array([]), "non-empty"),
        (np.zeros((20, 2)), "single response"),
    ],
)
def test_y_validation(bad_y, match):
    X, _ = make_data(20)
    if bad_y is not None and bad_y.size == 0:
        X = X[:0]
    with pytest.raises(ValueError, match=match):
        XYOnion().fit(X, bad_y)


def test_empty_y_with_nonempty_X():
    X, _ = make_data(20)
    with pytest.raises(ValueError, match="non-empty"):
        XYOnion().fit(X, [])


def test_column_vector_y_accepted():
    X, y = make_data(40)
    a = XYOnion(random_state=0).fit(X, y)
    b = XYOnion(random_state=0).fit(X, y.reshape(-1, 1))
    np.testing.assert_array_equal(a.test_mask_, b.test_mask_)


@pytest.mark.parametrize(
    "params",
    [
        {"test_size": 0.0},
        {"test_size": 1.0},
        {"loop_fraction": 0.0},
        {"metric": "cosine"},
        {"clustering": "dbscan"},
    ],
)
def test_invalid_params(params):
    X, y = make_data(40)
    with pytest.raises(ValueError):
        XYOnion(**params).fit(X, y)


def test_transform_and_fit_transform():
    X, y = make_data(40)
    est = XYOnion(random_state=0)
    out = est.fit_transform(X, y)
    assert out.shape == (40, 1)
    np.testing.assert_array_equal(out.ravel().astype(bool), est.test_mask_)
    assert list(est.get_feature_names_out()) == ["is_test"]
    with pytest.raises(ValueError, match="samples passed to fit"):
        est.transform(X[:10])


def test_split_as_cv():
    X, y = make_data(80)
    cv = XYOnion(random_state=0)
    assert cv.get_n_splits() == 1
    (train, test), = list(cv.split(X, y))
    assert len(test) == 20
    assert not hasattr(cv, "test_mask_")  # split fits a clone
    search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=cv).fit(X, y)
    assert search.n_splits_ == 1


def test_train_test_split_helper_numpy_and_pandas():
    X, y = make_data(60)
    Xtr, Xte, ytr, yte = xyonion_train_test_split(X, y, test_size=0.2, random_state=0)
    assert Xte.shape == (12, 2) and ytr.shape == (48,)
    Xdf, ys = pd.DataFrame(X, columns=["a", "b"]), pd.Series(y)
    Xtr, Xte, ytr, yte = xyonion_train_test_split(Xdf, ys, random_state=0)
    assert isinstance(Xtr, pd.DataFrame) and isinstance(yte, pd.Series)


def test_get_params_and_clone():
    est = XYOnion(test_size=0.3, loop_fraction=0.05, metric="mahalanobis",
                  clustering="hdbscan", clustering_params={"min_cluster_size": 5},
                  random_state=3)
    assert clone(est).get_params() == est.get_params()


def test_sklearn_check_estimator():
    # Runs scikit-learn's API conformance suite. Checks that make no sense for
    # a splitter whose transform only applies to the fitted samples are
    # expected to fail and listed here.
    expected_failed = {
        "check_transformer_general": "transform only defined on fitted samples",
        "check_methods_subset_invariance": "transform only defined on fitted samples",
        "check_methods_sample_order_invariance": "split depends on the whole sample set",
    }
    check_estimator(
        XYOnion(random_state=0),
        expected_failed_checks=expected_failed,
        on_fail="raise",
    )
