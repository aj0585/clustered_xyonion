"""scikit-learn compatible (clustered) XYOnion train/test splitter."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions, HasMethods
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ._core import (
    METRICS,
    apportion_test_counts,
    round_half_up,
    xyonion_group_split,
)

CLUSTERERS = {"kmeans": KMeans, "hdbscan": HDBSCAN}


def _check_y(y, n_samples: int) -> np.ndarray:
    """XYOnion needs a single numeric response: reject missing/empty/multi-column y."""
    if y is None:
        raise ValueError("XYOnion requires y; got y=None.")
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("XYOnion requires a non-empty y.")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(
            f"XYOnion requires a single response column; got y with shape {y.shape}."
        )
    if y.shape[0] != n_samples:
        raise ValueError(
            f"X has {n_samples} samples but y has {y.shape[0]}."
        )
    y = check_array(y, ensure_2d=False, dtype="numeric", input_name="y")
    return y.astype(float, copy=False)


class XYOnion(TransformerMixin, BaseEstimator):
    """XYOnion train/test splitting, optionally run separately on each cluster.

    The standard XYOnion method assumes a roughly convex data set. For data
    with complex (e.g. long, string-like) clusters, set ``clustering`` so the
    data is first given hard cluster labels; XYOnion then runs on each cluster
    on its own and the per-cluster train/test sets are recombined.

    Parameters
    ----------
    test_size : float in (0, 1), default=0.25
        Fraction of samples placed in the test (validation) set. The test count
        is ``round(test_size * n_samples)`` overall, apportioned across clusters
        with the largest-remainder method so the total is exact.
    loop_fraction : float in (0, 1], default=0.1
        Each layer holds ``max(1, round(loop_fraction * test_size * n_k))``
        samples, where ``n_k`` is the size of the cluster being split.
    metric : {"euclidean", "mahalanobis"}, default="euclidean"
        Distance used in the X and y blocks within each cluster.
    clustering : {None, "kmeans", "hdbscan"} or estimator, default=None
        Pre-clustering step. ``None`` runs plain XYOnion on the whole data set.
        An estimator instance must provide ``fit_predict``; its parameters can
        then be set through this estimator, e.g. ``clustering__n_clusters``.
        HDBSCAN noise points (label -1) are treated as one extra group.
    clustering_params : dict, default=None
        Keyword arguments for the clustering algorithm named by ``clustering``
        (e.g. ``{"n_clusters": 5}`` for kmeans, ``{"min_cluster_size": 20}``
        for hdbscan). Applied on top of an estimator instance's own params.
    random_state : int, RandomState instance or None, default=None
        Controls every source of randomness: the random split of the final
        core samples in each cluster and, where it has a ``random_state``
        parameter that isn't set explicitly, the clustering algorithm.

    Attributes
    ----------
    test_mask_ : ndarray of bool, shape (n_samples,)
        True where the fitted sample is in the test set.
    train_indices_, test_indices_ : ndarray of int
        Indices of the fitted samples in the train and test sets.
    cluster_labels_ : ndarray of int, shape (n_samples,)
        Hard cluster labels used (all zeros when ``clustering=None``).
    layer_ : ndarray of int, shape (n_samples,)
        Layer index within the sample's cluster (0 = outermost), or -1 if the
        sample was assigned in the final random split.
    clusterer_ : estimator or None
        The fitted clustering estimator.
    n_features_in_ : int
    feature_names_in_ : ndarray of str
        Only defined when X has string feature names.

    Notes
    -----
    ``transform`` returns the test-set indicator of the *fitted* samples as a
    single 0/1 column, so ``fit_transform(X, y)`` gives the split directly. Use
    ``split`` to use it as a ``cv`` argument, or :func:`xyonion_train_test_split`
    for a drop-in replacement of ``sklearn.model_selection.train_test_split``.
    """

    _parameter_constraints = {
        "test_size": [Interval(Real, 0, 1, closed="neither")],
        "loop_fraction": [Interval(Real, 0, 1, closed="right")],
        "metric": [StrOptions(set(METRICS))],
        "clustering": [
            None,
            StrOptions(set(CLUSTERERS)),
            HasMethods(["fit_predict"]),
        ],
        "clustering_params": [dict, None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        test_size=0.25,
        loop_fraction=0.1,
        metric="euclidean",
        clustering=None,
        clustering_params=None,
        random_state=None,
    ):
        self.test_size = test_size
        self.loop_fraction = loop_fraction
        self.metric = metric
        self.clustering = clustering
        self.clustering_params = clustering_params
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.single_output = True
        tags.target_tags.multi_output = False
        return tags

    def _make_clusterer(self):
        if self.clustering is None:
            return None
        params = dict(self.clustering_params or {})
        if isinstance(self.clustering, str):
            clusterer = CLUSTERERS[self.clustering]()
        else:
            clusterer = clone(self.clustering)
        own = clusterer.get_params(deep=False)
        if (
            "random_state" in own
            and own["random_state"] is None
            and "random_state" not in params
        ):
            params["random_state"] = self.random_state
        if params:
            clusterer.set_params(**params)
        return clusterer

    def fit(self, X, y):
        """Compute the train/test split of ``X`` and ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Required, numeric, single response.
        """
        self._validate_params()
        X = validate_data(self, X, dtype="numeric", ensure_min_samples=2)
        y = _check_y(y, X.shape[0])
        n_samples = X.shape[0]

        n_test_total = round_half_up(self.test_size * n_samples)
        if not 0 < n_test_total < n_samples:
            raise ValueError(
                f"test_size={self.test_size} with n_samples={n_samples} gives "
                f"{n_test_total} test samples; both train and test sets must "
                "be non-empty."
            )

        self.clusterer_ = self._make_clusterer()
        if self.clusterer_ is None:
            labels = np.zeros(n_samples, dtype=int)
        else:
            labels = np.asarray(self.clusterer_.fit_predict(X), dtype=int)

        groups = np.unique(labels)
        members = [np.flatnonzero(labels == g) for g in groups]
        n_test_per_group = apportion_test_counts([m.size for m in members], self.test_size)

        rng = check_random_state(self.random_state)
        test_mask = np.zeros(n_samples, dtype=bool)
        layer = np.full(n_samples, -1, dtype=int)
        for idx, n_test in zip(members, n_test_per_group):
            layer_size = max(1, round_half_up(self.loop_fraction * self.test_size * idx.size))
            group_test, group_layer = xyonion_group_split(
                X[idx], y[idx], int(n_test), layer_size, self.metric, rng
            )
            test_mask[idx] = group_test
            layer[idx] = group_layer

        self.cluster_labels_ = labels
        self.test_mask_ = test_mask
        self.layer_ = layer
        self.train_indices_ = np.flatnonzero(~test_mask)
        self.test_indices_ = np.flatnonzero(test_mask)
        return self

    def transform(self, X):
        """Return the test-set indicator (1 = test, 0 = train) of the fitted
        samples as an array of shape (n_samples, 1).

        ``X`` must be the data passed to ``fit``; a split cannot be extended to
        new samples.
        """
        check_is_fitted(self, "test_mask_")
        X = validate_data(self, X, dtype="numeric", reset=False)
        if X.shape[0] != self.test_mask_.shape[0]:
            raise ValueError(
                f"transform expects the {self.test_mask_.shape[0]} samples "
                f"passed to fit, got {X.shape[0]}."
            )
        return self.test_mask_.astype(int).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        """Name of the single output column."""
        check_is_fitted(self, "test_mask_")
        return np.asarray(["is_test"], dtype=object)

    # --- cross-validator interface (usable as ``cv=``) -------------------

    def split(self, X, y=None, groups=None):
        """Yield one ``(train_indices, test_indices)`` pair, so this object can
        be passed as ``cv`` to ``GridSearchCV``, ``cross_validate`` etc.
        Fits a clone, so ``self`` is not modified."""
        fitted = clone(self).fit(X, y)
        yield fitted.train_indices_, fitted.test_indices_

    def get_n_splits(self, X=None, y=None, groups=None):
        """Always 1: XYOnion produces a single train/test split."""
        return 1


def xyonion_train_test_split(X, y, **params):
    """Split ``X`` and ``y`` with :class:`XYOnion`, like
    ``sklearn.model_selection.train_test_split``.

    ``params`` are passed to :class:`XYOnion`. Returns
    ``X_train, X_test, y_train, y_test``, indexing the inputs as given (pandas
    objects stay pandas objects).
    """
    splitter = XYOnion(**params).fit(X, y)
    train, test = splitter.train_indices_, splitter.test_indices_

    def take(a, idx):
        if hasattr(a, "iloc"):
            return a.iloc[idx]
        return np.asarray(a)[idx]

    return take(X, train), take(X, test), take(y, train), take(y, test)
