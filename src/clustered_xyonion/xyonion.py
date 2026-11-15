"""(Clustered) XYOnion train/test splitting, in the style of
``sklearn.model_selection.train_test_split`` and ``ShuffleSplit``.

Written for scikit-learn 1.0.2.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import Bunch, _safe_indexing, check_random_state
from sklearn.utils.validation import check_array, indexable

from ._clusterers import HDBSCAN_KNN
from ._core import (
    METRICS,
    apportion_test_counts,
    round_half_up,
    xyonion_group_split,
)

CLUSTERERS = {"kmeans": KMeans, "hdbscan": HDBSCAN_KNN}


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
        raise ValueError(f"X has {n_samples} samples but y has {y.shape[0]}.")
    y = check_array(y, ensure_2d=False, dtype="numeric")
    return y.astype(float, copy=False)


class XYOnionSplit(BaseCrossValidator):
    """XYOnion train/test splitter, optionally run separately on each cluster.

    The standard XYOnion method assumes a roughly convex data set. For data
    with complex (e.g. long, string-like) clusters, set ``clustering`` so the
    data is first given hard cluster labels; XYOnion then runs on each cluster
    on its own and the per-cluster train/test sets are recombined.

    Like ``ShuffleSplit`` with ``n_splits=1``, this yields a single
    ``(train_indices, test_indices)`` pair, so it can be passed as ``cv`` to
    ``GridSearchCV``, ``cross_validate`` etc. Use
    :func:`xyonion_train_test_split` for a ``train_test_split``-style call.

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
        Pre-clustering step, run on X only. ``None`` runs plain XYOnion on the
        whole data set. ``"kmeans"`` uses ``sklearn.cluster.KMeans``;
        ``"hdbscan"`` uses :class:`HDBSCAN_KNN` (currently a placeholder that
        raises ``NotImplementedError``). Any estimator with ``fit_predict`` is
        accepted; it is cloned, never modified.
    clustering_params : dict, default=None
        Keyword arguments for the clustering algorithm, e.g.
        ``{"n_clusters": 5}`` for kmeans or
        ``{"min_cluster_size": 20, "n_neighbors": 5}`` for hdbscan. Applied on
        top of an estimator instance's own parameters.
    random_state : int, RandomState instance or None, default=None
        Controls every source of randomness: the random split of the final
        core samples in each cluster and, where it has a ``random_state``
        parameter that isn't set explicitly, the clustering algorithm.
    """

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

    # --- validation -----------------------------------------------------

    def _validate_params(self):
        def is_real(v):
            return isinstance(v, Real) and not isinstance(v, (bool, Integral))

        if not (is_real(self.test_size) and 0 < self.test_size < 1):
            raise ValueError(
                f"test_size must be a float in (0, 1), got {self.test_size!r}."
            )
        if not (is_real(self.loop_fraction) and 0 < self.loop_fraction <= 1):
            raise ValueError(
                f"loop_fraction must be a float in (0, 1], got {self.loop_fraction!r}."
            )
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}.")
        if isinstance(self.clustering, str):
            if self.clustering not in CLUSTERERS:
                raise ValueError(
                    f"clustering must be None, one of {sorted(CLUSTERERS)} or an "
                    f"estimator with fit_predict; got {self.clustering!r}."
                )
        elif self.clustering is not None and not hasattr(self.clustering, "fit_predict"):
            raise ValueError(
                "clustering estimator must have a fit_predict method; got "
                f"{type(self.clustering).__name__}."
            )
        if self.clustering_params is not None and not isinstance(self.clustering_params, dict):
            raise ValueError(
                f"clustering_params must be a dict or None, got "
                f"{type(self.clustering_params).__name__}."
            )

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

    # --- the split ------------------------------------------------------

    def split_info(self, X, y):
        """Compute the split and return it with diagnostics.

        Returns
        -------
        info : sklearn.utils.Bunch with keys
            ``train_indices``, ``test_indices`` : ndarray of int
            ``test_mask`` : ndarray of bool, True where a sample is in test
            ``cluster_labels`` : hard cluster label of each sample (all zeros
            when ``clustering=None``)
            ``layer`` : layer index within the sample's cluster (0 =
            outermost), or -1 if assigned in the final random split
            ``clusterer`` : the fitted clustering estimator, or None
        """
        self._validate_params()
        X = check_array(X, dtype="numeric", ensure_min_samples=2)
        y = _check_y(y, X.shape[0])
        n_samples = X.shape[0]

        n_test_total = round_half_up(self.test_size * n_samples)
        if not 0 < n_test_total < n_samples:
            raise ValueError(
                f"test_size={self.test_size} with n_samples={n_samples} gives "
                f"{n_test_total} test samples; both train and test sets must "
                "be non-empty."
            )

        clusterer = self._make_clusterer()
        if clusterer is None:
            labels = np.zeros(n_samples, dtype=int)
        else:
            labels = np.asarray(clusterer.fit_predict(X), dtype=int)

        # Each distinct label (including -1, should a clusterer return noise)
        # is split as its own group.
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

        return Bunch(
            train_indices=np.flatnonzero(~test_mask),
            test_indices=np.flatnonzero(test_mask),
            test_mask=test_mask,
            cluster_labels=labels,
            layer=layer,
            clusterer=clusterer,
        )

    def split(self, X, y=None, groups=None):
        """Yield one ``(train_indices, test_indices)`` pair.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Required (XYOnion uses it in its distance), numeric, single column.
        groups : ignored
            Present for API compatibility.
        """
        info = self.split_info(X, y)
        yield info.train_indices, info.test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Always 1: XYOnion produces a single train/test split."""
        return 1


def xyonion_train_test_split(
    X,
    y,
    *arrays,
    test_size=0.25,
    loop_fraction=0.1,
    metric="euclidean",
    clustering=None,
    clustering_params=None,
    random_state=None,
):
    """Split arrays into train and test subsets with (clustered) XYOnion.

    Works like ``sklearn.model_selection.train_test_split``, except that ``X``
    and ``y`` are required because XYOnion uses both to decide the split.
    Any extra ``arrays`` (e.g. sample weights or IDs) are split the same way.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,) or (n_samples, 1)
    *arrays : further indexables with the same number of samples
    test_size, loop_fraction, metric, clustering, clustering_params, random_state
        See :class:`XYOnionSplit`.

    Returns
    -------
    splitting : list, length 2 * (2 + len(arrays))
        ``[X_train, X_test, y_train, y_test, a_train, a_test, ...]``. Each
        output keeps its input's type (pandas objects stay pandas objects).
    """
    X, y, *arrays = indexable(X, y, *arrays)
    splitter = XYOnionSplit(
        test_size=test_size,
        loop_fraction=loop_fraction,
        metric=metric,
        clustering=clustering,
        clustering_params=clustering_params,
        random_state=random_state,
    )
    train, test = next(splitter.split(X, y))
    out = []
    for a in (X, y, *arrays):
        out.extend((_safe_indexing(a, train), _safe_indexing(a, test)))
    return out
