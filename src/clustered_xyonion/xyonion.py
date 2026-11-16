"""(Clustered) XYOnion train/validation/test splitting, in the style of
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
    TEST,
    TRAIN,
    VALIDATION,
    _whiten,
    apportion_counts,
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


def _is_float(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, Integral))


class XYOnionSplit(BaseCrossValidator):
    """XYOnion train/(validation/)test splitter, optionally run per cluster.

    The standard XYOnion method assumes a roughly convex data set. For data
    with complex (e.g. long, string-like) clusters, set ``clustering`` so the
    data is first given hard cluster labels; XYOnion then runs on each cluster
    on its own and the per-cluster subsets are recombined.

    With ``validation_fraction=0`` (the default) this works like
    ``ShuffleSplit(n_splits=1)``: ``split`` yields one ``(train, test)`` pair,
    so it can be passed as ``cv`` to ``GridSearchCV``, ``cross_validate`` etc.
    With a validation set, use :meth:`split_info` or
    :func:`xyonion_train_val_test_split` instead.

    Parameters
    ----------
    test_fraction : float in (0, 1), default=0.25
        Fraction of samples placed in the test set. The overall test count is
        ``round(test_fraction * n_samples)``, shared across clusters so the
        total is exact.
    validation_fraction : float in [0, 1), default=0.0
        Fraction of samples placed in the validation set (0 = no validation
        set). ``test_fraction + validation_fraction`` must be below 1.
    n_onion : int >= 0, default=3
        Number of onion rounds (``nonion`` in the reference implementation).
        Each round peels one layer per subset, in the order train, test,
        validation. Samples left after the last round are split at random.
    loop_fraction : float in (0, 1], default=0.1
        Fraction of the remaining samples peeled per round (fixed at 0.1 in
        the reference implementation). In each round, the layer for a subset
        holds ``round(loop_fraction * m0 * f)`` samples, where ``m0`` is the
        number of the cluster's samples still unassigned at the start of the
        round and ``f`` is that subset's fraction (train's fraction is
        ``1 - test_fraction - validation_fraction``). So layers shrink round by
        round, and with the defaults about 27% of each cluster is layered.
    metric : {"euclidean", "mahalanobis"}, default="euclidean"
        Distance used for X within each cluster. Mahalanobis whitens each
        cluster with its own covariance (pseudoinverse, so singular
        covariances work). y is always compared by absolute difference; its
        scale doesn't matter because X and y distances are each divided by
        their maximum before being added.
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
        test_fraction=0.25,
        validation_fraction=0.0,
        n_onion=3,
        loop_fraction=0.1,
        metric="euclidean",
        clustering=None,
        clustering_params=None,
        random_state=None,
    ):
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.n_onion = n_onion
        self.loop_fraction = loop_fraction
        self.metric = metric
        self.clustering = clustering
        self.clustering_params = clustering_params
        self.random_state = random_state

    # --- validation -----------------------------------------------------

    def _validate_params(self):
        if not (_is_float(self.test_fraction) and 0 < self.test_fraction < 1):
            raise ValueError(
                f"test_fraction must be a float in (0, 1), got {self.test_fraction!r}."
            )
        vf = self.validation_fraction
        if not ((_is_float(vf) or vf == 0) and not isinstance(vf, bool) and 0 <= vf < 1):
            raise ValueError(
                f"validation_fraction must be a float in [0, 1), got {vf!r}."
            )
        if self.test_fraction + vf >= 1:
            raise ValueError(
                f"test_fraction + validation_fraction must be below 1, got "
                f"{self.test_fraction} + {vf}."
            )
        if not (isinstance(self.n_onion, Integral) and not isinstance(self.n_onion, bool)
                and self.n_onion >= 0):
            raise ValueError(f"n_onion must be an int >= 0, got {self.n_onion!r}.")
        if not (_is_float(self.loop_fraction) and 0 < self.loop_fraction <= 1):
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
            ``train_indices``, ``validation_indices``, ``test_indices`` :
            ndarray of int (``validation_indices`` is empty when
            ``validation_fraction=0``)
            ``subset`` : ndarray of int, 0 = train, 1 = validation, 2 = test
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
        tf, vf = self.test_fraction, self.validation_fraction

        n_test_total = round_half_up(tf * n_samples)
        n_val_total = round_half_up(vf * n_samples)
        if n_test_total < 1 or n_test_total + n_val_total >= n_samples or (vf > 0 and n_val_total < 1):
            raise ValueError(
                f"test_fraction={tf} and validation_fraction={vf} with "
                f"n_samples={n_samples} give {n_samples - n_test_total - n_val_total} "
                f"train, {n_val_total} validation and {n_test_total} test samples; "
                "every requested set must be non-empty."
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
        sizes = np.array([m.size for m in members])
        n_test_per_group = apportion_counts(sizes, tf)
        n_val_per_group = apportion_counts(sizes, vf, capacity=sizes - n_test_per_group)

        # (train, validation, test) fractions, used for the layer sizes.
        fractions = (1.0 - tf - vf, vf, tf)

        rng = check_random_state(self.random_state)
        subset = np.full(n_samples, TRAIN, dtype=int)
        layer = np.full(n_samples, -1, dtype=int)
        for idx, n_test, n_val in zip(members, n_test_per_group, n_val_per_group):
            # Mahalanobis: whiten each cluster with its own covariance, so plain
            # Euclidean distance inside the core is the cluster's Mahalanobis
            # distance.
            X_group = _whiten(X[idx]) if self.metric == "mahalanobis" else X[idx]
            counts = (idx.size - int(n_test) - int(n_val), int(n_val), int(n_test))
            group_subset, group_layer = xyonion_group_split(
                X_group, y[idx], counts, fractions, self.n_onion,
                self.loop_fraction, rng,
            )
            subset[idx] = group_subset
            layer[idx] = group_layer

        return Bunch(
            train_indices=np.flatnonzero(subset == TRAIN),
            validation_indices=np.flatnonzero(subset == VALIDATION),
            test_indices=np.flatnonzero(subset == TEST),
            subset=subset,
            cluster_labels=labels,
            layer=layer,
            clusterer=clusterer,
        )

    def split(self, X, y=None, groups=None):
        """Yield one ``(train_indices, test_indices)`` pair.

        Only available with ``validation_fraction=0``; for three subsets use
        :meth:`split_info` or :func:`xyonion_train_val_test_split`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, 1)
            Required (XYOnion uses it in its distance), numeric, single column.
        groups : ignored
            Present for API compatibility.
        """
        if self.validation_fraction:
            raise ValueError(
                "split() yields (train, test) pairs, so it needs "
                "validation_fraction=0. Use split_info() or "
                "xyonion_train_val_test_split() for train/validation/test."
            )
        info = self.split_info(X, y)
        yield info.train_indices, info.test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Always 1: XYOnion produces a single split."""
        return 1


def _split_arrays(index_sets, X, y, arrays):
    out = []
    for a in (X, y, *arrays):
        out.extend(_safe_indexing(a, idx) for idx in index_sets)
    return out


def xyonion_train_test_split(
    X,
    y,
    *arrays,
    test_fraction=0.25,
    n_onion=3,
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
    test_fraction, n_onion, loop_fraction, metric, clustering, clustering_params, random_state
        See :class:`XYOnionSplit`.

    Returns
    -------
    splitting : list, length 2 * (2 + len(arrays))
        ``[X_train, X_test, y_train, y_test, a_train, a_test, ...]``. Each
        output keeps its input's type (pandas objects stay pandas objects).
    """
    X, y, *arrays = indexable(X, y, *arrays)
    info = XYOnionSplit(
        test_fraction=test_fraction,
        n_onion=n_onion,
        loop_fraction=loop_fraction,
        metric=metric,
        clustering=clustering,
        clustering_params=clustering_params,
        random_state=random_state,
    ).split_info(X, y)
    return _split_arrays((info.train_indices, info.test_indices), X, y, arrays)


def xyonion_train_val_test_split(
    X,
    y,
    *arrays,
    test_fraction=0.2,
    validation_fraction=0.2,
    n_onion=3,
    loop_fraction=0.1,
    metric="euclidean",
    clustering=None,
    clustering_params=None,
    random_state=None,
):
    """Split arrays into train, validation and test subsets with XYOnion.

    Like :func:`xyonion_train_test_split` with a third subset. Validation and
    test are treated the same way: both get layers from every depth of the
    data, and the outermost layer always goes to train.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,) or (n_samples, 1)
    *arrays : further indexables with the same number of samples
    test_fraction : float in (0, 1), default=0.2
    validation_fraction : float in (0, 1), default=0.2
    n_onion, loop_fraction, metric, clustering, clustering_params, random_state
        See :class:`XYOnionSplit`.

    Returns
    -------
    splitting : list, length 3 * (2 + len(arrays))
        ``[X_train, X_val, X_test, y_train, y_val, y_test, ...]``.
    """
    if not (_is_float(validation_fraction) and validation_fraction > 0):
        raise ValueError(
            "validation_fraction must be a float above 0; use "
            "xyonion_train_test_split for a train/test split."
        )
    X, y, *arrays = indexable(X, y, *arrays)
    info = XYOnionSplit(
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        n_onion=n_onion,
        loop_fraction=loop_fraction,
        metric=metric,
        clustering=clustering,
        clustering_params=clustering_params,
        random_state=random_state,
    ).split_info(X, y)
    index_sets = (info.train_indices, info.validation_indices, info.test_indices)
    return _split_arrays(index_sets, X, y, arrays)
