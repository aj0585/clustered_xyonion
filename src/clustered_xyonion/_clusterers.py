"""Clustering algorithms used as the XYOnion pre-processing step."""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClusterMixin


class HDBSCAN_KNN(ClusterMixin, BaseEstimator):
    """HDBSCAN clustering followed by a KNN step that assigns noise points.

    HDBSCAN labels low-density samples as noise (-1). This clusterer is meant
    to then give every noise point the label of its nearest clustered
    neighbours (a KNN vote over ``n_neighbors`` neighbours), so every sample
    ends up with a hard cluster label.

    .. warning::
       Placeholder only. ``fit`` raises ``NotImplementedError`` until the real
       algorithm is implemented. Its parameters are in place so XYOnion can
       already pass them down through ``clustering_params``.

    Parameters
    ----------
    min_cluster_size : int, default=5
        Smallest group of samples HDBSCAN treats as a cluster.
    min_samples : int or None, default=None
        Neighbourhood size used for core distances. None means
        ``min_cluster_size``.
    cluster_selection_epsilon : float, default=0.0
        Distance threshold below which clusters are merged.
    metric : str, default="euclidean"
        Distance metric used by HDBSCAN and the KNN step.
    n_neighbors : int, default=5
        Number of clustered neighbours that vote on each noise point's label.
    """

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        n_neighbors=5,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        """Not implemented yet: placeholder for HDBSCAN + KNN noise assignment."""
        raise NotImplementedError(
            "HDBSCAN_KNN is a placeholder and is not implemented yet. Use "
            "clustering='kmeans', clustering=None, or pass your own clusterer "
            "with a fit_predict method."
        )
