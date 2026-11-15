"""HDBSCAN_KNN + XYOnion on curved, non-convex clusters with noise.

HDBSCAN_KNN isn't implemented yet, so this uses PretendHDBSCAN_KNN: it takes
the same parameters (min_cluster_size, n_neighbors, ...) and XYOnion passes
them down through clustering_params as usual, but it returns made-up labels
instead of clustering. The made-up labels are what HDBSCAN + KNN should
produce here: each moon and the blob as its own cluster, with the scattered
noise points given to the cluster whose centre is nearest.

Run:  uv run --group examples python examples/03_moons_hdbscan_knn.py
"""

import numpy as np
from _common import PretendHDBSCAN_KNN, describe, plot_split
from sklearn.datasets import make_moons

from clustered_xyonion import XYOnionSplit

rng = np.random.RandomState(0)

# Two interleaved moons, a round blob, and a sprinkle of uniform noise.
X_moons, moon_labels = make_moons(n_samples=600, noise=0.07, random_state=0)
X_blob = rng.normal(loc=[2.6, 1.3], scale=0.25, size=(200, 2))
X_noise = rng.uniform(low=[-1.5, -1.0], high=[3.4, 2.0], size=(60, 2))
X = np.vstack([X_moons, X_blob, X_noise])

# Fake response: changes along each moon and is offset per cluster.
true_labels = np.concatenate([moon_labels, np.full(len(X_blob), 2), np.full(len(X_noise), -1)])

# Made-up "HDBSCAN + KNN" labels: noise points join the nearest cluster centre.
centres = np.array([X[true_labels == k].mean(axis=0) for k in range(3)])
fake_labels = true_labels.copy()
noise = true_labels == -1
fake_labels[noise] = np.argmin(
    ((X[noise, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)

y = (np.arctan2(X[:, 1] - 0.25, X[:, 0] - 0.5) + 1.5 * fake_labels
     + 0.15 * rng.normal(size=len(X)))

cv = XYOnionSplit(
    test_fraction=0.2,
    validation_fraction=0.2,
    metric="mahalanobis",
    clustering=PretendHDBSCAN_KNN(fake_labels=fake_labels),
    clustering_params={"min_cluster_size": 25, "n_neighbors": 7},
    random_state=0,
)
info = cv.split_info(X, y)

# The clustering parameters really were passed down to the clusterer:
params = info.clusterer.get_params()
print("clusterer got min_cluster_size =", params["min_cluster_size"],
      "and n_neighbors =", params["n_neighbors"])

plot_split(X, y, info, "Moons + blob: (pretend) HDBSCAN_KNN + XYOnion",
           describe(cv).replace("PretendHDBSCAN_KNN", "HDBSCAN_KNN (pretend labels)"),
           "03_moons_hdbscan_knn.png")
