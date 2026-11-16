"""Euclidean vs Mahalanobis distance on stretched blobs (kmeans).

Four well-separated blobs, each stretched 4:1 in its own direction, with a
response that is constant within each blob (see below for why).
With Euclidean distance the long ends of each blob are the "outside", so the
layers come mostly from the ends and the unlayered core is rounder. With
Mahalanobis distance each blob is first whitened with its own covariance, so
the layers form an even rim all the way round and the unlayered core keeps the
blob's stretched shape.

Also shows a plain train/test split (validation_fraction=0) next to a
train/validation/test split.

Run:  uv run --group examples python examples/02_blobs_euclidean_vs_mahalanobis.py
"""

import numpy as np
from _common import describe, plot_split

from clustered_xyonion import XYOnionSplit

rng = np.random.RandomState(3)
centres = [(-7, 0), (0, 6), (7, 0), (0, -6)]
angles_deg = [20, 75, -35, 120]           # direction each blob is stretched in
blobs, labels = [], []
for k, (centre, angle) in enumerate(zip(centres, angles_deg)):
    a = np.radians(angle)
    rotate = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    stretch = np.diag([2.0, 0.5])         # 4:1 long axis : short axis
    blobs.append(rng.normal(size=(200, 2)) @ (rotate @ stretch).T + centre)
    labels.append(np.full(200, k))
X = np.vstack(blobs)
labels = np.concatenate(labels)

# Fake response: constant within each blob. XYOnion gives y the SAME weight as
# X in its distance (each scaled by its own maximum), so any spread of y inside
# a blob shapes the layers as much as X does and would blur the comparison.
# With y constant per blob, the y term drops out and the layers show the X
# metric alone.
y = 2.0 * labels + 1.0

kmeans4 = {"clustering": "kmeans", "clustering_params": {"n_clusters": 4}}
runs = [
    ("euclidean", XYOnionSplit(test_fraction=0.25, metric="euclidean",
                               loop_fraction=0.2, random_state=0, **kmeans4)),
    ("mahalanobis", XYOnionSplit(test_fraction=0.25, metric="mahalanobis",
                                 loop_fraction=0.2, random_state=0, **kmeans4)),
    ("mahalanobis_3way", XYOnionSplit(test_fraction=0.2, validation_fraction=0.2,
                                      metric="mahalanobis", loop_fraction=0.2,
                                      random_state=0, **kmeans4)),
]
for name, cv in runs:
    info = cv.split_info(X, y)
    plot_split(X, y, info, f"Stretched blobs: {cv.metric} distance", describe(cv),
               f"02_blobs_{name}.png")
