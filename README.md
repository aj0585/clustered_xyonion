# clustered_xyonion

A Python implementation of the **XYOnion** train/test (calibration/validation)
splitting method, with an optional **clustered** variant for data whose
clusters are not convex (e.g. long, string-like clusters).

XYOnion (Ezenarro, *Anal. Chim. Acta* 1364 (2025) 344229,
[doi:10.1016/j.aca.2025.344229](https://doi.org/10.1016/j.aca.2025.344229))
combines SPXY-style joint X–y distances with Onion-style layered selection:
samples are peeled off in shells from the outside in, the outermost shell goes
to training (so the test set is never extrapolated), and layers are shared
between train and test so both sets span the whole data space.

Standard XYOnion assumes the data set is roughly convex. The clustered variant
first gives every sample a hard cluster label (kmeans or HDBSCAN), runs XYOnion
on each cluster on its own, and recombines the per-cluster splits into one
train set and one test set.

## Install

```bash
uv sync
```

or `pip install -e .`. Requires Python ≥ 3.10, numpy and scikit-learn ≥ 1.6.

## Usage

```python
from clustered_xyonion import XYOnion, xyonion_train_test_split

# Drop-in replacement for sklearn.model_selection.train_test_split
X_train, X_test, y_train, y_test = xyonion_train_test_split(
    X, y,
    test_size=0.25,
    loop_fraction=0.1,
    metric="mahalanobis",          # or "euclidean"
    clustering="hdbscan",          # or "kmeans", or None for plain XYOnion
    clustering_params={"min_cluster_size": 20},
    random_state=0,
)

# As a scikit-learn estimator / transformer
splitter = XYOnion(clustering="kmeans", clustering_params={"n_clusters": 5},
                   random_state=0).fit(X, y)
splitter.train_indices_, splitter.test_indices_   # the split
splitter.cluster_labels_                            # clusters used
splitter.layer_                                     # layer per sample (-1 = random core)
is_test = splitter.fit_transform(X, y)              # (n_samples, 1) 0/1 column

# As a cv object (one split)
from sklearn.model_selection import GridSearchCV
GridSearchCV(model, grid, cv=XYOnion(random_state=0)).fit(X, y)

# Clustering hyperparameters through the XYOnion API
from sklearn.cluster import KMeans
splitter = XYOnion(clustering=KMeans(n_clusters=3))
splitter.set_params(clustering__n_clusters=6)
```

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `test_size` | `0.25` | Fraction of samples in the test set, in (0, 1). |
| `loop_fraction` | `0.1` | Layer size is `max(1, round(loop_fraction * test_size * n_k))` for a cluster of `n_k` samples. |
| `metric` | `"euclidean"` | `"euclidean"` or `"mahalanobis"`, used within each cluster. |
| `clustering` | `None` | `None`, `"kmeans"`, `"hdbscan"`, or any estimator with `fit_predict`. |
| `clustering_params` | `None` | Dict of keyword arguments for the clustering algorithm. |
| `random_state` | `None` | Seeds the random core split and, unless set explicitly, the clusterer's `random_state`. |

`y` is required and must be a single numeric response (shape `(n,)` or
`(n, 1)`); a missing, empty or multi-column `y` raises `ValueError`.

Clustering runs on `X` only, so scale `X` beforehand if its features have
different units. HDBSCAN noise points (label −1) are split together as one
extra group.

## Assumptions

The paper leaves some details open, and the author's Matlab code is available
only on request. Until it is available, the implementation uses these rules,
all in `xyonion_group_split` in `src/clustered_xyonion/_core.py`:

1. **Distance:** `d(p, q) = d_x(p, q) / max d_x + d_y(p, q) / max d_y`, with the
   maxima being the largest pairwise distances in the cluster. Mahalanobis uses
   the pseudoinverse of the cluster's X covariance and `var(y)`.
2. **Layer membership:** each layer takes the remaining samples furthest from
   the centroid by that combined distance. For Euclidean, the centroid is
   recomputed from the remaining samples at each layer; for Mahalanobis it is
   computed once from the whole cluster.
3. **Layer size:** `round(loop_fraction * test_size * n_k)`, or 1 sample per
   layer when that rounds to 0 (small clusters).
4. **Layer assignment:** the first layer goes to train. Each later layer goes
   to whichever set keeps the running test ratio closest to `test_size`,
   without exceeding the exact train or test count.
5. **Final split:** samples left over when a full layer no longer fits are
   split at random so the cluster's test count is exact.
6. **Rounding:** the overall test count is `round(test_size * n_samples)`
   (halves round up), shared between clusters by the largest-remainder method
   so the total is exact.

## Development

```bash
uv run pytest
```
