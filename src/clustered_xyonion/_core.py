"""Core XYOnion layer-building routines operating on a single (convex) group.

The XYOnion paper (Ezenarro, Anal. Chim. Acta 1364 (2025) 344229) leaves several
details unspecified. Until the author's reference implementation is available,
this module implements the following interim rules (see README "Assumptions"):

* Distances are the SPXY-style combined metric
  ``d(p, q) = d_x(p, q) / max d_x + d_y(p, q) / max d_y`` where the maxima are
  the maximum pairwise distances within the group, in Euclidean or Mahalanobis
  form.
* Each layer is filled with the ``layer_size`` remaining samples that are
  furthest (by the combined metric) from the centroid. For the Euclidean metric
  the centroid is recomputed from the remaining samples at every layer; for the
  Mahalanobis metric it is computed once from the whole group.
* ``layer_size = max(1, round(loop_fraction * test_size * n_samples))``.
* The first layer goes to the calibration (train) set. Each following layer
  goes to whichever subset keeps the running test ratio closest to the target,
  never overshooting the exact target train/test counts.
* Once fewer than ``layer_size`` samples remain (or no full layer fits in either
  subset), the remaining samples are split at random so the group's test count
  is exactly ``n_test``.

All of the interim rules live in :func:`xyonion_group_split` so they can be
replaced when the reference implementation arrives.
"""

from __future__ import annotations

import numpy as np

METRICS = ("euclidean", "mahalanobis")


def round_half_up(x: float) -> int:
    """Round to the nearest integer, with halves rounded up (not to even)."""
    return int(np.floor(x + 0.5))


def apportion_test_counts(group_sizes, test_size: float) -> np.ndarray:
    """Split ``round(test_size * N)`` test samples across groups.

    Uses the largest-remainder method so every group gets ``floor`` or
    ``ceil`` of its own quota ``test_size * n_k`` and the total is exact.
    Ties in the remainder go to the larger group, then to the earlier group.
    """
    sizes = np.asarray(group_sizes, dtype=int)
    total = round_half_up(test_size * sizes.sum())
    quotas = test_size * sizes
    counts = np.floor(quotas).astype(int)
    leftover = total - counts.sum()
    if leftover > 0:
        remainders = quotas - counts
        # lexsort sorts by the last key first: remainder desc, size desc, index asc
        order = np.lexsort((np.arange(len(sizes)), -sizes, -remainders))
        counts[order[:leftover]] += 1
    return counts


def _whiten(X: np.ndarray) -> np.ndarray:
    """Return ``Z`` such that Euclidean distances in ``Z`` equal Mahalanobis
    distances in ``X``, using the pseudoinverse of the covariance matrix."""
    Xc = X - X.mean(axis=0)
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 0))
    cov = np.atleast_2d(np.cov(Xc, rowvar=False))
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Same cut-off numpy.linalg.pinv uses for its default rcond.
    tol = max(eigvals.max(initial=0.0), 0.0) * max(cov.shape) * np.finfo(float).eps
    keep = eigvals > tol
    return Xc @ (eigvecs[:, keep] / np.sqrt(eigvals[keep]))


def _max_pairwise_distance(Z: np.ndarray, chunk_size: int = 1024) -> float:
    """Maximum Euclidean distance between any two rows of ``Z``.

    Works in row chunks so memory stays at ``chunk_size * n`` floats.
    """
    n = Z.shape[0]
    if n < 2 or Z.shape[1] == 0:
        return 0.0
    sq = np.einsum("ij,ij->i", Z, Z)
    best_sq = 0.0
    for start in range(0, n, chunk_size):
        block = Z[start : start + chunk_size]
        d2 = sq[start : start + chunk_size, None] + sq[None, :] - 2.0 * block @ Z.T
        best_sq = max(best_sq, float(d2.max()))
    return float(np.sqrt(max(best_sq, 0.0)))


def _safe_inverse(value: float) -> float:
    """1/value, or 0 when the block has no spread (its term then drops out)."""
    return 1.0 / value if value > 0 else 0.0


def xyonion_group_split(
    X: np.ndarray,
    y: np.ndarray,
    n_test: int,
    layer_size: int,
    metric: str,
    rng: np.random.RandomState,
):
    """Run XYOnion on one group of samples.

    Parameters
    ----------
    X : ndarray of shape (n, n_features)
    y : ndarray of shape (n,)
    n_test : int
        Exact number of samples of this group to put in the test set.
    layer_size : int
        Number of samples per layer (>= 1).
    metric : {"euclidean", "mahalanobis"}
    rng : numpy RandomState used for the final random split.

    Returns
    -------
    is_test : ndarray of bool, shape (n,)
    layer : ndarray of int, shape (n,)
        Layer index of each sample (0 = outermost); -1 for samples assigned in
        the final random split.
    """
    n = X.shape[0]
    if not 0 <= n_test <= n:
        raise ValueError(f"n_test={n_test} must be between 0 and {n}.")
    if layer_size < 1:
        raise ValueError(f"layer_size={layer_size} must be >= 1.")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")

    is_test = np.zeros(n, dtype=bool)
    layer = np.full(n, -1, dtype=int)
    if n == 0:
        return is_test, layer

    # Put both blocks in a space where plain Euclidean distance is the chosen
    # metric, then scale each block by the inverse of its max pairwise distance.
    if metric == "mahalanobis":
        Zx = _whiten(X)
        y_std = y.std(ddof=1) if n > 1 else 0.0
        zy = (y - y.mean()) * _safe_inverse(y_std)
    else:
        Zx = X.astype(float, copy=True)
        zy = y.astype(float, copy=True)
    Zx = Zx * _safe_inverse(_max_pairwise_distance(Zx))
    zy = zy * _safe_inverse(float(np.ptp(zy)) if n > 1 else 0.0)

    if metric == "mahalanobis":
        fixed_centroid = (Zx.mean(axis=0), zy.mean())

    n_train = n - n_test
    target_ratio = n_test / n
    test_count = train_count = 0
    remaining = np.arange(n)
    layer_id = 0

    while remaining.size >= layer_size:
        assigned = test_count + train_count
        can_test = test_count + layer_size <= n_test
        can_train = train_count + layer_size <= n_train
        # Go to test if that leaves the running test ratio closer to target.
        want_test = test_count + layer_size / 2 < target_ratio * (assigned + layer_size)
        if layer_id == 0 and can_train:
            to_test = False
        elif want_test and can_test:
            to_test = True
        elif can_train:
            to_test = False
        elif can_test:
            to_test = True
        else:
            break

        if metric == "mahalanobis":
            cx, cy = fixed_centroid
        else:
            cx, cy = Zx[remaining].mean(axis=0), zy[remaining].mean()
        dist = np.linalg.norm(Zx[remaining] - cx, axis=1) + np.abs(zy[remaining] - cy)
        # Stable sort, so ties go to the lower original index (deterministic).
        order = np.argsort(-dist, kind="stable")
        chosen = remaining[order[:layer_size]]
        remaining = remaining[np.sort(order[layer_size:])]

        is_test[chosen] = to_test
        layer[chosen] = layer_id
        if to_test:
            test_count += layer_size
        else:
            train_count += layer_size
        layer_id += 1

    # Final random split of the core samples, hitting n_test exactly.
    still_needed = n_test - test_count
    if still_needed:
        is_test[rng.choice(remaining, size=still_needed, replace=False)] = True
    return is_test, layer
