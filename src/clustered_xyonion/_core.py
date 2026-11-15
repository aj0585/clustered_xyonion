"""Core XYOnion layer-building routines operating on a single (convex) group.

Only numpy is needed here, so this module can be used and tested without
scikit-learn.

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
* The first layer goes to the train (calibration) set. Each following layer
  goes to whichever subset (train, validation or test) is furthest below its
  target share, never overshooting any subset's exact count.
* Once fewer than ``layer_size`` samples remain (or no full layer fits in any
  subset), the remaining samples are split at random so each subset's count is
  exact.

All of the interim rules live in :func:`xyonion_group_split` so they can be
replaced when the reference implementation arrives.
"""

from __future__ import annotations

import numpy as np

METRICS = ("euclidean", "mahalanobis")

# Codes used in the returned ``subset`` arrays.
TRAIN, VALIDATION, TEST = 0, 1, 2
SUBSET_NAMES = ("train", "validation", "test")


def round_half_up(x: float) -> int:
    """Round to the nearest integer, with halves rounded up (not to even)."""
    return int(np.floor(x + 0.5))


def apportion_counts(group_sizes, fraction: float, capacity=None) -> np.ndarray:
    """Share ``round(fraction * N)`` samples out across groups.

    Each group ``k`` has a quota ``fraction * n_k``. Every group first gets the
    whole-number part of its quota; the samples still needed to reach the exact
    total then go one each to the groups with the largest leftover fractions
    (the "largest-remainder" method). Ties go to the larger group, then the
    earlier group.

    Parameters
    ----------
    group_sizes : sequence of int
        Number of samples ``n_k`` in each group.
    fraction : float in [0, 1)
    capacity : sequence of int or None
        Maximum count each group can take (used so a group's validation and
        test counts never add up to more than its size). Defaults to the group
        sizes.

    Returns
    -------
    counts : ndarray of int, one per group, summing to ``round(fraction * N)``.
    """
    sizes = np.asarray(group_sizes, dtype=int)
    cap = sizes if capacity is None else np.asarray(capacity, dtype=int)
    total = round_half_up(fraction * sizes.sum())
    quotas = fraction * sizes
    counts = np.minimum(np.floor(quotas).astype(int), cap)
    leftover = total - counts.sum()
    if leftover > cap.sum() - counts.sum():
        raise ValueError("Not enough room in the groups for the requested count.")
    remainders = quotas - counts
    while leftover > 0:
        has_room = counts < cap
        # lexsort sorts by its LAST key first: remainder desc, size desc, index asc.
        order = np.lexsort((np.arange(len(sizes)), -sizes, -remainders))
        order = order[has_room[order]]
        take = order[:leftover]
        counts[take] += 1
        remainders[take] -= 1.0
        leftover -= take.size
    return counts


def _whiten(X: np.ndarray) -> np.ndarray:
    """Transform ``X`` so plain Euclidean distance equals Mahalanobis distance.

    The Mahalanobis distance between two samples ``p`` and ``q`` is

        d_M(p, q) = sqrt( (x_p - x_q)^T  S^+  (x_p - x_q) )

    where ``S`` is the covariance matrix of ``X`` and ``S^+`` its
    (pseudo)inverse. Computing that for every pair would be slow, so instead we
    find a matrix ``W`` such that, with ``Z = X @ W``,

        d_M(p, q) = || z_p - z_q ||        (ordinary Euclidean distance)

    How ``W`` is found:

    1. ``S`` is symmetric, so it can be written as ``S = V @ diag(lam) @ V.T``
       (an eigen-decomposition). Each column of ``V`` is a direction in feature
       space, and ``lam`` holds the variance of the data along that direction.
    2. Its pseudoinverse is ``S^+ = V @ diag(1 / lam) @ V.T``, keeping only the
       directions with non-zero variance (dividing by zero variance is
       meaningless, so those directions are dropped, which is what makes it a
       *pseudo*-inverse).
    3. Substituting, with ``d = x_p - x_q``:

           d^T S^+ d = d^T V diag(1/lam) V^T d = || diag(1/sqrt(lam)) V^T d ||^2

       so ``W = V @ diag(1 / sqrt(lam))``: rotate onto the principal
       directions, then divide each by its standard deviation.

    The data is centred (mean subtracted) first; that doesn't change any
    distance but keeps the numbers small.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    Z : ndarray of shape (n_samples, n_kept_directions)
        Whitened data. Has zero columns if no direction has any variance (or
        fewer than 2 samples), in which case all distances are 0.
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.zeros((n_samples, 0))

    # Centre the data.
    X_centred = X - X.mean(axis=0)

    # Covariance matrix S, shape (n_features, n_features). Same as
    # np.cov(X, rowvar=False): sum of outer products divided by (n - 1).
    S = (X_centred.T @ X_centred) / (n_samples - 1)

    # Step 1: eigen-decomposition S = V diag(lam) V^T. eigh is the routine for
    # symmetric matrices; it returns real eigenvalues in ascending order.
    lam, V = np.linalg.eigh(S)

    # Step 2: drop directions with (numerically) zero variance. This cut-off is
    # the same one numpy.linalg.pinv uses by default.
    tolerance = max(lam.max(), 0.0) * max(S.shape) * np.finfo(float).eps
    keep = lam > tolerance
    V_kept = V[:, keep]
    lam_kept = lam[keep]

    # Step 3: W = V diag(1 / sqrt(lam)), then Z = X_centred @ W.
    W = V_kept @ np.diag(1.0 / np.sqrt(lam_kept))
    return X_centred @ W


def _max_pairwise_distance(Z: np.ndarray, chunk_size: int = 1024) -> float:
    """Largest Euclidean distance between any two rows of ``Z``.

    This is the ``max d`` used to normalise each block in the combined XYOnion
    distance. Comparing every pair of ``n`` samples needs an ``n x n`` table of
    distances, which for large ``n`` doesn't fit in memory, so the table is
    built ``chunk_size`` rows at a time and only its maximum is kept.

    Each chunk uses the identity

        || a - b ||^2 = ||a||^2 + ||b||^2 - 2 (a . b)

    which turns all the pairwise distances of a chunk into one matrix product
    (``chunk @ Z.T`` holds every dot product ``a . b`` at once).

    Parameters
    ----------
    Z : ndarray of shape (n_samples, n_features)
    chunk_size : int
        Rows compared per step; memory use is about ``chunk_size * n_samples``
        numbers.

    Returns
    -------
    float
        The maximum distance, or 0.0 with fewer than 2 samples or no features.
    """
    n_samples = Z.shape[0]
    if n_samples < 2 or Z.shape[1] == 0:
        return 0.0

    # ||b||^2 for every row b: square every entry, sum across each row.
    squared_norms = np.sum(Z**2, axis=1)  # shape (n_samples,)

    largest_squared = 0.0
    for start in range(0, n_samples, chunk_size):
        chunk = Z[start : start + chunk_size]                  # rows a
        chunk_norms = squared_norms[start : start + chunk_size]

        dot_products = chunk @ Z.T                             # a . b for all pairs
        # Squared distances: column vector + row vector - 2 * dot products.
        # chunk_norms[:, None] has shape (chunk, 1) and squared_norms[None, :]
        # has shape (1, n), so adding them gives a (chunk, n) table.
        squared = chunk_norms[:, None] + squared_norms[None, :] - 2.0 * dot_products
        largest_squared = max(largest_squared, float(squared.max()))

    # Rounding can make a squared distance a tiny bit negative; clip at 0.
    return float(np.sqrt(max(largest_squared, 0.0)))


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
    n_validation: int = 0,
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
    n_validation : int, default=0
        Exact number of samples of this group to put in the validation set.

    Returns
    -------
    subset : ndarray of int, shape (n,)
        ``TRAIN`` (0), ``VALIDATION`` (1) or ``TEST`` (2) for each sample.
    layer : ndarray of int, shape (n,)
        Layer index of each sample (0 = outermost); -1 for samples assigned in
        the final random split.
    """
    n = X.shape[0]
    if n_test < 0 or n_validation < 0 or n_test + n_validation > n:
        raise ValueError(
            f"n_test={n_test} and n_validation={n_validation} must be >= 0 and "
            f"add up to at most {n}."
        )
    if layer_size < 1:
        raise ValueError(f"layer_size={layer_size} must be >= 1.")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")

    subset = np.full(n, TRAIN, dtype=int)
    layer = np.full(n, -1, dtype=int)
    if n == 0:
        return subset, layer

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

    target = np.array([n - n_validation - n_test, n_validation, n_test])
    target_share = target / n
    count = np.zeros(3, dtype=int)
    remaining = np.arange(n)
    layer_id = 0

    while remaining.size >= layer_size:
        assigned = count.sum()
        fits = count + layer_size <= target
        if not fits.any():
            break
        if layer_id == 0 and fits[TRAIN]:
            # The outermost layer always goes to train, so neither validation
            # nor test ever needs extrapolating to.
            destination = TRAIN
        else:
            # How far each subset would be below its target share if this
            # layer went elsewhere. Give the layer to the biggest shortfall
            # among the subsets it fits in; ties go to train, then validation.
            shortfall = target_share * (assigned + layer_size) - count
            shortfall[~fits] = -np.inf
            destination = int(np.argmax(shortfall))

        if metric == "mahalanobis":
            cx, cy = fixed_centroid
        else:
            cx, cy = Zx[remaining].mean(axis=0), zy[remaining].mean()
        dist = np.linalg.norm(Zx[remaining] - cx, axis=1) + np.abs(zy[remaining] - cy)
        # Stable sort, so ties go to the lower original index (deterministic).
        order = np.argsort(-dist, kind="stable")
        chosen = remaining[order[:layer_size]]
        remaining = remaining[np.sort(order[layer_size:])]

        subset[chosen] = destination
        layer[chosen] = layer_id
        count[destination] += layer_size
        layer_id += 1

    # Final random split of the core samples, hitting every count exactly.
    still_needed_test = target[TEST] - count[TEST]
    if still_needed_test:
        picked = rng.choice(remaining, size=still_needed_test, replace=False)
        subset[picked] = TEST
        remaining = np.setdiff1d(remaining, picked)
    still_needed_val = target[VALIDATION] - count[VALIDATION]
    if still_needed_val:
        picked = rng.choice(remaining, size=still_needed_val, replace=False)
        subset[picked] = VALIDATION
    return subset, layer
