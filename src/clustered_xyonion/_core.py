"""Core XYOnion routines operating on a single (roughly convex) group.

Only numpy is needed here, so this module can be used and tested without
scikit-learn.

This follows the author's reference MATLAB implementation, ``XYonion.m``
(https://github.com/JokinEzenarro/XYOnion, MIT License, Copyright (c) 2025
Jokin Ezenarro), of the method in Ezenarro, Anal. Chim. Acta 1364 (2025)
344229:

* The data is peeled in ``n_onion`` rounds. Each round takes one layer for
  each subset, in the order train, test, validation. A layer for subset ``s``
  holds ``round(loop_fraction * m0 * fraction_s)`` samples, where ``m0`` is the
  number of samples still unassigned at the start of the round.
* Each layer is chosen by DISTSLCT (:func:`distslct`), which picks spread-out
  samples on the outside of the remaining data, using an SPXY-style combined X
  and y distance. X and y are re-centred on the remaining samples for every
  layer.
* Whatever is left after the rounds is split at random.

Differences from ``XYonion.m`` (all deliberate):

* A validation set can be requested as well as train (calibration) and test.
* The final random split hits each subset's exact count (the reference rounds
  per layer and then uses ``ceil``, so its counts are approximate), and no
  layer is allowed to overshoot a subset's count.
* Mahalanobis whitening uses the pseudoinverse, so singular covariances work,
  and is done per cluster by the caller.
* DISTSLCT never picks the same sample twice and treats a block with no
  spread (e.g. constant y) as contributing nothing, where the reference
  could divide by zero.
"""

from __future__ import annotations

import numpy as np

METRICS = ("euclidean", "mahalanobis")

# Codes used in the returned ``subset`` arrays.
TRAIN, VALIDATION, TEST = 0, 1, 2
SUBSET_NAMES = ("train", "validation", "test")

# Order in which each onion round takes its layers (train first, so the
# outermost samples always go to train).
LAYER_ORDER = (TRAIN, TEST, VALIDATION)


def round_half_up(x: float) -> int:
    """Round to the nearest integer, with halves rounded up (not to even).

    Matches MATLAB's ``round`` for the non-negative values used here.
    """
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

    The reference implementation uses ``V @ diag(1/sqrt(lam)) @ V.T`` instead,
    which only adds a final rotation back; rotations don't change distances,
    so both give the same Mahalanobis distances (here also for singular ``S``).

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


def _normalise(d: np.ndarray) -> np.ndarray:
    """``d / max(d)``, or all zeros if ``d`` has no positive value.

    This is how XYOnion balances X against y: each block's distances are
    divided by that block's largest distance among the current candidates, so
    both blocks run from 0 to 1 and carry equal weight. A block with no spread
    (e.g. constant y) contributes nothing instead of dividing by zero.
    """
    top = d.max() if d.size else 0.0
    return d / top if top > 0 else np.zeros_like(d)


def _stdsslct(x: np.ndarray, y: np.ndarray, n_select: int) -> np.ndarray:
    """First stage of DISTSLCT: pick samples that span the main directions.

    Port of ``stdsslct_xyonion`` in ``XYonion.m``. ``x`` and ``y`` must already
    be centred. Repeatedly:

    1. score every sample by ``|x|^2 / max|x|^2 + y^2 / max y^2`` (how far it is
       from the centre in X and in y) and pick the highest;
    2. remove the picked sample and make every other sample's X orthogonal to
       it (subtract its projection onto the picked sample), and subtract the
       picked sample's y from every y.

    Step 2 means each new pick lies along a direction the earlier picks don't
    already cover, which spreads the picks across the data's extremes.

    Returns the indices (into ``x``) of the ``n_select`` picked samples.
    """
    x = x.astype(float, copy=True)
    y = y.astype(float, copy=True)
    original_index = np.arange(x.shape[0])
    picked = []
    for _ in range(n_select):
        squared_norm_x = np.sum(x**2, axis=1)
        squared_norm_y = y**2
        score = _normalise(squared_norm_x) + _normalise(squared_norm_y)
        k = int(np.argmax(score))            # ties go to the first sample
        picked.append(int(original_index[k]))

        rx0, ry0 = x[k].copy(), y[k]
        x = np.delete(x, k, axis=0)
        y = np.delete(y, k)
        original_index = np.delete(original_index, k)

        # Remove each sample's component along rx0: x - (x . rx0 / |rx0|^2) rx0.
        rx0_squared = rx0 @ rx0
        if rx0_squared > 0:
            x = x - np.outer((x @ rx0) / rx0_squared, rx0)
        y = y - ry0
    return np.array(picked, dtype=int)


def distslct(x: np.ndarray, y: np.ndarray, n_select: int) -> np.ndarray:
    """Pick ``n_select`` spread-out samples on the outside of the data (DISTSLCT).

    Port of ``distslct_xyonion`` in ``XYonion.m``, the layer-selection step of
    XYOnion.

    1. Centre ``x`` and ``y`` on the samples given (so the centre moves as the
       data is peeled).
    2. If fewer samples are wanted than ``x`` has columns, use
       :func:`_stdsslct` for all of them.
    3. Otherwise pick the first ``n_features`` with :func:`_stdsslct`, then add
       samples one at a time: keep, for every sample, the running SUM of its
       distances (in X, and separately in y) to all samples picked so far, and
       pick the sample with the largest ``sumX / max sumX + sumY / max sumY``.
       Samples far from everything already picked score highest, so the picks
       spread around the outside of the data.

    Parameters
    ----------
    x : ndarray of shape (n, n_features)
    y : ndarray of shape (n,)
    n_select : int
        Number of samples to pick (at most ``n``).

    Returns
    -------
    ndarray of int
        Indices (into ``x``) of the picked samples, in the order picked.
    """
    n, n_features = x.shape
    n_select = min(int(n_select), n)
    if n_select <= 0:
        return np.array([], dtype=int)

    x = x - x.mean(axis=0)
    y = y - y.mean()

    if n_select < n_features:
        return _stdsslct(x, y, n_select)

    picked = list(_stdsslct(x, y, n_features))
    is_picked = np.zeros(n, dtype=bool)
    is_picked[picked] = True

    # Running sums of distances from every sample to the picked ones.
    sum_dist_x = np.zeros(n)
    sum_dist_y = np.zeros(n)
    for p in picked:
        sum_dist_x += np.linalg.norm(x - x[p], axis=1)
        sum_dist_y += np.abs(y - y[p])

    while len(picked) < n_select:
        score = _normalise(np.where(is_picked, 0.0, sum_dist_x)) \
            + _normalise(np.where(is_picked, 0.0, sum_dist_y))
        score[is_picked] = -np.inf           # never pick a sample twice
        k = int(np.argmax(score))
        picked.append(k)
        is_picked[k] = True
        sum_dist_x += np.linalg.norm(x - x[k], axis=1)
        sum_dist_y += np.abs(y - y[k])
    return np.array(picked, dtype=int)


def xyonion_group_split(
    X: np.ndarray,
    y: np.ndarray,
    counts,
    fractions,
    n_onion: int,
    loop_fraction: float,
    rng: np.random.RandomState,
):
    """Run XYOnion on one group of samples.

    ``X`` must already be in the space whose Euclidean distance is the chosen
    metric (i.e. whitened by the caller for Mahalanobis).

    Parameters
    ----------
    X : ndarray of shape (n, n_features)
    y : ndarray of shape (n,)
    counts : sequence of 3 int
        Exact number of samples of this group for (train, validation, test).
        Must add up to ``n``.
    fractions : sequence of 3 float
        The overall (train, validation, test) fractions, used for layer sizes.
    n_onion : int
        Number of onion rounds (``nonion`` in the reference); each round takes
        one layer per subset that has a non-zero fraction.
    loop_fraction : float
        Fraction of the remaining samples taken per round (``loopfraction`` in
        the reference, fixed there at 0.1).
    rng : numpy RandomState used for the final random split.

    Returns
    -------
    subset : ndarray of int, shape (n,)
        ``TRAIN`` (0), ``VALIDATION`` (1) or ``TEST`` (2) for each sample.
    layer : ndarray of int, shape (n,)
        Layer index of each sample (0 = outermost, i.e. the first train
        layer); -1 for samples assigned in the final random split.
    """
    n = X.shape[0]
    target = np.asarray(counts, dtype=int)
    if target.shape != (3,) or (target < 0).any() or target.sum() != n:
        raise ValueError(f"counts={list(counts)} must be 3 non-negative ints adding up to {n}.")
    if n_onion < 0:
        raise ValueError(f"n_onion={n_onion} must be >= 0.")

    subset = np.full(n, TRAIN, dtype=int)
    layer = np.full(n, -1, dtype=int)
    count = np.zeros(3, dtype=int)
    remaining = np.arange(n)
    layer_id = 0

    for _ in range(n_onion):
        # Layer sizes come from the samples left at the START of the round.
        m0 = remaining.size
        sizes = {s: round_half_up(loop_fraction * m0 * fractions[s]) for s in LAYER_ORDER}
        for s in LAYER_ORDER:
            size = min(sizes[s], target[s] - count[s], remaining.size)
            if size <= 0:
                continue
            picked = distslct(X[remaining], y[remaining], size)
            chosen = remaining[picked]
            subset[chosen] = s
            layer[chosen] = layer_id
            count[s] += size
            remaining = np.delete(remaining, picked)
            layer_id += 1

    # Final random split of what's left, hitting every count exactly.
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
