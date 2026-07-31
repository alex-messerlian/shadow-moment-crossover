"""Range-constrained variants of the single-copy moment U-statistic.

The exact U-statistic of :mod:`anrl.benchmark.moment_ustats` is unbiased but
unconstrained: at large ``n`` and modest budgets it routinely returns values
outside the physically attainable range of ``Tr(rho^k)``, including negative
purities and purities above one.  Any deployed pipeline would project such an
estimate back into the feasible set.  This module supplies two such projections
so the effect on the reported error can be measured.

For an ``n``-qubit state, ``Tr(rho^k)`` lies in

    [d^{1-k}, 1],   d = 2^n,

the lower end attained at the maximally mixed state and the upper end at any
pure state.  At ``k = 2`` this is ``[2^{-n}, 1]``.

**CLIPPED** is the Euclidean projection onto that interval.  Because the
interval is convex and contains the true value, projection is a contraction
toward the truth: for every realization,

    |clip(x) - mu| <= |x - mu|                    whenever mu is in the interval,

so the squared error is weakly reduced *pointwise*, with no distributional
assumption.  The improvement is therefore free, and the RMSE of a clipped
estimator can never exceed the interval width, which is below one.  This is the
mathematical content of the objection this module exists to measure.

**SHRINKAGE** additionally pulls the estimate toward the depolarizing floor by a
linear factor.  Unlike clipping this is *not* free: it trades variance for bias
and can increase the error on an individual sample.  The rule used here is

    a = sigma^2 / (sigma^2 + (theta_clip - floor)^2),
    theta_shrunk = clip((1 - a) * theta_hat + a * floor),

with ``sigma`` the exact Hoeffding standard deviation for the cell (available
from the projection variances without looking at the realized sample) and
``theta_clip`` the clipped raw estimate standing in for the unknown mean.  This
is the MSE-optimal linear shrinkage toward the floor for a Gaussian estimator of
known variance, with the unknown mean replaced by its clipped plug-in.  The
result is clipped, since clipping is free.  When ``sigma`` is small relative to
the distance from the floor, ``a -> 0`` and the rule returns the clipped raw
estimate; when ``sigma`` dominates, ``a -> 1`` and it returns the floor.
"""

from __future__ import annotations

import numpy as np


def physical_range(n: int, k: int) -> tuple[float, float]:
    """Attainable range ``[d^{1-k}, 1]`` of ``Tr(rho^k)`` for ``n`` qubits, ``d = 2^n``.

    The lower end is the maximally mixed value ``2^{n(1-k)}``; the upper end is
    the pure-state value ``1``.  Both ends are attained, so the interval cannot
    be tightened without further assumptions on the state.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    return 2.0 ** (n * (1 - k)), 1.0


def clip_moment(theta, n: int, k: int):
    """Euclidean projection of a moment estimate onto ``[d^{1-k}, 1]``.

    Weakly reduces squared error pointwise against any true value inside the
    range, with no assumptions.  Accepts scalars or arrays.
    """
    lo, hi = physical_range(n, k)
    return np.clip(theta, lo, hi)


def shrinkage_coefficient(theta, n: int, k: int, sigma: float):
    """Plug-in MSE-optimal shrinkage weight toward the depolarizing floor.

    ``a = sigma^2 / (sigma^2 + (clip(theta) - floor)^2)``, in ``[0, 1]``.
    ``sigma`` is the exact Hoeffding standard deviation of the estimator for the
    cell, which the pipeline knows from the projection variances; it is not read
    off the realized sample.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    lo, _ = physical_range(n, k)
    gap = clip_moment(theta, n, k) - lo
    denom = sigma * sigma + gap * gap
    return np.where(denom > 0, sigma * sigma / np.where(denom > 0, denom, 1.0), 0.0)


def shrink_moment(theta, n: int, k: int, sigma: float):
    """Linear shrinkage toward the floor by :func:`shrinkage_coefficient`, then clipped.

    Not free: unlike clipping this can increase the error on an individual
    sample.  It is included as the natural next thing a practitioner would try
    after clipping.
    """
    lo, _ = physical_range(n, k)
    a = shrinkage_coefficient(theta, n, k, sigma)
    return clip_moment((1.0 - a) * np.asarray(theta, dtype=float) + a * lo, n, k)
