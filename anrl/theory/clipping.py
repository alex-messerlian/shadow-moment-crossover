"""Clipped-estimator RMSE for a (near-)Gaussian purity estimate.

The single-copy purity U-statistic is exactly unbiased with mean mu = Tr(rho^k) and
standard deviation sigma from the exact Hoeffding formula (:mod:`anrl.theory.variance`).
A pipeline that CLIPS estimates to a physical range [a, b] (default [0, 1]) reports a
smaller RMSE: clipping projects out-of-range estimates back toward the truth.

For X ~ N(mu, sigma^2) and Y = clip(X, a, b), with alpha = (a-mu)/sigma,
beta = (b-mu)/sigma, phi/Phi the standard normal pdf/cdf:

    E[(Y - mu)^2] = (a-mu)^2 Phi(alpha)                      # mass below a -> a
                  + (b-mu)^2 (1 - Phi(beta))                 # mass above b -> b
                  + sigma^2 [ (Phi(beta) - Phi(alpha))       # in-range truncated variance
                              + alpha phi(alpha) - beta phi(beta) ]

using int_alpha^beta z^2 phi(z) dz = (Phi(beta)-Phi(alpha)) + alpha phi(alpha) - beta phi(beta).
``clipped_rmse`` returns sqrt of this. NOTE: the anrl pipeline does NOT clip its
single-copy estimator (it reports the raw U-statistic), so this is a model of what a
CLIPPING pipeline would report; not a default correction applied to anrl's own RMSE.
"""

from __future__ import annotations

import math


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def clipped_mse(mu: float, sigma: float, a: float = 0.0, b: float = 1.0) -> float:
    """E[(clip(X,a,b) - mu)^2] for X ~ N(mu, sigma^2). Exact closed form."""
    if sigma <= 0.0:
        c = min(max(mu, a), b)
        return (c - mu) ** 2
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    below = (a - mu) ** 2 * _Phi(alpha)
    above = (b - mu) ** 2 * (1.0 - _Phi(beta))
    inrange = sigma * sigma * ((_Phi(beta) - _Phi(alpha)) + alpha * _phi(alpha) - beta * _phi(beta))
    return float(max(0.0, below + above + inrange))


def clipped_rmse(mu: float, sigma: float, a: float = 0.0, b: float = 1.0) -> float:
    """RMSE about mu of a Gaussian estimate clipped to [a, b]. sqrt(:func:`clipped_mse`)."""
    return math.sqrt(clipped_mse(mu, sigma, a, b))
