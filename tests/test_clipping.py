"""Tests for the clipped-estimator RMSE closed form (anrl.theory.clipping).

Locks the derivation E[(clip(X,a,b)-mu)^2] for X ~ N(mu, sigma^2) against Monte
Carlo and against the boundary identities (mu at an edge -> RMSE = sigma/sqrt2).
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.theory.clipping import clipped_mse, clipped_rmse


@pytest.mark.parametrize("mu,sigma", [
    (0.5, 0.1), (0.5, 0.3), (0.2, 0.1), (0.9, 0.1), (0.9, 0.3),
    (1.0, 0.2), (0.0, 0.2), (0.05, 0.1),
])
def test_matches_gaussian_mc(mu, sigma):
    rng = np.random.default_rng(int(1000 * mu + 100 * sigma))
    x = rng.normal(mu, sigma, 3_000_000)
    mc = float(np.sqrt(np.mean((np.clip(x, 0.0, 1.0) - mu) ** 2)))
    assert clipped_rmse(mu, sigma) == pytest.approx(mc, rel=0.01)


def test_clipping_reduces_or_preserves_rmse():
    """Clipping to [0,1] never increases RMSE about mu in [0,1]."""
    for mu in (0.1, 0.5, 0.9, 1.0):
        for sigma in (0.05, 0.2, 0.5):
            assert clipped_rmse(mu, sigma) <= sigma + 1e-12


def test_negligible_when_far_from_boundary():
    """mu deep inside [0,1] with small sigma -> clipping negligible (RMSE ~= sigma)."""
    assert clipped_rmse(0.5, 0.02) == pytest.approx(0.02, rel=1e-6)


def test_boundary_identity_mu_equals_1():
    """At mu = 1 the upper half clips to 1 (contributes 0), leaving sigma^2/2 -> sigma/sqrt2.
    Exact only in the far-boundary limit (0 many sigma away), so use small sigma."""
    for sigma in (0.02, 0.05, 0.1):
        assert clipped_rmse(1.0, sigma) == pytest.approx(sigma / np.sqrt(2.0), rel=1e-6)


def test_boundary_identity_mu_equals_0():
    for sigma in (0.02, 0.05, 0.1):
        assert clipped_rmse(0.0, sigma) == pytest.approx(sigma / np.sqrt(2.0), rel=1e-6)


def test_sigma_zero():
    assert clipped_rmse(0.5, 0.0) == pytest.approx(0.0)
    assert clipped_rmse(1.3, 0.0) == pytest.approx(0.3)  # clip(1.3)->1.0, |1.0-1.3|
    assert clipped_mse(-0.2, 0.0) == pytest.approx(0.04)  # clip(-0.2)->0, (0-(-0.2))^2


def test_custom_range():
    """Formula respects an arbitrary [a,b]; check vs MC on [0.2, 0.8]."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.75, 0.2, 3_000_000)
    mc = float(np.sqrt(np.mean((np.clip(x, 0.2, 0.8) - 0.75) ** 2)))
    assert clipped_rmse(0.75, 0.2, a=0.2, b=0.8) == pytest.approx(mc, rel=0.01)
