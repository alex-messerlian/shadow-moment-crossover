"""Tests for the theory stress-test: new ensembles, extreme-noise bias, generality.

Pins: the new ensembles produce states with the expected purity (haar_pure == 1.0
to machine precision); the bias laws stay exact at g=0.3 to 1e-9 on all three
ensembles; the state-agnostic estimator is unbiased and matches direct MC.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank
from anrl.benchmark.budget import moment_ustat_linear
from anrl.benchmark.moments import moment
from anrl.theory.bias import brute_force_collective_value, collective_value
from anrl.theory.general import (
    estimate_hoeffding_components_general,
    predicted_collective_rmse_general,
    sample_batched_general,
)
from anrl.theory.variance import exact_single_copy_rmse


# ---------------------------------------------------------------------------
# 1 — new ensembles produce states with the expected purity
# ---------------------------------------------------------------------------
def test_haar_pure_purity_is_one() -> None:
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 5):
        state = haar_pure(n, rng)
        assert state.purity() == pytest.approx(1.0, abs=1e-12)  # exactly 1 to machine precision
        rho = state.density_matrix()
        assert float(np.trace(rho @ rho).real) == pytest.approx(1.0, abs=1e-12)


def test_low_rank_and_ghz_purity() -> None:
    rng = np.random.default_rng(1)
    for n in (3, 4, 5):
        lr = low_rank(n, 2, rng)
        # rank-2 Ginibre purity is structured, ~0.5-0.65, and matches the dense Tr(rho^2).
        assert 0.45 < lr.purity() < 0.7
        assert lr.purity() == pytest.approx(float(np.trace(lr.density_matrix() @ lr.density_matrix()).real), abs=1e-12)
        gz = ghz_noisy(n, 0.15, rng)
        d = 2 ** n
        # (1-q)^2 + 2(1-q)q/d + q^2/d, with Tr(P^2)=1 for the pure GHZ component.
        expected = 0.85 ** 2 + 2 * 0.85 * 0.15 / d + 0.15 ** 2 / d
        assert gz.purity() == pytest.approx(expected, abs=1e-12)


def test_ghz_is_deterministic() -> None:
    # ghz_noisy ignores rng (fixed GHZ vector).
    a = ghz_noisy(4, 0.15, np.random.default_rng(0))
    b = ghz_noisy(4, 0.15, np.random.default_rng(999))
    assert np.allclose(a.components, b.components)


# ---------------------------------------------------------------------------
# 2 — bias laws stay exact at extreme noise g = 0.3 on all three ensembles
# ---------------------------------------------------------------------------
def test_bias_laws_exact_at_extreme_noise() -> None:
    rng = np.random.default_rng(2)
    makers = {"haar_pure": lambda n: haar_pure(n, rng),
              "low_rank": lambda n: low_rank(n, 2, rng),
              "ghz_noisy": lambda n: ghz_noisy(n, 0.15, rng)}
    max_err = 0.0
    for name, make in makers.items():
        for n in (2, 3):
            rho = make(n).density_matrix()
            for nm in ("depolarizing", "amplitude_damping", "dephasing"):
                for k in (2, 3):  # k=4 at n=3 is a 4096-dim brute force; covered in the experiment
                    for g in (0.2, 0.3):
                        err = abs(collective_value(rho, k, nm, g, n)
                                  - brute_force_collective_value(rho, k, nm, g, n))
                        max_err = max(max_err, err)
                        assert err < 1e-9, (name, n, nm, k, g, err)
    assert max_err < 1e-9


# ---------------------------------------------------------------------------
# 3 — state-agnostic estimator: unbiased sampler, variance matches MC
# ---------------------------------------------------------------------------
def test_general_sampler_unbiased() -> None:
    rng = np.random.default_rng(3)
    for make in (lambda: low_rank(3, 2, rng), lambda: ghz_noisy(3, 0.15, rng)):
        state = make()
        truth = moment(state.density_matrix(), 2)
        est = [moment_ustat_linear(sample_batched_general(state, 800, rng), 2) for _ in range(120)]
        sem = float(np.std(est) / np.sqrt(len(est)))
        assert abs(float(np.mean(est)) - truth) < 4 * sem


def test_general_variance_matches_mc_out_of_ensemble() -> None:
    # The exact-Hoeffding RMSE must match a direct MC on low_rank (rank-2, NOT the
    # noisy-pure spectrum) and ghz (structured) -- the out-of-ensemble test.
    rng = np.random.default_rng(4)
    for make in (lambda: low_rank(4, 2, rng), lambda: ghz_noisy(4, 0.15, rng)):
        state = make()
        for k in (2, 3):
            comps = estimate_hoeffding_components_general(state, k, 120000, np.random.default_rng([4, k]))
            ests = [moment_ustat_linear(sample_batched_general(state, 4000, rng), k) for _ in range(400)]
            rmse_mc = float(np.sqrt(np.var(ests, ddof=1)))
            rmse_th = exact_single_copy_rmse(comps, k, 4000)
            assert 0.8 < rmse_th / rmse_mc < 1.25, (k, rmse_th, rmse_mc)


def test_predicted_collective_rmse_general_runs() -> None:
    rng = np.random.default_rng(5)
    rhos = [low_rank(3, 2, rng).density_matrix() for _ in range(3)]
    for nm in ("depolarizing", "amplitude_damping", "dephasing"):
        r = predicted_collective_rmse_general(rhos, 2, nm, 0.1, 2000, 3)
        assert r >= 0.0 and np.isfinite(r)
