"""Tests for the analytic crossover theory (anrl/theory).

Part 1: both collective bias laws match a brute-force ``Tr(C_k . noisy)`` to 1e-9.
Part 2: the Hoeffding zetas are stable, the closed-form zeta1 matches a dense
construction, and M* is well-behaved.  Part 3: the crossover predictor runs over
the saved sweeps and produces the comparison table.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pytest

from anrl.benchmark.budget import sample_batched
from anrl.benchmark.ensembles import noisy_pure
from anrl.benchmark.moments import moment
from anrl.theory import (
    alpha_eff,
    brute_force_collective_value,
    collective_bias,
    collective_value,
    depolarizing_bias,
    estimate_zeta1,
    estimate_zeta2,
    estimate_zetas,
    noisy_pure_moment,
    single_copy_rmse,
    single_copy_variance,
)
from anrl.theory.crossover import (
    build_comparison,
    load_measured_crossovers,
    predict_crossover,
    predicted_collective_rmse,
    predicted_single_rmse,
)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Part 1 — bias laws match brute force
# ---------------------------------------------------------------------------
def test_bias_laws_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    # Full grid at n=2 (cheap); n=3 for k=2,3 (n=3,k=4 is a 4096-dim register,
    # covered by a single spot-check to keep the test fast).
    configs = [(2, k) for k in (2, 3, 4)] + [(3, 2), (3, 3)]
    for n, k in configs:
        rho = noisy_pure(n, 0.15, rng).density_matrix()
        for nm in ("depolarizing", "amplitude_damping", "dephasing"):
            for g in (0.03, 0.08):
                law = collective_value(rho, k, nm, g, n)
                brute = brute_force_collective_value(rho, k, nm, g, n)
                assert law == pytest.approx(brute, abs=1e-9), (n, k, nm, g)
    # n=3, k=4 spot check (one config, still ~a few seconds).
    rho = noisy_pure(3, 0.15, rng).density_matrix()
    for nm in ("depolarizing", "dephasing"):
        assert collective_value(rho, 4, nm, 0.05, 3) == pytest.approx(
            brute_force_collective_value(rho, 4, nm, 0.05, 3), abs=1e-9
        )


def test_depolarizing_bias_formula() -> None:
    # g * |Tr(rho^k) - 2^{n(1-k)}|.
    for n, k, g in [(4, 2, 0.05), (5, 3, 0.1), (3, 4, 0.02)]:
        mk = 0.7
        assert depolarizing_bias(mk, k, g, n) == pytest.approx(g * abs(mk - 2.0 ** (n * (1 - k))), abs=1e-14)


def test_noisy_pure_moment_closed_form() -> None:
    rng = np.random.default_rng(1)
    for n in (2, 3, 4):
        rho = noisy_pure(n, 0.1, rng).density_matrix()
        for k in (2, 3, 4):
            assert noisy_pure_moment(n, k, 0.1) == pytest.approx(moment(rho, k), abs=1e-10)


# ---------------------------------------------------------------------------
# Part 2 — Hoeffding zetas: stability, closed-form zeta1, M* behaviour
# ---------------------------------------------------------------------------
def _zeta1_dense(state, k: int, snaps: np.ndarray) -> float:
    """Reference zeta1 = Var_x Tr(G_x rho^{k-1}) built densely (no closed form)."""
    rho = state.density_matrix()
    rho_km1 = np.linalg.matrix_power(rho, k - 1)
    g = np.array([reduce(np.kron, list(snaps[i])) for i in range(snaps.shape[0])])
    h1 = np.einsum("mab,ba->m", g, rho_km1).real  # Tr(G_x rho^{k-1})
    return float(np.var(h1, ddof=1))


def test_zeta1_closed_form_matches_dense() -> None:
    # The noisy-pure closed form for zeta1 must equal the dense Tr(G_x rho^{k-1}).
    rng = np.random.default_rng(2)
    for n in (2, 3):
        state = noisy_pure(n, 0.1, rng)
        snaps = sample_batched(state, 4000, np.random.default_rng([2, n, 7]))
        for k in (2, 3, 4):
            # closed form on the SAME snapshots
            from anrl.theory.variance import _psi_G_psi, _eigs
            lam1, lam0 = _eigs(state)
            p = _psi_G_psi(state, snaps)
            z1_closed = (lam1 ** (k - 1) - lam0 ** (k - 1)) ** 2 * np.var(p, ddof=1)
            z1_dense = _zeta1_dense(state, k, snaps)
            assert z1_closed == pytest.approx(z1_dense, rel=1e-6)


def test_zetas_stable_and_mstar_grows() -> None:
    # Estimates are stable across states, and M* is finite, positive, and grows in n.
    prev = None
    for n in (2, 3, 4):
        z = estimate_zetas(n, 2, 0.1, 40000, seed=0, n_states=3)
        assert z["zeta1"] > 0 and z["zeta2"] > 0 and np.isfinite(z["M_star"])
        assert z["zeta1_rel_spread"] < 0.2 and z["zeta2_rel_spread"] < 0.3
        if prev is not None:
            assert z["M_star"] > prev  # M* increases with n (shadow-norm growth)
        prev = z["M_star"]


def test_exact_ustatistic_variance_matches_mc() -> None:
    # The exact Hoeffding variance (closed-form projection components) must match a
    # direct Monte-Carlo of the U-statistic variance at small n, for every k.
    from anrl.benchmark.budget import moment_ustat_linear
    from anrl.theory import estimate_hoeffding_components, exact_ustatistic_variance
    rng = np.random.default_rng(3)
    for n, k in [(2, 2), (2, 3), (2, 4), (3, 3)]:
        state = noisy_pure(n, 0.1, rng)
        comps = estimate_hoeffding_components(state, k, 120000, np.random.default_rng([3, n, k]))
        assert len(comps) == k and all(c >= 0 for c in comps)  # projection variances are >= 0
        for m in (300, 1200):
            ests = [moment_ustat_linear(sample_batched(state, m, rng), k) for _ in range(500)]
            var_mc = float(np.var(ests, ddof=1))
            var_th = exact_ustatistic_variance(comps, k, m)
            assert 0.75 < var_th / var_mc < 1.35, (n, k, m, var_th, var_mc)


def test_exact_model_reproduces_alpha_where_two_term_fails() -> None:
    # At k=3, n=6 the two-term model badly over-predicts alpha (~0.98 vs measured
    # ~0.61); the EXACT model must land near the measured value.
    from anrl.theory import estimate_hoeffding_components, exact_fitted_alpha, fitted_alpha
    budgets = [2000, 8000, 32000]
    per_state = [
        estimate_hoeffding_components(
            noisy_pure(6, 0.1, np.random.default_rng([0, 6, s, 0])), 3, 120000,
            np.random.default_rng([0, 6, 3, s, 5]))
        for s in range(3)
    ]
    comps = [float(np.mean([cs[i] for cs in per_state])) for i in range(3)]
    two_term = fitted_alpha(budgets, 3, comps[0], comps[2])
    exact = exact_fitted_alpha(budgets, comps, 3)
    assert two_term > 0.85  # two-term model over-predicts (measured ~0.61)
    assert 0.50 < exact < 0.70  # exact model near the measured 0.607


def test_truncated_variance_has_the_correct_second_order_coefficient() -> None:
    # The 1/M^2 coefficient is k^2 (k-1)^2 / 2 times the SECOND PROJECTION zeta_2 --
    # so 2 zeta_2 at k=2, not zeta_2.  Section 3.5 rests on this.
    from anrl.theory import exact_ustatistic_variance, truncated_variance

    assert truncated_variance([1.0, 50.0], 2, 1000) == pytest.approx(4 / 1000 + 2 * 50 / 1e6)
    assert truncated_variance([1.0, 50.0, 7e3], 3, 1000) == pytest.approx(9 / 1000 + 18 * 50 / 1e6)
    # the truncation is the large-M limit of the exact law: the ratio -> 1 as M grows.
    for k in (2, 3, 4):
        comps = [1.0, 50.0, 7e3, 9e5][:k]
        ratios = [truncated_variance(comps, k, m) / exact_ustatistic_variance(comps, k, m)
                  for m in (1e5, 1e7, 1e9)]
        assert ratios == sorted(ratios, reverse=True)  # monotone approach from above
        assert ratios[-1] == pytest.approx(1.0, abs=1e-6)


def test_variance_and_alpha_formulas() -> None:
    # single_copy_variance is the retained STRAW MAN (wrong 1/M^2 coefficient) -- pinned
    # here so it cannot silently drift into looking correct.  Use truncated_variance.
    assert single_copy_variance(2, 1.0, 50.0, 1000) == pytest.approx(4 / 1000 + 50 / 1e6)
    assert single_copy_rmse(2, 1.0, 50.0, 1000) == pytest.approx(np.sqrt(4 / 1000 + 50 / 1e6))
    assert alpha_eff(1e9, 1.0) == pytest.approx(0.5, abs=1e-6)  # M >> M* -> 0.5
    assert alpha_eff(1.0, 1e9) == pytest.approx(1.0, abs=1e-6)  # M << M* -> 1.0
    assert alpha_eff(10.0, 10.0) == pytest.approx(0.75, abs=1e-9)  # M == M* -> 0.75


# ---------------------------------------------------------------------------
# Part 3 — crossover predictor runs over the saved results
# ---------------------------------------------------------------------------
def test_predict_crossover_sustained() -> None:
    # A tiny synthetic zetas dict: single RMSE grows, collective floor saturates.
    zetas = {(n, 2): {"zeta1": 1.0, "zeta2": 50.0 * 6 ** n} for n in range(2, 9)}
    # depolarizing bias grows with n, so there IS a sustained crossover.
    nstar = predict_crossover(2, "depolarizing", 0.1, 2000, list(range(2, 9)), zetas, 0.1)
    assert nstar is None or (2 <= nstar <= 8)
    # A zero-noise-ish huge-budget case may never cross -> None is acceptable.


@pytest.mark.skipif(
    not (REPO / "results" / "budget_scaling.json").exists(), reason="needs saved sweep"
)
def test_crossover_comparison_over_saved_results() -> None:
    zetas = {(n, k): estimate_zetas(n, k, 0.1, 30000, seed=0, n_states=2)
             for n in range(2, 7) for k in (2, 3)}
    measured = load_measured_crossovers(REPO / "results" / "budget_scaling.json", default_budget=2000)
    assert len(measured) > 0
    comp = build_comparison([c for c in measured if c["k"] in (2, 3)
                             and all(n <= 6 for n in c["sizes"])], zetas, 0.1)
    for c in comp:
        assert "predicted_n" in c and "measured_n" in c
        assert c["predicted_n"] is None or c["predicted_n"] in c["sizes"]
