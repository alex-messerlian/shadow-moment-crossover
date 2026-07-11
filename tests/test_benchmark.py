"""Tests for the single-copy vs collective purity benchmark.

Pinned to known values (purity references, noise-model formula) plus the
approximate-unbiasedness of both estimators and the core anchor: single-copy
shadow RMSE grows exponentially with n while the collective SWAP test stays
small and wins at n=4 across the noise range.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.physics import depolarize, ghz, maximally_mixed, random_density
from anrl.benchmark import (
    collective_purity_estimate,
    collective_signal,
    evaluate_estimator,
    gates_all_to_all,
    gates_linear_1d,
    make_collective_estimator,
    make_shadow_estimator,
    purity,
    shadow_purity_estimate,
)


def _noisy_states(n: int, k: int, seed: int) -> list[np.ndarray]:
    dim = 2 ** n
    rng = np.random.default_rng(seed)
    return [
        depolarize(random_density(dim, int(rng.integers(1, dim + 1)), rng), rng.uniform(0.0, 0.3))
        for _ in range(k)
    ]


# ---------------------------------------------------------------------------
# 1 — purity reference values
# ---------------------------------------------------------------------------
def test_purity_reference_values() -> None:
    for n in (2, 3):
        assert purity(ghz(n)) == pytest.approx(1.0, abs=1e-9)  # pure
        assert purity(maximally_mixed(2 ** n)) == pytest.approx(1.0 / 2 ** n, abs=1e-9)


# ---------------------------------------------------------------------------
# 2 — the noise model is correct (signal formula, tested directly)
# ---------------------------------------------------------------------------
def test_signal_formula_direct() -> None:
    for n in (2, 3, 4):
        for p in (0.0, 0.03, 0.1):
            for gfn in (gates_all_to_all, gates_linear_1d):
                pur = 0.37  # arbitrary purity value
                p_eff = 1.0 - (1.0 - p) ** gfn(n)
                expected = (1.0 - p_eff) * pur + p_eff / 2 ** n
                assert collective_signal(pur, p, n, gfn) == pytest.approx(expected, abs=1e-12)


def test_collective_zero_bias_at_p_gate_zero() -> None:
    # p_gate = 0 => p_eff = 0 => signal == purity exactly (no bias).
    for n in (2, 3, 4):
        for gfn in (gates_all_to_all, gates_linear_1d):
            assert collective_signal(0.42, 0.0, n, gfn) == pytest.approx(0.42, abs=1e-12)


def _swap_operator(dim: int) -> np.ndarray:
    """SWAP on two ``dim``-dimensional registers: |k>|l> -> |l>|k>."""
    swap = np.zeros((dim * dim, dim * dim))
    for k in range(dim):
        for l in range(dim):
            swap[l * dim + k, k * dim + l] = 1.0
    return swap


def test_signal_formula_matches_explicit_swap_construction() -> None:
    # Independent physical oracle: the noisy signal must equal
    # Tr(SWAP @ [(1 - p_eff) rho(x)rho + p_eff I/2^(2n)]), the global-depolarized
    # two-copy state measured by the ideal SWAP.  (Also validates Tr(SWAP)=2^n and
    # ideal <SWAP> = Tr(rho^2) at p_gate=0.)
    rng = np.random.default_rng(3)
    for n in (2, 3):
        dim = 2 ** n
        rho = depolarize(random_density(dim, dim, rng), 0.2)
        swap = _swap_operator(dim)
        maximally_mixed_2copy = np.eye(dim * dim) / (dim * dim)
        for p in (0.0, 0.05, 0.1):
            for gfn in (gates_all_to_all, gates_linear_1d):
                p_eff = 1.0 - (1.0 - p) ** gfn(n)
                two_copy = (1.0 - p_eff) * np.kron(rho, rho) + p_eff * maximally_mixed_2copy
                expected = float(np.trace(swap @ two_copy).real)
                assert collective_signal(purity(rho), p, n, gfn) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# 3 — both estimators are approximately unbiased at their target
# ---------------------------------------------------------------------------
def test_shadow_unbiased_high_snapshots() -> None:
    rng = np.random.default_rng(0)
    rho = depolarize(random_density(4, 4, rng), 0.2)  # n=2
    true = purity(rho)
    # Average several low-variance (high snapshot, dense pairing) estimates so
    # the mean pins the expectation, not a single noisy draw.
    estimates = [shadow_purity_estimate(rho, 4000, rng, n_pairs=60000) for _ in range(10)]
    assert abs(float(np.mean(estimates)) - true) < 0.03


def test_collective_unbiased_zero_noise() -> None:
    rng = np.random.default_rng(1)
    rho = depolarize(random_density(4, 4, rng), 0.2)  # n=2
    true = purity(rho)
    est = collective_purity_estimate(rho, 40000, 0.0, 2, gates_all_to_all, rng)
    assert abs(est - true) < 0.02


# ---------------------------------------------------------------------------
# 4 — harness accounts for copy cost and pairs by state
# ---------------------------------------------------------------------------
def test_evaluate_estimator_copy_accounting_and_pairing() -> None:
    states = _noisy_states(2, 6, seed=5)
    budget = 2000
    shadow = evaluate_estimator(make_shadow_estimator(), states, budget, np.random.default_rng(0))
    collective = evaluate_estimator(
        make_collective_estimator(0.0, gates_all_to_all), states, budget, np.random.default_rng(0)
    )
    # single-copy spends 1 copy/snapshot; collective spends 2 copies/SWAP test.
    assert shadow["n_uses"] == budget
    assert collective["n_uses"] == budget // 2
    # Errors are aligned by state (same states list -> paired comparison).
    assert shadow["errors"].shape == (len(states),)
    assert collective["errors"].shape == (len(states),)


# ---------------------------------------------------------------------------
# 5 — the core anchor: exponential single-copy growth; collective wins at n=4
# ---------------------------------------------------------------------------
def test_anchor_exponential_growth_and_collective_win() -> None:
    budget = 2000
    n_states = 12

    shadow_rmse = {}
    for n in (2, 3, 4):
        states = _noisy_states(n, n_states, seed=1000 + n)
        res = evaluate_estimator(
            make_shadow_estimator(), states, budget, np.random.default_rng(n)
        )
        shadow_rmse[n] = res["rmse"]

    # Exponential growth: n=4 shadow RMSE is at least several times the n=2 one.
    assert shadow_rmse[4] >= 3.0 * shadow_rmse[2]
    assert shadow_rmse[4] > 0.8  # the estimator has genuinely blown up at n=4

    # Collective beats single-copy at n=4 across the noise range, both gate models.
    states4 = _noisy_states(4, n_states, seed=1000 + 4)
    for gfn in (gates_all_to_all, gates_linear_1d):
        for p_gate in (0.0, 0.05, 0.1):
            res = evaluate_estimator(
                make_collective_estimator(p_gate, gfn), states4, budget, np.random.default_rng(7)
            )
            assert res["rmse"] < shadow_rmse[4]
            assert res["rmse"] < 0.2  # collective stays small even under noise
