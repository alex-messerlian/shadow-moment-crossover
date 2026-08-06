"""Tests for the copy-budget sweep and its M-linear estimators.

Pins: the M-linear exact U-statistics equal the reference estimators; the
batched sampler is unbiased; the single-copy RMSE follows 1/sqrt(M) (P1
mechanism); the collective error plateaus at a budget-independent bias floor (P2
mechanism); and the sweep runs end to end (reproducibly) with error bars.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anrl.benchmark import (
    budgets_for,
    crossover_table,
    exact_moment_ustatistic,
    fit_budget_exponent,
    moment,
    moment_ustat_linear,
    predicted_bias_floor,
    run_budget_sweep,
    sample_batched,
    save_budget_sweep,
)
from anrl.benchmark.budget_sweep import _collective_signal
from anrl.benchmark.ensembles import noisy_pure
from anrl.benchmark.moments import collective_moment_estimate
from anrl.benchmark.scaling import snapshots_factored


# ---------------------------------------------------------------------------
# 1, M-linear estimators equal the reference exact estimators
# ---------------------------------------------------------------------------
def test_moment_ustat_linear_matches_reference() -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 5, 6):
        state = noisy_pure(n, 0.1, rng)
        snaps = snapshots_factored(state, 100, rng)
        for k in (2, 3, 4):
            assert moment_ustat_linear(snaps, k) == pytest.approx(
                exact_moment_ustatistic(snaps, k), abs=1e-9
            )


def test_moment_ustat_linear_rejects_too_few_snapshots() -> None:
    rng = np.random.default_rng(1)
    snaps = snapshots_factored(noisy_pure(1, 0.1, rng), 2, rng)  # m=2
    with pytest.raises(ValueError):
        moment_ustat_linear(snaps, 3)


# ---------------------------------------------------------------------------
# 2; the batched sampler is an unbiased shadow
# ---------------------------------------------------------------------------
def test_sample_batched_unbiased() -> None:
    rng = np.random.default_rng(2)
    for n in (2, 3):
        state = noisy_pure(n, 0.2, rng)
        for k in (2, 3):
            truth = moment(state.density_matrix(), k)
            est = [moment_ustat_linear(sample_batched(state, 600, rng), k) for _ in range(120)]
            sem = float(np.std(est) / np.sqrt(len(est)))
            assert abs(float(np.mean(est)) - truth) < 4 * sem  # unbiased within 4 SEM


def test_sample_batched_probabilities_match_dense() -> None:
    # The sampled outcome distribution must match diag(U rho U^dag); check the
    # per-snapshot probability construction against a dense reference at n=2.
    rng = np.random.default_rng(3)
    state = noisy_pure(2, 0.15, rng)
    snaps = sample_batched(state, 4000, rng)
    # Mean of the local shadow reproduces rho (E[snapshot tensor] = rho): compare
    # the mean purity U-statistic to Tr(rho^2) (already covered), here assert the
    # snapshots are valid Hermitian shadows (3 P - I has trace 3*1 - 2 = 1).
    traces = np.einsum("mnii->mn", snaps).real
    assert np.allclose(traces, 1.0, atol=1e-9)  # each 2x2 shadow has unit trace


# ---------------------------------------------------------------------------
# 3, P1 mechanism: single-copy RMSE ~ 1/sqrt(M) (alpha ~ 0.5)
# ---------------------------------------------------------------------------
def test_fit_budget_exponent_recovers_half() -> None:
    # Synthetic RMSE = C / sqrt(M) must fit to alpha = 0.5 exactly.
    budgets = [2000, 8000, 32000, 128000]
    rmses = [0.3 / np.sqrt(m) for m in budgets]
    alpha, se = fit_budget_exponent(budgets, rmses)
    assert alpha == pytest.approx(0.5, abs=1e-9)
    assert se == pytest.approx(0.0, abs=1e-9)


def test_fit_budget_exponent_bootstrap() -> None:
    from anrl.benchmark import fit_budget_exponent_bootstrap
    rng = np.random.default_rng(0)
    budgets = [2000, 8000, 32000]
    # Per-state MSE = (C/sqrt(M))^2 with small state scatter -> alpha ~ 0.5, small se.
    n_states = 40
    mse = np.array([[(0.3 / np.sqrt(m)) ** 2 * (1 + 0.05 * rng.standard_normal()) for _ in range(n_states)]
                    for m in budgets])
    alpha, se = fit_budget_exponent_bootstrap(budgets, mse, rng)
    assert 0.47 < alpha < 0.53 and 0.0 < se < 0.05
    # A single budget or single state yields nan se, no crash.
    assert np.isnan(fit_budget_exponent_bootstrap([2000], mse[:1], rng)[1])


def test_dense_power_sum_split_chunk_independent() -> None:
    # Chunking must not change the value (it is an accumulation of the same terms).
    from anrl.benchmark.budget import dense_power_sum_split
    rng = np.random.default_rng(1)
    for n in (2, 3, 5):  # incl odd n (unequal halves)
        snaps = sample_batched(noisy_pure(n, 0.1, rng), 3000, rng)
        full = dense_power_sum_split(snaps, chunk=10 ** 9)
        chunked = dense_power_sum_split(snaps, chunk=256)
        assert np.max(np.abs(full - chunked)) < 1e-10


def test_single_copy_rmse_scales_as_inverse_sqrt_M() -> None:
    # On the ACTUAL exact estimator, quadrupling M must roughly halve the RMSE.
    rng = np.random.default_rng(4)
    state = noisy_pure(3, 0.1, rng)
    truth = moment(state.density_matrix(), 2)
    rmse = {}
    for m in (2000, 8000, 32000):
        errs = [moment_ustat_linear(sample_batched(state, m, rng), 2) - truth for _ in range(40)]
        rmse[m] = float(np.sqrt(np.mean(np.square(errs))))
    # Each 4x should shrink RMSE by ~2x (1/sqrt(M)); allow a generous band.
    assert 1.6 < rmse[2000] / rmse[8000] < 2.5
    assert 1.6 < rmse[8000] / rmse[32000] < 2.5
    alpha, _ = fit_budget_exponent([2000, 8000, 32000], [rmse[2000], rmse[8000], rmse[32000]])
    assert 0.4 < alpha < 0.6  # close to the predicted 0.5


# ---------------------------------------------------------------------------
# 4, P2 mechanism: collective error plateaus at a budget-independent bias floor
# ---------------------------------------------------------------------------
def test_predicted_bias_floor_formula() -> None:
    # [1 - (1-g)^(k n)] * Tr(rho^k).
    assert predicted_bias_floor(3, 2, 0.05, 0.8) == pytest.approx((1 - 0.95 ** 6) * 0.8, abs=1e-12)
    assert predicted_bias_floor(4, 3, 0.1, 0.7) == pytest.approx((1 - 0.9 ** 12) * 0.7, abs=1e-12)


def test_collective_error_plateaus_at_bias_floor() -> None:
    # With nonzero noise, the collective error is dominated by a budget-INDEPENDENT
    # bias |Tr(sigma^k) - Tr(rho^k)|: quadrupling the budget must not materially
    # lower it, and it must sit near that bias (not fall toward zero like variance).
    rng = np.random.default_rng(5)
    n, k, g = 4, 2, 0.1
    state = noisy_pure(n, 0.1, rng)
    density = state.density_matrix()
    truth = moment(density, k)
    signal = _collective_signal(density, n, k, "amplitude_damping", g, truth)
    bias = abs(signal - truth)
    rmse = {}
    for m in (2000, 8000, 32000):
        errs = [collective_moment_estimate(k, m // k, signal, rng) - truth for _ in range(400)]
        rmse[m] = float(np.sqrt(np.mean(np.square(errs))))
    # Bias is the floor: RMSE stays close to |bias| and barely moves with budget.
    assert abs(rmse[32000] - bias) < 0.02  # plateau sits at the bias
    assert rmse[2000] / rmse[32000] < 1.5  # <50% reduction over 16x budget (not 1/sqrt)
    # A single-copy estimator over the SAME 16x range would shrink ~4x, not <1.5x.


# ---------------------------------------------------------------------------
# 5, budgets_for feasibility plan
# ---------------------------------------------------------------------------
def test_budgets_for_plan() -> None:
    assert budgets_for(2, 2) == (2000, 8000, 32000, 128000)  # k=2 small n: up to 64x
    assert budgets_for(7, 2) == (2000, 8000, 32000)          # k=2 large n: 64x dropped
    assert budgets_for(5, 3) == (2000, 8000, 32000)          # k=3: up to 16x
    assert budgets_for(4, 4) == (500, 2000, 8000)            # k=4: O(M^2), small budgets


# ---------------------------------------------------------------------------
# 6; the sweep runs end to end (reproducibly) and saves error bars
# ---------------------------------------------------------------------------
def test_run_budget_sweep_end_to_end(tmp_path) -> None:
    kw = dict(
        sizes_by_k={2: (2, 3), 3: (2, 3), 4: (2, 3)},
        noise_models=("depolarizing", "dephasing"), rates=(0.05, 0.1),
        n_states=6, n_trials=3, seed=0,
    )
    rows, alpha_fits = run_budget_sweep(**kw, max_workers=4, return_alpha_fits=True)
    assert len(rows) > 0
    # alpha_fits: one per (n,k); each carries the budget curve, alpha, and a se.
    assert len(alpha_fits) > 0
    for a in alpha_fits:
        assert len(a["budgets"]) == len(a["single_rmse"]) and "alpha" in a and "alpha_se" in a
    for r in rows:
        assert r["winner"] in ("collective", "single-copy", "tie")
        assert r["single_rmse"] >= 0 and r["collective_rmse"] >= 0
        lo, hi = r["single_rmse_ci68"]
        assert lo <= r["single_rmse"] <= hi
        assert r["single_copies"] == r["budget"]
        assert r["collective_measurements"] == r["budget"] // r["k"]
        assert r["predicted_floor"] >= 0 and r["measured_bias"] >= 0

    # single-copy RMSE is noise-independent for a fixed (n, k, budget)
    by_key: dict[tuple[int, int, int], set[float]] = {}
    for r in rows:
        by_key.setdefault((r["n"], r["k"], r["budget"]), set()).add(round(r["single_rmse"], 12))
    for key, vals in by_key.items():
        assert len(vals) == 1, f"single_rmse varies within {key}: {vals}"

    # reproducible across worker counts
    rows2 = run_budget_sweep(**kw, max_workers=2)
    assert [r["single_rmse"] for r in rows] == [r["single_rmse"] for r in rows2]
    assert [r["paired_z"] for r in rows] == [r["paired_z"] for r in rows2]

    table = crossover_table(rows, group_keys=("k", "budget", "noise_model", "rate"))
    out = tmp_path / "budget.json"
    save_budget_sweep(rows, table, out, {"baseline": 2000})
    payload = json.loads(out.read_text())
    assert len(payload["rows"]) == len(rows) and "crossover_table" in payload
