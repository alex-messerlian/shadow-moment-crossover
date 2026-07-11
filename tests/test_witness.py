"""Tests for the witness-based negativity estimator (Phase 2).

Each test pins an exact property:

1. Defining identity  — sum_P w_P(rho) tr(rho P) == negativity(rho) (1e-9).
2. High-shot recovery — the realizable estimator recovers the true negativity.
3. Variational lower bound (Finding C) — a wrong witness direction underestimates.
4. Efficiency benchmark (Finding B) — at n=4 and a fixed modest shot budget the
   witness estimator (true witness, uniform measurement) beats full
   reconstruction on mean negativity error.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from anrl.physics import (
    all_local_pauli_settings,
    bell_phi_plus,
    depolarize,
    estimate_negativity_witness,
    estimate_pauli_expectations,
    negativity,
    negativity_witness_estimator,
    pauli_string,
    random_density,
    reconstruct,
    simulate_settings,
    witness_weights,
)


def _exact_expectations(rho: np.ndarray, n: int) -> dict[tuple[str, ...], float]:
    """Exact Pauli expectations {P: tr(rho P)} over all 4**n Pauli strings."""
    return {
        term: float(np.trace(rho @ pauli_string(term)).real)
        for term in itertools.product("IXYZ", repeat=n)
    }


# ---------------------------------------------------------------------------
# Property 1 — defining identity: sum_P w_P(rho) <P> == negativity(rho)
# ---------------------------------------------------------------------------
def test_witness_defining_identity() -> None:
    rng = np.random.default_rng(0xC0FFEE)
    max_err = 0.0
    n_states = 0
    for n in (2, 3):
        dim = 2 ** n
        for _ in range(30):
            rank = int(rng.integers(1, dim + 1))
            rho = depolarize(random_density(dim, rank, rng), rng.uniform(0.0, 0.4))
            exps = _exact_expectations(rho, n)
            weights = witness_weights(rho)
            est = estimate_negativity_witness(exps, weights)
            err = abs(est - negativity(rho))
            max_err = max(max_err, err)
            n_states += 1
            assert est == pytest.approx(negativity(rho), abs=1e-9)
    # Report-worthy: the identity holds essentially to machine precision.
    assert max_err < 1e-9, f"max identity error {max_err:.2e} over {n_states} states"


# ---------------------------------------------------------------------------
# Property 2 — high-shot recovery of the realizable estimator
# ---------------------------------------------------------------------------
def test_high_shot_witness_recovery() -> None:
    rng = np.random.default_rng(2026)
    rho = depolarize(bell_phi_plus(), 0.2)  # true negativity = 0.35
    true_neg = negativity(rho)
    counts = simulate_settings(rho, all_local_pauli_settings(2), shots=20000, rng=rng)
    measured = estimate_pauli_expectations(counts, n=2)
    est = negativity_witness_estimator(measured, n=2)
    assert est == pytest.approx(true_neg, abs=0.02)


# ---------------------------------------------------------------------------
# Property 3 — variational lower bound (Finding C): a mismatched witness
# direction underestimates the true negativity.
# ---------------------------------------------------------------------------
def test_variational_lower_bound() -> None:
    rng = np.random.default_rng(7)
    n, dim = 2, 4
    worst_slack = -np.inf  # max over pairs of (estimate - negativity); must be <= 0
    positive_bounds = 0  # count of strictly-positive (non-trivial) lower bounds
    for _ in range(200):
        rho = depolarize(random_density(dim, int(rng.integers(1, dim + 1)), rng), rng.uniform(0.0, 0.5))
        sigma = depolarize(random_density(dim, int(rng.integers(1, dim + 1)), rng), rng.uniform(0.0, 0.5))
        # sigma's witness evaluated on rho's TRUE expectations.
        est = estimate_negativity_witness(_exact_expectations(rho, n), witness_weights(sigma))
        assert est <= negativity(rho) + 1e-9
        worst_slack = max(worst_slack, est - negativity(rho))
        if est > 1e-6:
            positive_bounds += 1
    # The bound is tight only when sigma matches rho's negative eigenspace.
    assert worst_slack <= 1e-9
    # Non-vacuous: mismatched witnesses still produce many strictly-positive
    # lower bounds (a degenerate all-zero witness would make this 0).
    assert positive_bounds > 0
    # Achievability: the MATCHED witness (sigma = rho) attains the bound exactly,
    # so the inequality is not passing merely because every estimate is ~0.
    rho_tight = depolarize(bell_phi_plus(), 0.2)  # negativity 0.35
    matched = estimate_negativity_witness(
        _exact_expectations(rho_tight, n), witness_weights(rho_tight)
    )
    assert matched == pytest.approx(negativity(rho_tight), abs=1e-9)


# ---------------------------------------------------------------------------
# Property 4 — efficiency benchmark (Finding B) at n=4: the witness estimator
# (true witness, uniform measurement) has meaningfully lower mean negativity
# error than full reconstruction at the same modest shot budget.
# ---------------------------------------------------------------------------
def test_efficiency_benchmark_n4() -> None:
    n, shots, batch = 4, 200, 40
    rng = np.random.default_rng(2030)  # locked seed; observed gap ~1.9x
    settings = all_local_pauli_settings(n)

    witness_errors, recon_errors = [], []
    for _ in range(batch):
        rank = int(rng.integers(1, 2 ** n + 1))
        rho = depolarize(random_density(2 ** n, rank, rng), rng.uniform(0.05, 0.35))
        true_neg = negativity(rho)

        counts = simulate_settings(rho, settings, shots=shots, rng=rng)
        measured = estimate_pauli_expectations(counts, n=n)

        # Witness estimator with the TRUE witness direction, clipped at 0.
        est_witness = max(0.0, estimate_negativity_witness(measured, witness_weights(rho)))
        # Full-reconstruction estimator at the same budget.
        est_recon = negativity(reconstruct(measured, n))

        witness_errors.append(abs(est_witness - true_neg))
        recon_errors.append(abs(est_recon - true_neg))

    witness_mean = float(np.mean(witness_errors))
    recon_mean = float(np.mean(recon_errors))

    # Strictly better, and meaningfully so.  The expected gap is roughly 2x and
    # is ~1.9x for this locked seed; the >= 1.3x floor stays below the batch-40
    # seed-to-seed fluctuation band (observed min ~1.37x) so the margin clause is
    # robust, not just satisfied by this one seed.
    assert witness_mean < recon_mean
    assert witness_mean < recon_mean / 1.3, (
        f"witness_mean={witness_mean:.4f} not meaningfully below "
        f"recon_mean={recon_mean:.4f} (ratio {recon_mean / witness_mean:.2f}x)"
    )
