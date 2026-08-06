"""Tests for single-copy local-Pauli measurement and reconstruction.

Group 7 (single-copy tomography recovers the true negativity at high shots) is
the pinned requirement; the remaining tests are sanity checks on the measurement
primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.physics import (
    all_local_pauli_settings,
    bell_phi_plus,
    depolarize,
    estimate_pauli_expectations,
    ghz,
    negativity,
    outcome_probabilities,
    pauli_string,
    reconstruct,
    simulate_settings,
    single_qubit_rotation,
)


# ---------------------------------------------------------------------------
# Group 7, single-copy tomography recovers the true negativity within 0.02
# ---------------------------------------------------------------------------
def test_single_copy_tomography_recovers_negativity() -> None:
    rng = np.random.default_rng(2026_07_11)
    # A noisy but still-entangled 2-qubit state (true negativity 0.5 - 0.75*0.2).
    rho = depolarize(bell_phi_plus(), 0.2)
    true_neg = negativity(rho)
    assert true_neg == pytest.approx(0.35, abs=1e-9)

    settings = all_local_pauli_settings(2)  # 9 settings
    counts = simulate_settings(rho, settings, shots=20000, rng=rng)
    expectations = estimate_pauli_expectations(counts, n=2)
    rho_hat = reconstruct(expectations, n=2)

    assert negativity(rho_hat) == pytest.approx(true_neg, abs=0.02)


# ---------------------------------------------------------------------------
# Sanity checks on the measurement primitives
# ---------------------------------------------------------------------------
def test_single_qubit_rotations_are_unitary() -> None:
    for pauli in ("X", "Y", "Z"):
        u = single_qubit_rotation(pauli)
        assert np.allclose(u @ u.conj().T, np.eye(2), atol=1e-12)


def test_outcome_probabilities_normalized_and_nonnegative() -> None:
    rho = ghz(2)
    for setting in all_local_pauli_settings(2):
        probs = outcome_probabilities(rho, setting)
        assert probs.shape == (4,)
        assert np.all(probs >= -1e-15)
        assert probs.sum() == pytest.approx(1.0, abs=1e-12)


def test_estimated_expectations_match_exact_at_high_shots() -> None:
    # With many shots the shot-weighted Pauli estimates should track tr(rho P).
    rng = np.random.default_rng(4242)
    rho = depolarize(bell_phi_plus(), 0.3)
    counts = simulate_settings(rho, all_local_pauli_settings(2), shots=40000, rng=rng)
    expectations = estimate_pauli_expectations(counts, n=2)

    # Identity term is exactly 1 by construction.
    assert expectations[("I", "I")] == pytest.approx(1.0, abs=1e-12)

    # Check a few representative terms against the exact tr(rho P).
    for term in [("Z", "Z"), ("X", "X"), ("Y", "Y"), ("Z", "I"), ("I", "X")]:
        exact = float(np.trace(rho @ pauli_string(term)).real)
        assert expectations[term] == pytest.approx(exact, abs=0.02)


def test_reconstruct_returns_valid_density_matrix() -> None:
    rng = np.random.default_rng(999)
    rho = depolarize(bell_phi_plus(), 0.25)
    counts = simulate_settings(rho, all_local_pauli_settings(2), shots=20000, rng=rng)
    rho_hat = reconstruct(estimate_pauli_expectations(counts, n=2), n=2)

    # Hermitian, unit trace, positive semidefinite.
    assert np.allclose(rho_hat, rho_hat.conj().T, atol=1e-10)
    assert np.trace(rho_hat).real == pytest.approx(1.0, abs=1e-9)
    eigvals = np.linalg.eigvalsh(rho_hat)
    assert eigvals.min() >= -1e-9
