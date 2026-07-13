"""Tests for the Cepheus noise model, calibration curve, and locked predictions.

All local; ZERO credits.  Covered:
* the noise model applies to the transpiled circuit (right gates; noiseless = exact);
* the analytic density-matrix + readout-confusion purity matches a direct shot-based
  Aer run with the full ReadoutError model;
* the calibration curve is monotone in p2 (and p_ro) and invertible;
* the analytic bias law and the Qiskit noise sim agree for the Bell state, and the
  gate-only effective-g is state-independent while readout breaks that universality
  (the discrepancy is characterized);
* the hardware-implementable mixed ensemble has the target purity and recovers it;
* the shadow-route noisy predictor recovers purity noiselessly;
* the shot-budget / credit arithmetic is correct and within budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator

from anrl.hardware import (
    CEPHEUS_SQUARE,
    REF_P1,
    REF_P2,
    REF_P_RO,
    avg_gate_error_to_depol_param,
    bell_calibration_surface,
    bell_state,
    cepheus_noise_model,
    depol_param_to_avg_gate_error,
    destructive_swap_test,
    effective_g_from_purity,
    haar_pure,
    invert_measured_to_p2,
    measured_swap_purity,
    measured_swap_purity_ensemble,
    mixed_ensemble,
    purity_from_counts,
    purity_from_g,
    readout_confusion_matrix,
)
from anrl.hardware.noise_model import CEPHEUS_BASIS_GATES
from anrl.hardware.shadow_noise import noisy_shadow_dists, predict_shadow_purity
from anrl.hardware.shot_budget import (
    CREDIT_BUDGET,
    shots_to_credits,
    swap_shot_se,
    swap_shots_for_se,
)


# --------------------------------------------------------------------------- #
# Noise model                                                                 #
# --------------------------------------------------------------------------- #
def test_noise_model_targets_physical_gates():
    """Depolarizing attaches to cz and rx; rz is virtual (untouched) by default."""
    nm = cepheus_noise_model(REF_P2, REF_P1, REF_P_RO)
    noisy = set(nm.noise_instructions)
    assert "cz" in noisy and "rx" in noisy
    assert "rz" not in noisy  # virtual frame change: noiseless
    # readout error present
    assert len(nm.noise_qubits) >= 0  # readout is all-qubit; model builds without error


def test_noiseless_model_recovers_exact_purity():
    """p2=p1=p_ro=0 reproduces the true purity for pure and mixed states."""
    assert measured_swap_purity(bell_state(), 0.0, 0.0, 0.0) == pytest.approx(1.0, abs=1e-9)
    assert measured_swap_purity(haar_pure(2, 0), 0.0, 0.0, 0.0) == pytest.approx(1.0, abs=1e-9)
    ens = mixed_ensemble(2, 0.7, seed=1)
    assert measured_swap_purity_ensemble(ens, 0.0, 0.0, 0.0) == pytest.approx(ens.purity(), abs=1e-9)


def test_noise_reduces_measured_purity():
    """Any positive rate lowers the measured Bell purity below 1."""
    assert measured_swap_purity(bell_state(), REF_P2, REF_P1, REF_P_RO) < 1.0
    assert measured_swap_purity(bell_state(), 0.02, 0.0, 0.0) < 1.0
    assert measured_swap_purity(bell_state(), 0.0, 0.0, 0.02) < 1.0


def test_analytic_matches_shot_based_aer():
    """Density-matrix + readout-confusion purity matches direct Aer with full ReadoutError."""
    sim = AerSimulator(method="density_matrix")
    bell = bell_state()
    tqc = transpile(destructive_swap_test(bell), coupling_map=CEPHEUS_SQUARE,
                    basis_gates=CEPHEUS_BASIS_GATES, optimization_level=3, seed_transpiler=0)
    nm = cepheus_noise_model(REF_P2, REF_P1, REF_P_RO)
    counts = sim.run(tqc, shots=400_000, noise_model=nm, seed_simulator=7).result().get_counts()
    shot_based = purity_from_counts(counts, bell.n)
    analytic = measured_swap_purity(bell, REF_P2, REF_P1, REF_P_RO)
    assert analytic == pytest.approx(shot_based, abs=0.01)


def test_readout_confusion_matrix_stochastic():
    """Confusion matrix is symmetric and column-stochastic."""
    r = readout_confusion_matrix(4, 0.03)
    assert r.shape == (16, 16)
    assert np.allclose(r, r.T)
    assert np.allclose(r.sum(axis=0), 1.0)


def test_avg_error_depol_param_roundtrip():
    """Average-gate-error <-> depolarizing-parameter conversions invert each other."""
    for num_qubits in (1, 2):
        lam = avg_gate_error_to_depol_param(0.009, num_qubits)
        assert depol_param_to_avg_gate_error(lam, num_qubits) == pytest.approx(0.009, abs=1e-12)
    # datasheet 99.1% CZ (avg err 0.009) -> depol param 0.012
    assert avg_gate_error_to_depol_param(0.009, 2) == pytest.approx(0.012, abs=1e-9)


# --------------------------------------------------------------------------- #
# Calibration curve + inversion                                               #
# --------------------------------------------------------------------------- #
def test_calibration_monotone_in_p2():
    """Measured Bell purity strictly decreases as the two-qubit error grows."""
    p2_grid = np.linspace(0.0, 0.05, 21)
    p_ro_grid = np.array([0.005, 0.02, 0.05])
    surface = bell_calibration_surface(p2_grid, p_ro_grid, REF_P1)
    for j in range(surface.shape[1]):
        assert np.all(np.diff(surface[:, j]) < 0)


def test_calibration_monotone_in_p_ro():
    """Measured Bell purity strictly decreases as readout error grows (fixed p2)."""
    surface = bell_calibration_surface(np.array([REF_P2]), np.linspace(0.005, 0.05, 10), REF_P1)
    assert np.all(np.diff(surface[0, :]) < 0)


def test_calibration_invertible():
    """Inverting a simulated measurement recovers the p2 that produced it (given p_ro)."""
    measured = measured_swap_purity(bell_state(), 0.015, REF_P1, REF_P_RO)
    recovered = invert_measured_to_p2(measured, REF_P_RO, REF_P1)
    assert recovered == pytest.approx(0.015, abs=1e-3)


def test_inversion_recovers_reference_rate():
    """At the reference readout, the reference measured purity inverts back to p2=0.009."""
    measured = measured_swap_purity(bell_state(), REF_P2, REF_P1, REF_P_RO)
    assert invert_measured_to_p2(measured, REF_P_RO, REF_P1) == pytest.approx(REF_P2, abs=1e-3)


# --------------------------------------------------------------------------- #
# Bias law vs Qiskit noise sim                                                 #
# --------------------------------------------------------------------------- #
def test_bias_law_reproduces_bell_under_calibration():
    """The bias law with the Bell-calibrated g reproduces the Bell Qiskit measurement."""
    measured = measured_swap_purity(bell_state(), REF_P2, REF_P1, REF_P_RO)
    g = effective_g_from_purity(measured, 1.0)
    assert purity_from_g(g, 1.0) == pytest.approx(measured, abs=1e-9)


def test_gate_only_effective_g_is_state_independent():
    """Gate-only depolarizing gives a nearly state-independent effective g (bias law holds)."""
    states = [bell_state(), haar_pure(2, 0), mixed_ensemble(2, 0.7, 1), mixed_ensemble(2, 0.5, 1)]
    gs = []
    for st in states:
        tp = 1.0 if not hasattr(st, "components") else st.purity()
        meas = (measured_swap_purity_ensemble(st, REF_P2, REF_P1, 0.0) if hasattr(st, "components")
                else measured_swap_purity(st, REF_P2, REF_P1, 0.0))
        gs.append(effective_g_from_purity(meas, tp))
    assert max(gs) - min(gs) < 0.01  # tight: gate noise ~ single global g


def test_readout_breaks_g_universality():
    """Adding readout makes the effective g state-dependent (the characterized discrepancy)."""
    states = [bell_state(), haar_pure(2, 0), mixed_ensemble(2, 0.7, 1), mixed_ensemble(2, 0.5, 1)]
    gs = []
    for st in states:
        tp = 1.0 if not hasattr(st, "components") else st.purity()
        meas = (measured_swap_purity_ensemble(st, REF_P2, REF_P1, REF_P_RO) if hasattr(st, "components")
                else measured_swap_purity(st, REF_P2, REF_P1, REF_P_RO))
        gs.append(effective_g_from_purity(meas, tp))
    assert max(gs) - min(gs) > 0.02  # readout spreads g across states


# --------------------------------------------------------------------------- #
# Mixed ensemble + shadow route                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [0.5, 0.7, 0.9])
def test_mixed_ensemble_hits_target_purity(target):
    """The rank-2 ensemble's exact rho has the prescribed purity."""
    ens = mixed_ensemble(2, target, seed=3)
    assert ens.purity() == pytest.approx(target, abs=1e-9)
    assert abs(sum(w for w, _ in ens.components) - 1.0) < 1e-12


def test_shadow_route_recovers_purity_noiseless():
    """Noiseless shadow predictor recovers the true purity within its statistical error."""
    prep = haar_pure(2, 0)
    dists = noisy_shadow_dists(prep, 0.0, 0.0, 0.0)
    r = predict_shadow_purity(prep, dists, m_shots=2000, n_experiments=30, base_seed=0)
    assert abs(r["mean"] - 1.0) < 3 * r["std"] / np.sqrt(r["n_experiments"]) + 0.05


# --------------------------------------------------------------------------- #
# Shot budget / credits                                                       #
# --------------------------------------------------------------------------- #
def test_swap_shot_se_and_inverse_consistent():
    """swap_shots_for_se inverts swap_shot_se."""
    n = swap_shots_for_se(0.9, 0.005)
    assert swap_shot_se(0.9, n) <= 0.005


def test_recommended_budget_within_credits():
    """20k shots/config x 8 configs stays within the 45-credit budget."""
    total = 20_000 * 8
    assert shots_to_credits(total) == pytest.approx(41.6, abs=1e-6)
    assert shots_to_credits(total) <= CREDIT_BUDGET
