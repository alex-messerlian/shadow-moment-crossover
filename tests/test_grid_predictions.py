"""Tests for the measured-parameter grid-prediction machinery (all local; ZERO credits)."""

from __future__ import annotations

import numpy as np
import pytest

from anrl.hardware import avg_gate_error_to_depol_param, bell_state, swap_sign
from anrl.hardware.calibration import gate_noisy_probs
from anrl.hardware.grid_predict import predict_swap, transpile_swap
from anrl.hardware.readout_model import correlated_confusion, qubit_rates
from anrl.hardware.state_prep import ghz_state, haar_pure


# --------------------------------------------------------------------------- #
# Correlated readout model                                                    #
# --------------------------------------------------------------------------- #
def test_confusion_column_stochastic():
    """Each column of the joint confusion sums to 1 (a true->measured distribution)."""
    for corr in (True, False):
        R = correlated_confusion([0, 1, 9, 10], correlated=corr)
        assert R.shape == (16, 16)
        assert np.allclose(R.sum(axis=0), 1.0)
        assert np.all(R >= 0)


def test_correlation_raises_p10_with_excited_neighbors():
    """$0's false-1 rate grows with the number of excited other qubits (measured crosstalk)."""
    R = correlated_confusion([0, 1, 9, 10], correlated=True)
    # true = 0000 (all idle) -> $0 (clbit0) flips to 1 with p10_idle; true with 2 others excited -> higher.
    # column index = true bitstring (clbit0 LSB). P($0 reads 1 | true) summed over other bits.
    def p0_flip(true_idx):
        return sum(R[m, true_idx] for m in range(16) if (m & 1) == 1)
    idle = p0_flip(0b0000)          # no other excited
    two_excited = p0_flip(0b1100)   # clbits 2,3 excited ($9,$10), $0 still 0
    assert idle < two_excited
    assert idle == pytest.approx(0.0165, abs=1e-3)         # measured idle
    assert two_excited == pytest.approx(0.1685, abs=1e-3)  # measured excited (w=2)


def test_uncharacterized_qubit_uses_mean():
    idle, excited, p01 = qubit_rates(50)  # not in {0,1,9,10}
    assert idle == excited                # no correlation assumed
    assert 0.04 < idle < 0.06 and 0.07 < p01 < 0.09


# --------------------------------------------------------------------------- #
# GHZ prep + transpilation                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 4])
def test_ghz_prep_pure(n):
    g = ghz_state(n)
    assert float(np.trace(g.rho @ g.rho).real) == pytest.approx(1.0, abs=1e-9)
    assert g.circuit is not None


@pytest.mark.parametrize("n,expected_cz", [(2, 4), (3, 7), (4, 10)])
def test_ghz_swap_zero_routing(n, expected_cz):
    """GHZ SWAP test maps onto a ladder with no routing: CZ = 3n-2."""
    _, cz_dev, _, routing, _ = transpile_swap(ghz_state(n))
    assert cz_dev == expected_cz == 3 * n - 2
    assert routing == 0


def test_haar_swap_routes_at_n3_n4():
    """Dense Haar prep needs routing SWAPs at n=3,4 (unlike GHZ)."""
    for n in (3, 4):
        _, cz_dev, cz_free, routing, _ = transpile_swap(haar_pure(n, 0))
        assert routing > 0
        assert cz_dev > cz_free


# --------------------------------------------------------------------------- #
# Prediction consistency                                                      #
# --------------------------------------------------------------------------- #
def test_ghz2_reproduces_bell_reconciliation():
    """GHZ(2) SWAP prediction == the Step-1 Bell reconciliation (~0.7163 at spec CZ)."""
    p2 = avg_gate_error_to_depol_param(0.009, 2)
    r = predict_swap(ghz_state(2), p2, 0.001)
    assert r["measured_purity"] == pytest.approx(0.7163, abs=0.003)
    # penalty decomposition adds up: gate + readout = 1 - measured
    assert r["gate_penalty"] + r["readout_penalty"] == pytest.approx(1.0 - r["measured_purity"], abs=1e-9)


def test_correlated_closes_residual_vs_independent():
    """Correlated readout reproduces measured Bell 0.7184 far better than the independent model."""
    signs = np.array([swap_sign(format(b, "04b"), 2) for b in range(16)])
    q = gate_noisy_probs(bell_state(), avg_gate_error_to_depol_param(0.009, 2), 0.001)
    corr = float(signs @ (correlated_confusion([0, 1, 9, 10], True) @ q))
    ind = float(signs @ (correlated_confusion([0, 1, 9, 10], False) @ q))
    assert abs(0.7184 - corr) < 0.01          # correlated closes it
    assert abs(0.7184 - ind) > 0.02           # independent leaves a residual
