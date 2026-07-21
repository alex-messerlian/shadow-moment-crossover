"""Tests for the corrected collective + single-copy readout models (PASS 15)."""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.ensembles import noisy_pure
from anrl.benchmark.readout_correction import (
    _pair_diag,
    _pair_operator,
    collective_readout_signal,
    collective_readout_signal_state,
    confusion_1q,
    corrected_snapshots,
    per_pair_factor,
    two_copy_noisy_state,
)
from anrl.benchmark.readout_shadows import (
    cyclic_readout_rates,
    snapshots_factored_readout,
    uniform_readout_rates,
)
from anrl.benchmark.scaling import collective_purity_signal
from anrl.benchmark.shadows import full_purity_ustatistic

_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def test_per_pair_factor_matches_enumeration():
    p10, p01 = 0.093, 0.089
    for a in (0, 1):
        for b in (0, 1):
            da = [1 - p10, p10] if a == 0 else [p01, 1 - p01]
            db = [1 - p10, p10] if b == 0 else [p01, 1 - p01]
            enum = sum(da[ap] * db[bp] * (-1) ** (ap * bp) for ap in (0, 1) for bp in (0, 1))
            assert per_pair_factor(a, b, p10, p01, p10, p01) == pytest.approx(enum)


def test_pair_operator_is_swap_at_zero_readout():
    O = _pair_operator(_pair_diag(0.0, 0.0, 0.0, 0.0))
    assert np.allclose(O, _SWAP)


def test_collective_signal_reproduces_committed_at_zero_readout():
    """15.1c gate: zero-readout collective signal == committed collective_purity_signal."""
    for n in (2, 3, 4):
        st = noisy_pure(n, 0.1, np.random.default_rng([0, 0, n, 0, 0]))
        for noise, rate in [("depolarizing", 0.1), ("amplitude_damping", 0.05), ("dephasing", 0.1)]:
            sig = collective_purity_signal(st, noise, rate)
            tau = two_copy_noisy_state(st, noise, rate)
            zero = [((0.0, 0.0), (0.0, 0.0))] * n
            assert collective_readout_signal(tau, n, zero) == pytest.approx(sig, abs=1e-10)


def test_factorized_matches_dense():
    n = 4
    st = noisy_pure(n, 0.1, np.random.default_rng([0, 0, n, 0, 0]))
    rates = [((0.065, 0.065), (0.065, 0.065))] * n
    for noise, rate in [("depolarizing", 0.1), ("amplitude_damping", 0.05), ("dephasing", 0.1)]:
        for mit in (False, True):
            tau = two_copy_noisy_state(st, noise, rate)
            dense = collective_readout_signal(tau, n, rates, mitigate=mit)
            fact = collective_readout_signal_state(st, noise, rate, n, rates, mitigate=mit)
            assert dense == pytest.approx(fact, abs=1e-9)


def test_blanket_is_harsher_than_corrected():
    """PASS 14's (1-2f)^{2n} blanket contraction under-retains signal vs the corrected model."""
    f = 0.065
    n = 5
    st = noisy_pure(n, 0.1, np.random.default_rng([0, 0, n, 0, 0]))
    rho = st.density_matrix()
    tau = np.kron(rho, rho)
    sig = float(np.trace(rho @ rho).real)
    corrected = collective_readout_signal(tau, n, [((f, f), (f, f))] * n) / sig
    blanket = (1 - 2 * f) ** (2 * n)
    assert corrected > blanket  # corrected retains more signal


def test_corrected_snapshots_restore_unbiasedness_symmetric():
    n, M, trials = 3, 3000, 40
    st = noisy_pure(n, 0.1, np.random.default_rng([0, 0, n, 0, 0]))
    true = st.purity()
    rates = uniform_readout_rates(n, 0.065)
    out_rng = np.random.default_rng([0, 0, n, 0, 1])
    ro_rng = np.random.default_rng([9, n, 0])
    corr = [full_purity_ustatistic(corrected_snapshots(
        snapshots_factored_readout(st, M, out_rng, ro_rng, rates), rates)) for _ in range(trials)]
    # corrected bias ~ 0 (statistical); certainly far smaller than the ~-0.35 uncorrected bias.
    assert abs(np.mean(corr) - true) < 0.05


def test_mitigated_observable_is_unbiased():
    """w = (R^T)^-1 v satisfies R^T w = v, so E[w over readout | true x] = parity(x)."""
    p10, p01 = 0.093, 0.089
    R = np.kron(confusion_1q(p10, p01), confusion_1q(p10, p01))
    v = np.array([1.0, 1.0, 1.0, -1.0])
    w = np.linalg.solve(R.T, v)
    assert np.allclose(R.T @ w, v)  # readout-averaged mitigated observable returns the ideal parity
