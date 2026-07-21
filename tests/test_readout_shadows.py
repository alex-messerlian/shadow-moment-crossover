"""Tests for uncorrected readout on the single-copy shadow route (PASS 14)."""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.ensembles import noisy_pure
from anrl.benchmark.readout_shadows import (
    _READOUT_CYCLE,
    collective_parity_contraction,
    cyclic_readout_rates,
    snapshots_factored_readout,
    uniform_readout_rates,
)
from anrl.benchmark.scaling import snapshots_factored
from anrl.benchmark.shadows import full_purity_ustatistic
from anrl.hardware.readout_model import MEASURED_READOUT, qubit_rates


def test_zero_readout_reproduces_noiseless_snapshots_exactly():
    """At zero rates the readout snapshots equal the noiseless ones byte-for-byte."""
    n, M = 4, 300
    state = noisy_pure(n, 0.1, np.random.default_rng([0, 0, n, 0, 0]))
    base = snapshots_factored(state, M, np.random.default_rng([0, 0, n, 0, 1]))
    ro = snapshots_factored_readout(
        state,
        M,
        np.random.default_rng([0, 0, n, 0, 1]),  # SAME outcome stream
        np.random.default_rng([7, 7, 7]),  # separate readout stream (never used at 0)
        rates=[(0.0, 0.0)] * n,
    )
    assert np.array_equal(base, ro)
    # ... and therefore the exact same purity estimate.
    assert full_purity_ustatistic(base) == full_purity_ustatistic(ro)


def test_nonzero_readout_changes_snapshots():
    n, M = 4, 300
    state = noisy_pure(n, 0.1, np.random.default_rng([0, 0, n, 0, 0]))
    base = snapshots_factored(state, M, np.random.default_rng([0, 0, n, 0, 1]))
    ro = snapshots_factored_readout(
        state,
        M,
        np.random.default_rng([0, 0, n, 0, 1]),
        np.random.default_rng([1, 2, 3]),
        rates=uniform_readout_rates(n, 0.065),
    )
    assert not np.array_equal(base, ro)


def test_cyclic_rates_match_measured_device():
    """cyclic_readout_rates uses mean-P(1|0) and measured P(0|1), cycled over {0,1,9,10}."""
    rates = cyclic_readout_rates(6)
    for qb in range(6):
        phys = _READOUT_CYCLE[qb % 4]
        idle, excited, p01 = qubit_rates(phys)
        assert rates[qb] == pytest.approx((0.5 * (idle + excited), p01))
    # q0 must reproduce the task's stated 9.3% / 8.9%.
    idle, excited, p01 = MEASURED_READOUT[0]
    assert 0.5 * (idle + excited) == pytest.approx(0.0925, abs=1e-4)
    assert p01 == pytest.approx(0.0892, abs=1e-4)


def test_uniform_rates_symmetric():
    rates = uniform_readout_rates(5, 0.065)
    assert rates == [(0.065, 0.065)] * 5


def test_collective_contraction_matches_symmetric_formula():
    n = 3
    f = 0.065
    rates = uniform_readout_rates(2 * n, f)  # 2n measured qubits for a 2-copy test
    assert collective_parity_contraction(rates) == pytest.approx((1.0 - 2.0 * f) ** (2 * n))
    # zero readout -> no contraction.
    assert collective_parity_contraction([(0.0, 0.0)] * 4) == pytest.approx(1.0)
    # contraction is < 1 and shrinks with more qubits.
    assert collective_parity_contraction(rates) < 1.0


def test_readout_biases_the_estimator():
    """The noiseless U-statistic is ~unbiased; under readout it acquires a real bias."""
    n, M, trials = 3, 400, 60
    rng = np.random.default_rng([0, 0, n, 0, 0])
    state = noisy_pure(n, 0.1, rng)
    true = state.purity()
    out_rng = np.random.default_rng([0, 0, n, 0, 1])
    ro_rng = np.random.default_rng([5, 5, 5])
    rates = uniform_readout_rates(n, 0.065)
    ro_est = [
        full_purity_ustatistic(snapshots_factored_readout(state, M, out_rng, ro_rng, rates))
        for _ in range(trials)
    ]
    base_rng = np.random.default_rng([0, 0, n, 0, 1])
    base_est = [
        full_purity_ustatistic(snapshots_factored(state, M, base_rng)) for _ in range(trials)
    ]
    bias_ro = float(np.mean(ro_est)) - true
    bias_base = float(np.mean(base_est)) - true
    # noiseless bias is ~0 (statistical), readout bias is materially larger in magnitude.
    assert abs(bias_ro) > abs(bias_base)
    assert abs(bias_ro) > 0.01
