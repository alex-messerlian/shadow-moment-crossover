"""Collective (2-copy SWAP-test) purity estimation under gate noise.

The SWAP test on two copies of an n-qubit state measures ``Tr(rho^2)``: an ideal
run returns +/-1 with ``P(+1) = (1 + Tr(rho^2)) / 2``.  We model the entangling
gate noise as an effective *global depolarizing* channel on the two-copy system,
applied before the ideal SWAP measurement, with rate

    p_eff = 1 - (1 - p_gate) ** gate_count,   gate_count = gate_count_fn(n_qubits).

Global depolarizing replaces the state with the maximally mixed state on the
two-copy (``2 * 2^n`` qubit) system with probability ``p_eff``.  The SWAP
operator has eigenvalues +/-1 and trace ``2^n`` on two n-qubit registers, so its
expectation on the maximally mixed two-copy state is ``2^n / 2^(2n) = 1 / 2^n``.
Hence the noisy signal is

    signal = (1 - p_eff) * Tr(rho^2) + p_eff / 2^n.
"""

from __future__ import annotations

import numpy as np

from anrl.physics import purity


def gates_all_to_all(n: int) -> int:
    """Entangling-gate count for an all-to-all (dense) SWAP-test circuit."""
    return n


def gates_linear_1d(n: int) -> int:
    """Entangling-gate count for a linear / 1D-connectivity SWAP-test circuit."""
    return max(n, n * (n - 1) // 2)


def collective_signal(
    purity_value: float, p_gate: float, n_qubits: int, gate_count_fn
) -> float:
    """Exact noisy SWAP-test signal ``(1 - p_eff) * purity + p_eff / 2^n``.

    Separated from sampling so the noise model can be validated directly.
    """
    gate_count = gate_count_fn(n_qubits)
    p_eff = 1.0 - (1.0 - p_gate) ** gate_count
    return (1.0 - p_eff) * purity_value + p_eff / (2 ** n_qubits)


def collective_purity_estimate(
    rho: np.ndarray,
    n_measurements: int,
    p_gate: float,
    n_qubits: int,
    gate_count_fn,
    rng: np.random.Generator,
) -> float:
    """Noisy 2-copy SWAP-test purity estimate from ``n_measurements`` shots.

    Each shot is a binary SWAP-test outcome (+/-1) with ``P(+1) = (1+signal)/2``;
    the estimate is ``2 * (fraction of +1) - 1``, which is unbiased for the noisy
    ``signal`` (and thus for ``Tr(rho^2)`` exactly when ``p_gate = 0``).
    """
    if n_measurements < 1:
        raise ValueError(f"n_measurements must be >= 1, got {n_measurements}")
    signal = collective_signal(purity(rho), p_gate, n_qubits, gate_count_fn)
    p_plus = float(np.clip((1.0 + signal) / 2.0, 0.0, 1.0))
    n_plus = int(rng.binomial(n_measurements, p_plus))
    fraction_plus = n_plus / n_measurements
    return 2.0 * fraction_plus - 1.0
