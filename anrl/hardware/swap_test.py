"""The destructive (Bell-basis) SWAP test — the collective route.

Two independent copies of an ``n``-qubit state sit on qubits ``0..n-1`` (copy A)
and ``n..2n-1`` (copy B).  For each pair ``(i, i+n)`` we apply ``CNOT(i -> i+n)``
then ``H(i)`` and measure all ``2n`` qubits.  This uses exactly ``n`` two-qubit
gates (shallow, NISQ-friendly) — unlike the ancilla Fredkin SWAP test.

Sign rule (VERIFIED against the exact simulator to ~1e-16, pure and mixed):

    Tr(rho^2) = sum_bitstring  (-1)^{#pairs (a_i, b_i) with a_i = b_i = 1}  P(bitstring)

i.e. each pair contributes a sign ``-1`` iff BOTH its measured bits are 1.  The
outcome bit for qubit ``q`` is read from the (little-endian) Qiskit bitstring at
position ``q``.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix

from .state_prep import PreparedState


def _swap_test_body(n: int) -> QuantumCircuit:
    """The measurement-basis-change body: ``CNOT(i, i+n)`` then ``H(i)`` per pair."""
    qc = QuantumCircuit(2 * n, name="swap_body")
    for i in range(n):
        qc.cx(i, i + n)  # control = copy-A qubit i, target = copy-B qubit i+n
        qc.h(i)
    return qc


def destructive_swap_test(prep: PreparedState, prep_b: PreparedState | None = None) -> QuantumCircuit:
    """Full ``2n``-qubit destructive SWAP-test circuit (two copies + basis change + measure).

    ``prep`` must carry a prep circuit (pure states).  Copy B defaults to a second
    independent copy of the same state.
    """
    if prep.circuit is None:
        raise ValueError(f"state {prep.label!r} has no prep circuit; use exact_swap_purity for mixed states")
    prep_b = prep_b or prep
    if prep_b.circuit is None:
        raise ValueError("copy B has no prep circuit")
    n = prep.n
    qc = QuantumCircuit(2 * n, 2 * n, name=f"swap_test_{prep.label}")
    qc.compose(prep.circuit, qubits=range(n), inplace=True)
    qc.compose(prep_b.circuit, qubits=range(n, 2 * n), inplace=True)
    qc.barrier()
    qc.compose(_swap_test_body(n), qubits=range(2 * n), inplace=True)
    qc.measure(range(2 * n), range(2 * n))
    return qc


def swap_sign(bitstring: str, n: int) -> int:
    """``(-1)^{#pairs (a_i, b_i) with both bits = 1}`` for a Qiskit (little-endian) bitstring."""
    bits = bitstring[::-1]  # index q -> qubit q
    both_one = sum(1 for i in range(n) if bits[i] == "1" and bits[i + n] == "1")
    return -1 if (both_one % 2) else 1


def purity_from_counts(counts: dict, n: int) -> float:
    """Estimate ``Tr(rho^2)`` from measured SWAP-test counts (the sign-rule sum)."""
    shots = sum(counts.values())
    return float(sum(swap_sign(b.replace(" ", ""), n) * c for b, c in counts.items()) / shots)


def exact_swap_purity(rho: np.ndarray, n: int) -> float:
    """Exact SWAP-test estimator on ``rho (x) rho`` (circuit-free; validates the sign rule).

    Evolves the two-copy state through the SWAP-test body and applies the sign rule
    to the exact outcome distribution; equals ``Tr(rho^2)`` for any ``rho``.
    """
    two = np.kron(rho, rho)
    final = DensityMatrix(two).evolve(_swap_test_body(n))
    probs = final.probabilities_dict()
    return float(sum(swap_sign(b, n) * p for b, p in probs.items()))
