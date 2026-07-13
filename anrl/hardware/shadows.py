"""The single-copy route — local classical shadows as hardware circuits.

For each shot we pick a random single-qubit Pauli basis per qubit (X, Y, or Z),
apply the corresponding pre-measurement rotation, and measure.  The local shadow
snapshot for a qubit measured in basis ``P`` with outcome ``b`` is the standard
``3 R^dag |b><b| R - I`` (``R`` = the rotation applied before measurement), whose
per-qubit expectation inverts the measurement channel so ``E[snapshot] = rho``.
Purity is then the EXACT copy-fair U-statistic already in ``anrl`` — the same
estimator used throughout the study — so the two routes are compared like-for-like
at the same shot budget.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from anrl.benchmark.shadows import _I2, _KET_BRA, full_purity_ustatistic
from .state_prep import PreparedState

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
# Pre-measurement rotation R (as a matrix) for each Pauli basis: measuring in the
# P eigenbasis == applying R then measuring in Z.
_ROT_MATRIX = {"Z": _I2, "X": _H, "Y": _H @ _SDG}
_BASES = ("X", "Y", "Z")


def _apply_basis(qc: QuantumCircuit, q: int, basis: str) -> None:
    if basis == "X":
        qc.h(q)
    elif basis == "Y":
        qc.sdg(q)
        qc.h(q)
    # Z: no rotation


def pauli_shadow_circuits(prep: PreparedState, n_shots: int, seed: int) -> tuple[list[QuantumCircuit], np.ndarray]:
    """``n_shots`` shadow circuits (prep + random per-qubit Pauli basis + measure).

    Returns the circuits and the ``(n_shots, n)`` array of basis choices (needed to
    build the snapshots from the measured bitstrings).
    """
    if prep.circuit is None:
        raise ValueError(f"state {prep.label!r} has no prep circuit")
    n = prep.n
    rng = np.random.default_rng(seed)
    bases = rng.integers(0, 3, size=(n_shots, n))  # 0=X, 1=Y, 2=Z
    circuits = []
    for s in range(n_shots):
        qc = QuantumCircuit(n, n, name=f"shadow_{prep.label}_{s}")
        qc.compose(prep.circuit, qubits=range(n), inplace=True)
        qc.barrier()
        for q in range(n):
            _apply_basis(qc, q, _BASES[bases[s, q]])
        qc.measure(range(n), range(n))
        circuits.append(qc)
    return circuits, bases


def _snapshot(basis: str, bit: int) -> np.ndarray:
    """Local shadow ``3 R^dag |b><b| R - I`` for one qubit."""
    r = _ROT_MATRIX[basis]
    return 3.0 * (r.conj().T @ _KET_BRA[bit] @ r) - _I2


def snapshots_from_outcomes(bases: np.ndarray, bits: np.ndarray, n: int) -> np.ndarray:
    """``(M, n, 2, 2)`` shadow snapshots from basis choices and measured bits ``(M, n)``."""
    m = bases.shape[0]
    snaps = np.empty((m, n, 2, 2), dtype=np.complex128)
    for s in range(m):
        for q in range(n):
            snaps[s, q] = _snapshot(_BASES[bases[s, q]], int(bits[s, q]))
    return snaps


def shadow_purity(bases: np.ndarray, bits: np.ndarray, n: int) -> float:
    """Copy-fair single-copy purity estimate ``Tr(rho^2)`` (exact full U-statistic)."""
    return full_purity_ustatistic(snapshots_from_outcomes(bases, bits, n))


def bits_from_bitstring(bitstring: str, n: int) -> list[int]:
    """Qiskit (little-endian) bitstring -> per-qubit bits ``[b_0, ..., b_{n-1}]``."""
    b = bitstring.replace(" ", "")[::-1]
    return [int(b[q]) for q in range(n)]
