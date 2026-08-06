"""State preparation for the hardware purity experiment.

Each prepared state records its EXACT classical density matrix, so the true
purity ``Tr(rho^2)`` is always known and the two measurement routes can be scored
against ground truth.  Pure states carry a Qiskit prep circuit (runnable on
hardware); mixed states carry only the density matrix (used for exact,
circuit-free validation of the estimators via ``rho (x) rho``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_statevector, random_unitary

from anrl.physics import random_density


@dataclass(frozen=True)
class PreparedState:
    """A prepared ``n``-qubit state: exact ``rho`` (always) and an optional circuit."""

    n: int
    rho: np.ndarray  # 2^n x 2^n exact density matrix
    label: str
    circuit: QuantumCircuit | None  # n-qubit prep on |0...0>; None for mixed states

    def purity(self) -> float:
        return float(np.trace(self.rho @ self.rho).real)


def bell_state() -> PreparedState:
    """The 2-qubit Bell state ``|Phi+> = (|00> + |11>)/sqrt2`` (purity exactly 1.0)."""
    qc = QuantumCircuit(2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = psi[3] = 1.0 / np.sqrt(2.0)
    return PreparedState(2, np.outer(psi, psi.conj()), "bell", qc)


def ghz_state(n: int) -> PreparedState:
    """The ``n``-qubit GHZ state ``(|0...0> + |1...1>)/sqrt2`` (purity 1.0).

    Prepared by ``H(0)`` then a CNOT chain ``0->1->...->(n-1)`` (linear connectivity),
    which is NISQ-friendly and maps onto a qubit path with little routing.
    """
    from anrl.physics.states import ghz

    qc = QuantumCircuit(n, name=f"ghz_n{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return PreparedState(n, ghz(n), f"ghz_n{n}", qc)


def haar_pure(n: int, seed: int) -> PreparedState:
    """A Haar-random ``n``-qubit pure state (seeded, reproducible), purity 1.0.

    The state vector is prepared with Qiskit's ``prepare_state`` so it runs on
    hardware; the exact ``rho = |psi><psi|`` is recorded classically.
    """
    psi = random_statevector(2 ** n, seed=seed).data
    qc = QuantumCircuit(n, name=f"haar_n{n}_s{seed}")
    qc.prepare_state(psi, range(n))
    return PreparedState(n, np.outer(psi, psi.conj()), f"haar_n{n}_s{seed}", qc)


def random_mixed(n: int, seed: int, rank: int | None = None) -> PreparedState:
    """A random ``n``-qubit mixed state ``G G^dag`` (purity < 1), density matrix only.

    Used to validate the estimators on mixed inputs via ``rho (x) rho`` (no physical
    prep circuit is needed for that check).
    """
    dim = 2 ** n
    rho = random_density(dim, rank if rank is not None else dim, np.random.default_rng(seed))
    return PreparedState(n, rho, f"mixed_n{n}_s{seed}", None)


@dataclass(frozen=True)
class MixedEnsemble:
    """A mixed state realized as a per-shot classical ensemble of pure prep circuits.

    ``rho = sum_i w_i |v_i><v_i|`` with the ``|v_i>`` an orthonormal eigenbasis.
    On hardware: for each shot, sample component ``i`` with probability ``w_i`` and
    run its pure prep circuit; NO mid-circuit measurement, no ancilla.  For the
    two-copy SWAP test the two copies sample independently, so the measured
    observable averages to ``Tr(rho^2)``.  ``rho`` is the exact recorded matrix.
    """

    n: int
    rho: np.ndarray
    label: str
    components: tuple[tuple[float, PreparedState], ...]  # (weight, pure component)

    def purity(self) -> float:
        return float(np.trace(self.rho @ self.rho).real)


def mixed_ensemble(n: int, target_purity: float, seed: int, rank: int = 2) -> MixedEnsemble:
    """A rank-``rank`` mixed state with a prescribed purity, as a hardware ensemble.

    The eigenbasis is Haar-random (seeded, reproducible); the spectrum is chosen to
    hit ``target_purity`` exactly.  For ``rank = 2`` the two weights are
    ``(a, 1-a)`` with ``a = (1 + sqrt(2P - 1)) / 2`` (requires ``P >= 0.5``); each
    eigenvector is a Haar-random pure state prepared with ``prepare_state``.
    """
    if rank != 2:
        raise NotImplementedError("only rank-2 ensembles are supported (covers P in [0.5, 1))")
    if not (0.5 <= target_purity < 1.0):
        raise ValueError(f"rank-2 purity must be in [0.5, 1), got {target_purity}")
    a = 0.5 * (1.0 + np.sqrt(2.0 * target_purity - 1.0))
    weights = (a, 1.0 - a)

    dim = 2 ** n
    u = random_unitary(dim, seed=seed).data  # Haar-random; columns are orthonormal
    components = []
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for idx, w in enumerate(weights):
        v = u[:, idx]
        rho += w * np.outer(v, v.conj())
        qc = QuantumCircuit(n, name=f"mix{idx}")
        qc.prepare_state(v, range(n))
        label = f"mixed_ens_n{n}_P{target_purity:.2f}_s{seed}_c{idx}"
        components.append((float(w), PreparedState(n, np.outer(v, v.conj()), label, qc)))
    return MixedEnsemble(
        n, rho, f"mixed_ens_n{n}_P{target_purity:.2f}_s{seed}", tuple(components)
    )
