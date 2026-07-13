"""Tests for the Cepheus purity-experiment circuits (all local; ZERO credits).

Covered:
* destructive SWAP test recovers exact ``Tr(rho^2)`` noiselessly (pure + mixed);
* the parity/sign rule is pinned against the exact simulator;
* the single-copy shadow route recovers purity (exact snapshot unbiasedness +
  a shot-based end-to-end check);
* the transpiled circuit is verified equivalent to the logical one, and maps onto
  Cepheus's real coupling map without routing overhead at n=2.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Operator, random_statevector
from qiskit_aer import AerSimulator

from anrl.hardware import (
    CEPHEUS_BASIS_GATES,
    bell_state,
    cepheus_coupling_map,
    destructive_swap_test,
    exact_swap_purity,
    haar_pure,
    pauli_shadow_circuits,
    purity_from_counts,
    random_mixed,
    shadow_purity,
)
from anrl.hardware.shadows import _BASES, _snapshot, bits_from_bitstring
from anrl.hardware.swap_test import _swap_test_body, swap_sign


# --------------------------------------------------------------------------- #
# Destructive SWAP test — exact recovery of Tr(rho^2)                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prep", [bell_state(), haar_pure(2, 0), haar_pure(2, 7), haar_pure(3, 1)])
def test_swap_test_recovers_pure_purity_exactly(prep):
    """SWAP test on rho(x)rho returns 1.0 for pure states (exact outcome distribution)."""
    est = exact_swap_purity(prep.rho, prep.n)
    assert est == pytest.approx(1.0, abs=1e-9)
    assert est == pytest.approx(prep.purity(), abs=1e-9)


@pytest.mark.parametrize("seed", [3, 9, 21])
def test_swap_test_recovers_mixed_purity_exactly(seed):
    """SWAP test recovers Tr(rho^2) for mixed states to machine precision."""
    prep = random_mixed(2, seed)
    est = exact_swap_purity(prep.rho, prep.n)
    assert est == pytest.approx(prep.purity(), abs=1e-12)
    assert prep.purity() < 1.0  # genuinely mixed


def test_swap_test_recovers_purity_shot_based():
    """End-to-end: the actual measured circuit on Aer recovers purity within sampling error."""
    sim = AerSimulator()
    prep = bell_state()
    qc = transpile(destructive_swap_test(prep), sim, optimization_level=1)
    counts = sim.run(qc, shots=100_000, seed_simulator=1).result().get_counts()
    est = purity_from_counts(counts, prep.n)
    assert est == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------- #
# Sign rule pinned against the exact simulator                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3])
def test_sign_rule_matches_exact_simulator(n):
    """Tr(rho^2) = sum (-1)^{#pairs both-1} P(bitstring), verified vs DensityMatrix evolution."""
    rng = np.random.default_rng(100 + n)
    psi = random_statevector(2 ** n, seed=int(rng.integers(1 << 30))).data
    rho = np.outer(psi, psi.conj())
    two = np.kron(rho, rho)
    probs = DensityMatrix(two).evolve(_swap_test_body(n)).probabilities_dict()
    via_rule = sum(swap_sign(b, n) * p for b, p in probs.items())
    assert via_rule == pytest.approx(float(np.trace(rho @ rho).real), abs=1e-9)


def test_swap_sign_values():
    """The sign is -1 iff an odd number of pairs have both measured bits = 1 (little-endian)."""
    # n=1, bitstring "b1 b0" little-endian -> qubit0=a, qubit1=b
    assert swap_sign("00", 1) == 1
    assert swap_sign("01", 1) == 1  # only a=1
    assert swap_sign("10", 1) == 1  # only b=1
    assert swap_sign("11", 1) == -1  # both 1 -> one odd pair
    # n=2: qubits 0,1 = copy A; 2,3 = copy B. Pairs (0,2),(1,3).
    # reversed("0101")="1010": a0=1,a1=0,b0=1,b1=0; pair0 (a0,b0)=(1,1) both 1 -> one odd pair -> -1
    assert swap_sign("0101", 2) == -1
    # reversed("1111")="1111": both pairs (1,1) -> two odd pairs -> +1
    assert swap_sign("1111", 2) == 1
    # reversed("0001")="1000": a0=1 only -> no pair both 1 -> +1
    assert swap_sign("0001", 2) == 1


# --------------------------------------------------------------------------- #
# Single-copy Pauli shadows                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 4, 11])
def test_shadow_snapshot_unbiased(seed):
    """E_{basis,Born}[3 R^dag|b><b|R - I] = rho exactly for a single-qubit state (no sampling)."""
    psi = random_statevector(2, seed=seed).data
    rho = np.outer(psi, psi.conj())
    acc = np.zeros((2, 2), dtype=np.complex128)
    for basis in _BASES:
        # Born probabilities of the two outcomes in this measurement basis
        from anrl.hardware.shadows import _ROT_MATRIX

        r = _ROT_MATRIX[basis]
        rot = r @ rho @ r.conj().T  # state after rotation, measured in Z
        for bit in (0, 1):
            p = float(np.real(rot[bit, bit]))
            acc += (1.0 / 3.0) * p * _snapshot(basis, bit)
    assert np.allclose(acc, rho, atol=1e-12)


def test_shadow_route_recovers_purity():
    """The single-copy circuits + copy-fair U-statistic recover Tr(rho^2) within sampling error."""
    sim = AerSimulator()
    prep = haar_pure(2, 0)
    circs, bases = pauli_shadow_circuits(prep, 8_000, seed=2)
    circs = transpile(circs, sim, optimization_level=1)
    res = sim.run(circs, shots=1, seed_simulator=5).result()
    bits = np.array([bits_from_bitstring(next(iter(res.get_counts(i))), prep.n) for i in range(len(circs))])
    est = shadow_purity(bases, bits, prep.n)
    assert est == pytest.approx(prep.purity(), abs=0.08)  # ~few sigma at M=8000, n=2


def test_bits_from_bitstring_little_endian():
    """Qiskit little-endian bitstring maps position q -> qubit q."""
    assert bits_from_bitstring("00", 2) == [0, 0]
    assert bits_from_bitstring("10", 2) == [0, 1]  # qubit1=1
    assert bits_from_bitstring("01", 2) == [1, 0]  # qubit0=1


# --------------------------------------------------------------------------- #
# Transpilation to Cepheus                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prep", [bell_state(), haar_pure(2, 0)])
def test_transpiled_equivalent_to_logical(prep):
    """Native-gate decomposition (cz/rx/rz) is unitarily equivalent to the logical circuit."""
    logical = destructive_swap_test(prep).remove_final_measurements(inplace=False)
    native = transpile(logical, basis_gates=CEPHEUS_BASIS_GATES, optimization_level=3, seed_transpiler=0)
    assert set(native.count_ops()) <= set(CEPHEUS_BASIS_GATES) | {"barrier"}
    assert Operator(logical).equiv(Operator(native))


def test_transpile_onto_cepheus_no_routing_overhead():
    """At n=2 the SWAP test maps onto a real Cepheus square with no added SWAPs."""
    cm = cepheus_coupling_map()
    assert cm.size() == 108 and len(cm.get_edges()) == 193
    logical = destructive_swap_test(bell_state())
    n2q_before = sum(1 for i in logical.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
    dev = transpile(logical, coupling_map=cm, basis_gates=CEPHEUS_BASIS_GATES,
                    optimization_level=3, seed_transpiler=0)
    n2q_after = sum(1 for i in dev.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
    assert n2q_after == n2q_before == 4  # 4 CX -> 4 CZ, zero routing SWAPs
    phys = {dev.find_bit(b).index for i in dev.data for b in i.qubits if i.operation.name != "barrier"}
    assert len(phys) == 4  # exactly 4 physical qubits used
