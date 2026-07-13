"""Pauli (randomized-compiling) twirl of the destructive SWAP circuit.

Randomized compiling converts coherent gate error into stochastic (Pauli) error.
For each CZ we insert a random 2-qubit Pauli ``T`` immediately BEFORE it and the
unique correcting Pauli ``C = CZ·T·CZ`` immediately AFTER, so

    C · CZ · T = CZ   (exactly)

because ``CZ·CZ = I`` and ``T·T = I``.  Each CZ's twirl is therefore locally
self-correcting: the ideal circuit unitary is unchanged, so the SWAP-test
outcome distribution is bit-for-bit identical to the untwirled circuit and NO
measurement-frame correction is needed.  Meanwhile the coherent error channel on
each CZ is independently Pauli-twirled toward a stochastic channel — the whole
point of the experiment.

Everything here is verified numerically (``verify_twirl``): the correcting Pauli,
the native gate decompositions, and the full-circuit logical equivalence.
Zero credits — pure construction + local simulation.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

from anrl.hardware import CEPHEUS_BASIS_GATES, cepheus_coupling_map, destructive_swap_test
from anrl.hardware.state_prep import ghz_state
from anrl.hardware.swap_test import swap_sign

# clbit c -> physical qubit; copy A on {0..n-1}, copy B on the {9,10,11,12} ladder
LADDER = {2: [0, 1, 9, 10], 3: [0, 1, 2, 9, 10, 11], 4: [0, 1, 2, 3, 9, 10, 11, 12]}

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PMAT = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
_CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

# native rz/rx decomposition (up to global phase) of each single-qubit Pauli,
# given in CIRCUIT order (first gate applied first). VERIFIED in _check_natives().
_PNATIVE = {
    "I": [],
    "X": [("rx", np.pi)],
    "Z": [("rz", np.pi)],
    "Y": [("rx", np.pi), ("rz", np.pi)],  # operator rz(pi)@rx(pi) = -iY
}
_PAULIS = ("I", "X", "Y", "Z")


def _rx(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def _native_op(gates: list) -> np.ndarray:
    """Single-qubit operator from a circuit-order native gate list (first applied first)."""
    op = _I
    for name, ang in gates:
        g = _rx(ang) if name == "rx" else _rz(ang)
        op = g @ op  # later gates multiply on the left
    return op


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff a == e^{i phi} b for some global phase."""
    w = b.conj().T @ a
    c = w[0, 0]
    if abs(c) < tol:
        # pick any nonzero diagonal to fix the phase
        idx = int(np.argmax(np.abs(np.diag(w))))
        c = w[idx, idx]
    return np.allclose(w, c * np.eye(w.shape[0]), atol=tol) and abs(abs(c) - 1.0) < 1e-6


def _check_natives() -> None:
    for p in _PAULIS:
        assert _equal_up_to_phase(_native_op(_PNATIVE[p]), _PMAT[p]), f"native decomp wrong for {p}"


def correcting_pauli(ta: str, tb: str) -> tuple[str, str]:
    """Return (ca, cb) with (Ca⊗Cb) = CZ·(Ta⊗Tb)·CZ (a product Pauli, up to phase)."""
    cmat = _CZ @ np.kron(_PMAT[ta], _PMAT[tb]) @ _CZ
    for ca in _PAULIS:
        for cb in _PAULIS:
            if _equal_up_to_phase(cmat, np.kron(_PMAT[ca], _PMAT[cb])):
                return ca, cb
    raise RuntimeError(f"no product Pauli correction for CZ·({ta}{tb})·CZ")


def _verify_pair(ta: str, tb: str) -> None:
    """Assert C·CZ·T = CZ (up to global phase) using the native-gate operators."""
    ca, cb = correcting_pauli(ta, tb)
    t_op = np.kron(_native_op(_PNATIVE[ta]), _native_op(_PNATIVE[tb]))
    c_op = np.kron(_native_op(_PNATIVE[ca]), _native_op(_PNATIVE[cb]))
    assert _equal_up_to_phase(c_op @ _CZ @ t_op, _CZ), f"twirl of ({ta}{tb}) does not preserve CZ"


def verify_pauli_algebra() -> None:
    """One-time check of the native decompositions and every 2-qubit twirl correction."""
    _check_natives()
    for ta in _PAULIS:
        for tb in _PAULIS:
            _verify_pair(ta, tb)


# --------------------------------------------------------------------------- #
# Native instruction lists (physical-qubit addressing)                        #
# --------------------------------------------------------------------------- #
# An instruction is (name, params_tuple, phys_qubits_tuple, clbit_or_None).

_CM = cepheus_coupling_map()


def untwirled_instrs(n: int) -> list:
    """Native (rz/rx/cz) instruction list for the transpiled SWAP circuit on the ladder."""
    logical = destructive_swap_test(ghz_state(n))
    tqc = transpile(logical, coupling_map=_CM, basis_gates=CEPHEUS_BASIS_GATES,
                    initial_layout=LADDER[n], optimization_level=3, seed_transpiler=0)
    instrs = []
    for inst in tqc.data:
        name = inst.operation.name
        qs = tuple(tqc.find_bit(b).index for b in inst.qubits)
        if name in ("rz", "rx"):
            instrs.append((name, (float(inst.operation.params[0]),), qs, None))
        elif name == "cz":
            instrs.append(("cz", (), qs, None))
        elif name == "barrier":
            instrs.append(("barrier", (), qs, None))
        elif name == "measure":
            c = tqc.find_bit(inst.clbits[0]).index
            instrs.append(("measure", (), qs, c))
        else:
            raise SystemExit(f"unexpected op {name}")
    return instrs


def twirl_instrs(instrs: list, rng: np.random.Generator) -> tuple[list, list]:
    """Insert an independent random Pauli twirl around every CZ. Returns (twirled, twirls).

    ``twirls`` records the sampled (a, b, Ta, Tb, Ca, Cb) per CZ for the manifest.
    """
    out, record = [], []
    for (name, params, qs, cl) in instrs:
        if name != "cz":
            out.append((name, params, qs, cl))
            continue
        a, b = qs
        ta = _PAULIS[rng.integers(4)]
        tb = _PAULIS[rng.integers(4)]
        ca, cb = correcting_pauli(ta, tb)
        for gn, ang in _PNATIVE[ta]:
            out.append((gn, (ang,), (a,), None))
        for gn, ang in _PNATIVE[tb]:
            out.append((gn, (ang,), (b,), None))
        out.append(("cz", (), (a, b), None))
        for gn, ang in _PNATIVE[ca]:
            out.append((gn, (ang,), (a,), None))
        for gn, ang in _PNATIVE[cb]:
            out.append((gn, (ang,), (b,), None))
        record.append({"a": int(a), "b": int(b), "Ta": ta, "Tb": tb, "Ca": ca, "Cb": cb})
    return out, record


def emit_qasm(instrs: list, n_clbits: int) -> str:
    """OpenQASM3 with physical ``$N`` addressing from a native instruction list."""
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{n_clbits}] c;"]
    for name, params, qs, cl in instrs:
        if name in ("rz", "rx"):
            lines.append(f"{name}({params[0]!r}) ${qs[0]};")
        elif name == "cz":
            lines.append(f"cz ${qs[0]}, ${qs[1]};")
        elif name == "barrier":
            lines.append("barrier " + ", ".join(f"${q}" for q in qs) + ";")
        elif name == "measure":
            lines.append(f"c[{cl}] = measure ${qs[0]};")
        else:
            raise SystemExit(f"unexpected op {name}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Local (noiseless) verification of logical equivalence                       #
# --------------------------------------------------------------------------- #

def _compact_circuit(instrs: list, n: int) -> tuple[QuantumCircuit, list]:
    """Relabel physical qubits to compact clbit order; drop measure/barrier for statevector sim."""
    clbit_to_phys = {cl: qs[0] for (nm, _, qs, cl) in instrs if nm == "measure"}
    phys_order = [clbit_to_phys[c] for c in range(2 * n)]
    p2c = {p: c for c, p in enumerate(phys_order)}
    qc = QuantumCircuit(2 * n)
    for name, params, qs, cl in instrs:
        if name in ("measure", "barrier"):
            continue
        cq = [p2c[q] for q in qs]
        if name == "rz":
            qc.rz(params[0], cq[0])
        elif name == "rx":
            qc.rx(params[0], cq[0])
        elif name == "cz":
            qc.cz(cq[0], cq[1])
    return qc, phys_order


def _outcome_probs(qc: QuantumCircuit, n: int) -> np.ndarray:
    """Outcome distribution over 2^(2n) in clbit order (index bit c = compact qubit c)."""
    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2
    # Statevector index i has bit q at (i>>q)&1 for qubit q == clbit q -> already clbit order
    return np.clip(np.real(probs), 0.0, None)


def purity_from_probs(probs: np.ndarray, n: int) -> float:
    signs = np.array([swap_sign(format(b, f"0{2 * n}b"), n) for b in range(2 ** (2 * n))], dtype=float)
    return float(signs @ probs)


def ideal_distribution(instrs: list, n: int) -> tuple[np.ndarray, float]:
    """Noiseless outcome distribution (clbit order) and SWAP-test purity of a circuit."""
    qc, _ = _compact_circuit(instrs, n)
    probs = _outcome_probs(qc, n)
    return probs, purity_from_probs(probs, n)


def insert_x_before_cz(instrs: list, phys_q: int, cz_pair: tuple) -> list:
    """Positive control: insert one uncorrected rx(pi) on ``phys_q`` immediately before
    the first CZ acting on ``cz_pair``.  Logically NON-trivial — its outcome distribution
    differs from the untwirled circuit, so matching it on hardware proves that single-qubit
    gates inserted around a mid-circuit CZ actually execute (are not stripped/resynthesized
    away)."""
    out, done = [], False
    for (name, params, qs, cl) in instrs:
        if not done and name == "cz" and set(qs) == set(cz_pair):
            out.append(("rx", (float(np.pi),), (phys_q,), None))
            done = True
        out.append((name, params, qs, cl))
    if not done:
        raise ValueError(f"no CZ acting on {cz_pair}")
    return out


def verify_twirl(twirled_instrs: list, untw_instrs: list, n: int, tol: float = 1e-9) -> dict:
    """Assert the twirled circuit is logically identical to the untwirled one on GHZ.

    Checks (1) the two noiseless outcome distributions are bit-for-bit identical and
    (2) both give the SWAP-test purity 1.0 (GHZ is pure) — so the same swap_sign
    analysis applies to the twirled counts with NO frame correction.
    """
    qc_t, _ = _compact_circuit(twirled_instrs, n)
    qc_u, _ = _compact_circuit(untw_instrs, n)
    pt = _outcome_probs(qc_t, n)
    pu = _outcome_probs(qc_u, n)
    max_dev = float(np.max(np.abs(pt - pu)))
    pur_t = purity_from_probs(pt, n)
    pur_u = purity_from_probs(pu, n)
    ok = max_dev < tol and abs(pur_t - 1.0) < 1e-6 and abs(pur_u - 1.0) < 1e-6
    return {"identical_distribution": bool(max_dev < tol), "max_prob_deviation": max_dev,
            "purity_twirled": pur_t, "purity_untwirled": pur_u, "ok": bool(ok)}
