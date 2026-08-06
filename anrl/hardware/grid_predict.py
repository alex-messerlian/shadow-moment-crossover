"""General-n gate-level prediction for the collective (SWAP) and single-copy routes.

Transpiles the actual circuit onto the real Cepheus coupling map (so routing SWAPs
show up in the CZ budget), compactifies it to only the active qubits for a
density-matrix noise simulation, and applies the measured correlated-readout model.
Zero credits, pure local simulation.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from anrl.benchmark.shadows import full_purity_ustatistic
from .backend import cepheus_coupling_map
from .noise_model import CEPHEUS_BASIS_GATES, cepheus_noise_model
from .readout_model import correlated_confusion
from .shadows import _BASES, _apply_basis, snapshots_from_outcomes
from .state_prep import PreparedState
from .swap_test import destructive_swap_test, swap_sign

_DM = AerSimulator(method="density_matrix")
_CM = cepheus_coupling_map()


def transpile_swap(prep: PreparedState, prep_b: PreparedState | None = None):
    """Transpile the 2n-qubit SWAP test onto real Cepheus; report CZ + routing.

    Returns ``(device_circuit, cz_device, cz_no_routing, routing_overhead,
    phys_in_clbit_order)``.  ``cz_no_routing`` is the CZ count with the same native
    decomposition but no coupling constraint; the difference is the routing overhead.
    """
    logical = destructive_swap_test(prep, prep_b)
    dev = transpile(logical, coupling_map=_CM, basis_gates=CEPHEUS_BASIS_GATES,
                    optimization_level=3, seed_transpiler=0)
    free = transpile(logical, basis_gates=CEPHEUS_BASIS_GATES,
                     optimization_level=3, seed_transpiler=0)
    cz_dev = sum(1 for i in dev.data if i.operation.name == "cz")
    cz_free = sum(1 for i in free.data if i.operation.name == "cz")
    clbit_to_phys = {}
    for inst in dev.data:
        if inst.operation.name == "measure":
            clbit_to_phys[dev.find_bit(inst.clbits[0]).index] = dev.find_bit(inst.qubits[0]).index
    phys_order = [clbit_to_phys[c] for c in range(len(clbit_to_phys))]
    return dev, cz_dev, cz_free, cz_dev - cz_free, phys_order


def _compactify(dev, phys_order):
    """Relabel the device circuit's active qubits to clbit order and drop measurement.

    Adds ``save_density_matrix``; the diagonal is the ideal-readout outcome
    distribution over the measured qubits in clbit order.
    """
    phys_to_compact = {p: c for c, p in enumerate(phys_order)}
    m = len(phys_order)
    qc = QuantumCircuit(m)
    for inst in dev.data:
        name = inst.operation.name
        if name in ("measure", "barrier"):
            continue
        qubits = [phys_to_compact[dev.find_bit(b).index] for b in inst.qubits]
        qc.append(inst.operation, qubits)
    qc.save_density_matrix()
    return qc


def swap_gate_noisy_probs(prep, p2, p1, prep_b=None):
    """Ideal-readout SWAP outcome distribution under gate noise (device-transpiled)."""
    dev, _, _, _, phys_order = transpile_swap(prep, prep_b)
    qc = _compactify(dev, phys_order)
    rho = _DM.run(qc, noise_model=cepheus_noise_model(p2=p2, p1=p1, p_ro=0.0)).result().data(0)["density_matrix"]
    return np.clip(np.real(rho.probabilities()), 0.0, None), phys_order


def swap_signs(n: int) -> np.ndarray:
    return np.array([swap_sign(format(b, f"0{2*n}b"), n) for b in range(2 ** (2 * n))], dtype=float)


def predict_swap(prep, p2, p1, correlated=True):
    """Predicted collective purity + gate/readout penalty decomposition (measured readout).

    Returns dict with measured purity, gate-only purity, gate_penalty, readout_penalty,
    cz counts, routing overhead, physical qubits.
    """
    n = prep.n
    dev, cz_dev, cz_free, routing, phys_order = transpile_swap(prep)
    q, _ = swap_gate_noisy_probs(prep, p2, p1)
    signs = swap_signs(n)
    R = correlated_confusion(phys_order, correlated=correlated)
    gate_only = float(signs @ q)
    measured = float(signs @ (R @ q))
    return {
        "measured_purity": measured,
        "gate_only_purity": gate_only,
        "gate_penalty": 1.0 - gate_only,
        "readout_penalty": gate_only - measured,
        "cz_device": cz_dev, "cz_no_routing": cz_free, "routing_overhead": routing,
        "phys_qubits": phys_order,
    }


# --------------------------------------------------------------------------- #
# Single-copy (local Pauli shadow) route                                      #
# --------------------------------------------------------------------------- #
# copy-A physical qubits of the SWAP ladder (the single copy the shadow route uses);
# a line among them, matching Cepheus connectivity (0-1-2-3 are edges on the used patch).
SHADOW_PHYS = {2: [0, 1], 3: [0, 1, 2], 4: [0, 1, 2, 3]}


def _shadow_gate_dist(prep, combo, p2, p1):
    """Ideal-readout distribution for one Pauli-basis combo (device gate noise, n-qubit line)."""
    n = prep.n
    line = CouplingMap(couplinglist=[(i, i + 1) for i in range(n - 1)]) if n > 1 else None
    qc = QuantumCircuit(n)
    qc.compose(prep.circuit, qubits=range(n), inplace=True)
    for q in range(n):
        _apply_basis(qc, q, _BASES[combo[q]])
    dev = transpile(qc, coupling_map=line, basis_gates=CEPHEUS_BASIS_GATES,
                    optimization_level=3, seed_transpiler=0)
    dev.save_density_matrix()
    rho = _DM.run(dev, noise_model=cepheus_noise_model(p2=p2, p1=p1, p_ro=0.0)).result().data(0)["density_matrix"]
    return np.clip(np.real(rho.probabilities()), 0.0, None), SHADOW_PHYS[n]


def predict_shadow(prep, p2, p1, m_ref=2000, n_exp=40, base_seed=0, correlated=True):
    """Predicted single-copy shadow purity + statistical error, under measured noise.

    Precomputes the exact noisy outcome distribution for each of the ``3^n`` Pauli-basis
    combinations (device gate noise + correlated readout), then samples ``n_exp``
    independent ``m_ref``-snapshot experiments to get the mean and the per-experiment SE
    (extrapolated to 10k shots as a conservative ``1/sqrt(M)`` bound).
    """
    n = prep.n
    combos = list(product(range(3), repeat=n))
    # phys layout from the Z-basis combo (prep fixes the 2q layout; rotations are 1q)
    _, phys = _shadow_gate_dist(prep, (2,) * n, p2, p1)
    R = correlated_confusion(phys, correlated=correlated)
    dists = {}
    for combo in combos:
        g, _ = _shadow_gate_dist(prep, combo, p2, p1)
        dists[combo] = R @ g
    combo_arr = np.array(combos)
    dist_arr = np.array([dists[c] for c in combos])
    ests = np.empty(n_exp)
    for e in range(n_exp):
        rng = np.random.default_rng(base_seed + e)
        idx = rng.integers(0, len(combos), size=m_ref)
        bases = combo_arr[idx]
        outs = np.array([rng.choice(2 ** n, p=dist_arr[ci]) for ci in idx])
        bits = np.stack([np.array([(int(o) >> q) & 1 for q in range(n)]) for o in outs])
        ests[e] = full_purity_ustatistic(snapshots_from_outcomes(bases, bits, n))
    mean, std = float(ests.mean()), float(ests.std(ddof=1))
    return {"measured_purity": mean, "se_at_Mref": std, "m_ref": m_ref,
            "se_at_10k": std * np.sqrt(m_ref / 10000.0), "phys_qubits": phys}
