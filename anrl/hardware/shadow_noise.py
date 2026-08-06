"""Noisy prediction for the single-copy (classical-shadow) route.

The shadow route measures ONE copy per snapshot: prepare the state, apply a random
single-qubit Pauli basis rotation per qubit, measure.  Under the Cepheus model the
prep and rotation gates depolarize and the readout flips bits.  We predict the
measured purity (copy-fair U-statistic) and its statistical error at a shot budget.

Exact-then-sample.  There are only ``3^n`` basis combinations at ``n`` qubits, so
we solve each once (density matrix under gate noise) and apply readout confusion,
giving the exact noisy outcome distribution per basis.  Snapshots are then drawn
from those distributions in pure NumPy, many independent ``M``-shot experiments,
whose spread is the statistical error at budget ``M``.  No 60k-circuit Aer run.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from .noise_model import CEPHEUS_BASIS_GATES, cepheus_noise_model, readout_confusion_matrix
from .shadows import _BASES, _apply_basis, snapshots_from_outcomes
from .state_prep import MixedEnsemble, PreparedState
from anrl.benchmark.shadows import full_purity_ustatistic

_DM_SIM = AerSimulator(method="density_matrix")
# n-qubit line coupling (prep may carry 2-qubit gates); 2 qubits -> a single edge.
_LINE = {2: CouplingMap(couplinglist=[(0, 1)])}


def _components(state) -> list[tuple[float, PreparedState]]:
    if isinstance(state, MixedEnsemble):
        return list(state.components)
    return [(1.0, state)]


def noisy_shadow_dists(state, p2: float, p1: float, p_ro: float) -> dict[tuple[int, ...], np.ndarray]:
    """Exact noisy outcome distribution per Pauli-basis combination (``3^n`` entries).

    For a mixture the per-basis distribution is the weight-averaged distribution of
    the components, matching a per-shot classical ensemble.
    """
    comps = _components(state)
    n = comps[0][1].n
    coupling = _LINE.get(n)
    confusion = readout_confusion_matrix(n, p_ro)
    nm = cepheus_noise_model(p2=p2, p1=p1, p_ro=0.0)
    dists: dict[tuple[int, ...], np.ndarray] = {}
    for combo in product(range(3), repeat=n):
        mixed = np.zeros(2 ** n, dtype=np.float64)
        for w, comp in comps:
            qc = QuantumCircuit(n, name="shadow_nm")
            qc.compose(comp.circuit, qubits=range(n), inplace=True)
            for q in range(n):
                _apply_basis(qc, q, _BASES[combo[q]])
            tqc = transpile(qc, coupling_map=coupling, basis_gates=CEPHEUS_BASIS_GATES,
                            optimization_level=3, seed_transpiler=0)
            tqc.save_density_matrix()  # after translation (save is not a basis gate)
            rho = _DM_SIM.run(tqc, noise_model=nm).result().data(0)["density_matrix"]
            mixed += w * np.clip(np.real(rho.probabilities()), 0.0, None)
        dists[combo] = confusion @ mixed
    return dists


def _outcome_bits(outcome: int, n: int) -> np.ndarray:
    """Integer outcome -> per-qubit bits ``[b_0, ..., b_{n-1}]`` (qubit q = bit q, little-endian)."""
    return np.array([(outcome >> q) & 1 for q in range(n)], dtype=np.int64)


def predict_shadow_purity(state, dists: dict[tuple[int, ...], np.ndarray], m_shots: int,
                          n_experiments: int, base_seed: int = 0) -> dict:
    """Predicted noisy shadow purity at budget ``m_shots`` (mean + statistical error).

    Runs ``n_experiments`` independent ``m_shots``-snapshot experiments sampled from
    the exact noisy distributions; returns the mean estimate and the std across
    experiments (the standard error of a single ``m_shots`` run).

    Scaling note: the U-statistic SE has a ``4 zeta1 / M`` term (asymptotically
    ``1/sqrt(M)``) plus a ``2 zeta2 / (M(M-1))`` term (``~1/M``).  At small ``M``
    the second term still contributes, so the SE falls slightly FASTER than
    ``1/sqrt(M)``; extrapolating a small-``M`` SE to a larger budget by ``1/sqrt(M)``
    is therefore a conservative UPPER bound on the true SE at that budget.
    """
    comps = _components(state)
    n = comps[0][1].n
    combos = list(product(range(3), repeat=n))
    dist_arr = np.array([dists[c] for c in combos])  # (3^n, 2^n)
    combo_arr = np.array(combos)  # (3^n, n) basis index per qubit

    ests = np.empty(n_experiments, dtype=np.float64)
    for e in range(n_experiments):
        rng = np.random.default_rng(base_seed + e)
        combo_idx = rng.integers(0, len(combos), size=m_shots)  # which basis combo per shot
        bases = combo_arr[combo_idx]  # (M, n)
        outcomes = np.array([rng.choice(2 ** n, p=dist_arr[ci]) for ci in combo_idx])
        bits = np.stack([_outcome_bits(int(o), n) for o in outcomes])  # (M, n)
        snaps = snapshots_from_outcomes(bases, bits, n)
        ests[e] = full_purity_ustatistic(snaps)
    return {
        "mean": float(ests.mean()),
        "std": float(ests.std(ddof=1)),
        "m_shots": m_shots,
        "n_experiments": n_experiments,
    }
