"""Build circuits + lock predictions for the GHZ-ladder hardware experiment (no credits here).

Collective route: destructive SWAP test on GHZ at n=2,3,4 (physical ladder qubits, zero
routing).  Single-copy anchor: a FIXED, pre-registered set of 15 random local-Pauli bases
x 600 shots on the copy-A qubits {0,1} at n=2 — simulated with the measured device
parameters to lock the prediction BEFORE running (the exact same 15 bases run on hardware).

Writes the QASM circuits to results/hardware/ and the locked predictions to
results/hardware/hg_locked.json.  Run:  python -m experiments.hardware_grid_build
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile

from anrl.hardware import (avg_gate_error_to_depol_param, cepheus_coupling_map,
                           CEPHEUS_BASIS_GATES, destructive_swap_test)
from anrl.hardware.grid_predict import _shadow_gate_dist
from anrl.hardware.readout_model import correlated_confusion
from anrl.hardware.shadows import _BASES, _apply_basis, snapshots_from_outcomes
from anrl.hardware.state_prep import ghz_state
from anrl.benchmark.shadows import full_purity_ustatistic

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
LADDER = {2: [0, 1, 9, 10], 3: [0, 1, 2, 9, 10, 11], 4: [0, 1, 2, 3, 9, 10, 11, 12]}
ANCHOR_SEED = 20260713
N_BASES = 15
SHOTS_PER_BASIS = 600
CZ_MID = avg_gate_error_to_depol_param(0.009, 2)
CZ_LO = avg_gate_error_to_depol_param(0.005, 2)
CZ_HI = avg_gate_error_to_depol_param(0.015, 2)
P1 = 0.001
_CM = cepheus_coupling_map()


def emit_physical_qasm3(tqc, n_clbits: int) -> str:
    """Emit OpenQASM3 with physical ``$N`` addressing from a device-transpiled circuit."""
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{n_clbits}] c;"]
    for inst in tqc.data:
        name = inst.operation.name
        qs = [tqc.find_bit(b).index for b in inst.qubits]
        if name in ("rz", "rx"):
            lines.append(f"{name}({inst.operation.params[0]!r}) ${qs[0]};")
        elif name == "cz":
            lines.append(f"cz ${qs[0]}, ${qs[1]};")
        elif name == "barrier":
            lines.append("barrier " + ", ".join(f"${q}" for q in qs) + ";")
        elif name == "measure":
            c = tqc.find_bit(inst.clbits[0]).index
            lines.append(f"c[{c}] = measure ${qs[0]};")
        else:
            raise SystemExit(f"unexpected op {name}")
    return "\n".join(lines) + "\n"


def build_collective():
    out = {}
    for n in (2, 3, 4):
        logical = destructive_swap_test(ghz_state(n))
        tqc = transpile(logical, coupling_map=_CM, basis_gates=CEPHEUS_BASIS_GATES,
                        initial_layout=LADDER[n], optimization_level=3, seed_transpiler=0)
        cz = sum(1 for i in tqc.data if i.operation.name == "cz")
        assert cz == 3 * n - 2, (n, cz)  # zero routing
        clbit_to_phys = {tqc.find_bit(i.clbits[0]).index: tqc.find_bit(i.qubits[0]).index
                         for i in tqc.data if i.operation.name == "measure"}
        phys_order = [clbit_to_phys[c] for c in range(2 * n)]
        (HW / f"hg_coll_n{n}.qasm").write_text(emit_physical_qasm3(tqc, 2 * n))
        out[n] = {"cz": cz, "phys_clbit_order": phys_order}
    return out


def anchor_bases():
    rng = np.random.default_rng(ANCHOR_SEED)
    return rng.integers(0, 3, size=(N_BASES, 2))  # 2 copy-A qubits {0,1}


def build_single_anchor(bases):
    """15 GHZ(2)-prep + fixed random-Pauli-basis circuits on physical {0,1}."""
    for i, basis in enumerate(bases):
        qc = QuantumCircuit(2, 2)
        qc.compose(ghz_state(2).circuit, qubits=[0, 1], inplace=True)
        for q in range(2):
            _apply_basis(qc, q, _BASES[int(basis[q])])
        qc.measure([0, 1], [0, 1])
        tqc = transpile(qc, coupling_map=_CM, basis_gates=CEPHEUS_BASIS_GATES,
                        initial_layout=[0, 1], optimization_level=3, seed_transpiler=0)
        (HW / f"hg_single_n2_b{i:02d}.qasm").write_text(emit_physical_qasm3(tqc, 2))


def lock_anchor_prediction(bases):
    """Simulate the EXACT 15 bases x 600 shots with measured params -> locked prediction.

    Uses the biased full U-statistic (same estimator hardware will use), so the prediction
    is directly comparable to the hardware result — this validates the pipeline, not purity.
    """
    R = correlated_confusion([0, 1], correlated=True)         # measured readout on {0,1}
    results = {}
    for tag, p2 in (("lo", CZ_LO), ("mid", CZ_MID), ("hi", CZ_HI)):
        dists = {}
        for basis in bases:
            key = tuple(int(x) for x in basis)
            if key not in dists:
                g, _ = _shadow_gate_dist(ghz_state(2), key, p2, P1)
                dists[key] = R @ g                              # gate noise + measured readout
        ests = []
        for seed in range(200):
            rng = np.random.default_rng(1000 + seed)
            allb, allo = [], []
            for basis in bases:
                key = tuple(int(x) for x in basis)
                outs = rng.choice(4, size=SHOTS_PER_BASIS, p=dists[key])
                for o in outs:
                    allb.append(basis); allo.append([(int(o) >> q) & 1 for q in range(2)])
            ests.append(full_purity_ustatistic(snapshots_from_outcomes(np.array(allb), np.array(allo), 2)))
        results[tag] = {"mean": float(np.mean(ests)), "se": float(np.std(ests, ddof=1))}
    return results


def main() -> None:
    coll = build_collective()
    bases = anchor_bases()
    build_single_anchor(bases)
    anchor = lock_anchor_prediction(bases)

    v2 = json.loads((HW / "locked_grid_predictions_v2.json").read_text())
    v2_ghz = {c["n"]: c for c in v2["grid"] if c["state"] == "ghz"}

    locked = {
        "collective": {n: {"cz": coll[n]["cz"], "phys_clbit_order": coll[n]["phys_clbit_order"],
                           "v2_swap_band": v2_ghz[n]["swap"]["purity_band"],
                           "v2_swap_mid": v2_ghz[n]["swap"]["purity_mid"],
                           "v2_gate_penalty": v2_ghz[n]["swap"]["gate_penalty"],
                           "v2_readout_penalty": v2_ghz[n]["swap"]["readout_penalty"]} for n in (2, 3, 4)},
        "single_anchor_n2": {"n_bases": N_BASES, "shots_per_basis": SHOTS_PER_BASIS,
                             "total_shots": N_BASES * SHOTS_PER_BASIS, "seed": ANCHOR_SEED,
                             "bases": bases.tolist(), "locked_prediction": anchor,
                             "estimator": "full copy-fair U-statistic (biased under grouped bases; "
                                          "same estimator on sim and hardware -> validates the pipeline)"},
    }
    (HW / "hg_locked.json").write_text(json.dumps(locked, indent=2))

    print("=== LOCKED PREDICTIONS (printed before any submission) ===\n")
    print("Collective (SWAP) — v2 measured-parameter predictions:")
    print(f"  {'n':>2} {'CZ':>3} | {'purity band lo/mid/hi':>22} | {'gate pen':>8} {'readout pen':>11}")
    for n in (2, 3, 4):
        c = locked["collective"][n]; b = c["v2_swap_band"]
        print(f"  {n:>2} {c['cz']:>3} | {b['lo']:.3f}/{b['mid']:.3f}/{b['hi']:.3f}      | "
              f"{c['v2_gate_penalty']:>8.3f} {c['v2_readout_penalty']:>11.3f}")
    a = locked["single_anchor_n2"]
    print(f"\nSingle-copy anchor n=2 ({a['n_bases']} fixed bases x {a['shots_per_basis']} shots, "
          f"seed {a['seed']}):")
    print(f"  LOCKED predicted U-statistic (biased protocol): "
          f"lo {a['locked_prediction']['lo']['mean']:.3f}+/-{a['locked_prediction']['lo']['se']:.3f}, "
          f"mid {a['locked_prediction']['mid']['mean']:.3f}+/-{a['locked_prediction']['mid']['se']:.3f}, "
          f"hi {a['locked_prediction']['hi']['mean']:.3f}+/-{a['locked_prediction']['hi']['se']:.3f}")
    print("  (goal: hardware reproduces this exact number -> single-copy pipeline validated)")


if __name__ == "__main__":
    main()
