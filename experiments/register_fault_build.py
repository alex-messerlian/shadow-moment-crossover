"""Build + verify circuits for the register-fault localization experiment (no credits).

Three SWAP cells:
  n3_std : {0,1,2,9,10,11}          (byte-identical to hg_coll_n3.qasm)
  n4     : {0,1,2,3,9,10,11,12}     (byte-identical to hg_coll_n4.qasm)
  n3_alt : {1,2,3,10,11,12}         (NEW localization ladder; parity pairs (1,10)(2,11)(3,12)
                                     -> includes the suspect pair {3,12})

Verifies: n3_alt transpiles with ZERO routing (7 CZ) and clbit order == the ladder;
n3_alt is a valid GHZ SWAP (noiseless purity 1.0); the standard circuits are byte-identical
to the committed QASM. Writes results/hardware/rf_n3alt.qasm.
Run:  PYTHONPATH=. python -m experiments.register_fault_build
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from anrl.hardware import CEPHEUS_BASIS_GATES, cepheus_coupling_map, destructive_swap_test, swap_sign
from anrl.hardware.state_prep import ghz_state
from experiments.hardware_grid_build import emit_physical_qasm3

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
ALT_LADDER = [1, 2, 3, 10, 11, 12]  # clbit c -> physical; copy A {1,2,3}, copy B {10,11,12}
_CM = cepheus_coupling_map()


def _phys_order(tqc, n_clbits):
    c2p = {tqc.find_bit(i.clbits[0]).index: tqc.find_bit(i.qubits[0]).index
           for i in tqc.data if i.operation.name == "measure"}
    return [c2p[c] for c in range(n_clbits)]


def _noiseless_purity(qasm_phys_order, tqc, n):
    """Simulate the transpiled circuit compacted to clbit order; SWAP-sign purity of GHZ."""
    p2c = {p: c for c, p in enumerate(qasm_phys_order)}
    qc = QuantumCircuit(2 * n)
    for inst in tqc.data:
        nm = inst.operation.name
        if nm in ("measure", "barrier"):
            continue
        qs = [p2c[tqc.find_bit(b).index] for b in inst.qubits]
        getattr(qc, nm)(*(list(inst.operation.params) + qs))
    probs = np.abs(Statevector.from_instruction(qc).data) ** 2
    signs = np.array([swap_sign(format(b, f"0{2*n}b"), n) for b in range(2 ** (2 * n))])
    return float(signs @ probs)


def build_alt():
    logical = destructive_swap_test(ghz_state(3))
    tqc = transpile(logical, coupling_map=_CM, basis_gates=CEPHEUS_BASIS_GATES,
                    initial_layout=ALT_LADDER, optimization_level=3, seed_transpiler=0)
    cz = sum(1 for i in tqc.data if i.operation.name == "cz")
    phys = _phys_order(tqc, 6)
    assert cz == 3 * 3 - 2 == 7, f"alt ladder NOT zero-routing: {cz} CZ (expected 7)"
    assert phys == ALT_LADDER, f"clbit order {phys} != ladder {ALT_LADDER}"
    pur = _noiseless_purity(phys, tqc, 3)
    assert abs(pur - 1.0) < 1e-9, f"alt circuit noiseless purity {pur} != 1.0"
    (HW / "rf_n3alt.qasm").write_text(emit_physical_qasm3(tqc, 6))
    return cz, phys, pur


def verify_byte_identity():
    """The n3_std / n4 SWAP circuits we submit are the committed hg_coll_n{3,4}.qasm."""
    out = {}
    for n in (3, 4):
        f = f"results/hardware/hg_coll_n{n}.qasm"
        committed = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
        ondisk = (HW / f"hg_coll_n{n}.qasm").read_text()
        out[n] = (committed == ondisk and len(committed) > 0)
    return out


def main():
    cz, phys, pur = build_alt()
    print(f"n3_alt {{1,2,3,10,11,12}}: {cz} CZ (zero routing), clbit->phys {phys}, "
          f"noiseless purity {pur:.6f}  -> wrote rf_n3alt.qasm")
    print(f"  parity pairs (clbit i, i+3): (1,10) (2,11) (3,12)  <- suspect {{3,12}} = pair 2")
    bi = verify_byte_identity()
    print(f"byte-identity vs committed: hg_coll_n3 {bi[3]}, hg_coll_n4 {bi[4]}")
    assert all(bi.values()), "byte-identity check FAILED"
    print("all circuit checks PASS")


if __name__ == "__main__":
    main()
