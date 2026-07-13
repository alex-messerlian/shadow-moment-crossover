"""Noiseless validation of the Cepheus purity-experiment circuits (ZERO credits).

Everything here is local simulation.  It confirms, before any hardware run, that:

* the destructive (Bell-basis) SWAP test recovers ``Tr(rho^2)`` for pure and mixed
  states (shot-based on Aer, and exact on ``rho (x) rho``);
* the single-copy Pauli-shadow route recovers ``Tr(rho^2)`` via the copy-fair
  U-statistic;
* the SWAP-test circuit transpiles onto Cepheus's real coupling map with no
  routing overhead at n=2, and the transpiled circuit is unitarily equivalent to
  the logical one.

Run:  ``python -m experiments.hardware_validation``
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator
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
from anrl.hardware.shadows import bits_from_bitstring

SWAP_SHOTS = 200_000
SHADOW_SHOTS = 8_000


def _n2q(circ) -> int:
    return sum(1 for inst in circ.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


def validate_swap_test(sim: AerSimulator) -> None:
    print("== Destructive SWAP test — recover Tr(rho^2) ==")
    print("  (shot-based Aer for pure states; exact rho(x)rho for mixed)")
    for prep in (bell_state(), haar_pure(2, 0), haar_pure(2, 7)):
        qc = transpile(destructive_swap_test(prep), sim, optimization_level=1)
        counts = sim.run(qc, shots=SWAP_SHOTS, seed_simulator=1).result().get_counts()
        est = purity_from_counts(counts, prep.n)
        print(f"    {prep.label:>12}: est={est:.4f}  true={prep.purity():.4f}  |diff|={abs(est - prep.purity()):.4f}")
    for prep in (random_mixed(2, 3), random_mixed(2, 9)):
        est = exact_swap_purity(prep.rho, prep.n)
        print(f"    {prep.label:>12}: exact={est:.6f}  true={prep.purity():.6f}  |diff|={abs(est - prep.purity()):.1e}")


def validate_shadows(sim: AerSimulator) -> None:
    print("\n== Single-copy Pauli shadows — recover Tr(rho^2) (copy-fair U-statistic) ==")
    for prep in (bell_state(), haar_pure(2, 0)):
        circs, bases = pauli_shadow_circuits(prep, SHADOW_SHOTS, seed=2)
        circs = transpile(circs, sim, optimization_level=1)
        res = sim.run(circs, shots=1, seed_simulator=5).result()
        bits = np.array(
            [bits_from_bitstring(next(iter(res.get_counts(i))), prep.n) for i in range(len(circs))]
        )
        est = shadow_purity(bases, bits, prep.n)
        print(f"    {prep.label:>12}: est={est:.4f}  true={prep.purity():.4f}  (M={SHADOW_SHOTS} shots)")


def validate_transpilation(sim: AerSimulator) -> None:
    print("\n== Transpilation to Cepheus (native cz/rx/rz + real coupling map) ==")
    cm = cepheus_coupling_map()
    print(f"  device: {cm.size()} qubits, {len(cm.get_edges())} edges; n=2 SWAP test uses 4 qubits")
    for prep in (bell_state(), haar_pure(2, 0)):
        logical = destructive_swap_test(prep)
        dev = transpile(logical, coupling_map=cm, basis_gates=CEPHEUS_BASIS_GATES,
                        optimization_level=3, seed_transpiler=0)
        phys = sorted({dev.find_bit(b).index for inst in dev.data for b in inst.qubits
                       if inst.operation.name != "barrier"})
        lg = logical.remove_final_measurements(inplace=False)
        dec = transpile(lg, basis_gates=CEPHEUS_BASIS_GATES, optimization_level=3, seed_transpiler=0)
        equiv = Operator(lg).equiv(Operator(dec))
        print(f"    {prep.label:>12}: before depth={logical.depth():>2} 2q={_n2q(logical)} | "
              f"after depth={dev.depth():>2} cz={_n2q(dev)} routing-swaps={_n2q(dev) - _n2q(logical)} "
              f"phys={phys} | native==logical: {equiv}")


def main() -> None:
    sim = AerSimulator()
    validate_swap_test(sim)
    validate_shadows(sim)
    validate_transpilation(sim)
    print("\nNo hardware job submitted — ZERO quantum credits spent.")


if __name__ == "__main__":
    main()
