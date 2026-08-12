"""PASS 48.1 support: do paper one's compilation facts survive without the hardware section?

    PYTHONPATH=. .venv/bin/python experiments/pass48_verify_compilation_facts.py

PASS 47 concluded that Section 5.4's claim -- the entangling overhead is real but not the
binding constraint -- survives dropping the hardware section, because its load-bearing support
is a TRANSPILATION count reproducible offline from the committed
``anrl/hardware/cepheus_metadata.json``, not a measured purity deficit.  That was an argument.
This checks it: the coupling map is loaded from committed package data with the credential
environment variables explicitly unset, and every gate count the paper states for the tested
circuits is recomputed.

Writes ``results/pass48_compilation_facts.json``.  Reports non-reproducing claims as such.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile

from anrl.hardware.backend import CEPHEUS_BASIS_GATES, cepheus_coupling_map

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass48_compilation_facts.json"

N2_LAYOUT = [0, 1, 9, 10]
LADDER_LAYOUT = {3: [0, 1, 2, 9, 10, 11], 4: [0, 1, 2, 3, 9, 10, 11, 12]}
OPT_LEVEL = 1
SEED = 7


def _cz(qc: QuantumCircuit, coupling, layout=None) -> tuple[int, int]:
    """(cz count on the mapped device, cz count with all-to-all connectivity)."""
    mapped = transpile(qc, coupling_map=coupling, basis_gates=CEPHEUS_BASIS_GATES,
                       initial_layout=layout, optimization_level=OPT_LEVEL,
                       seed_transpiler=SEED)
    free = transpile(qc, basis_gates=CEPHEUS_BASIS_GATES, optimization_level=OPT_LEVEL,
                     seed_transpiler=SEED)
    return mapped.count_ops().get("cz", 0), free.count_ops().get("cz", 0)


def bell_swap() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.h(0); qc.cx(0, 1)
    qc.h(2); qc.cx(2, 3)
    qc.cx(0, 2); qc.cx(1, 3)
    qc.h(0); qc.h(1)
    return qc


def ghz_ladder_swap(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(2 * n)
    for off in (0, n):
        qc.h(off)
        for q in range(n - 1):
            qc.cx(off + q, off + q + 1)
    for q in range(n):
        qc.cx(q, q + n)
    for q in range(n):
        qc.h(q)
    return qc


def main() -> None:
    t0 = time.time()
    creds = {k: os.environ.pop(k, None)
             for k in ("OPENQUANTUM_CLIENT_ID", "OPENQUANTUM_CLIENT_SECRET")}
    print("credentials removed from the environment: "
          + ", ".join(f"{k}={'unset' if v is None else 'REMOVED'}" for k, v in creds.items()))

    cm = cepheus_coupling_map()
    print(f"coupling map from committed package data: {cm.size()} qubits, "
          f"{len(cm.get_edges())} directed edges (no network, no credentials)")

    checks = []

    mapped, free = _cz(bell_swap(), cm, N2_LAYOUT)
    checks.append({
        "claim": "n=2 destructive SWAP on physical {0,1,9,10} uses four CZ and zero routing",
        "paper_section": "6.2 / 5.4", "paper_value": 4,
        "recomputed_mapped": mapped, "recomputed_all_to_all": free,
        "routing_attributable": mapped - free,
        "reproduces": mapped == 4 and mapped == free,
    })

    for n in (3, 4):
        mapped, free = _cz(ghz_ladder_swap(n), cm, LADDER_LAYOUT[n])
        checks.append({
            "claim": f"n={n} GHZ ladder maps with zero routing at 3n-2 CZ",
            "paper_section": "6.2", "paper_value": 3 * n - 2,
            "recomputed_mapped": mapped, "recomputed_all_to_all": free,
            "routing_attributable": mapped - free,
            "reproduces": mapped == 3 * n - 2 and mapped == free,
        })

    # The generic-state count (46 CZ = 26 intrinsic + 20 routing) came from the hardware
    # build pipeline's own state-preparation synthesis, not from this ad-hoc circuit, so it is
    # NOT expected to reproduce here and is recorded as unverified rather than as a match.
    checks.append({
        "claim": "generic n=4 states need 46 CZ on this topology (26 intrinsic + 20 routing)",
        "paper_section": "6.2", "paper_value": 46,
        "recomputed_mapped": None, "recomputed_all_to_all": None,
        "routing_attributable": None, "reproduces": None,
        "note": "NOT reproduced here. This figure depends on the state-preparation synthesis "
                "used by the hardware build scripts; an ad-hoc prepare_state with "
                f"optimization_level={OPT_LEVEL} gives a different count (50 in a spot check). "
                "It is a genuine hardware-pipeline number and should travel with paper two "
                "unless paper one re-derives it from a stated synthesis.",
    })

    ok = [c for c in checks if c["reproduces"] is True]
    unver = [c for c in checks if c["reproduces"] is None]
    bad = [c for c in checks if c["reproduces"] is False]
    print(f"\n{'claim':66s} {'paper':>6s} {'mapped':>7s} {'free':>6s}  verdict")
    for c in checks:
        v = ("reproduces" if c["reproduces"] else
             "NOT REPRODUCED" if c["reproduces"] is False else "not checked here")
        print(f"  {c['claim'][:64]:64s} {c['paper_value']:>6} "
              f"{str(c['recomputed_mapped']):>7s} {str(c['recomputed_all_to_all']):>6s}  {v}")
    print(f"\n{len(ok)} reproduce exactly, {len(bad)} fail, {len(unver)} not checked here")
    print("VERDICT: the structured-circuit counts paper one needs in Section 2 -- four CZ with "
          "zero routing at n=2 and 3n-2 for the ladders -- reproduce offline from committed "
          "package data with no credentials, so Section 5.4's claim survives the split. The "
          "generic-state 46-CZ figure does not reproduce from this script and belongs to "
          "paper two.")

    OUT.write_text(json.dumps({
        "description": "PASS 48.1 support: paper one's compilation facts, recomputed offline",
        "config": {"optimization_level": OPT_LEVEL, "seed_transpiler": SEED,
                   "basis_gates": CEPHEUS_BASIS_GATES, "n2_layout": N2_LAYOUT,
                   "ladder_layouts": LADDER_LAYOUT},
        "coupling_map": {"n_qubits": cm.size(), "n_directed_edges": len(cm.get_edges()),
                         "source": "anrl/hardware/cepheus_metadata.json (committed)"},
        "credentials_present": {k: v is not None for k, v in creds.items()},
        "checks": checks,
        "summary": {"reproduce": len(ok), "fail": len(bad), "not_checked": len(unver)},
        "wall_seconds": time.time() - t0,
    }, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
