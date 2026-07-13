"""Analyze the bracketed same-session GHZ-ladder experiment (post-processing; no credits).

Drift A->C (per-qubit readout, with CIs); collective measured vs the locked (Block A)
bands and vs Block C predictions; monotonicity; readout-vs-gate decomposition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.same_session_lib import (
    build_tables, parse_readout, predict_swap, swap_purity_from_counts, PHYS_8,
)
from anrl.hardware import swap_sign

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"


def drift(rates_a, rates_c):
    rows = []
    for q in PHYS_8:
        a, c = rates_a[q], rates_c[q]
        # significant if CIs disjoint
        p10_sig = a["p10_ci"][1] < c["p10_ci"][0] or c["p10_ci"][1] < a["p10_ci"][0]
        p01_sig = a["p01_ci"][1] < c["p01_ci"][0] or c["p01_ci"][1] < a["p01_ci"][0]
        rows.append({"qubit": q,
                     "p10_A": round(a["p10"], 4), "p10_C": round(c["p10"], 4),
                     "p10_shift": round(c["p10"] - a["p10"], 4), "p10_sig": bool(p10_sig),
                     "p01_A": round(a["p01"], 4), "p01_C": round(c["p01"], 4),
                     "p01_shift": round(c["p01"] - a["p01"], 4), "p01_sig": bool(p01_sig)})
    return rows


def boot_ci(counts, n, seed=0):
    signs = np.array([swap_sign(b.replace(" ", ""), n) for b, c in counts.items() for _ in range(c)], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(signs, size=len(signs), replace=True).mean() for _ in range(4000)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    rates_a = parse_readout("A")
    rates_c = parse_readout("C")
    tables_c = build_tables(rates_c)
    locked = json.loads((HW / "locked_same_session.json").read_text())
    a_preds = locked["predictions"]

    drift_rows = drift(rates_a, rates_c)
    n_sig = sum(1 for r in drift_rows if r["p10_sig"] or r["p01_sig"])
    mean_p10_shift = float(np.mean([abs(r["p10_shift"]) for r in drift_rows]))
    mean_p01_shift = float(np.mean([abs(r["p01_shift"]) for r in drift_rows]))

    cells = []
    for n in (2, 3, 4):
        counts = {k: int(v) for k, v in json.loads((HW / f"ss_B_n{n}_counts.json").read_text()).items()}
        mu = swap_purity_from_counts(counts, n)
        ci = boot_ci(counts, n)
        a_band = a_preds[str(n)]["band"]
        c_pred = predict_swap(n, tables_c)
        c_band = c_pred["band"]
        inside_a = a_band["hi"] <= mu <= a_band["lo"]
        # is the measurement bracketed between A and C predictions (drift-consistent)?
        lo_env = min(a_band["hi"], c_band["hi"]); hi_env = max(a_band["lo"], c_band["lo"])
        bracketed = lo_env <= mu <= hi_env
        cells.append({"n": n, "measured": round(mu, 4), "ci95": [round(x, 4) for x in ci],
                      "A_band": a_band, "C_band": c_band, "inside_A_band": bool(inside_a),
                      "bracketed_by_A_and_C": bool(bracketed),
                      "gate_penalty": a_preds[str(n)]["gate_penalty"],
                      "readout_penalty": a_preds[str(n)]["readout_penalty"],
                      "dist_outside_A": 0.0 if inside_a else round(min(abs(mu - a_band["lo"]), abs(mu - a_band["hi"])), 4)})

    monotonic = cells[0]["measured"] > cells[1]["measured"] > cells[2]["measured"]
    result = {"drift": {"rows": drift_rows, "n_qubits_significant": n_sig,
                        "mean_abs_p10_shift": round(mean_p10_shift, 4),
                        "mean_abs_p01_shift": round(mean_p01_shift, 4)},
              "cells": cells, "monotonic_n2_gt_n3_gt_n4": bool(monotonic)}
    (HW / "same_session_analysis.json").write_text(json.dumps(result, indent=2, default=float))

    print("=== DRIFT A->C (per-qubit readout) ===")
    print(f"  significant-shift qubits: {n_sig}/8 | mean |Δp10|={mean_p10_shift:.4f} mean |Δp01|={mean_p01_shift:.4f}")
    for r in drift_rows:
        flag = " *" if (r["p10_sig"] or r["p01_sig"]) else ""
        print(f"  ${r['qubit']}: p10 {r['p10_A']:.3f}->{r['p10_C']:.3f} ({r['p10_shift']:+.3f}), "
              f"p01 {r['p01_A']:.3f}->{r['p01_C']:.3f} ({r['p01_shift']:+.3f}){flag}")
    print("\n=== COLLECTIVE measured vs locked (A) band [and C band] ===")
    for c in cells:
        v = "INSIDE-A" if c["inside_A_band"] else f"OUTSIDE-A by {c['dist_outside_A']:.3f}"
        br = " (bracketed by A&C)" if c["bracketed_by_A_and_C"] else ""
        print(f"  n={c['n']}: measured {c['measured']:.4f} (CI {c['ci95']}) | A-band "
              f"{c['A_band']['hi']:.3f}-{c['A_band']['lo']:.3f} | C-band {c['C_band']['hi']:.3f}-{c['C_band']['lo']:.3f}"
              f" -> {v}{br}")
    print(f"\nMonotonic (n2>n3>n4)? {monotonic}")


if __name__ == "__main__":
    main()
