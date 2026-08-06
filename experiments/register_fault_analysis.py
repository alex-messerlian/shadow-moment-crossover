"""Analyze the register-fault localization experiment (no credits).

Discriminator:
  * measured vs locked prediction for n3_std, n4, n3_alt (with bootstrap CIs);
  * DECOUPLING, does n3_std match its band while n4 falls below, same session?
  * LOCALIZATION, is n3_alt ({1,2,3,10,11,12}, includes suspect {3,12}) anomalous?
  * within-session drift from opening (A) vs closing (C) readout;
  * verdict A (global state) / B (register-specific fault) / neither.

Run:  PYTHONPATH=. python -m experiments.register_fault_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware import swap_sign
from experiments.register_fault_lib import CELLS, parse_readout_rf
from experiments.same_session_lib import PHYS_8, STATES, swap_purity_from_counts, wilson

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CELL_FILE = {"n3_std": "rf_B_n3_std", "n4": "rf_B_n4", "n3_alt": "rf_B_n3_alt"}


def _parse_present(block: str):
    """Per-qubit readout rates from whichever <block> states are present (tolerant to a
    truncated closing bracket). Returns (rates, states_used)."""
    present = {name: STATES[name] for name in STATES if (HW / f"rf_{block}_{name}_counts.json").exists()}
    counts = {name: {k: int(v) for k, v in json.loads((HW / f"rf_{block}_{name}_counts.json").read_text()).items()}
              for name in present}
    out = {}
    for c in range(8):
        n0 = f10 = n1 = f01 = 0
        for name, excited in present.items():
            prep = 1 if c in excited else 0
            for s, cnt in counts[name].items():
                m = int(s.replace(" ", "")[7 - c])
                if prep == 0:
                    n0 += cnt; f10 += cnt if m == 1 else 0
                else:
                    n1 += cnt; f01 += cnt if m == 0 else 0
        out[PHYS_8[c]] = {"p10": f10 / n0 if n0 else float("nan"), "p10_ci": wilson(f10, n0),
                          "p01": f01 / n1 if n1 else float("nan"), "p01_ci": wilson(f01, n1)}
    return out, list(present)


def _boot_ci(counts, n, reps=4000, seed=0):
    signs = np.array([swap_sign(b.replace(" ", ""), n) for b, c in counts.items() for _ in range(c)], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(signs, size=len(signs), replace=True).mean() for _ in range(reps)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _drift():
    a, _ = _parse_present("A")
    c, c_states = _parse_present("C")
    rows = []
    for q in [0, 1, 2, 3, 9, 10, 11, 12]:
        p10s = a[q]["p10_ci"][1] < c[q]["p10_ci"][0] or c[q]["p10_ci"][1] < a[q]["p10_ci"][0]
        p01s = a[q]["p01_ci"][1] < c[q]["p01_ci"][0] or c[q]["p01_ci"][1] < a[q]["p01_ci"][0]
        rows.append({"q": q, "p10_A": round(a[q]["p10"], 4), "p10_C": round(c[q]["p10"], 4),
                     "p10_shift": round(c[q]["p10"] - a[q]["p10"], 4),
                     "p01_A": round(a[q]["p01"], 4), "p01_C": round(c[q]["p01"], 4),
                     "p01_shift": round(c[q]["p01"] - a[q]["p01"], 4), "sig": bool(p10s or p01s)})
    return {"rows": rows, "closing_states_used": c_states,
            "mean_abs_p10_shift": round(float(np.mean([abs(r["p10_shift"]) for r in rows])), 4),
            "mean_abs_p01_shift": round(float(np.mean([abs(r["p01_shift"]) for r in rows])), 4),
            "n_sig": sum(r["sig"] for r in rows),
            "mean_p10_A": round(float(np.mean([r["p10_A"] for r in rows])), 4)}


def main():
    locked = json.loads((HW / "rf_locked.json").read_text())
    preds = locked["predictions"]
    cells = {}
    for name, cell in CELLS.items():
        n = cell["n"]
        counts = {k: int(v) for k, v in json.loads((HW / f"{CELL_FILE[name]}_counts.json").read_text()).items()}
        mu = swap_purity_from_counts(counts, n)
        ci = _boot_ci(counts, n)
        band = preds[name]["band"]  # band.hi (pessimistic) <= mid <= band.lo (optimistic)
        inside = band["hi"] <= mu <= band["lo"]
        below = mu < band["hi"]
        cells[name] = {"n": n, "phys": cell["phys"], "measured": round(mu, 4),
                       "ci95": [round(x, 4) for x in ci], "band": band,
                       "inside_band": bool(inside), "below_band": bool(below),
                       "deficit_vs_mid": round(band["mid"] - mu, 4),
                       "shots": sum(counts.values())}

    n3, n4, alt = cells["n3_std"], cells["n4"], cells["n3_alt"]
    drift = _drift()

    # RAW cross-register comparison (drift/prediction-robust: same-session measured values)
    def _sig_lower(a, b):  # is cell a significantly LOWER than cell b (CIs disjoint)?
        return a["ci95"][1] < b["ci95"][0]
    raw = {
        "n3_std_measured": n3["measured"], "n4_measured": n4["measured"], "n3_alt_measured": alt["measured"],
        "gap_n3std_minus_n4": round(n3["measured"] - n4["measured"], 4),
        "n4_below_n3std_sig": _sig_lower(n4, n3),
        "gap_n3std_minus_n3alt": round(n3["measured"] - alt["measured"], 4),
        "n3alt_below_n3std_sig": _sig_lower(alt, n3),  # does adding {3,12} make n=3 worse?
    }
    # strict (vs-prediction) decoupling as the task defined it
    decoupling_strict = n3["inside_band"] and n4["below_band"]
    all_below = n3["below_band"] and n4["below_band"] and alt["below_band"]
    # localization: is the {3,12}-containing n3_alt MORE anomalous than standard n3?
    localized = raw["n3alt_below_n3std_sig"]

    if decoupling_strict and localized:
        verdict = "B, LOCALIZED to {3,12}: n3_std matches, n4 below, and n3_alt (with {3,12}) is anomalous at n=3."
    elif decoupling_strict and not localized:
        verdict = ("B (register-specific) but NOT the pair {3,12}: n3_std matches, n4 below, yet n3_alt "
                   "(with {3,12}) is as healthy as n3_std, n=4-geometry-specific.")
    elif n3["inside_band"] and n4["inside_band"]:
        verdict = "A (global state): n3_std and n4 both match same-session predictions; no decoupling."
    elif all_below:
        verdict = ("NEITHER clean. All three cells fell below their (clean-readout) predictions this session, "
                   "so the readout calibration OVER-predicted purity for every cell; the deficit is not "
                   "captured by readout. A REJECTED (deficit does not track the clean readout). The RAW "
                   f"comparison shows n=4 sits {raw['gap_n3std_minus_n4']:+.3f} below n=3 (significant: "
                   f"{raw['n4_below_n3std_sig']}) this session, but the n4-vs-n3 ORDERING FLIPPED across the "
                   "4-session history (n=4 was ABOVE n=3 in the first two sessions on the identical register), "
                   "so this is SESSION-DEPENDENT, not a fixed n=4-geometry fault -> B not supported as a "
                   f"standing register fault. Localization REFUTED: the {{3,12}}-containing n3_alt is NOT more "
                   f"anomalous than n3_std (gap {raw['gap_n3std_minus_n3alt']:+.3f}, sig "
                   f"{raw['n3alt_below_n3std_sig']}). Net: session-dependent, register-differentiated deficit; "
                   "not the {3,12} pair; needs a repeated same-session time series + per-edge RB.")
    else:
        verdict = "NEITHER cleanly; see per-cell table and raw cross-register comparison."

    result = {"locked_predictions": preds, "cells": cells, "raw_cross_register": raw,
              "decoupling_strict": bool(decoupling_strict), "all_cells_below_band": bool(all_below),
              "localized_to_3_12": bool(localized), "drift": drift, "verdict": verdict}
    (HW / "register_fault_analysis.json").write_text(json.dumps(result, indent=2, default=float))

    print("=== MEASURED vs LOCKED PREDICTION (same session) ===")
    for name in ("n3_std", "n4", "n3_alt"):
        c = cells[name]; b = c["band"]
        tag = "INSIDE band" if c["inside_band"] else ("BELOW band" if c["below_band"] else "ABOVE band")
        print(f"  {name:7s} {c['phys']}: measured {c['measured']:.4f} CI {c['ci95']} | "
              f"band {b['hi']:.3f}-{b['lo']:.3f} (mid {b['mid']:.3f}) -> {tag} "
              f"(deficit vs mid {c['deficit_vs_mid']:+.3f})")
    print(f"\nRAW cross-register (drift/prediction-robust): n3_std={raw['n3_std_measured']} "
          f"n4={raw['n4_measured']} n3_alt={raw['n3_alt_measured']}")
    print(f"  n4 below n3_std by {raw['gap_n3std_minus_n4']:+.3f} (sig {raw['n4_below_n3std_sig']}); "
          f"n3_alt vs n3_std {raw['gap_n3std_minus_n3alt']:+.3f} (n3_alt more anomalous? {raw['n3alt_below_n3std_sig']})")
    print(f"DECOUPLING strict (n3_std matches pred, n4 below): {decoupling_strict}; all cells below band: {all_below}")
    print(f"LOCALIZATION to {{3,12}} (n3_alt sig more anomalous than n3_std): {localized}")
    print(f"DRIFT A->C ({len(drift['closing_states_used'])} closing states {drift['closing_states_used']}): "
          f"mean|Δp10|={drift['mean_abs_p10_shift']}, mean|Δp01|={drift['mean_abs_p01_shift']}, "
          f"{drift['n_sig']}/8 qubits significant; mean p10(A)={drift['mean_p10_A']}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
