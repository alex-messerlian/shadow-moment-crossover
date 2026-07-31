"""PASS 36: assemble the clipping audit into one artifact and print the verdict.

Reads the three PASS-36 result files and writes
``results/pass36_clipping_audit_summary.json``.  Adds one comparison the other
scripts do not make: the RMSE of a data-free constant guess inside the physical
range.  An estimator whose RMSE exceeds that of a constant is carrying no usable
information, which is a metric-independent way to say "unusable" -- unlike a raw
RMSE, it cannot be inflated by an unbounded estimator.

Run:  PYTHONPATH=. python -m experiments.build_clipping_verdict
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
HEADLINE_NS = (2, 4, 6, 8, 10)


def main() -> None:
    audit = json.loads((R / "pass36_clipping_audit.json").read_text())
    grid = json.loads((R / "pass36_clipping_grid.json").read_text())
    paired = json.loads((R / "pass36_clipping_paired_grid.json").read_text())
    hardened = json.loads((R / "scaling_hardened.json").read_text())

    by_n = {int(k): v for k, v in audit["by_n"].items()}

    # collective RMSE at the paper's headline configuration (dephasing, g=0.05)
    coll = {r["n"]: r["collective_rmse"] for r in hardened["rows"]
            if r.get("ensemble") == "noisy_pure" and r["noise_model"] == "dephasing"
            and abs(r["rate"] - 0.05) < 1e-12}

    print("=" * 78)
    print("36.2  HEADLINE SEQUENCE, noisy-pure q=0.1, M=2000, 48 states x 10 trials")
    print("=" * 78)
    print(f"  {'n':>3} {'true':>7} {'RAW':>10} {'CLIPPED':>9} {'SHRUNK':>9} "
          f"{'outside':>9} {'clip rel':>9} {'collective':>11}")
    for n in HEADLINE_NS:
        d = by_n[n]
        print(f"  {n:>3} {d['true_purity']:>7.4f} {d['rmse_raw']:>10.4f} "
              f"{d['rmse_clipped']:>9.4f} {d['rmse_shrunk']:>9.4f} "
              f"{100*d['frac_outside_range']:>8.1f}% {d['rel_err_clipped']:>8.1%} "
              f"{coll.get(n, float('nan')):>11.4f}")

    # ---- data-free baselines
    print("\n" + "=" * 78)
    print("36.4(d)  IS THE CLIPPED ESTIMATOR STILL UNUSABLE?  vs a data-free guess")
    print("=" * 78)
    baselines = {}
    print(f"  {'n':>3} {'clipped RMSE':>13} {'const-0.5 RMSE':>15} {'minimax':>9} "
          f"{'clip rel err':>13}  verdict")
    for n in sorted(by_n):
        d = by_n[n]
        lo, hi = d["physical_range"]
        mu = d["true_purity"]
        const_mid = abs(mu - 0.5 * (lo + hi))     # always guess the interval midpoint
        minimax = 0.5 * (hi - lo)                  # worst-case RMSE of that guess
        beats = d["rmse_clipped"] < const_mid
        baselines[n] = {"const_midpoint_rmse": const_mid, "minimax_rmse": minimax,
                        "clipped_beats_constant": bool(beats)}
        print(f"  {n:>3} {d['rmse_clipped']:>13.4f} {const_mid:>15.4f} {minimax:>9.4f} "
              f"{d['rel_err_clipped']:>12.1%}  "
              f"{'informative' if beats else 'WORSE THAN A CONSTANT GUESS'}")

    # ---- verdict fields
    n10 = by_n[10]
    fifteen_x_raw = n10["rel_err_raw"]
    fifteen_x_clip = n10["rel_err_clipped"]

    ptab = {(e["noise_model"], e["rate"]): e["crossover_n"]
            for e in paired["crossover_tables"]["raw"]}
    ctab = {(e["noise_model"], e["rate"]): e["crossover_n"]
            for e in paired["crossover_tables"]["clipped"]}
    stab = {(e["noise_model"], e["rate"]): e["crossover_n"]
            for e in paired["crossover_tables"]["shrunk"]}

    shifts = [ctab[k] - ptab[k] for k in ptab
              if ptab.get(k) is not None and ctab.get(k) is not None]

    summary = {
        "description": "PASS 36 verdict: how much of the crossover result survives a "
                       "range-constrained single-copy estimator",
        "paper_claim_checked": "single-copy purity RMSE 0.043, 0.072, 0.270, 1.62, 11.98 at "
                               "n = 2,4,6,8,10; 'fifteen times the quantity being estimated'",
        "headline_table": {str(n): {
            "true_purity": by_n[n]["true_purity"],
            "rmse_raw": by_n[n]["rmse_raw"],
            "rmse_clipped": by_n[n]["rmse_clipped"],
            "rmse_shrunk": by_n[n]["rmse_shrunk"],
            "frac_outside_physical_range": by_n[n]["frac_outside_range"],
            "rel_err_raw": by_n[n]["rel_err_raw"],
            "rel_err_clipped": by_n[n]["rel_err_clipped"],
            "collective_rmse_dephasing_g0.05": coll.get(n),
        } for n in sorted(by_n)},
        "data_free_baselines": baselines,
        "fifteen_times_claim": {
            "raw_relative_error_n10": fifteen_x_raw,
            "clipped_relative_error_n10": fifteen_x_clip,
            "raw_multiple_of_estimand": fifteen_x_raw,
            "clipped_multiple_of_estimand": fifteen_x_clip,
            "survives": bool(fifteen_x_clip >= 15.0),
        },
        "clipping_is_free_check": {
            "samples_checked": sum(by_n[n]["n_samples"] for n in by_n),
            "clip_increased_squared_error_count":
                sum(by_n[n]["clip_increased_sq_error_count"] for n in by_n),
            "shrink_increased_squared_error_count":
                sum(by_n[n]["shrink_increased_sq_error_count"] for n in by_n),
        },
        "crossover_paired_k2_noisy_pure": {
            "raw": {f"{a}@{b}": v for (a, b), v in ptab.items()},
            "clipped": {f"{a}@{b}": v for (a, b), v in ctab.items()},
            "shrunk": {f"{a}@{b}": v for (a, b), v in stab.items()},
            "shift_raw_to_clipped": shifts,
            "mean_shift": float(np.mean(shifts)) if shifts else None,
            "validation": paired["validation"],
        },
        "crossover_grid_96_cells": {
            "validation_raw_reproduces_committed":
                grid["validation_raw_reproduces_committed"],
            "resolved": {kind: sum(1 for r in grid["rows"]
                                   if r[f"crossover_{kind}"] is not None)
                         for kind in ("raw", "clipped", "shrunk")},
            "n_cells": len(grid["rows"]),
        },
    }
    (R / "pass36_clipping_audit_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\nwrote {R / 'pass36_clipping_audit_summary.json'}")


if __name__ == "__main__":
    main()
