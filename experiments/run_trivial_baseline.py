"""PASS 37.3: both routes against a data-free constant, the replacement headline.

The paper opened, at the time this was written, with "an error fifteen times the
quantity being estimated".  That number belongs to the unconstrained U-statistic; a
clipped estimator cannot exceed the width of the physical range (PASS 36).  The claim
that survives is stronger and metric-independent: past a certain size the single-copy
estimator is beaten by a rule that never looks at the data.  It is what the paper now
opens with.

Three constants are scored, and they are not equally legitimate:

  MIDPOINT   c = (2^{-n} + 1)/2.  Uses only the physical range, which is known a
             priori.  This is the honest data-free competitor, and it is minimax
             optimal over the range.
  ENSEMBLE   c = mean_s Tr(rho_s^2) over the benchmark states.  Knowing that mean
             is knowing the answer, so this is NOT a fair competitor; it is
             reported only to show how concentrated the ensemble is.
  ORACLE     c = the per-cell true value.  RMSE 0 by construction; listed to make
             explicit that "the constant minimizing RMSE against the true value"
             is degenerate and cannot be the intended baseline.

Writes ``results/pass37_trivial_baseline.json``.
Run:  PYTHONPATH=. python -m experiments.run_trivial_baseline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
K = 2


def main() -> None:
    audit = json.loads((R / "pass36_clipping_audit.json").read_text())
    units = audit["raw_units"]
    hardened = json.loads((R / "scaling_hardened.json").read_text())
    coll = {r["n"]: r["collective_rmse"] for r in hardened["rows"]
            if r.get("ensemble") == "noisy_pure" and r["noise_model"] == "dephasing"
            and abs(r["rate"] - 0.05) < 1e-12}

    rows = []
    for n in sorted({u["n"] for u in units}):
        us = [u for u in units if u["n"] == n]
        truths = np.asarray([u["truth"] for u in us])
        lo, hi = 2.0 ** (n * (1 - K)), 1.0
        mid = 0.5 * (lo + hi)
        d = audit["by_n"][str(n)]
        # a constant c is evaluated against every state, as the estimators are
        rmse_mid = float(np.sqrt(np.mean((mid - truths) ** 2)))
        c_best = float(truths.mean())
        rmse_ens = float(np.sqrt(np.mean((c_best - truths) ** 2)))
        rows.append({
            "n": n,
            "true_purity_mean": float(truths.mean()),
            "true_purity_std": float(truths.std()),
            "physical_range": [lo, hi],
            "single_raw": d["rmse_raw"],
            "single_clipped": d["rmse_clipped"],
            "single_shrunk": d["rmse_shrunk"],
            "collective": coll.get(n),
            "const_midpoint": mid,
            "rmse_const_midpoint": rmse_mid,
            "rmse_const_ensemble_mean": rmse_ens,
            "rmse_const_oracle": 0.0,
            "clipped_beats_midpoint": bool(d["rmse_clipped"] < rmse_mid),
            "collective_beats_midpoint": bool(coll.get(n, np.inf) < rmse_mid),
        })

    print("=" * 92)
    print("37.3  BOTH ROUTES vs A DATA-FREE CONSTANT   (noisy-pure q=0.1, M=2000, "
          "collective: dephasing g=0.05)")
    print("=" * 92)
    print(f"  {'n':>3}{'single RAW':>12}{'single CLIP':>13}{'collective':>12}"
          f"{'const-mid':>11}{'const-ens':>11}   who beats const-mid")
    for r in rows:
        who = []
        if r["clipped_beats_midpoint"]:
            who.append("single(clip)")
        if r["collective_beats_midpoint"]:
            who.append("collective")
        print(f"  {r['n']:>3}{r['single_raw']:>12.4f}{r['single_clipped']:>13.4f}"
              f"{r['collective']:>12.4f}{r['rmse_const_midpoint']:>11.4f}"
              f"{r['rmse_const_ensemble_mean']:>11.4f}   {', '.join(who) or 'NEITHER'}")

    cross_single = next((r["n"] for r in rows if not r["clipped_beats_midpoint"]), None)
    cross_coll = next((r["n"] for r in rows if not r["collective_beats_midpoint"]), None)
    print(f"\n  Clipped single-copy stops beating the data-free constant at n = {cross_single}")
    print(f"  Collective stops beating it at n = {cross_coll if cross_coll else 'never in range'}")

    (R / "pass37_trivial_baseline.json").write_text(json.dumps({
        "description": "PASS 37.3: single-copy (raw/clipped) and collective against "
                       "data-free constant guesses",
        "baselines": {
            "midpoint": "c = (2^-n + 1)/2, uses only the physical range; minimax optimal; "
                        "the honest data-free competitor",
            "ensemble_mean": "c = mean true purity; NOT a fair competitor (knowing it is "
                             "knowing the answer); shows ensemble concentration",
            "oracle": "c = the true value; RMSE 0 by construction; degenerate",
        },
        "single_copy_stops_beating_midpoint_at_n": cross_single,
        "collective_stops_beating_midpoint_at_n": cross_coll,
        "rows": rows,
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass37_trivial_baseline.json'}")


if __name__ == "__main__":
    main()
