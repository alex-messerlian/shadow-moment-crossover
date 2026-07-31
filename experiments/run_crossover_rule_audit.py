"""PASS 37.1: which rule produced each crossover number the paper reports.

Appendix C defines the crossover as "the smallest size from which the single-copy
RMSE exceeds the collective RMSE and stays above it for the remainder of the swept
range (the sustained crossover of ``predict_crossover``)".  That describes exactly
one of the three rules the codebase actually uses:

  RULE 1  ``anrl.theory.crossover.predict_crossover``
          both sides from theory; sustained RMSE comparison.
          -> every PREDICTED n* in the paper.

  RULE 2  ``anrl.benchmark.hardened.crossover_table``
          measured single vs measured collective, PAIRED state-level z-test,
          first n whose verdict is "collective" (no sustained requirement).
          -> the MEASURED n* of the 68 noisy-pure cells.

  RULE 3  ``experiments/run_stress_test.py`` part 4
          measured single RMSE vs the EXACT COLLECTIVE FLOOR from theory;
          sustained.  -> the MEASURED n* of the 27 held-out cells.

Rules 2 and 3 are not what Appendix C describes.  This script quantifies the
consequence: it re-scores the noisy-pure sweeps under Appendix C's stated rule
(measured-vs-measured RMSE, sustained) and lists every cell where that disagrees
with the committed measured n*, with the margin by which each flips.

Writes ``results/pass37_rule_audit.json``.
Run:  PYTHONPATH=. python -m experiments.run_crossover_rule_audit
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def _sustained(sizes, single, collective):
    ns = sorted(sizes)
    wins = {n: single[n] > collective[n] for n in ns}
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None


def _first_win(sizes, single, collective):
    """Rule-2 shape without the significance test: first n where single > collective."""
    for n in sorted(sizes):
        if single[n] > collective[n]:
            return n
    return None


def main() -> None:
    sources = [(R / "budget_scaling.json", 2000), (R / "moment_sweep_corrected.json", 2000)]
    rows = []
    for path, default_budget in sources:
        payload = json.loads(path.read_text())
        cells: dict[tuple, dict] = {}
        for r in payload["rows"]:
            key = (r["k"], r["noise_model"], float(r["rate"]),
                   int(r.get("budget", default_budget)))
            c = cells.setdefault(key, {"single": {}, "collective": {}, "z": {}, "winner": {}})
            n = int(r["n"])
            c["single"][n] = float(r["single_rmse"])
            c["collective"][n] = float(r["collective_rmse"])
            c["z"][n] = r.get("paired_z")
            c["winner"][n] = r.get("winner")
        committed = {(int(e["k"]), e["noise_model"], float(e["rate"]),
                      int(e.get("budget", default_budget))): e for e in payload["crossover_table"]}
        for key, c in cells.items():
            sizes = sorted(c["single"])
            appendix_c = _sustained(sizes, c["single"], c["collective"])
            first_rmse = _first_win(sizes, c["single"], c["collective"])
            comm = committed.get(key)
            rows.append({
                "source": path.stem, "k": key[0], "noise_model": key[1],
                "rate": key[2], "budget": key[3], "sizes": sizes,
                "committed_measured_n": None if comm is None else comm["crossover_n"],
                "committed_rule": "paired z-test, first 'collective' (RULE 2)",
                "appendix_c_rule_n": appendix_c,
                "first_rmse_crossing_n": first_rmse,
                "agrees": (None if comm is None else comm["crossover_n"]) == appendix_c,
                "single_rmse": {str(n): c["single"][n] for n in sizes},
                "collective_rmse": {str(n): c["collective"][n] for n in sizes},
                "paired_z": {str(n): c["z"][n] for n in sizes},
                "winner": {str(n): c["winner"][n] for n in sizes},
            })

    dis = [r for r in rows if not r["agrees"]]
    print(f"Noisy-pure cells scored: {len(rows)}")
    print(f"Appendix-C rule agrees with the committed measured n* in "
          f"{len(rows) - len(dis)}/{len(rows)} cells; {len(dis)} disagree\n")
    print(f"  {'k':>2} {'channel':<18}{'rate':>6}{'budget':>8}"
          f"{'committed':>10}{'AppxC':>7}  margin at the disputed size")
    for r in sorted(dis, key=lambda x: (x["k"], x["noise_model"], x["rate"], x["budget"])):
        comm, ac = r["committed_measured_n"], r["appendix_c_rule_n"]
        detail = []
        for n in sorted({x for x in (comm, ac) if x is not None}):
            s = r["single_rmse"][str(n)]
            cl = r["collective_rmse"][str(n)]
            z = r["paired_z"][str(n)]
            detail.append(f"n={n}: single {s:.4f} vs coll {cl:.4f} "
                          f"(ratio {s/cl:.3f}, z={z:+.2f}, verdict {r['winner'][str(n)]})")
        print(f"  {r['k']:>2} {r['noise_model']:<18}{r['rate']:>6}{r['budget']:>8}"
              f"{str(comm):>10}{str(ac):>7}")
        for d in detail:
            print(f"       {d}")

    (R / "pass37_rule_audit.json").write_text(json.dumps({
        "description": "PASS 37.1: crossover-rule audit. Appendix C states one rule; "
                       "the pipeline uses three.",
        "rules": {
            "RULE_1_predicted": "anrl.theory.crossover.predict_crossover -- theory vs "
                                "theory, sustained RMSE. Produces every PREDICTED n*. "
                                "This is what Appendix C describes.",
            "RULE_2_measured_noisy_pure": "anrl.benchmark.hardened.crossover_table -- "
                                          "measured vs measured, PAIRED z-test, first "
                                          "'collective' verdict, not sustained. Produces "
                                          "the MEASURED n* of the 68 noisy-pure cells.",
            "RULE_3_measured_heldout": "experiments/run_stress_test.py part 4 -- measured "
                                       "single RMSE vs the THEORY collective floor, "
                                       "sustained. Produces the MEASURED n* of the 27 "
                                       "held-out cells.",
        },
        "n_cells": len(rows), "n_disagree": len(dis),
        "disagreeing_cells": dis, "all_cells": rows,
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass37_rule_audit.json'}")


if __name__ == "__main__":
    main()
