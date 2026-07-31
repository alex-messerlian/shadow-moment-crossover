"""PASS 38.2/38.3: reconcile the exact noisy-pure re-run against the analytic path,
and print the final numbers for the revision.

PASS 37 scored the noisy-pure half of the clipping aggregate with an analytic RMSE
rule because the exact paired-test re-run had not completed.  This reads the
completed exact run and reports, cell by cell, where the two differ and why, then
emits the table the revision will print.

Writes ``results/pass38_final.json``.
Run:  PYTHONPATH=. python -m experiments.build_pass38_final
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
N_SWEPT = 123
SRC = {"budget": "budget_scaling", "moment": "moment_sweep_corrected"}


def score(pairs):
    res = [(p, m) for p, m in pairs if p is not None and m is not None]
    w1 = sum(1 for p, m in res if abs(p - m) <= 1)
    ex = sum(1 for p, m in res if p == m)
    both_none = sum(1 for p, m in pairs if p is None and m is None)
    return w1, ex, len(res), both_none


def exact_map(noisy, kind):
    m = {}
    for tab, src in SRC.items():
        for e in noisy["crossover_tables"][kind][tab]:
            budget = e.get("budget", 2000)
            m[(src, e["k"], e["noise_model"], e["rate"], budget)] = e["crossover_n"]
    return m


def main() -> None:
    noisy = json.loads((R / "pass37_noisypure_clipping.json").read_text())
    grid = json.loads((R / "pass36_clipping_grid.json").read_text())
    heldout = json.loads((R / "pass37_heldout_clipping.json").read_text())
    rule = json.loads((R / "pass37_rule_audit.json").read_text())
    cp = json.loads((R / "pass37_clipped_prediction.json").read_text())["rows"]
    triv = json.loads((R / "pass37_trivial_baseline.json").read_text())
    ct = json.loads((R / "crossover_theory.json").read_text())["comparison"]

    print("=" * 100)
    print("38.1(b)  VALIDATION FIRST")
    print("=" * 100)
    print(f"  RAW column reproduces the committed noisy-pure crossover tables in "
          f"{noisy['validation_raw_reproduces_committed']} cells")

    ex = {k: exact_map(noisy, k) for k in ("raw", "clipped", "shrunk")}
    an = {k: {} for k in ("raw", "clipped", "shrunk")}
    for r in grid["rows"]:
        key = (r["source"], r["k"], r["noise_model"], r["rate"], r["budget"])
        for k in an:
            an[k][key] = r[f"crossover_{k}"]

    print("\n" + "=" * 100)
    print("38.1(c)  NOISY-PURE CELLS UNDER EACH ESTIMATOR (exact paired test)")
    print("=" * 100)
    keys = sorted(ex["raw"], key=str)
    for kind in ("raw", "clipped", "shrunk"):
        res = sum(1 for k in keys if ex[kind][k] is not None)
        print(f"  {kind:9s}: {res}/{len(keys)} resolve")
    for kind in ("clipped", "shrunk"):
        sh = [ex[kind][k] - ex["raw"][k] for k in keys
              if ex["raw"][k] is not None and ex[kind][k] is not None]
        lost = sum(1 for k in keys if ex["raw"][k] is not None and ex[kind][k] is None)
        gained = sum(1 for k in keys if ex["raw"][k] is None and ex[kind][k] is not None)
        print(f"  raw -> {kind}: unchanged {sum(1 for d in sh if d == 0)}, "
              f"shifted {sum(1 for d in sh if d != 0)} "
              f"{dict(sorted(collections.Counter(sh).items()))}, lost {lost}, gained {gained}")

    print("\n" + "=" * 100)
    print("38.2  EXACT vs ANALYTIC, cell by cell")
    print("=" * 100)
    disagree = []
    for kind in ("raw", "clipped", "shrunk"):
        d = [(k, an[kind].get(k), ex[kind][k]) for k in keys if an[kind].get(k) != ex[kind][k]]
        print(f"  {kind:9s}: {len(keys) - len(d)}/{len(keys)} agree, {len(d)} differ")
        for k, a, e in d:
            disagree.append({"estimator": kind, "cell": list(k), "analytic": a, "exact": e})
            print(f"     k={k[1]} {k[2]}@{k[3]} M={k[4]} [{k[0]}]: analytic {a} vs exact {e}")

    # ---- aggregate scores
    n_ms = sum(1 for r in rule["all_cells"] if r["source"] == "moment_sweep_corrected")
    pred = {}
    for i, c in enumerate(ct):
        s = "moment_sweep_corrected" if i < n_ms else "budget_scaling"
        pred[(s, c["k"], c["noise_model"], c["rate"], c["budget"])] = c["predicted_n_exact"]
    pred_clip = {(r["source"], r["k"], r["noise_model"], r["rate"], r["budget"]):
                 r["predicted_clipped"] for r in cp}
    ho_pred = {(r["ensemble"], r["noise"], r["rate"]): r["predicted_n"] for r in heldout["rows"]}
    ho = {k: {(r["ensemble"], r["noise"], r["rate"]): r[f"measured_n_{k}"]
              for r in heldout["rows"]} for k in ("raw", "clipped", "shrunk")}

    scores = {}

    def add(label, p_np, m_np, m_ho):
        pairs = ([(p_np.get(k), m_np.get(k)) for k in keys]
                 + [(ho_pred.get(k), m_ho.get(k)) for k in ho_pred])
        w1, e, nres, bn = score(pairs)
        scores[label] = {"resolved": nres, "within_one": w1, "exact": e,
                         "within_one_pct": w1 / nres, "exact_pct": e / nres,
                         "all_cells_within_one": w1 + bn,
                         "all_cells_pct": (w1 + bn) / N_SWEPT}
        print(f"  {label:<34} resolved {nres:>3}  within-1 {w1:>3}/{nres:<3} "
              f"({w1/nres:>5.1%})  exact {e:>3}/{nres:<3} ({e/nres:>5.1%})  "
              f"all-cells {w1+bn}/{N_SWEPT} = {(w1+bn)/N_SWEPT:.1%}")

    print("\n" + "=" * 100)
    print("38.3  FINAL NUMBERS  (paper: 83 resolved, 82/83 = 98.8%, 73/83 = 88.0%, "
          "118/123 = 95.9%)")
    print("=" * 100)
    add("published (raw / raw)", pred, ex["raw"], ho["raw"])
    add("CLIPPED meas, raw criterion", pred, ex["clipped"], ho["clipped"])
    add("CLIPPED meas, CLIPPED criterion", pred_clip, ex["clipped"], ho["clipped"])
    add("SHRUNK meas, raw criterion", pred, ex["shrunk"], ho["shrunk"])

    hl = [e for e in noisy["crossover_tables"]["raw"]["moment"]
          if e["k"] == 2 and e["noise_model"] == "dephasing" and abs(e["rate"] - 0.05) < 1e-12]
    hl_by = {}
    for kind in ("raw", "clipped", "shrunk"):
        m = [e for e in noisy["crossover_tables"][kind]["moment"]
             if e["k"] == 2 and e["noise_model"] == "dephasing" and abs(e["rate"] - 0.05) < 1e-12]
        hl_by[kind] = m[0]["crossover_n"] if m else None
    print(f"\n  38.3(d) headline config (k=2, noisy-pure, dephasing g=0.05, M=2000): "
          f"RAW {hl_by['raw']}, CLIPPED {hl_by['clipped']}, SHRUNK {hl_by['shrunk']}")

    print("\n  38.3(e) n=2..10, headline configuration")
    print(f"  {'n':>3}{'clipped single':>16}{'collective':>12}{'const-midpoint':>16}")
    for r in triv["rows"]:
        print(f"  {r['n']:>3}{r['single_clipped']:>16.4f}{r['collective']:>12.4f}"
              f"{r['rmse_const_midpoint']:>16.4f}")
    print(f"  crossings: clipped single loses to the constant at n = "
          f"{triv['single_copy_stops_beating_midpoint_at_n']}; collective at n = "
          f"{triv['collective_stops_beating_midpoint_at_n']}")

    (R / "pass38_final.json").write_text(json.dumps({
        "description": "PASS 38: exact noisy-pure clipping re-run, reconciled against the "
                       "PASS-37 analytic path, with the final figures for the revision",
        "validation_raw_reproduces_committed":
            noisy["validation_raw_reproduces_committed"],
        "exact_vs_analytic_disagreements": disagree,
        "n_noisy_pure_cells": len(keys),
        "resolved_by_estimator": {k: sum(1 for c in keys if ex[k][c] is not None)
                                  for k in ("raw", "clipped", "shrunk")},
        "accuracy": scores,
        "headline_crossover": hl_by,
        "trivial_baseline_rows": triv["rows"],
        "crossings": {
            "single_clipped_loses_to_constant_at_n":
                triv["single_copy_stops_beating_midpoint_at_n"],
            "collective_loses_to_constant_at_n":
                triv["collective_stops_beating_midpoint_at_n"]},
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass38_final.json'}")


if __name__ == "__main__":
    main()
