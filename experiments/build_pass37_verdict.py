"""PASS 37.4: assemble the two gap closures into results/pass37_gaps.json.

Computes the paper's own accuracy figures -- "82 of the 83 within one qubit, 73
exact" and "95.9% of all 123 swept cells" -- under
  (i)  each estimator (RAW / CLIPPED / SHRUNK), and
  (ii) the measured-crossover rule Appendix C actually describes, as the
       counterfactual cost of "fix the code instead of the prose".

Run:  PYTHONPATH=. python -m experiments.build_pass37_verdict
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def score(pairs):
    """(within-one, exact, n_resolved) over (predicted, measured) pairs."""
    res = [(p, m) for p, m in pairs if p is not None and m is not None]
    w1 = sum(1 for p, m in res if abs(p - m) <= 1)
    ex = sum(1 for p, m in res if p == m)
    return w1, ex, len(res)


def main() -> None:
    rule = json.loads((R / "pass37_rule_audit.json").read_text())
    heldout = json.loads((R / "pass37_heldout_clipping.json").read_text())
    npath = R / "pass37_noisypure_clipping.json"
    noisy = json.loads(npath.read_text()) if npath.exists() else None
    ct = json.loads((R / "crossover_theory.json").read_text())["comparison"]
    trivial = json.loads((R / "pass37_trivial_baseline.json").read_text())

    # The two noisy-pure sweeps share 18 (k, noise, rate, budget) keys, so every
    # map below is keyed by SOURCE as well.  crossover_theory.json's comparison is
    # built as moment_sweep_corrected (36 rows) followed by budget_scaling (60).
    n_ms = sum(1 for r in rule["all_cells"] if r["source"] == "moment_sweep_corrected")
    assert len(ct) == len(rule["all_cells"]), (len(ct), len(rule["all_cells"]))
    pred_np = {}
    for i, c in enumerate(ct):
        src = "moment_sweep_corrected" if i < n_ms else "budget_scaling"
        pred_np[(src, c["k"], c["noise_model"], c["rate"], c["budget"])] = c["predicted_n_exact"]

    comm_np, appx_np = {}, {}
    for r in rule["all_cells"]:
        key = (r["source"], r["k"], r["noise_model"], r["rate"], r["budget"])
        comm_np[key] = r["committed_measured_n"]
        appx_np[key] = r["appendix_c_rule_n"]
    assert len(comm_np) == len(pred_np) == 96, (len(comm_np), len(pred_np))

    est_np = {}
    if noisy is not None:
        for kind in ("raw", "clipped", "shrunk"):
            m = {}
            for e in noisy["crossover_tables"][kind]["budget"]:
                m[("budget_scaling", e["k"], e["noise_model"], e["rate"],
                   e["budget"])] = e["crossover_n"]
            for e in noisy["crossover_tables"][kind]["moment"]:
                m[("moment_sweep_corrected", e["k"], e["noise_model"], e["rate"],
                   2000)] = e["crossover_n"]
            est_np[kind] = m

    # Fallback / cross-check: the analytic RMSE-rule grid from PASS 36, applied
    # IDENTICALLY to all three estimators.  Its absolute baseline differs from the
    # committed one on the 8 rule cells of 37.1, but the SHIFT it reports is
    # apples-to-apples because one rule is used throughout.
    grid = json.loads((R / "pass36_clipping_grid.json").read_text())
    alt_np = {kind: {} for kind in ("raw", "clipped", "shrunk")}
    for r in grid["rows"]:
        key = (r["source"], r["k"], r["noise_model"], r["rate"], r["budget"])
        for kind in alt_np:
            alt_np[kind][key] = r[f"crossover_{kind}"]

    # held-out: predicted (RULE 1, unchanged) and measured under each estimator
    ho_pred = {(r["ensemble"], r["noise"], r["rate"]): r["predicted_n"] for r in heldout["rows"]}
    ho_est = {kind: {(r["ensemble"], r["noise"], r["rate"]): r[f"measured_n_{kind}"]
                     for r in heldout["rows"]} for kind in ("raw", "clipped", "shrunk")}

    N_SWEPT = 123
    out_scores = {}

    def report(label, np_measured, ho_measured):
        pairs = ([(pred_np.get(k), np_measured.get(k)) for k in comm_np]
                 + [(ho_pred.get(k), ho_measured.get(k)) for k in ho_pred])
        w1, ex, nres = score(pairs)
        # all-cells figure: an unresolved cell counts as correct when BOTH sides agree
        allcells_w1 = w1 + sum(1 for p, m in pairs if p is None and m is None)
        allcells_ex = ex + sum(1 for p, m in pairs if p is None and m is None)
        out_scores[label] = {
            "n_resolved": nres, "within_one": w1, "exact": ex,
            "within_one_pct_of_resolved": w1 / nres if nres else None,
            "exact_pct_of_resolved": ex / nres if nres else None,
            "all_cells_within_one": allcells_w1,
            "all_cells_within_one_pct": allcells_w1 / N_SWEPT,
        }
        print(f"  {label:<28} resolved {nres:>3}   within-1 {w1:>3}/{nres:<3} "
              f"({w1/nres:.1%})   exact {ex:>3}/{nres:<3} ({ex/nres:.1%})   "
              f"all-cells within-1 {allcells_w1}/{N_SWEPT} = {allcells_w1/N_SWEPT:.1%}")

    print("=" * 104)
    print("37.2(c)  THE PAPER'S ACCURACY FIGURES UNDER EACH ESTIMATOR")
    print("  paper reports: 83 resolved, 82 within one (98.8%), 73 exact (88.0%), "
          "95.9% of 123 all-cells")
    print("=" * 104)
    report("committed (paper)", comm_np, ho_est["raw"])
    if est_np:
        print("  -- exact pipeline, RULE 2 paired test, PASS-37 re-run --")
        for kind in ("raw", "clipped", "shrunk"):
            report(f"exact: {kind.upper()}", est_np[kind], ho_est[kind])
    else:
        print("  -- exact noisy-pure re-run not available; analytic path only --")
    print("  -- analytic RMSE rule, one rule throughout (noisy-pure) + exact held-out --")
    for kind in ("raw", "clipped", "shrunk"):
        report(f"analytic: {kind.upper()}", alt_np[kind], ho_est[kind])

    # ---- the consistent comparison: criterion re-derived for a clipped estimator
    cp = json.loads((R / "pass37_clipped_prediction.json").read_text())["rows"]
    pred_clip = {(r["source"], r["k"], r["noise_model"], r["rate"], r["budget"]):
                 r["predicted_clipped"] for r in cp}
    pairs_consistent = ([(pred_clip.get(k), alt_np["clipped"].get(k)) for k in comm_np]
                        + [(ho_pred.get(k), ho_est["clipped"].get(k)) for k in ho_pred])
    w1, ex, nres = score(pairs_consistent)
    both_none = sum(1 for p, m in pairs_consistent if p is None and m is None)
    out_scores["consistent: CLIPPED pred vs CLIPPED meas"] = {
        "n_resolved": nres, "within_one": w1, "exact": ex,
        "within_one_pct_of_resolved": w1 / nres, "exact_pct_of_resolved": ex / nres,
        "all_cells_within_one": w1 + both_none,
        "all_cells_within_one_pct": (w1 + both_none) / N_SWEPT,
        "note": "noisy-pure prediction re-derived for a clipped estimator; the 27 "
                "held-out predictions are left raw (not re-derived)",
    }
    print("\n  -- CONSISTENT: criterion re-derived for a clipped estimator, both sides --")
    print(f"  {'consistent: CLIPPED/CLIPPED':<28} resolved {nres:>3}   "
          f"within-1 {w1:>3}/{nres:<3} ({w1/nres:.1%})   exact {ex:>3}/{nres:<3} "
          f"({ex/nres:.1%})   all-cells within-1 {w1+both_none}/{N_SWEPT} = "
          f"{(w1+both_none)/N_SWEPT:.1%}")

    print("\n" + "=" * 104)
    print("37.1(d)  COUNTERFACTUAL: if the CODE were changed to match Appendix C's prose")
    print("=" * 104)
    report("Appendix-C rule (measured)", appx_np, ho_est["raw"])

    summary = {
        "description": "PASS 37: crossover-rule audit + held-out ensembles under clipping",
        "gap_1_crossover_rule": {
            "appendix_c_text": "the smallest size from which the single-copy RMSE exceeds "
                               "the collective RMSE and stays above it for the remainder of "
                               "the swept range (the sustained crossover of predict_crossover)",
            "rules_actually_used": rule["rules"],
            "n_noisy_pure_cells": rule["n_cells"],
            "n_disagree_with_appendix_c": rule["n_disagree"],
            "disagreeing_cells": [
                {k: c[k] for k in ("k", "noise_model", "rate", "budget",
                                   "committed_measured_n", "appendix_c_rule_n")}
                for c in rule["disagreeing_cells"]],
            "diagnosis": "code correct, prose incomplete: 7 of 8 are the paired test "
                         "declining a statistically insignificant call (|z| < 1.5, RMSE "
                         "ratio < 1.06) and waiting one qubit for a decisive one "
                         "(z > 4.3); the 8th is already flagged ambiguous=True by the "
                         "pipeline",
        },
        "gap_2_heldout_under_clipping": {
            "validation_raw_reproduces_committed":
                heldout["validation_raw_reproduces_committed"],
            "per_ensemble": {},
        },
        "accuracy_scores": out_scores,
        "trivial_baseline": {
            "single_copy_stops_beating_midpoint_at_n":
                trivial["single_copy_stops_beating_midpoint_at_n"],
            "collective_stops_beating_midpoint_at_n":
                trivial["collective_stops_beating_midpoint_at_n"],
            "rows": trivial["rows"],
            "caveat": "for the noisy-pure ensemble Tr(rho^2) = 0.81 + 0.19/2^n exactly, "
                      "independent of the state, so the 'best constant' is exact and the "
                      "only meaningful data-free baseline is the range midpoint",
        },
    }
    for e in ("haar_pure", "low_rank", "ghz_noisy"):
        sel = [r for r in heldout["rows"] if r["ensemble"] == e]
        both = [r for r in sel if r["measured_n_raw"] is not None
                and r["measured_n_clipped"] is not None]
        summary["gap_2_heldout_under_clipping"]["per_ensemble"][e] = {
            "cells": len(sel),
            "resolved_raw": sum(1 for r in sel if r["measured_n_raw"] is not None),
            "resolved_clipped": sum(1 for r in sel if r["measured_n_clipped"] is not None),
            "unchanged": sum(1 for r in both
                             if r["measured_n_clipped"] == r["measured_n_raw"]),
            "shifts": [r["measured_n_clipped"] - r["measured_n_raw"] for r in both],
            "lost": sum(1 for r in sel if r["measured_n_raw"] is not None
                        and r["measured_n_clipped"] is None),
        }

    (R / "pass37_gaps.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\nwrote {R / 'pass37_gaps.json'}")


if __name__ == "__main__":
    main()
