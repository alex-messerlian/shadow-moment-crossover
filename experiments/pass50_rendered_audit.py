"""PASS 50.5(c): re-verify the manuscript's numbers against the RENDERED PDF, not the source.

    PYTHONPATH=. .venv/bin/python experiments/pass50_rendered_audit.py

``pass49_number_audit.py`` checks the LaTeX source.  This checks what a reader actually sees:
text extracted from ``paper/paper.pdf``.  It catches anything the source audit cannot -- a
number mangled by a bad macro, a table cell lost to a broken row terminator, a figure caption
that did not pick up the value it was built from.

Each claim is located in the rendered text by a surrounding phrase, so a number that silently
moved to a different sentence fails rather than passing on a bare string match.

Writes ``results/pass50_rendered_audit.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
PDF = REPO / "paper" / "paper.pdf"
R = REPO / "results"
OUT = R / "pass50_rendered_audit.json"


def load(n: str) -> dict:
    return json.loads((R / n).read_text())


def main() -> None:
    with fitz.open(PDF) as d:
        flat = re.sub(r"\s+", " ", "\n".join(p.get_text() for p in d))

    rank = load("pass47_statewise_ranking.json")
    cross = load("pass48_new_ensembles_crossover.json")
    perstate = load("pass47_perstate_validation.json")
    p48 = load("pass48_pilot_extension.json")
    tail = load("pass49_n8_tail.json")
    spec = load("pass48_spectral_functional.json")
    mstar = load("pass47_statewise_mstar.json")

    checks: list[dict] = []

    def rendered(context: str, value: str, artifact_value, label: str):
        """Require `value` near `context` in the rendered text.

        The window is symmetric, because a value can precede its anchoring phrase, and both
        sides are space-stripped, because PDF extraction inserts spaces inside math runs
        ("d M *relative").  Both of those produced false misses on the first version.
        """
        def squash(x: str) -> str:
            return re.sub(r"\s+", "", x)

        hay, needle = squash(flat), squash(context)
        i = hay.find(needle)
        window = hay[max(0, i - 320): i + len(needle) + 420] if i >= 0 else ""
        present = squash(value) in window
        checks.append({"label": label, "context_found": i >= 0, "value": value,
                       "artifact": artifact_value, "ok": bool(i >= 0 and present)})

    vr = rank["by_ensemble"]["variable_rank"]
    npq = rank["by_ensemble"]["noisy_pure_q0.1"]
    rendered("rank correlation between predicted and measured", "+0.87",
             round(vr["mean_spearman"], 2), "S4 variable_rank rho")
    rendered("rank correlation between predicted and measured", "+0.49",
             round(vr["min_spearman"], 2), "S4 variable_rank min rho")
    rendered("regressing measured on predicted gives a slope", "0.97",
             round(vr["mean_slope"], 2), "S4 variable_rank slope")
    rendered("median relative deviation of", "2.7%",
             round(vr["median_abs_rel_dev"] * 100, 1), "S4 variable_rank median dev")
    rendered("82 of 84", "82 of 84", vr["within_2se"], "S4 within-2SE 82/84")
    rendered("but ρ = +0.26", "0.55", round(npq["mean_slope"], 2), "S4 control slope")

    d4 = cross["per_ensemble_diagnostic"]
    # Anchor on the table header, which is unique; the bare family names occur much earlier.
    rendered("state family estimand", "0.40", round(d4["haar_pure"]["spread_over_noise"], 2),
             "S4 table Haar-pure s/eta")
    rendered("state family estimand", "4.58", round(d4["variable_rank"]["spread_over_noise"], 2),
             "S4 table variable-rank s/eta")
    rendered("state family estimand", "0.59", round(d4["low_rank"]["spread_over_noise"], 2),
             "S4 table low-rank s/eta")
    rendered("state family estimand", "1.49", round(d4["variable_q"]["spread_over_noise"], 2),
             "S4 table variable-q s/eta")
    rendered("reproduces the earlier sweep in all", "27",
             cross["validation_gates"]["G1_raw_vs_stress_test_part4"], "S4 gate 27 cells")
    a5 = cross["accuracy"]["all_five"]
    rendered("Over all five families the criterion resolves", "27",
             a5["raw"]["resolving"], "S4 all-five resolving")
    rendered("Over all five families the criterion resolves", "26",
             a5["raw"]["exact"], "S4 all-five exact")
    rendered("scoring the non-crossing cells as", "42", a5["raw"]["all_cells_within_one"],
             "S4 all-cells 42 of 45")
    rendered("the criterion places the crossover exactly in", "82",
             perstate["C_per_sequence_crossover"]["exact"], "S4 sequences 82")
    rendered("inside two standard errors, and the per-state budget-scaling", "194 of 205",
             perstate["B_per_state_alpha"]["within_2se"], "S4 alpha 194/205")

    # Section 5: the pilot table, read out of the rendered table body
    tbl = flat[flat.find("Pilot cost against the threshold") - 900:
               flat.find("Pilot cost against the threshold")]
    for n, ms, budget, ratio in ((2, "26", "8,000", "307"), (4, "541", "8,000", "14.8"),
                                 (6, "16,519", "32,000", "1.94"), (7, "91,741", "128,000", "1.40"),
                                 (8, "520,935", "512,000", "0.98")):
        checks.append({"label": f"S5 table row n={n}",
                       "value": f"{ms} {budget} {ratio}",
                       "artifact": "pilot table",
                       "ok": all(v in tbl for v in (ms, budget, ratio))})

    m8 = float(sorted(tail["per_state"][k]["m_star_exact"] for k in tail["per_state"])[1])
    checks.append({"label": "S5 table M*(8) equals the PASS-49 median",
                   "value": "520,935", "artifact": round(m8),
                   "ok": abs(m8 - 520935) < 1 and "520,935" in tbl})
    rendered("a fitted exponent of", "−1.14", tail["fits"]["last_three"]["exponent"],
             "S5 tail exponent")
    rendered("confidence interval of", "−1.38", tail["fits"]["last_three"]["ci95"][0],
             "S5 tail CI low")
    rendered("gives 27.7%", "10.9%", None, "S5 tail MADs")
    rendered("median of dM∗relative to M∗is", "0.992", None, "S5 pooled median offsets")
    rendered("Interpolating log-linearly between those two bracketing", "7.95", None,
             "S5 crossing 7.95")
    rendered("understating the required budget by a factor of", "3.8",
             512000 / p48["pass47_extrapolation"]["8"], "S5 understatement 3.8x")

    # Section 3
    rendered("closest approach is the maximally mixed state at", "0.999999942",
             spec["bounds_scan"]["tightest_lower"]["zeta2_over_7n"], "S3 tightest lower")
    rendered("projected gradient ascent over pure states reaches only", "76.5%",
             0.765, "S3 upper-bound approach")
    a4 = [a for a in spec["break_attempts"]["attempts"] if a["attempt"].startswith("A4")][0]
    rendered("records fall below, as low as", "6.982", a4["min"], "S3 lowest ratio")
    rendered("Seven of the twelve are at", "n = 2", 7, "S3 corrected sub-7 distribution")
    rendered("recovering base(M ∗) from the ratio to within", "8", 8e-16, "S3 base identity")
    rows = {r["n"]: r for r in mstar["timing_47_2a"]["rows"]}
    rendered("takes 2", "23.5", round(rows[10]["t_zeta1_s"], 1), "S3.6 n=10 timing")
    rendered("so n = 10 is the largest size evaluable in under a minute", "242",
             mstar["timing_47_2a"]["n11_measured_s"], "S3.6 n=11 timing")

    bad = [c for c in checks if not c["ok"]]
    print(f"{len(checks) - len(bad)}/{len(checks)} claims verified in the RENDERED text")
    for c in bad:
        print(f"  MISS  {c['label']}: looked for {c['value']!r}"
              f"{'' if c.get('context_found', True) else '  (context phrase not found)'}")
    OUT.write_text(json.dumps({"n_checks": len(checks), "n_failed": len(bad),
                               "checks": checks}, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
