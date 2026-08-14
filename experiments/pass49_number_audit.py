"""PASS 49.6: audit every numeric claim in the NEW prose against the committed artifacts.

    PYTHONPATH=. .venv/bin/python experiments/pass49_number_audit.py

The restructure introduced roughly a hundred new numeric tokens, in the abstract, the promoted
Section 3 material, the two new sections, and the conclusion.  Each is checked here against the
artifact it came from, so no figure in the manuscript rests on a transcription.

Writes ``results/pass49_number_audit.json``; exit status is non-zero if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
OUT = R / "pass49_number_audit.json"


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def main() -> None:
    rank = load("pass47_statewise_ranking.json")
    cross = load("pass48_new_ensembles_crossover.json")
    perstate = load("pass47_perstate_validation.json")
    pilot47 = load("pass47_pilot_estimator.json")
    pilot48 = load("pass48_pilot_extension.json")
    spec = load("pass48_spectral_functional.json")
    mstar = load("pass47_statewise_mstar.json")

    checks: list[dict] = []

    def chk(claim: str, stated, actual, tol=0.0, exact=False):
        if exact or isinstance(stated, str):
            ok = stated == actual
        else:
            ok = abs(float(stated) - float(actual)) <= tol
        checks.append({"claim": claim, "stated": stated, "artifact": actual, "ok": bool(ok)})

    # --- Section 4: statewise ---
    vr = rank["by_ensemble"]["variable_rank"]
    npq = rank["by_ensemble"]["noisy_pure_q0.1"]
    chk("S4 variable_rank mean Spearman = 0.87", 0.87, vr["mean_spearman"], 0.005)
    chk("S4 variable_rank min Spearman = 0.49", 0.49, vr["min_spearman"], 0.005)
    chk("S4 variable_rank slope = 0.97", 0.97, vr["mean_slope"], 0.005)
    chk("S4 variable_rank median |dev| = 2.7%", 0.027, vr["median_abs_rel_dev"], 0.0005)
    chk("S4 variable_rank within-2SE = 82/84", "82/84", vr["within_2se"], exact=True)
    chk("S4 variable_rank predicted spread = 21.9%", 0.219, vr["mean_predicted_spread"], 0.0005)
    chk("S4 noisy_pure Spearman = 0.26", 0.26, npq["mean_spearman"], 0.005)
    chk("S4 noisy_pure slope = 0.55", 0.55, npq["mean_slope"], 0.005)
    chk("S4 noisy_pure median |dev| = 2.4%", 0.024, npq["median_abs_rel_dev"], 0.0005)
    chk("S4 noisy_pure within-2SE = 79/84", "79/84", npq["within_2se"], exact=True)
    chk("S4 trial noise = 4.1%", 0.041, vr["mean_measurement_noise"], 0.0006)
    chk("S4 states per cell = 14", 14, rank["config"]["n_states"], 0)
    chk("S4 trials = 300", 300, rank["config"]["n_trials"], 0)

    d = cross["per_ensemble_diagnostic"]
    for name, key, ratio, spread, mm in (
            ("Haar-pure", "haar_pure", 0.40, 0.024, 1.20),
            ("low-rank", "low_rank", 0.59, 0.035, 1.42),
            ("GHZ-noisy", "ghz_noisy", 0.00, 0.0, 1.00),
            ("variable-q", "variable_q", 1.49, 0.088, 2.54),
            ("variable-rank", "variable_rank", 4.58, 0.270, 9.74)):
        chk(f"S4 table {name} spread/noise", ratio, d[key]["spread_over_noise"], 0.006)
        chk(f"S4 table {name} spread", spread, d[key]["predicted_rmse_rel_spread"], 0.0006)
        chk(f"S4 table {name} M* max/min", mm, d[key]["m_star_spread_ratio"], 0.006)
    chk("S4 table variable-q estimand 21.6%", 0.216, d["variable_q"]["estimand_rel_std"], 0.001)
    chk("S4 table variable-rank estimand 73.8%", 0.738, d["variable_rank"]["estimand_rel_std"], 0.001)
    chk("S4 table low-rank estimand 6.0%", 0.060, d["low_rank"]["estimand_rel_std"], 0.001)

    g = cross["validation_gates"]
    chk("S4 gate G1 27/27", "27/27", g["G1_raw_vs_stress_test_part4"], exact=True)
    chk("S4 gate G2 27/27", "27/27", g["G2_clipped_vs_pass37_heldout"], exact=True)
    chk("S4 gate G3 27/27", "27/27", g["G3_exact_inputs_vs_committed_mc_predictions"], exact=True)
    a5r, a5c = cross["accuracy"]["all_five"]["raw"], cross["accuracy"]["all_five"]["clipped"]
    chk("S4 all-five swept 45", 45, a5r["swept"], 0)
    chk("S4 all-five raw resolving 27", 27, a5r["resolving"], 0)
    chk("S4 all-five raw exact 26", 26, a5r["exact"], 0)
    chk("S4 all-five raw within-one 27", 27, a5r["within_one"], 0)
    chk("S4 all-five clipped resolving 26", 26, a5c["resolving"], 0)
    chk("S4 all-five clipped exact 21", 21, a5c["exact"], 0)
    chk("S4 all-five raw all-cells 42", 42, a5r["all_cells_within_one"], 0)
    chk("S4 all-five clipped all-cells 42", 42, a5c["all_cells_within_one"], 0)
    n2r = cross["accuracy"]["new_two"]["raw"]
    chk("S4 new-two resolving 12", 12, n2r["resolving"], 0)
    chk("S4 new-two exact 12", 12, n2r["exact"], 0)
    chk("S4 new-two all-cells 18/18", 18, n2r["all_cells_within_one"], 0)
    chk("S4 new-two swept 18", 18, n2r["swept"], 0)

    chk("S4 wider grid median |dev| 9.4%", 0.094, perstate["A_per_state_rmse"]["median_abs_rel_dev"], 0.0005)
    chk("S4 wider grid within-2SE 562/615", "562/615", perstate["A_per_state_rmse"]["within_2se"], exact=True)
    chk("S4 wider grid alpha 194/205", "194/205", perstate["B_per_state_alpha"]["within_2se"], exact=True)
    chk("S4 sequences exact 82/123", "82/123", perstate["C_per_sequence_crossover"]["exact"], exact=True)
    chk("S4 sequences within-one 123/123", "123/123", perstate["C_per_sequence_crossover"]["within_one"], exact=True)

    # --- Section 5: pilot ---
    for n, ex_mstar, budget, err in ((2, 26, 8000, 0.055), (3, 110, 8000, 0.065),
                                     (4, 541, 8000, 0.088), (5, 2884, 32000, 0.058),
                                     (6, 16519, 32000, 0.090)):
        s = pilot47["summary"][f"noisy_pure_q0.1|n{n}"]
        chk(f"S5 table n={n} M*", ex_mstar, round(s["m_star_exact_median"]), 1)
        idx = list(pilot47["config"]["pilots"]).index(budget)
        chk(f"S5 table n={n} error at {budget}", err, s["m_star_rel_mad"][idx], 0.0006)
    sv = pilot48["summary"]["noisy_pure_q0.1|n7"]
    chk("S5 table n=7 M*", 91741, round(sv["m_star_exact_median"]), 1)
    chk("S5 table n=7 10% budget", 128000, sv["first_budget_under_10pct"], 0)
    bl7 = [c for c in pilot48["config"]["cells"] if c[0] == "noisy_pure_q0.1" and c[1] == 7][0][2]
    chk("S5 table n=7 error at 128000", 0.084, sv["m_star_rel_mad"][bl7.index(128000)], 0.0006)
    # n = 8 is superseded by the PASS 49.1 re-measurement (40 reps, independent states)
    tail = load("pass49_n8_tail.json")
    m8 = float(np.median([tail["per_state"][k]["m_star_exact"] for k in tail["per_state"]]))
    chk("S5 table n=8 M*", 520955, round(m8), 1000)
    chk("S5 table n=8 10% budget", 512000, tail["first_budget_under_10pct"], 0)
    mad8 = {r["budget"]: r["mad"] for r in tail["pooled_mad"]}
    chk("S5 table n=8 error at 512000", 0.057, mad8[512000], 0.0006)
    chk("S5 tail MAD at 128000 = 27.7%", 0.277, mad8[128000], 0.0006)
    chk("S5 tail MAD at 256000 = 10.9%", 0.109, mad8[256000], 0.0006)
    chk("S5 tail exponent = -1.14", -1.14, tail["fits"]["last_three"]["exponent"], 0.006)
    chk("S5 tail CI low = -1.38", -1.38, tail["fits"]["last_three"]["ci95"][0], 0.006)
    chk("S5 tail CI high = -0.88", -0.88, tail["fits"]["last_three"]["ci95"][1], 0.006)
    chk("S5 tail verdict ABSENT", "ABSENT", tail["verdict"], exact=True)
    for M, off in ((128000, 1.059), (256000, 0.963), (512000, 1.009)):
        chk(f"S5 median offset at {M}", off,
            tail["median_offsets"][str(M)]["median_ratio"], 0.0006)
    for n, ratio in ((6, 1.94), (7, 1.40)):
        src = pilot48["summary"][f"noisy_pure_q0.1|n{n}"]
        chk(f"S5 pilot/M* at n={n}", ratio,
            src["first_budget_under_10pct"] / src["m_star_exact_median"], 0.006)
    chk("S5 pilot/M* at n=8", 0.98, 512000 / m8, 0.006)
    chk("S5 variable_rank n=7 M* = 503,444", 503444,
        round(pilot48["summary"]["variable_rank|n7"]["m_star_exact_median"]), 1)
    vr7 = pilot48["summary"]["variable_rank|n7"]
    bl = [c for c in pilot48["config"]["cells"] if c[0] == "variable_rank"][0][2]
    chk("S5 variable_rank n=7 error at 256k = 19.7%", 0.197,
        vr7["m_star_rel_mad"][bl.index(256000)], 0.0006)
    for n, budget in ((2, 8000), (4, 32000), (6, 128000)):
        s = pilot47["summary"][f"variable_rank|n{n}"]
        first = min((b for b, v in zip(pilot47["config"]["pilots"], s["m_star_rel_mad"])
                     if v is not None and v < 0.10), default=None)
        chk(f"S5 variable_rank n={n} 10% budget", budget, first, 0)
    z1 = [pilot48["per_unit"][f"noisy_pure_q0.1|8|{s}"]["32000"]["zeta1_rel_rmse"] for s in range(3)]
    z2 = [pilot48["per_unit"][f"noisy_pure_q0.1|8|{s}"]["32000"]["zeta2_rel_rmse"] for s in range(3)]
    nz = sum(pilot48["per_unit"][f"noisy_pure_q0.1|8|{s}"]["32000"]["n_nonpositive_zeta1"]
             for s in range(3))
    chk("S5 n=8 M=32k zeta1 relRMSE 205%", 2.05, float(np.median(z1)), 0.01)
    chk("S5 n=8 M=32k zeta2 relRMSE 12.7%", 0.127, float(np.median(z2)), 0.001)
    chk("S5 n=8 M=32k non-positive zeta1 = 17 of 48", 17, nz, 0)
    chk("S5 extrapolation understated by 1.85x at n=7", 1.85,
        pilot48["summary"]["noisy_pure_q0.1|n7"]["first_budget_under_10pct"]
        / pilot48["pass47_extrapolation"]["7"], 0.02)
    chk("S5 extrapolation understated by 3.8x at n=8", 3.8,
        512000 / pilot48["pass47_extrapolation"]["8"], 0.06)

    # --- Section 3: spectral ---
    chk("S3 zoo records = 70", 70, len(spec["bounds_scan"]["rows"]), 0)
    chk("S3 zoo violations = 0", 0, len(spec["bounds_scan"]["violations"]), 0)
    chk("S3 tightest lower zeta2/7^n = 0.999999942", 0.999999942,
        spec["bounds_scan"]["tightest_lower"]["zeta2_over_7n"], 1e-8)
    tu = spec["bounds_scan"]["tightest_upper"]
    chk("S3 upper-bound approach = 76.5%", 0.765, tu["zeta2"] / tu["upper_17_over_2_n"], 0.001)
    chk("S3 base identity to 8e-16", 8e-16, spec["base_relation"]["worst_deviation_from_1"], 3e-16)
    a4 = [a for a in spec["break_attempts"]["attempts"] if a["attempt"].startswith("A4")][0]
    chk("S3 records with zeta2^(1/n) < 7 = 12", 12, len(a4["outside"]), 0)
    chk("S3 lowest zeta2^(1/n) = 6.982", 6.982, a4["min"], 0.001)
    a6 = [a for a in spec["break_attempts"]["attempts"] if a["attempt"].startswith("A6")][0]
    chk("S3 product base(zeta1) fit n=3-5 -> 1.666", 1.666,
        a6["fitted_base_by_window"]["3-5"], 0.001)
    chk("S3 product base(zeta1) fit n=6-8 -> 1.540", 1.540,
        a6["fitted_base_by_window"]["6-8"], 0.001)

    # --- Section 3.6: cost ---
    rows = {r["n"]: r for r in mstar["timing_47_2a"]["rows"]}
    chk("S3.6 zeta1 at n=9 = 2.3 s", 2.3, rows[9]["t_zeta1_s"], 0.06)
    chk("S3.6 zeta1 at n=10 = 23.5 s", 23.5, rows[10]["t_zeta1_s"], 0.1)
    chk("S3.6 n=11 = 242 s", 242.07, mstar["timing_47_2a"]["n11_measured_s"], 0.01)
    chk("S3.6 largest n under 60 s = 10", 10, mstar["timing_47_2a"]["largest_n_under_60s"], 0)
    tr = mstar["input_requirements_47_2b"]["noisy_pure_q0.1|n6"]["truncation"]
    chk("S3.6 zeta2 error at w<=1 = 1.2%", 0.0124, tr[1]["zeta2_rel_err"], 0.0006)
    chk("S3.6 zeta2 error at w<=2 = 0.3%", 0.0028, tr[2]["zeta2_rel_err"], 0.0004)
    chk("S3.6 w<=2 keeps 154 strings", 154, tr[2]["n_terms"], 0)
    chk("S3.6 zeta1 diagonal error at w<=5 = 54%", 0.537, tr[5]["zeta1_diag_rel_err"], 0.004)
    chk("S3.6 w<=5 keeps 82% of strings", 0.822, tr[5]["frac_of_4n"], 0.002)

    bad = [c for c in checks if not c["ok"]]
    print(f"{len(checks) - len(bad)}/{len(checks)} numeric claims match their artifacts")
    for c in bad:
        print(f"  MISMATCH  {c['claim']}: manuscript {c['stated']} vs artifact {c['artifact']}")
    OUT.write_text(json.dumps({"description": "PASS 49.6 numeric audit of the new prose",
                               "n_checks": len(checks), "n_failed": len(bad),
                               "checks": checks}, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
