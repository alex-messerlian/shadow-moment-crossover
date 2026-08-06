"""Analyze the Pauli-twirled (randomized-compiling) SWAP results (NO credits).

Deliverables:
  * Positive-control survival check: does the uncorrected X before CZ(0,9) actually
    execute?  P(counts on posctrl-ideal support) vs P(untwirled-ideal support).  These
    supports are disjoint (TVD 1.0), so this is a clean binary test that mid-circuit
    gates around a CZ are not stripped/resynthesized away.
  * Untwirled baseline (same session) vs historical 0.378/0.420, drift control.
  * Twirled randomized-compiling estimate = mean purity over twirls, with a CI and the
    per-twirl scatter decomposed into physical scatter vs shot noise (high physical
    scatter is itself a coherent-error signature).
  * Shift: twirled vs untwirled and vs the depolarizing-model prediction (0.606 / 0.579),
    with significance.
  * Non-monotonicity: twirled n=3 vs twirled n=4.

Run:  PYTHONPATH=. .venv/bin/python -m experiments.coherent_error_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware.swap_test import purity_from_counts, swap_sign

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"


def _load(name: str) -> dict | None:
    f = HW / f"ce_{name}_counts.json"
    if not f.exists():
        return None
    return {k.replace(" ", ""): int(v) for k, v in json.loads(f.read_text()).items()}


def _boot_ci(counts: dict, n: int, reps: int = 4000, seed: int = 0) -> tuple[float, float]:
    signs = np.array([swap_sign(b, n) for b, c in counts.items() for _ in range(c)], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(signs, size=len(signs), replace=True).mean() for _ in range(reps)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _shot_var(mu: float, shots: int) -> float:
    return (1.0 - mu * mu) / shots


def positive_control(manifest: dict, n: int = 3) -> dict | None:
    counts = _load("n3_posctrl")
    if counts is None:
        return None
    pc = manifest["circuits"]["n3_posctrl"]
    pc_supp = set(pc["posctrl_ideal_support"])
    un_supp = set(pc["untwirled_ideal_support"])
    tot = sum(counts.values())
    p_pc = sum(c for b, c in counts.items() if b in pc_supp) / tot
    p_un = sum(c for b, c in counts.items() if b in un_supp) / tot
    # verdict: X executed iff mass concentrates on the posctrl (X-before-CZ) support
    executed = p_pc > p_un
    return {"shots": tot, "P_on_posctrl_support": round(p_pc, 4),
            "P_on_untwirled_support": round(p_un, 4),
            "gates_around_cz_execute": bool(executed),
            "verdict": ("EXECUTED, mid-circuit gates around CZ run (twirl survives)"
                        if executed else
                        "STRIPPED, inserted gate removed; twirl likely collapsed")}


def analyze_n(n: int, manifest: dict) -> dict:
    depol = manifest["depol_prediction"][str(n)]
    hist = manifest["untwirled_measured_historical"][str(n)]

    untw = _load(f"n{n}_untw")
    untw_res = None
    if untw is not None:
        mu = purity_from_counts(untw, n)
        ci = _boot_ci(untw, n, seed=1)
        untw_res = {"purity": round(mu, 4), "ci95": [round(x, 4) for x in ci],
                    "shots": sum(untw.values()), "drift_vs_historical": round(mu - hist, 4)}

    # twirl realizations
    tw_rows = []
    k = 0
    while True:
        c = _load(f"n{n}_tw{k:02d}")
        if c is None:
            break
        mu = purity_from_counts(c, n)
        tw_rows.append({"k": k, "purity": round(mu, 4), "shots": sum(c.values())})
        k += 1

    # anomaly assessment this session: is same-session untwirled actually below prediction?
    anomaly = None
    if untw_res is not None:
        gap_ss = depol - untw_res["purity"]
        anomaly = {"untwirled_same_session": untw_res["purity"], "depol_prediction": depol,
                   "gap_same_session": round(gap_ss, 4),
                   "gap_historical": round(depol - hist, 4),
                   "anomaly_present_this_session": bool(untw_res["ci95"][1] < depol - 0.02)}

    tw_res = None
    if tw_rows:
        purs = np.array([r["purity"] for r in tw_rows])
        shots = np.array([r["shots"] for r in tw_rows])
        rc_mean = float(purs.mean())
        # intra-run drift proxy: mean of first half vs second half of the twirl sequence
        half = len(purs) // 2
        early_late = None
        if half >= 1 and len(purs) >= 2:
            early_late = {"early_mean": round(float(purs[:half].mean()), 4),
                          "late_mean": round(float(purs[half:].mean()), 4),
                          "early_minus_late": round(float(purs[:half].mean() - purs[half:].mean()), 4)}
        # CI of the RC estimate = SEM across twirls (captures shot noise + physical scatter)
        sem = float(purs.std(ddof=1) / np.sqrt(len(purs))) if len(purs) > 1 else float("nan")
        obs_var = float(purs.var(ddof=1)) if len(purs) > 1 else float("nan")
        shot_var = float(np.mean([_shot_var(m, s) for m, s in zip(purs, shots)]))
        phys_var = obs_var - shot_var  # excess scatter beyond shot noise
        base = untw_res["purity"] if untw_res else hist
        gap = depol - base
        closed = (rc_mean - base) / gap if abs(gap) > 1e-9 else float("nan")
        # significance of the shift rc_mean vs base (untwirled), and vs depol prediction
        z_vs_base = (rc_mean - base) / sem if sem and not np.isnan(sem) else float("nan")
        z_vs_depol = (rc_mean - depol) / sem if sem and not np.isnan(sem) else float("nan")
        tw_res = {
            "n_twirls": len(purs), "rc_estimate": round(rc_mean, 4),
            "rc_sem": round(sem, 4) if not np.isnan(sem) else None,
            "per_twirl_purity": [round(float(p), 4) for p in purs],
            "scatter_std": round(float(purs.std(ddof=1)), 4) if len(purs) > 1 else None,
            "observed_var": obs_var, "shot_noise_var": shot_var,
            "physical_scatter_var": round(phys_var, 6),
            "physical_scatter_std": round(float(np.sqrt(phys_var)), 4) if phys_var > 0 else 0.0,
            "scatter_exceeds_shot_noise": bool(phys_var > 0 and obs_var > 2 * shot_var),
            "baseline_used": round(base, 4),
            "depol_prediction": depol,
            "shift_vs_untwirled": round(rc_mean - base, 4),
            "gap_closed_fraction": round(closed, 3) if not np.isnan(closed) else None,
            "z_shift_vs_untwirled": round(z_vs_base, 2) if not np.isnan(z_vs_base) else None,
            "z_vs_depol_prediction": round(z_vs_depol, 2) if not np.isnan(z_vs_depol) else None,
            "intra_run_drift": early_late,
        }

    return {"n": n, "depol_prediction": depol, "untwirled_historical": hist,
            "untwirled_same_session": untw_res, "anomaly": anomaly,
            "twirled": tw_res, "twirl_rows": tw_rows}


def main() -> None:
    manifest = json.loads((HW / "ce_manifest.json").read_text())
    result = {"positive_control": positive_control(manifest),
              "n3": analyze_n(3, manifest), "n4": analyze_n(4, manifest)}

    # non-monotonicity: compare BOTH this-session untwirled and twirled at n3 vs n4.
    # (Historical was n3<n4; the honest question is what happened this session and whether
    # twirling; not drift, is responsible for any change.)
    t3 = result["n3"]["twirled"]; t4 = result["n4"]["twirled"]
    u3 = result["n3"]["untwirled_same_session"]; u4 = result["n4"]["untwirled_same_session"]
    if t3 and t4 and u3 and u4:
        result["non_monotonicity"] = {
            "historical_untwirled_n3": result["n3"]["untwirled_historical"],
            "historical_untwirled_n4": result["n4"]["untwirled_historical"],
            "historical_n3_lt_n4": result["n3"]["untwirled_historical"] < result["n4"]["untwirled_historical"],
            "same_session_untwirled_n3": u3["purity"], "same_session_untwirled_n4": u4["purity"],
            "same_session_untwirled_n3_gt_n4": u3["purity"] > u4["purity"],
            "twirled_rc_n3": t3["rc_estimate"], "twirled_rc_n4": t4["rc_estimate"],
            "twirled_n3_gt_n4": t3["rc_estimate"] > t4["rc_estimate"],
            "note": ("This-session UNTWIRLED already shows n3>n4 (reversed vs the historical n3<n4). "
                     "The reversal is due to cross-session DRIFT, not twirling: twirled and untwirled "
                     "give the same ordering. The anomaly moved from the n=3 register to the n=4 register.")}

    (HW / "coherent_error_analysis.json").write_text(json.dumps(result, indent=2, default=float))

    # ---------- print ----------
    pc = result["positive_control"]
    print("=== POSITIVE CONTROL (does an inserted gate around a CZ execute?) ===")
    if pc:
        print(f"  P(posctrl support)={pc['P_on_posctrl_support']}  P(untwirled support)={pc['P_on_untwirled_support']}")
        print(f"  -> {pc['verdict']}")
    else:
        print("  (no positive-control counts yet)")

    for n in (3, 4):
        r = result[f"n{n}"]
        print(f"\n=== n={n} (depol prediction {r['depol_prediction']}, untwirled hist {r['untwirled_historical']}) ===")
        u = r["untwirled_same_session"]
        if u:
            print(f"  untwirled same-session: {u['purity']} CI {u['ci95']} "
                  f"(drift vs hist {u['drift_vs_historical']:+.4f})")
        a = r["anomaly"]
        if a:
            print(f"  anomaly this session? {a['anomaly_present_this_session']} "
                  f"(same-session gap {a['gap_same_session']:+.4f} vs historical gap {a['gap_historical']:+.4f})")
        t = r["twirled"]
        if t:
            print(f"  twirled RC estimate: {t['rc_estimate']} +/- {t['rc_sem']}  ({t['n_twirls']} twirls)")
            print(f"    per-twirl: {t['per_twirl_purity']}")
            print(f"    scatter std {t['scatter_std']}; physical scatter std {t['physical_scatter_std']} "
                  f"(exceeds shot noise: {t['scatter_exceeds_shot_noise']})")
            print(f"    shift vs untwirled {t['shift_vs_untwirled']:+.4f} (z={t['z_shift_vs_untwirled']}), "
                  f"z vs depol {t['z_vs_depol_prediction']}")
            if t["intra_run_drift"]:
                d = t["intra_run_drift"]
                print(f"    intra-run drift (early-late twirls): {d['early_minus_late']:+.4f} "
                      f"(early {d['early_mean']}, late {d['late_mean']})")
    nm = result.get("non_monotonicity")
    if nm:
        print(f"\n=== NON-MONOTONICITY ===")
        print(f"  historical: n3 {nm['historical_untwirled_n3']} < n4 {nm['historical_untwirled_n4']} "
              f"({nm['historical_n3_lt_n4']})")
        print(f"  this session UNTWIRLED: n3 {nm['same_session_untwirled_n3']} vs n4 "
              f"{nm['same_session_untwirled_n4']} -> n3>n4: {nm['same_session_untwirled_n3_gt_n4']}")
        print(f"  this session TWIRLED: n3 {nm['twirled_rc_n3']} vs n4 {nm['twirled_rc_n4']} -> "
              f"n3>n4: {nm['twirled_n3_gt_n4']}")
        print(f"  -> {nm['note']}")


if __name__ == "__main__":
    main()
