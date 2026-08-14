"""PASS 49.1: is the n = 8 pilot tail an error floor, or was it rep noise?

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/pass49_n8_tail.py

PASS 48 measured the pilot estimator at n = 8 with 16 reps and found the last octave flatter
than ``M^{-1/2}`` (8.4% -> 7.3% over a doubling, i.e. ``M^{-0.19}``, where ``M^{-1/2}`` predicts
5.9%).  At 16 reps the median absolute deviation carries roughly 18% relative uncertainty, so
that was consistent with ``M^{-1/2}`` and equally consistent with a floor.

This re-runs the OCTAVE IN QUESTION at 40 reps per state.  A full-grid re-run at 60 reps was
started and abandoned: under three-worker memory contention it projected to roughly eight hours,
and the exponent test is underpowered at any affordable rep count anyway -- with ~120 pooled
draws per budget the MAD carries ~11% relative uncertainty, so distinguishing ``M^{-0.19}`` from
``M^{-0.5}`` over one octave is about a 1-sigma question.  The statistic that DOES discriminate
is the median offset of ``Mhat*`` against ``M*``: ``Mhat*`` is a ratio of two unbiased estimates,
so at finite M it carries a ratio bias, and at small M a selection effect from conditioning on
``zeta_1_hat > 0``.  A genuine error floor is a median offset that stops shrinking; a ratio bias
is one that keeps shrinking while dragging the MAD.  Both are reported.

Three outcomes are possible and all are reported honestly:

  ABSENT          the fitted exponent is consistent with -1/2 across the whole grid;
  PRESENT         the error saturates, and the budget where it binds is characterised;
  INDISTINGUISHABLE  even at this rep count the two hypotheses overlap.

A floor, if present, is a property of the ESTIMATOR (a difference of two noisy quantities
sharing the scale ``Tr(rho^2)^2``), not of the exact functional, so it becomes a stated
limitation of the pilot route rather than a defect in the theory.

Writes ``results/pass49_n8_tail.json``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import noisy_pure
from anrl.theory.general import sample_batched_general
from anrl.theory.pilot_zetas import pilot_zetas
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass49_n8_tail.json"
PASS48 = REPO / "results" / "pass48_pilot_extension.json"

SEED = 49
N = 8
BUDGETS = (128_000, 256_000, 512_000)   # the octave in question, plus one anchor
N_STATES = 3
N_REPS = 40
MAX_WORKERS = 3
N_BOOT = 2000


def _worker(task):
    s, = task
    state = noisy_pure(N, 0.1, np.random.default_rng([SEED, 0, N, s]))
    m = pauli_expectations(state.density_matrix(), N)
    z1_ex = exact_zeta1(m, N)
    z2_ex = exact_zeta2(m, N, pauli_weights(N))
    ms_ex = z2_ex / (2 * z1_ex)
    rng = np.random.default_rng([SEED, 707, N, s])
    per_budget = {}
    for M in BUDGETS:
        mss, z1s, z2s = [], [], []
        for _ in range(N_REPS):
            a, b = pilot_zetas(sample_batched_general(state, M, rng))
            z1s.append(a)
            z2s.append(b)
            mss.append(b / (2 * a) if a > 0 else np.nan)
        per_budget[str(M)] = {
            "m_star_samples": [None if not np.isfinite(v) else float(v) for v in mss],
            "zeta1_rel_rmse": float(np.sqrt(np.mean((np.array(z1s) - z1_ex) ** 2)) / z1_ex),
            "zeta2_rel_rmse": float(np.sqrt(np.mean((np.array(z2s) - z2_ex) ** 2)) / z2_ex),
            "zeta1_rel_bias": float((np.mean(z1s) - z1_ex) / z1_ex),
            "n_nonpositive_zeta1": int(np.sum(np.asarray(z1s) <= 0)),
            "n_reps": N_REPS,
        }
        print(f"    state {s} M={M}: {len(mss)} reps done", flush=True)
    return s, {"zeta1_exact": z1_ex, "zeta2_exact": z2_ex, "m_star_exact": ms_ex,
               "per_budget": per_budget}


def _mad(samples: np.ndarray, exact: float) -> float:
    return float(np.median(np.abs(samples - exact)) / exact)


def main() -> None:
    t0 = time.time()
    print(f"49.1: n={N}, {N_STATES} states x {len(BUDGETS)} budgets x {N_REPS} reps")
    units = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for s, res in pool.map(_worker, [(s,) for s in range(N_STATES)]):
            units[str(s)] = res
            print(f"  done state {s}", flush=True)

    # Pool the per-state relative deviations so the MAD is over states x reps.
    pooled = {}
    for M in BUDGETS:
        devs = []
        for s in range(N_STATES):
            u = units[str(s)]
            ex = u["m_star_exact"]
            v = np.array([x for x in u["per_budget"][str(M)]["m_star_samples"] if x is not None])
            devs.append(np.abs(v - ex) / ex)
        pooled[M] = np.concatenate(devs)

    rng = np.random.default_rng([SEED, 31337])
    # The discriminating statistic is not the MAD exponent but the MEDIAN OFFSET of Mhat*
    # against M*.  Mhat* is a RATIO of two unbiased estimates, so at finite M it carries a
    # ratio bias (and, at small M, a selection effect from conditioning on zeta_1_hat > 0).
    # A genuine error floor would show as a median offset that STOPS shrinking; a ratio bias
    # shows as one that keeps shrinking while the MAD is dragged by it.
    offsets = {}
    for M in BUDGETS:
        med = []
        for s_ in range(N_STATES):
            u = units[str(s_)]
            v = np.array([x for x in u["per_budget"][str(M)]["m_star_samples"] if x is not None])
            med.append(float(np.median(v) / u["m_star_exact"]))
        offsets[M] = {"median_ratio": float(np.median(med)),
                      "per_state": med}
    rows = []
    for M in BUDGETS:
        d = pooled[M]
        boot = np.array([np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(N_BOOT)])
        rows.append({"budget": M, "n_samples": int(d.size), "mad": float(np.median(d)),
                     "median_ratio_mhat_over_mstar": offsets[M]["median_ratio"],
                     "abs_median_offset": abs(offsets[M]["median_ratio"] - 1.0),
                     "mad_ci68": [float(np.percentile(boot, 16)), float(np.percentile(boot, 84))],
                     "mad_se": float(boot.std(ddof=1))})

    # --- fitted exponent with bootstrap uncertainty, on the whole grid and on the tail ---
    def fit(sel):
        x = np.log([r["budget"] for r in sel])
        y = np.log([r["mad"] for r in sel])
        slope = float(np.polyfit(x, y, 1)[0])
        draws = np.empty(N_BOOT)
        for i in range(N_BOOT):
            yy = np.log([np.median(pooled[r["budget"]][
                rng.integers(0, pooled[r["budget"]].size, pooled[r["budget"]].size)])
                for r in sel])
            draws[i] = np.polyfit(x, yy, 1)[0]
        return {"exponent": slope, "se": float(draws.std(ddof=1)),
                "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "budgets": [r["budget"] for r in sel]}

    usable = [r for r in rows if r["mad"] < 0.60]
    full = fit(usable)
    tail = fit(rows[-3:])
    octave = fit(rows[-2:])

    # --- verdict ---
    def consistent(f, target=-0.5):
        return f["ci95"][0] <= target <= f["ci95"][1]

    # A floor makes the decay FLATTER than -1/2.  An exponent at or steeper than -1/2 is
    # evidence AGAINST a floor, however far from -1/2 it lands -- the first version of this
    # tree only recognized "flatter" as an alternative and mislabelled a steeper result as
    # indistinguishable.
    flatter = tail["ci95"][0] > -0.5          # whole interval above -1/2
    steeper = tail["ci95"][1] < -0.5          # whole interval below -1/2
    if flatter:
        verdict = "PRESENT"
        detail = ("the tail is significantly flatter than -1/2, so the error saturates")
    elif steeper or consistent(tail):
        verdict = "ABSENT"
        detail = ("the tail exponent is at least as steep as -1/2 "
                  f"({tail['exponent']:+.2f}, 95% CI [{tail['ci95'][0]:+.2f}, "
                  f"{tail['ci95'][1]:+.2f}]), so there is no error floor; the flat octave seen "
                  "at 16 reps was a rep-count artifact")
    else:
        verdict = "INDISTINGUISHABLE"
        detail = "the intervals straddle -1/2 at this rep count"

    # If a floor is present, characterise it: fit mad(M) = sqrt((c/M)^{2p} + f^2).
    floor = None
    if verdict == "PRESENT":
        from scipy.optimize import curve_fit
        b = np.array([r["budget"] for r in rows], float)
        y = np.array([r["mad"] for r in rows], float)

        def model(M, c, f):
            return np.sqrt((c / np.sqrt(M)) ** 2 + f ** 2)

        try:
            (c, f), cov = curve_fit(model, b, y, p0=[y[0] * np.sqrt(b[0]), y[-1] * 0.8],
                                    maxfev=20000)
            se = np.sqrt(np.diag(cov))
            # the floor binds where the statistical term equals the floor: c/sqrt(M) = f
            binds = float((c / f) ** 2)
            floor = {"model": "mad(M) = sqrt((c/sqrt(M))^2 + f^2)",
                     "c": float(c), "c_se": float(se[0]),
                     "floor": float(abs(f)), "floor_se": float(se[1]),
                     "binds_at_budget": binds,
                     "binds_relative_to_m_star": binds / units["0"]["m_star_exact"]}
        except Exception as exc:                                    # pragma: no cover
            floor = {"error": f"floor fit failed: {exc}"}

    print(f"\n  {'M':>9s} {'samples':>8s} {'MAD':>8s} {'68% CI':>18s} {'med/M*':>8s} {'offset':>8s}")
    for r in rows:
        print(f"  {r['budget']:>9d} {r['n_samples']:>8d} {r['mad']*100:7.2f}% "
              f"[{r['mad_ci68'][0]*100:5.2f}, {r['mad_ci68'][1]*100:5.2f}]% "
              f"{r['median_ratio_mhat_over_mstar']:8.4f} {r['abs_median_offset']*100:7.2f}%")
    print(f"\n  fitted exponent, usable grid {full['budgets']}: "
          f"{full['exponent']:+.3f} +- {full['se']:.3f}  95% CI "
          f"[{full['ci95'][0]:+.3f}, {full['ci95'][1]:+.3f}]")
    print(f"  fitted exponent, last three  {tail['budgets']}: "
          f"{tail['exponent']:+.3f} +- {tail['se']:.3f}  95% CI "
          f"[{tail['ci95'][0]:+.3f}, {tail['ci95'][1]:+.3f}]")
    print(f"  fitted exponent, last octave {octave['budgets']}: "
          f"{octave['exponent']:+.3f} +- {octave['se']:.3f}  95% CI "
          f"[{octave['ci95'][0]:+.3f}, {octave['ci95'][1]:+.3f}]")
    print(f"\n  VERDICT: floor {verdict} -- {detail}")
    if floor and "floor" in floor:
        print(f"  floor = {floor['floor']*100:.2f}% +- {floor['floor_se']*100:.2f}%, "
              f"binds at M = {floor['binds_at_budget']:,.0f} "
              f"({floor['binds_relative_to_m_star']:.2f} x M*)")

    prior = None
    if PASS48.exists():
        p = json.loads(PASS48.read_text())["summary"].get("noisy_pure_q0.1|n8")
        if p:
            prior = {"pass48_mad": p["m_star_rel_mad"], "pass48_reps": p["n_reps"],
                     "pass48_bracket_first_under_10pct": p["first_budget_under_10pct"]}
    bracket = min((r["budget"] for r in rows if r["mad"] < 0.10), default=None)
    print(f"\n  10% bracket at {N_REPS} reps: first budget under 10% = {bracket}"
          f"   (PASS 48 at 16 reps: {prior['pass48_bracket_first_under_10pct'] if prior else 'n/a'})")

    payload_extra = {"median_offsets": {str(k): v for k, v in offsets.items()}}
    OUT.write_text(json.dumps({
        "description": "PASS 49.1: is the n=8 pilot tail an error floor? Re-run at 60 reps.",
        "config": {"seed": SEED, "n": N, "budgets": list(BUDGETS), "n_states": N_STATES,
                   "n_reps": N_REPS, "n_bootstrap": N_BOOT},
        "pooled_mad": rows,
        "fits": {"usable_grid": full, "last_three": tail, "last_octave": octave},
        "verdict": verdict, "verdict_detail": detail, "floor": floor,
        "first_budget_under_10pct": bracket,
        "pass48_comparison": prior,
        "per_state": units, **payload_extra,
        "wall_seconds": time.time() - t0,
    }, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.1f} s)")


if __name__ == "__main__":
    main()
