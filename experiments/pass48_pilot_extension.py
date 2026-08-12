"""PASS 48.2: confirm the pilot/M* crossing DIRECTLY at n = 7 and n = 8.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/pass48_pilot_extension.py

PASS 47 measured the pilot budget needed to pin ``M*`` to 10% at n = 2..6 and found the ratio
``pilot/M*`` falling 0.39x per qubit, crossing 1 near n = 7 -- but by EXTRAPOLATION from a fit
whose exponent steepened past -0.5 at n >= 5, because at those sizes the small budgets are
still emerging from the regime where ``zeta_1_hat`` is useless.  The extrapolation is
therefore optimistic and has to be checked at the sizes it predicts.

This run measures n = 7 and n = 8 directly, with budgets chosen to bracket the 10% threshold
rather than to span a fixed decade, and re-runs n = 6 under PASS 47's exact configuration as a
reproduction gate on the promoted estimator (:mod:`anrl.theory.pilot_zetas`).

Writes ``results/pass48_pilot_extension.json``.  No paper edit.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import low_rank, noisy_pure
from anrl.theory.general import sample_batched_general
from anrl.theory.pilot_zetas import pilot_zetas
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass48_pilot_extension.json"
PASS47 = REPO / "results" / "pass47_pilot_estimator.json"

SEED = 47                      # PASS 47's seed, so the n=6 gate reproduces exactly
MAX_WORKERS = 4
_ENS_ID = {"noisy_pure_q0.1": 0, "variable_rank": 5}

# (ensemble, n, budgets, n_states, n_reps).  n=6 replays PASS 47 verbatim as the gate; the new
# sizes use brackets around the predicted 10% threshold (~75k at n=7, ~150k at n=8) and fewer
# reps, because cost is linear in the budget and the budgets are an order of magnitude larger.
CELLS = (
    ("noisy_pure_q0.1", 6, (500, 2000, 8000, 32000, 128000), 4, 40),
    ("noisy_pure_q0.1", 7, (8000, 32000, 64000, 128000, 256000), 3, 24),
    ("noisy_pure_q0.1", 8, (32000, 64000, 128000, 256000, 512000), 3, 16),
    ("variable_rank", 7, (8000, 32000, 64000, 128000, 256000), 3, 24),
)
PASS47_GATE_BUDGETS = (500, 2000, 8000, 32000, 128000)


def make_state(ens: str, n: int, s: int):
    rng = np.random.default_rng([SEED, _ENS_ID[ens], n, s])
    if ens == "noisy_pure_q0.1":
        return noisy_pure(n, 0.1, rng)
    return low_rank(n, int(rng.integers(1, min(8, 2 ** n) + 1)), rng)


def _worker(task):
    ens, n, budgets, s, reps = task
    state = make_state(ens, n, s)
    m = pauli_expectations(state.density_matrix(), n)
    z1_ex = exact_zeta1(m, n)
    z2_ex = exact_zeta2(m, n, pauli_weights(n))
    ms_ex = z2_ex / (2 * z1_ex)
    # Same stream as PASS 47 so the n=6 cell reproduces bit-for-bit.
    rng = np.random.default_rng([SEED, 909, _ENS_ID[ens], n, s])
    out = {"zeta1_exact": z1_ex, "zeta2_exact": z2_ex, "m_star_exact": ms_ex, "budgets": list(budgets)}
    for M in budgets:
        z1s, z2s, mss = [], [], []
        for _ in range(reps):
            snaps = sample_batched_general(state, M, rng)
            a, b = pilot_zetas(snaps)
            z1s.append(a)
            z2s.append(b)
            mss.append(b / (2 * a) if a > 0 else np.nan)
        z1s, z2s, mss = np.array(z1s), np.array(z2s), np.array(mss)
        fin = np.isfinite(mss)
        out[str(M)] = {
            "n_reps": reps,
            "zeta1_rel_rmse": float(np.sqrt(np.mean((z1s - z1_ex) ** 2)) / z1_ex),
            "zeta2_rel_rmse": float(np.sqrt(np.mean((z2s - z2_ex) ** 2)) / z2_ex),
            "n_nonpositive_zeta1": int((z1s <= 0).sum()),
            "m_star_rel_mad": (float(np.median(np.abs(mss[fin] - ms_ex)) / ms_ex) if fin.any() else None),
            "m_star_median": float(np.median(mss[fin])) if fin.any() else None,
        }
    return (ens, n, s), out


def _bracket_and_fit(rows: list[dict], budgets: tuple[int, ...]) -> dict:
    """Budget for 10% error: the measured bracket, and a log-log fit inside the usable range."""
    mad = [float(np.median([r[str(M)]["m_star_rel_mad"] for r in rows
                            if r[str(M)]["m_star_rel_mad"] is not None])) for M in budgets]
    under = [M for M, v in zip(budgets, mad) if v < 0.10]
    over = [M for M, v in zip(budgets, mad) if v >= 0.10]
    good = [(M, v) for M, v in zip(budgets, mad) if v < 0.60]
    slope = fitted = None
    if len(good) >= 2:
        c = np.polyfit(np.log([g[0] for g in good]), np.log([g[1] for g in good]), 1)
        slope = float(c[0])
        fitted = float(np.exp((np.log(0.10) - c[1]) / c[0]))
    return {
        "m_star_rel_mad": mad,
        "first_budget_under_10pct": min(under) if under else None,
        "last_budget_over_10pct": max(over) if over else None,
        "fitted_convergence_exponent": slope,
        "fitted_budget_for_10pct": fitted,
    }


def main() -> None:
    t0 = time.time()
    grid = [(ens, n, b, s, reps) for (ens, n, b, ns, reps) in CELLS for s in range(ns)]
    print(f"pilot extension: {len(grid)} units")
    per_unit: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for key, res in pool.map(_worker, grid):
            per_unit["|".join(map(str, key))] = res
            print(f"  done {key}", flush=True)

    # ---- reproduction gate: n=6 against the committed PASS 47 artifact ----
    gate = {"checked": [], "worst_rel_dev": 0.0, "status": "not run"}
    if PASS47.exists():
        old = json.loads(PASS47.read_text())
        for s in range(4):
            k = f"noisy_pure_q0.1|6|{s}"
            o = old["per_unit"].get(f"noisy_pure_q0.1|6|{s}")
            if k not in per_unit or o is None:
                continue
            for M in PASS47_GATE_BUDGETS:
                a = per_unit[k][str(M)]["zeta1_rel_rmse"]
                b = o[str(M)]["zeta1_rel_rmse"]
                dev = abs(a - b) / max(abs(b), 1e-30)
                gate["checked"].append({"unit": k, "budget": M, "new": a, "pass47": b, "rel_dev": dev})
                gate["worst_rel_dev"] = max(gate["worst_rel_dev"], dev)
        gate["status"] = "PASS" if gate["worst_rel_dev"] < 1e-9 else "MISMATCH"
        print(f"\nGATE n=6 vs PASS 47 ({len(gate['checked'])} comparisons): "
              f"{gate['status']}, worst relative deviation {gate['worst_rel_dev']:.3e}")

    # ---- the crossing ----
    print(f"\n{'cell':22s} " + " ".join(f"{'M='+str(M):>10s}" for M in (500, 2000, 8000, 32000, 64000, 128000, 256000, 512000)))
    summary = {}
    for (ens, n, budgets, ns, reps) in CELLS:
        rows = [per_unit[f"{ens}|{n}|{s}"] for s in range(ns)]
        info = _bracket_and_fit(rows, budgets)
        ms_ex = float(np.median([r["m_star_exact"] for r in rows]))
        info["m_star_exact_median"] = ms_ex
        info["n_reps"] = reps
        info["n_states"] = ns
        for label in ("first_budget_under_10pct", "fitted_budget_for_10pct"):
            v = info[label]
            info[f"ratio_from_{label}"] = (v / ms_ex) if v else None
        summary[f"{ens}|n{n}"] = info
        cells = {M: v for M, v in zip(budgets, info["m_star_rel_mad"])}
        line = " ".join(f"{cells[M]*100:9.1f}%" if M in cells else "          "
                        for M in (500, 2000, 8000, 32000, 64000, 128000, 256000, 512000))
        print(f"  {ens+' n='+str(n):22s}{line}")
        print(f"    {'':20s} M* = {ms_ex:,.0f}   bracket: >10% at {info['last_budget_over_10pct']}, "
              f"<10% at {info['first_budget_under_10pct']}   fitted {info['fitted_budget_for_10pct']:,.0f}"
              if info["fitted_budget_for_10pct"] else "")

    print("\n=== 48.2(b)(c) the crossing, measured vs PASS-47 extrapolation ===")
    extrap = None
    if PASS47.exists():
        old = json.loads(PASS47.read_text())
        ns_old = [2, 3, 4, 5, 6]
        p10 = [old["summary"][f"noisy_pure_q0.1|n{n}"]["pilot_budget_for_10pct"] for n in ns_old]
        c = np.polyfit(ns_old, np.log(p10), 1)
        extrap = {str(n): float(np.exp(np.polyval(c, n))) for n in (7, 8)}
        extrap["fitted_growth_per_qubit"] = float(np.exp(c[0]))
    for n in (7, 8):
        key = f"noisy_pure_q0.1|n{n}"
        if key not in summary:
            continue
        s = summary[key]
        meas = s["first_budget_under_10pct"]
        print(f"  n={n}: measured 10%-budget bracket ({s['last_budget_over_10pct']}, {meas}]   "
              f"fitted {s['fitted_budget_for_10pct']:,.0f}   "
              f"PASS-47 extrapolation {extrap[str(n)]:,.0f}" if extrap else "")
        if meas:
            print(f"        M* = {s['m_star_exact_median']:,.0f}   measured pilot/M* = "
                  f"{meas / s['m_star_exact_median']:.2f}   (crossing is at ratio 1)")

    payload = {
        "description": "PASS 48.2: direct measurement of the pilot/M* crossing at n = 7 and 8",
        "config": {"seed": SEED, "cells": [[e, n, list(b), ns, r] for (e, n, b, ns, r) in CELLS],
                   "estimator": "anrl.theory.pilot_zetas.pilot_zetas (four disjoint blocks, unbiased)"},
        "reproduction_gate_n6_vs_pass47": gate,
        "summary": summary,
        "pass47_extrapolation": extrap,
        "per_unit": per_unit,
        "wall_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
