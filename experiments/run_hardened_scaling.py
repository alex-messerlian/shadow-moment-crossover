"""Statistically hardened scaling study: single-copy vs collective, out to n=10.

    .venv/bin/python experiments/run_hardened_scaling.py

Re-runs the scaling grid with many states per cell and many measurement trials
per state, attaches bootstrap error bars to every RMSE, and calls each
single-vs-collective crossover with a PAIRED state-level test (collective /
single-copy / tie).  Extends the noisy-pure sweep to n=9 and n=10 (single-copy
via the O(M^2 n) factorized U-statistic; collective via the dense per-qubit
channel).  Saves the full grid + crossover table to
results/scaling_hardened.json and prints the three report questions.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from anrl.benchmark import crossover_table, run_hardened, save_hardened

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "scaling_hardened.json"

BUDGET = 2000
N_STATES = 48
N_TRIALS = 10
SEED = 0
ENSEMBLE_Q = 0.1
SIZES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.0, 0.02, 0.05, 0.1)
MAX_RANDOM_MIXED_N = 6


def _fit_factor(sizes: list[int], rmses: list[float]) -> float:
    """Geometric growth factor per qubit = exp(slope of log(rmse) vs n)."""
    xs = np.asarray(sizes, dtype=np.float64)
    ys = np.log(np.clip(np.asarray(rmses, dtype=np.float64), 1e-12, None))
    return float(np.exp(np.polyfit(xs, ys, 1)[0]))


def main() -> None:
    print(
        f"Hardened grid: sizes={SIZES}, noise={NOISE_MODELS}, rates={RATES}, "
        f"budget={BUDGET}, {N_STATES} states x {N_TRIALS} trials/cell, q={ENSEMBLE_Q} ..."
    )
    start = time.time()
    rows = run_hardened(
        ensembles=("noisy_pure", "random_mixed"),
        sizes=SIZES,
        noise_models=NOISE_MODELS,
        rates=RATES,
        ensemble_q=ENSEMBLE_Q,
        budget=BUDGET,
        n_states=N_STATES,
        n_trials=N_TRIALS,
        seed=SEED,
        max_random_mixed_n=MAX_RANDOM_MIXED_N,
    )
    table = crossover_table(rows)
    wall = time.time() - start

    metadata = {
        "budget": BUDGET, "n_states": N_STATES, "n_trials": N_TRIALS, "seed": SEED,
        "ensemble_q": ENSEMBLE_Q, "sizes": list(SIZES), "noise_models": list(NOISE_MODELS),
        "rates": list(RATES), "max_random_mixed_n": MAX_RANDOM_MIXED_N,
        "paired_test": "state-level MSE difference (single - collective); the n_states "
                       "delta(s) are iid so z=mean/SE is valid.  Winner called at |z| > 2, "
                       "else 'tie'.  Noise is modelled only on the collective route, so "
                       "single and collective per-state MSE are ~independent (pairing is a "
                       "valid framing, not a power gain).  No family-wise correction across "
                       "cells; the 12 (noise,rate) cells at fixed (ensemble,n) share the "
                       "single-copy sample so are positively correlated.  A crossover boundary "
                       "is flagged 'ambiguous' when the winner sequence is non-monotone in "
                       "single-copy<tie<collective order, a 'tie' sits below the crossover n, "
                       "OR the boundary |z| < 3.  RMSE intervals are 68% bootstrap over states.",
        "wall_seconds": round(wall, 1),
    }
    save_hardened(rows, table, OUT, metadata)
    print(f"Grid done in {wall:.1f}s -> {OUT.relative_to(REPO)} "
          f"({len(rows)} rows, {len(table)} crossover keys)\n")

    #, Part 3: single-copy RMSE growth vs n, with error bars, 
    print("Single-copy FAIR purity RMSE vs n  [noise-independent per (ensemble, n)]:")
    for ensemble in ("noisy_pure", "random_mixed"):
        seen: dict[int, dict] = {}
        for r in rows:
            if r["ensemble"] == ensemble and r["n"] not in seen:
                seen[r["n"]] = r
        ns = sorted(seen)
        print(f"\n  {ensemble}:")
        print(f"    {'n':>2} {'true_pur':>9} {'single_rmse':>12} {'68% CI':>19} {'factor':>8}")
        prev = None
        for n in ns:
            r = seen[n]
            rmse = r["single_rmse"]
            lo, hi = r["single_rmse_ci68"]
            factor = f"{rmse / prev:.2f}x" if prev else "-"
            print(f"    {n:>2} {r['mean_true_purity']:9.4f} {rmse:12.4f} "
                  f"[{lo:7.4f},{hi:7.4f}] {factor:>8}")
            prev = rmse
        rmses = [seen[n]["single_rmse"] for n in ns]
        full = _fit_factor(ns, rmses)
        tail = _fit_factor([n for n in ns if n >= 5], [seen[n]["single_rmse"] for n in ns if n >= 5]) \
            if sum(n >= 5 for n in ns) >= 2 else float("nan")
        print(f"    -> growth/qubit: full range {full:.2f}x; n>=5 tail {tail:.2f}x")

    #, Part 2: crossover table with error bars, 
    print("\n" + "=" * 78)
    print("CROSSOVER IN n (paired state-level test; C=collective s=single-copy .=tie):")
    for ensemble in ("noisy_pure", "random_mixed"):
        print(f"\n  {ensemble}:")
        print(f"    {'noise_model':>18} {'rate':>5} {'crossover_n':>11} {'flag':>10}   winners-by-n")
        symbol = {"collective": "C", "single-copy": "s", "tie": "."}
        for entry in table:
            if entry["ensemble"] != ensemble:
                continue
            winners = entry["winners_by_n"]  # int -> verdict
            marks = " ".join(f"{n}:{symbol[winners[n]]}" for n in sorted(winners))
            cx = entry["crossover_n"]
            cz = entry["crossover_z"]
            flag = "AMBIGUOUS" if entry["ambiguous"] else ("resolved" if cx else "no-cross")
            zstr = f"z={cz:+.1f}" if cz is not None else ""
            print(f"    {entry['noise_model']:>18} {entry['rate']:>5} "
                  f"{str(cx):>11} {flag:>10} {zstr:>8}   {marks}")

    #, Part 2 detail: paired difference at the crossover-relevant cells, 
    print("\nPaired MSE difference (single - collective) +- SE at noisy_pure cells "
          "near each boundary [z>2 => collective wins]:")
    for entry in table:
        if entry["ensemble"] != "noisy_pure" or entry["crossover_n"] is None:
            continue
        nm, rate, cx = entry["noise_model"], entry["rate"], entry["crossover_n"]
        cells = [r for r in rows if r["ensemble"] == "noisy_pure"
                 and r["noise_model"] == nm and r["rate"] == rate and cx - 1 <= r["n"] <= cx]
        for r in sorted(cells, key=lambda r: r["n"]):
            print(f"    {nm:>18} rate={rate} n={r['n']}: "
                  f"diff={r['paired_mse_diff']:+.5f} +- {r['paired_mse_diff_se']:.5f} "
                  f"(z={r['paired_z']:+.1f}) -> {r['winner']}")

    print("\n(single-copy RMSE is the ideal-baseline shadow estimator; noise degrades "
          "only the collective route.)")


if __name__ == "__main__":
    main()
