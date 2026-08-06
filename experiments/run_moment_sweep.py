"""Corrected, hardened moment-family sweep (k=2,3,4) on the noisy-pure ensemble.

    .venv/bin/python experiments/run_moment_sweep.py

Re-runs the single-copy vs collective moment sweep with the two defects fixed
(noisy-pure ensemble whose Tr(rho^k) stays O(1); sizes extended to n=2..8) and to
publication standard: >=48 states x >=10 trials per cell, bootstrap RMSE
intervals, and a paired state-level crossover test.  Uses the EXACT full
U-statistic for every k.  Saves the grid + crossover table to
results/moment_sweep_corrected.json and prints the four report questions.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from anrl.benchmark import crossover_table, run_moment_sweep, save_moment_sweep, skipped_cells

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "moment_sweep_corrected.json"

BUDGET = 2000
N_STATES = 48
N_TRIALS = 10
SEED = 0
ENSEMBLE_Q = 0.1
SIZES = (2, 3, 4, 5, 6, 7, 8)
KS = (2, 3, 4)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.0, 0.02, 0.05, 0.1)


def _fit_factor(sizes: list[int], rmses: list[float]) -> float:
    xs = np.asarray(sizes, dtype=np.float64)
    ys = np.log(np.clip(np.asarray(rmses, dtype=np.float64), 1e-12, None))
    return float(np.exp(np.polyfit(xs, ys, 1)[0]))


def main() -> None:
    print(
        f"Corrected moment sweep: sizes={SIZES}, k={KS}, noise={NOISE_MODELS}, rates={RATES}, "
        f"budget={BUDGET}, {N_STATES} states x {N_TRIALS} trials/cell, q={ENSEMBLE_Q} ..."
    )
    start = time.time()
    rows = run_moment_sweep(
        sizes=SIZES, ks=KS, noise_models=NOISE_MODELS, rates=RATES,
        ensemble_q=ENSEMBLE_Q, budget=BUDGET, n_states=N_STATES, n_trials=N_TRIALS, seed=SEED,
    )
    table = crossover_table(rows, group_keys=("k", "noise_model", "rate"))
    wall = time.time() - start
    skipped = skipped_cells(SIZES, KS)

    metadata = {
        "ensemble": "noisy_pure ((1-q)|psi><psi| + q I/2^n, Haar psi), q=%.3f" % ENSEMBLE_Q,
        "budget": BUDGET, "n_states": N_STATES, "n_trials": N_TRIALS, "seed": SEED,
        "sizes": list(SIZES), "ks": list(KS), "noise_models": list(NOISE_MODELS), "rates": list(RATES),
        "single_estimator": "EXACT full U-statistic for k=2,3,4 (exact_moment_ustatistic; "
                            "identical to the brute-force-verified reference estimators)",
        "budget_accounting": "single spends budget snapshots; collective spends k*(budget//k) copies",
        "paired_test": "state-level MSE diff (single - collective), |z|>2 winner else 'tie'; "
                       "68% bootstrap RMSE intervals. A crossover boundary is flagged "
                       "'ambiguous' when the winner sequence is non-monotone in "
                       "single-copy<tie<collective order, a 'tie' sits below the crossover n, "
                       "OR the boundary |z| < 3. Caveats: no family-wise correction; single-copy "
                       "sample shared across noise cells at a fixed (n,k) so those verdicts are "
                       "positively correlated.",
        "skipped_infeasible": skipped, "wall_seconds": round(wall, 1),
    }
    save_moment_sweep(rows, table, OUT, metadata)
    print(f"Sweep done in {wall:.1f}s -> {OUT.relative_to(REPO)} "
          f"({len(rows)} rows, {len(table)} crossover keys); skipped {len(skipped)} infeasible (n,k)\n")

    #, Q1: true Tr(rho^k) stays O(1), 
    print("Q1. Mean true Tr(rho^k) by (n, k)  [must stay O(1), not collapse to 0]:")
    print(f"    {'n':>2} " + " ".join(f"{'k='+str(k):>10}" for k in KS))
    for n in SIZES:
        cells = {k: next((r for r in rows if r["n"] == n and r["k"] == k), None) for k in KS}
        vals = " ".join(f"{cells[k]['mean_true_moment']:10.4f}" if cells[k] else f"{'--':>10}" for k in KS)
        print(f"    {n:>2} {vals}")

    #, Q2: single-copy RMSE growth per qubit, per k, 
    print("\nQ2. Single-copy FAIR RMSE vs n and growth factor per qubit, per k:")
    for k in KS:
        seen = {}
        for r in rows:
            if r["k"] == k and r["n"] not in seen:
                seen[r["n"]] = r["single_rmse"]
        ns = sorted(seen)
        print(f"\n  k={k}:  " + "  ".join(f"n{n}={seen[n]:.4f}" for n in ns))
        if len(ns) >= 2:
            full = _fit_factor(ns, [seen[n] for n in ns])
            tail_ns = [n for n in ns if n >= 5]
            tail = _fit_factor(tail_ns, [seen[n] for n in tail_ns]) if len(tail_ns) >= 2 else float("nan")
            print(f"        growth/qubit: full range {full:.2f}x; n>=5 tail {tail:.2f}x")

    #, Q3: crossover table, 
    print("\n" + "=" * 78)
    print("Q3. CROSSOVER IN n (paired test; C=collective s=single-copy .=tie):")
    symbol = {"collective": "C", "single-copy": "s", "tie": "."}
    for k in KS:
        print(f"\n  k={k}:")
        print(f"    {'noise_model':>18} {'rate':>5} {'xover_n':>8} {'flag':>10} {'z':>7}   winners-by-n")
        for entry in table:
            if entry["k"] != k:
                continue
            marks = " ".join(f"{n}:{symbol[entry['winners_by_n'][n]]}" for n in sorted(entry["winners_by_n"]))
            cx, cz = entry["crossover_n"], entry["crossover_z"]
            flag = "AMBIGUOUS" if entry["ambiguous"] else ("resolved" if cx else "no-cross")
            zstr = f"{cz:+.1f}" if cz is not None else ""
            print(f"    {entry['noise_model']:>18} {entry['rate']:>5} {str(cx):>8} {flag:>10} {zstr:>7}   {marks}")

    if skipped:
        print("\nSkipped (infeasible single-copy exact):", ", ".join(f"n{c['n']}k{c['k']}" for c in skipped))


if __name__ == "__main__":
    main()
