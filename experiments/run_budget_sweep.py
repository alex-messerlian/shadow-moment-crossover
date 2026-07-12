"""Out-of-sample test of the crossover law: copy-budget sweep.

    .venv/bin/python experiments/run_budget_sweep.py

Sweeps the copy budget over multipliers of the 2000 baseline (feasibility-capped
per (n,k)) on the noisy-pure ensemble, and reports the three locked-in
predictions with numbers.  No tuning: the law's parameters were fixed from
zero-noise data before this run.

  P1  single-copy RMSE prop M^{-alpha};   law: alpha = 0.5
  P2  collective error saturates at the budget-independent bias floor
      [1 - (1-g)^{k n}] Tr(rho^k)
  P3  the crossover n* moves +1 qubit per 4x budget
      (k=2, g=0.05 predicted n* = 7, 8, 9 at 1x, 4x, 16x)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from anrl.benchmark import (
    crossover_table,
    run_budget_sweep,
    save_budget_sweep,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "budget_scaling.json"

N_STATES = 48
N_TRIALS = 8
SEED = 0
ENSEMBLE_Q = 0.1
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.05, 0.1)
SIZES_BY_K = {2: tuple(range(2, 10)), 3: tuple(range(2, 9)), 4: tuple(range(2, 7))}
MULT = {2000: "1x", 8000: "4x", 32000: "16x", 128000: "64x", 500: "0.25x"}
# Locked-in P3 predictions (k=2, g=0.05): crossover n* at 1x, 4x, 16x.
P3_PREDICTED = {2000: 7, 8000: 8, 32000: 9}


def main() -> None:
    print(f"Budget sweep: sizes_by_k={SIZES_BY_K}, noise={NOISE_MODELS}, rates={RATES}, "
          f"{N_STATES} states x {N_TRIALS} trials, q={ENSEMBLE_Q} ...")
    start = time.time()
    rows, alpha_fits = run_budget_sweep(
        sizes_by_k=SIZES_BY_K, noise_models=NOISE_MODELS, rates=RATES,
        ensemble_q=ENSEMBLE_Q, n_states=N_STATES, n_trials=N_TRIALS, seed=SEED,
        return_alpha_fits=True,
    )
    table = crossover_table(rows, group_keys=("k", "budget", "noise_model", "rate"))
    wall = time.time() - start
    metadata = {
        "ensemble": "noisy_pure, q=%.3f" % ENSEMBLE_Q, "baseline_budget": 2000,
        "n_states": N_STATES, "n_trials": N_TRIALS, "seed": SEED,
        "sizes_by_k": {str(k): list(v) for k, v in SIZES_BY_K.items()},
        "noise_models": list(NOISE_MODELS), "rates": list(RATES),
        "single_estimator": "EXACT full U-statistic (M-linear for k=2,3; reference for k=4), "
                            "budgets nested within a trial (one M_max sample per trial).",
        "predictions": {"P1_alpha": 0.5, "P2_floor": "[1-(1-g)^(k n)] Tr(rho^k)",
                        "P3_k2_g0.05": P3_PREDICTED},
        "wall_seconds": round(wall, 1),
    }
    save_budget_sweep(rows, table, OUT, metadata, alpha_fits=alpha_fits)
    print(f"Sweep done in {wall:.1f}s -> {OUT.relative_to(REPO)} ({len(rows)} rows)\n")

    # ---- P1: budget-scaling exponent alpha (single-copy is noise-independent) ----
    print("=" * 78)
    print("P1. Single-copy RMSE budget-scaling exponent alpha (law: 0.5)")
    print("    single_rmse ~ M^-alpha; alpha with a state-bootstrap SE (nested budgets).")
    fit = {(a["n"], a["k"]): a for a in alpha_fits}
    for k in (2, 3, 4):
        print(f"\n  k={k}:  {'n':>2} | {'budgets (M: rmse)':<52} | {'alpha+-se':>12}")
        for n in SIZES_BY_K[k]:
            a = fit.get((n, k))
            if a is None:
                continue
            curve = "  ".join(f"{MULT.get(b, b)}:{r:.4f}" for b, r in zip(a["budgets"], a["single_rmse"]))
            print(f"       {n:>2} | {curve:<52} | {a['alpha']:6.3f}+-{a['alpha_se']:5.3f}")

    # ---- P2: collective bias-floor plateau, measured vs predicted ----
    print("\n" + "=" * 78)
    print("P2. Collective error plateaus at a budget-independent floor?")
    print("    measured_bias = |Tr(sigma^k)-Tr(rho^k)|; predicted = [1-(1-g)^(kn)] Tr(rho^k).")
    print("    coll_rmse(minB->maxB): falls toward the floor once variance dies; sat? = "
          "coll_rmse(maxB)/meas_bias <= 1.05 (still variance-limited if not).")
    for k in (2, 3, 4):
        for g in RATES:
            print(f"\n  k={k} g={g}:  {'noise':>18} {'n':>2} | "
                  f"{'coll_rmse: minB->maxB':>22} {'sat?':>4} | {'meas_bias':>9} {'pred_floor':>10} {'meas/pred':>9}")
            for nm in NOISE_MODELS:
                for n in SIZES_BY_K[k]:
                    cells = sorted((r for r in rows if r["k"] == k and r["n"] == n
                                    and r["noise_model"] == nm and r["rate"] == g),
                                   key=lambda r: r["budget"])
                    if len(cells) < 2 or n not in (2, 5, SIZES_BY_K[k][-1]):
                        continue
                    lo, hi = cells[0], cells[-1]
                    ratio = hi["measured_bias"] / hi["predicted_floor"] if hi["predicted_floor"] > 0 else float("nan")
                    sat = "yes" if hi["measured_bias"] > 0 and hi["collective_rmse"] / hi["measured_bias"] <= 1.05 else "no"
                    print(f"           {nm:>18} {n:>2} | {lo['collective_rmse']:8.4f} -> {hi['collective_rmse']:8.4f}  {sat:>4} | "
                          f"{hi['measured_bias']:9.4f} {hi['predicted_floor']:10.4f} {ratio:9.2f}")

    # ---- P3: crossover n* vs budget (k=2, g=0.05 is the locked prediction) ----
    print("\n" + "=" * 78)
    print("P3. Crossover n* vs budget (does it move +1 qubit per 4x?)")
    symbol = {"collective": "C", "single-copy": "s", "tie": "."}
    for k in (2, 3, 4):
        print(f"\n  k={k}:")
        print(f"    {'noise':>18} {'rate':>5} {'budget':>7} {'n*':>4} {'flag':>10}   winners-by-n"
              + ("   [P3 pred]" if k == 2 else ""))
        for nm in NOISE_MODELS:
            for g in RATES:
                for b in sorted({r["budget"] for r in rows if r["k"] == k}):
                    entry = next((e for e in table if e["k"] == k and e["budget"] == b
                                  and e["noise_model"] == nm and e["rate"] == g), None)
                    if entry is None:
                        continue
                    marks = " ".join(f"{n}:{symbol[entry['winners_by_n'][n]]}" for n in sorted(entry["winners_by_n"]))
                    cx = entry["crossover_n"]
                    flag = "AMBIGUOUS" if entry["ambiguous"] else ("resolved" if cx else "no-cross")
                    pred = ""
                    if k == 2 and abs(g - 0.05) < 1e-9 and b in P3_PREDICTED:
                        pred = f"   pred n*={P3_PREDICTED[b]}"
                    print(f"    {nm:>18} {g:>5} {MULT.get(b, b):>7} {str(cx):>4} {flag:>10}   {marks}{pred}")

    print("\n(single-copy improves ~1/sqrt(M); if the collective floor is budget-independent, "
          "the crossover n* rises with budget. Cells with |z|<3 / non-monotone flagged ambiguous.)")


if __name__ == "__main__":
    main()
