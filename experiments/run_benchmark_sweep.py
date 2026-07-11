"""Run the single-copy vs collective robustness sweep and report the result.

    .venv/bin/python experiments/run_benchmark_sweep.py

Runs the full (task k, noise model, rate, size) sweep at a fixed copy budget,
saves the machine-readable results to results/benchmark_sweep.json, and prints
the three questions that matter with their numbers, under BOTH single-copy
estimator conventions:

* subsampled — the O(M) = n_snapshots//k tuple convention (as specified).
* fair — the copy-optimal U-statistic (many more tuples; forming tuples costs
  no copies), i.e. the honest single-copy performance.
"""

from __future__ import annotations

import time
from pathlib import Path

from anrl.benchmark import run_sweep, save_sweep

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "benchmark_sweep.json"

BUDGET = 2000
N_STATES = 12
SEED = 0
KS = (2, 3, 4)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.0, 0.02, 0.05, 0.1)
SIZES = (2, 3, 4)


def main() -> None:
    print(f"Running sweep: k={KS}, noise={NOISE_MODELS}, rates={RATES}, sizes={SIZES}, "
          f"budget={BUDGET}, {N_STATES} states/cell ...")
    start = time.time()
    rows = run_sweep(SIZES, KS, NOISE_MODELS, RATES, BUDGET, N_STATES, seed=SEED)
    wall = time.time() - start

    metadata = {
        "budget": BUDGET, "n_states": N_STATES, "seed": SEED, "sizes": list(SIZES),
        "ks": list(KS), "noise_models": list(NOISE_MODELS), "rates": list(RATES),
        "single_estimators": {
            "subsampled": "O(M) = n_snapshots // k tuple U-statistic (the specified convention)",
            "fair": "copy-optimal U-statistic (50000 tuples); forming tuples costs no copies",
        },
        "note": "single-copy shadow RMSE is noise-independent; noise degrades only the collective route.",
    }
    save_sweep(rows, OUT, metadata)
    print(f"Sweep done in {wall:.1f}s -> {OUT.relative_to(REPO)} ({len(rows)} cells)\n")

    print("Single-copy shadow RMSE by (n, k)  [noise-independent]:")
    print(f"  {'(n,k)':8s} {'subsampled O(M)':>16s} {'fair (copy-optimal)':>20s}")
    seen = set()
    for r in rows:
        if (r["n"], r["k"]) not in seen:
            seen.add((r["n"], r["k"]))
            print(f"  n={r['n']} k={r['k']}   {r['single_rmse_subsampled']:16.3f} {r['single_rmse_fair']:20.3f}")

    for convention, wkey, fkey in (
        ("SUBSAMPLED O(M) single-copy (as specified)", "winner_subsampled", "factor_subsampled"),
        ("FAIR copy-optimal single-copy (honest comparison)", "winner_fair", "factor_fair"),
    ):
        print(f"\n================ {convention} ================")
        # Q1 — every task k?
        print("Q1. Collective vs single-copy per moment order k:")
        for k in KS:
            cells = [r for r in rows if r["k"] == k]
            wins = sum(r[wkey] == "collective" for r in cells)
            factors = [r[fkey] for r in cells]
            print(f"  k={k}: collective wins {wins}/{len(cells)}; factor {min(factors):.2f}x .. {max(factors):.0f}x")
        # Q2 — every noise model?
        print("Q2. Collective vs single-copy per noise model:")
        for noise in NOISE_MODELS:
            cells = [r for r in rows if r["noise_model"] == noise]
            wins = sum(r[wkey] == "collective" for r in cells)
            print(f"  {noise:18s}: collective wins {wins}/{len(cells)}")
        # Q3 — crossovers?
        crossovers = [r for r in rows if r[wkey] == "single-copy"]
        print(f"Q3. Crossovers (single-copy wins): {len(crossovers)}")
        for r in crossovers:
            print(f"  n={r['n']} k={r['k']} {r['noise_model']} rate={r['rate']}: "
                  f"single_fair={r['single_rmse_fair']:.4f} single_sub={r['single_rmse_subsampled']:.4f} "
                  f"collective={r['collective_rmse']:.4f}")
        if not crossovers:
            print("  None — collective wins every cell.")


if __name__ == "__main__":
    main()
