"""Scaling study: single-copy vs collective purity estimation vs system size.

    .venv/bin/python experiments/run_scaling.py

Measures the copy-fair single-copy shadow purity RMSE and the collective (2-copy
SWAP) RMSE as a function of ``n`` for two ensembles at a fixed copy budget, then
locates the crossover in ``n`` (the size at which collective starts beating
single-copy) per noise model and rate.  Saves the full grid to
results/scaling_crossover.json and prints the four report questions with numbers.

Ensembles
    noisy_pure, (1-q)|psi><psi| + q I/2^n, |psi> Haar; purity stays O(1)
                   (the realistic NISQ regime).  Swept for all n.
    random_mixed, full-rank Ginibre depolarized at q; purity collapses toward
                   2^-n (the old highly-mixed ensemble).  Capped at n<=6 (dense
                   O(2^{2n}) sampling / channel cost).

Both single-copy and collective use the copy-fair estimator: single-copy is the
EXACT full pairwise U-statistic over all snapshot pairs (tuples are free
post-processing), collective is budget//2 SWAP-test measurements (2 copies each).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from anrl.benchmark import run_scaling, save_scaling

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "scaling_crossover.json"

BUDGET = 2000
N_STATES = 12
SEED = 0
ENSEMBLE_Q = 0.1  # depolarizing weight of the ensemble states themselves
SIZES = (2, 3, 4, 5, 6, 7, 8)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.0, 0.02, 0.05, 0.1)
MAX_RANDOM_MIXED_N = 6  # dense-sampling cap for the collapsing ensemble


def _fit_growth_per_qubit(sizes: list[int], rmses: list[float]) -> float:
    """Least-squares slope of ``log(rmse)`` vs ``n`` -> multiplicative factor/qubit."""
    xs = np.asarray(sizes, dtype=np.float64)
    ys = np.log(np.clip(np.asarray(rmses, dtype=np.float64), 1e-12, None))
    slope = float(np.polyfit(xs, ys, 1)[0])
    return float(np.exp(slope))


def main() -> None:
    print(
        f"Running scaling grid: sizes={SIZES}, noise={NOISE_MODELS}, rates={RATES}, "
        f"budget={BUDGET}, {N_STATES} states/cell, q={ENSEMBLE_Q} ..."
    )
    start = time.time()
    rows = run_scaling(
        ensembles=("noisy_pure", "random_mixed"),
        sizes=SIZES,
        noise_models=NOISE_MODELS,
        rates=RATES,
        ensemble_q=ENSEMBLE_Q,
        budget=BUDGET,
        n_states=N_STATES,
        seed=SEED,
        max_random_mixed_n=MAX_RANDOM_MIXED_N,
    )
    wall = time.time() - start

    metadata = {
        "budget": BUDGET, "n_states": N_STATES, "seed": SEED, "ensemble_q": ENSEMBLE_Q,
        "sizes": list(SIZES), "noise_models": list(NOISE_MODELS), "rates": list(RATES),
        "max_random_mixed_n": MAX_RANDOM_MIXED_N,
        "estimators": {
            "single_copy": "copy-fair: EXACT full pairwise U-statistic over all snapshot "
                           "pairs (O(M^2 n) per-qubit factorization); tuples are free "
                           "post-processing, so no subsampling handicap.",
            "collective": "budget//2 two-copy SWAP-test measurements (2 copies each); "
                          "depolarizing closed form, amp-damp/dephasing via per-qubit Kraus.",
        },
        "note": "single-copy RMSE is noise-independent for a fixed (ensemble, n).",
    }
    save_scaling(rows, OUT, metadata)
    print(f"Grid done in {wall:.1f}s -> {OUT.relative_to(REPO)} ({len(rows)} rows)\n")

    #, single-copy RMSE vs n (noise-independent) + growth factor per qubit, 
    print("Single-copy FAIR purity RMSE vs n  [noise-independent per (ensemble, n)]:")
    for ensemble in ("noisy_pure", "random_mixed"):
        seen: dict[int, tuple[float, float]] = {}
        for r in rows:
            if r["ensemble"] == ensemble and r["n"] not in seen:
                seen[r["n"]] = (r["single_rmse"], r["mean_true_purity"])
        ns = sorted(seen)
        print(f"\n  {ensemble}:")
        print(f"    {'n':>2} {'true_purity':>12} {'single_rmse':>12} {'factor':>8}")
        prev = None
        for n in ns:
            rmse, pur = seen[n]
            factor = f"{rmse / prev:.2f}x" if prev else "-"
            print(f"    {n:>2} {pur:12.4f} {rmse:12.4f} {factor:>8}")
            prev = rmse
        growth = _fit_growth_per_qubit(ns, [seen[n][0] for n in ns])
        print(f"    -> geometric growth factor per qubit (log-linear fit): {growth:.2f}x")

    #, Q3: crossover in n per (noise model, rate), 
    print("\n" + "=" * 72)
    print("CROSSOVER IN n (smallest n where collective beats single-copy):")
    for ensemble in ("noisy_pure", "random_mixed"):
        print(f"\n  {ensemble}:")
        print(f"    {'noise_model':>18} {'rate':>6} {'crossover_n':>12}   winners-by-n")
        for noise in NOISE_MODELS:
            for rate in RATES:
                cells = sorted(
                    (r for r in rows if r["ensemble"] == ensemble
                     and r["noise_model"] == noise and r["rate"] == rate),
                    key=lambda r: r["n"],
                )
                winners = {c["n"]: c["winner"] for c in cells}
                crossover = next(
                    (c["n"] for c in cells if c["winner"] == "collective"), None
                )
                marks = " ".join(
                    f"{n}:{'C' if winners[n] == 'collective' else 's'}" for n in sorted(winners)
                )
                cx = str(crossover) if crossover is not None else "none"
                print(f"    {noise:>18} {rate:>6} {cx:>12}   {marks}")

    print("\n(C = collective wins, s = single-copy wins, at that n.)")


if __name__ == "__main__":
    main()
