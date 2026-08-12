"""PASS 47.2(c)/(d), sharpened: can the statewise law ORDER individual states?

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/pass47_statewise_ranking.py

``pass47_perstate_validation.py`` shows the exact statewise law predicts each state's RMSE
to within trial noise -- but on the paper's four committed ensembles the PREDICTED spread
across states inside a cell (1.6--5.3%) is far below the measurement noise of that run
(14.4% at 24 trials), so those ensembles cannot test statewise sensitivity at all: the law
could be reading only the ensemble mean and would score the same.

This run removes that confound.  It uses the ``variable_rank`` ensemble, whose statewise
``M*`` spans an order of magnitude, and raises the trial count until the measurement noise
is well below the predicted spread, so a positive result means the law tracks the STATE and
not the family.  Reported as the rank correlation between predicted and measured RMSE within
each ``(n, budget)`` cell, plus the paired slope of measured against predicted.

Writes ``results/pass47_statewise_ranking.json``.  No paper edit.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.budget import moment_ustat_linear
from anrl.benchmark.ensembles import low_rank, noisy_pure
from anrl.benchmark.moments import moment
from anrl.theory.general import sample_batched_general
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights
from anrl.theory.variance import exact_single_copy_rmse

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass47_statewise_ranking.json"

SEED = 47
K = 2
SIZES = (3, 4, 5)
BUDGETS = (2000, 8000)
N_STATES = 14
N_TRIALS = 300          # relative RMSE noise ~ 1/sqrt(2 T) = 4.1%
MAX_WORKERS = 3
ENSEMBLES = ("variable_rank", "noisy_pure_q0.1")   # the second as the negative control
_ENS_ID = {"variable_rank": 5, "noisy_pure_q0.1": 0}


def make_state(ens: str, n: int, s: int):
    rng = np.random.default_rng([SEED, 55, _ENS_ID[ens], n, s])
    if ens == "variable_rank":
        return low_rank(n, int(rng.integers(1, min(8, 2 ** n) + 1)), rng)
    return noisy_pure(n, 0.1, rng)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def _worker(task):
    ens, n, s = task
    state = make_state(ens, n, s)
    weights = pauli_weights(n)
    m = pauli_expectations(state.density_matrix(), n)
    z1, z2 = exact_zeta1(m, n), exact_zeta2(m, n, weights)
    truth = moment(state.density_matrix(), K)
    rng = np.random.default_rng([SEED, 66, _ENS_ID[ens], n, s])
    out = {"zeta1": z1, "zeta2": z2, "m_star": z2 / (2 * z1), "purity": truth,
           "rank": int(state.components.shape[1])}
    for b in BUDGETS:
        sq = np.array([(moment_ustat_linear(sample_batched_general(state, b, rng), K) - truth) ** 2
                       for _ in range(N_TRIALS)])
        out[f"measured_{b}"] = float(np.sqrt(sq.mean()))
        out[f"measured_se_{b}"] = float(sq.std(ddof=1) / (2 * np.sqrt(sq.mean()) * np.sqrt(N_TRIALS)))
        out[f"predicted_{b}"] = exact_single_copy_rmse([z1, z2], K, b)
    return task, out


def main() -> None:
    t0 = time.time()
    grid = [(e, n, s) for e in ENSEMBLES for n in SIZES for s in range(N_STATES)]
    print(f"ranking grid: {len(grid)} states x {len(BUDGETS)} budgets x {N_TRIALS} trials")
    units = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, (task, res) in enumerate(pool.map(_worker, grid), 1):
            units["|".join(map(str, task))] = res
            if i % 14 == 0:
                print(f"    {i}/{len(grid)}", flush=True)

    cells = []
    for e in ENSEMBLES:
        for n in SIZES:
            for b in BUDGETS:
                keys = [f"{e}|{n}|{s}" for s in range(N_STATES)]
                pr = np.array([units[k][f"predicted_{b}"] for k in keys])
                me = np.array([units[k][f"measured_{b}"] for k in keys])
                se = np.array([units[k][f"measured_se_{b}"] for k in keys])
                ms = np.array([units[k]["m_star"] for k in keys])
                slope = float(np.polyfit(pr, me, 1)[0])
                cells.append({
                    "ensemble": e, "n": n, "budget": b, "n_states": N_STATES,
                    "predicted_rel_spread": float(pr.std(ddof=1) / pr.mean()),
                    "measurement_rel_noise": float((se / me).mean()),
                    "spread_over_noise": float((pr.std(ddof=1) / pr.mean()) / (se / me).mean()),
                    "spearman": spearman(pr, me),
                    "slope_measured_on_predicted": slope,
                    "median_abs_rel_dev": float(np.median(np.abs(pr - me) / me)),
                    "within_2se": int((np.abs(pr - me) <= 2 * se).sum()),
                    "m_star_spread_ratio": float(ms.max() / ms.min()),
                })
                c = cells[-1]
                print(f"  {e:16s} n={n} M={b:>5d}: pred spread {c['predicted_rel_spread']*100:5.1f}%  "
                      f"noise {c['measurement_rel_noise']*100:4.1f}%  ratio {c['spread_over_noise']:5.2f}  "
                      f"rho={c['spearman']:+.3f}  slope={slope:5.2f}  "
                      f"median |dev| {c['median_abs_rel_dev']*100:4.1f}%  "
                      f"within-2SE {c['within_2se']}/{N_STATES}  M* max/min {c['m_star_spread_ratio']:.2f}",
                      flush=True)

    by_ens = {}
    for e in ENSEMBLES:
        sub = [c for c in cells if c["ensemble"] == e]
        by_ens[e] = {
            "cells": len(sub),
            "mean_spearman": float(np.mean([c["spearman"] for c in sub])),
            "min_spearman": float(np.min([c["spearman"] for c in sub])),
            "mean_slope": float(np.mean([c["slope_measured_on_predicted"] for c in sub])),
            "mean_predicted_spread": float(np.mean([c["predicted_rel_spread"] for c in sub])),
            "mean_measurement_noise": float(np.mean([c["measurement_rel_noise"] for c in sub])),
            "median_abs_rel_dev": float(np.median([c["median_abs_rel_dev"] for c in sub])),
            "within_2se": f"{sum(c['within_2se'] for c in sub)}/{len(sub) * N_STATES}",
        }
        print(f"\n{e}: mean rho {by_ens[e]['mean_spearman']:+.3f} (min {by_ens[e]['min_spearman']:+.3f}), "
              f"mean slope {by_ens[e]['mean_slope']:.2f}, "
              f"median |dev| {by_ens[e]['median_abs_rel_dev']*100:.1f}%, "
              f"within-2SE {by_ens[e]['within_2se']}")

    payload = {
        "description": "PASS 47.2(c) sharpened: statewise RANKING power of the exact law",
        "config": {"seed": SEED, "k": K, "sizes": list(SIZES), "budgets": list(BUDGETS),
                   "n_states": N_STATES, "n_trials": N_TRIALS, "ensembles": list(ENSEMBLES)},
        "cells": cells,
        "by_ensemble": by_ens,
        "units": units,
        "reading": (
            "A cell tests statewise sensitivity only if spread_over_noise > 1. On variable_rank it "
            "does; on the paper's noisy-pure family it does not, at any trial count that is cheap, "
            "because the statewise spread there is intrinsically a few percent."
        ),
        "wall_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
