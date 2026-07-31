"""PASS 37.2: the 27 held-out cells under RAW / CLIPPED / SHRUNK.

PASS 36 covered only the noisy-pure k=2 grid.  The held-out ensembles -- Haar-pure,
low-rank and GHZ-noisy -- supply 27 of the paper's crossover cells and were not
tested against clipping.

This reproduces ``experiments/run_stress_test.py`` part 4 exactly, on its committed
seeds, and re-scores it under the three estimators.  That part uses RULE 3 (see
``run_crossover_rule_audit.py``): the measured single-copy RMSE is compared against
the EXACT collective floor from theory, with the sustained rule.  The collective
side is untouched, so only the single-copy column moves.

Seeding, from ``run_stress_test.py``:
    state  rng = default_rng([SEED, ENS_ID[ens], n, s])
    draws  rng = default_rng([SEED, n, k, s, 2]), budgets in order (2000, 8000)

VALIDATION GATE: under RAW the measured n* must reproduce ``stress_test.json``
part 4 cell for cell.  Reported before anything else.

Writes ``results/pass37_heldout_clipping.json``.
Run:  PYTHONPATH=. python -m experiments.run_heldout_clipping
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.budget import moment_ustat_linear
from anrl.benchmark.constrained import clip_moment, shrink_moment
from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank
from anrl.benchmark.moments import moment
from anrl.theory.general import predicted_collective_rmse_general, sample_batched_general

R = Path(__file__).resolve().parent.parent / "results"

SEED = 0
BUDGETS = (2000, 8000)
BUDGET = 2000
K = 2
SIZES = (2, 3, 4, 5, 6)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.05, 0.1, 0.3)
ENSEMBLES = {"haar_pure": False, "low_rank": False, "ghz_noisy": True}
_ENS_ID = {"haar_pure": 0, "low_rank": 1, "ghz_noisy": 2}
N_MEAS_STATES = {False: 24, True: 1}
N_TRIALS = {False: 6, True: 36}


def _make_state(ens: str, n: int, s: int):
    rng = np.random.default_rng([SEED, _ENS_ID[ens], n, s])
    if ens == "haar_pure":
        return haar_pure(n, rng)
    if ens == "low_rank":
        return low_rank(n, 2, rng)
    return ghz_noisy(n, 0.15, rng)


def _unit(task):
    """Raw per-trial estimates at budget 2000 for one (ens, n, state), committed seeds."""
    ens, n, s = task
    state = _make_state(ens, n, s)
    truth = moment(state.density_matrix(), K)
    rng = np.random.default_rng([SEED, n, K, s, 2])
    # budgets are drawn in order; 2000 comes first, so its draws are the first block
    est = [moment_ustat_linear(sample_batched_general(state, BUDGET, rng), K)
           for _ in range(N_TRIALS[ENSEMBLES[ens]])]
    return {"ensemble": ens, "n": n, "state": s, "truth": float(truth),
            "raw": [float(e) for e in est]}


def _sustained(sizes, wins):
    ns = sorted(sizes)
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None


def main() -> None:
    tasks = [(e, n, s) for e in ENSEMBLES for n in SIZES
             for s in range(N_MEAS_STATES[ENSEMBLES[e]])]
    with ProcessPoolExecutor(max_workers=3) as pool:
        units = list(pool.map(_unit, tasks))

    # single-copy RMSE per (ens, n) under each estimator, pooled exactly as the sweep does
    rmse = {kind: {} for kind in ("raw", "clipped", "shrunk")}
    for e in ENSEMBLES:
        for n in SIZES:
            us = [u for u in units if u["ensemble"] == e and u["n"] == n]
            sig = float(np.sqrt(np.mean([(x - u["truth"]) ** 2
                                         for u in us for x in u["raw"]])))
            for kind in rmse:
                sq = []
                for u in us:
                    xs = np.asarray(u["raw"], dtype=float)
                    y = (xs if kind == "raw" else
                         clip_moment(xs, n, K) if kind == "clipped" else
                         shrink_moment(xs, n, K, sig))
                    sq.extend(((y - u["truth"]) ** 2).tolist())
                rmse[kind][(e, n)] = float(np.sqrt(np.mean(sq)))

    rows = []
    for e in ENSEMBLES:
        rho_by_n = {n: [_make_state(e, n, s).density_matrix()
                        for s in range(N_MEAS_STATES[ENSEMBLES[e]])] for n in SIZES}
        for nm in NOISE_MODELS:
            for g in RATES:
                coll = {n: predicted_collective_rmse_general(rho_by_n[n], K, nm, g, BUDGET, n)
                        for n in SIZES}
                cell = {"ensemble": e, "noise": nm, "rate": g,
                        "collective": {str(n): coll[n] for n in SIZES}}
                for kind in ("raw", "clipped", "shrunk"):
                    wins = {n: rmse[kind][(e, n)] > coll[n] for n in SIZES}
                    cell[f"measured_n_{kind}"] = _sustained(SIZES, wins)
                    cell[f"single_{kind}"] = {str(n): rmse[kind][(e, n)] for n in SIZES}
                rows.append(cell)

    # ---- VALIDATION GATE
    committed = {(c["ensemble"], c["noise"], c["rate"]): c
                 for c in json.loads((R / "stress_test.json").read_text())["part4"]}
    ok = tot = 0
    for r in rows:
        c = committed.get((r["ensemble"], r["noise"], r["rate"]))
        if c is None:
            continue
        tot += 1
        ok += (r["measured_n_raw"] == c["measured_n"])
    print(f"VALIDATION: RAW measured n* reproduces stress_test.json part 4 in {ok}/{tot} cells\n")

    print(f"  {'ensemble':>10}{'noise':>19}{'rate':>6}{'pred':>6}"
          f"{'RAW':>6}{'CLIP':>6}{'SHRINK':>8}")
    for r in rows:
        c = committed.get((r["ensemble"], r["noise"], r["rate"]))
        r["predicted_n"] = None if c is None else c["predicted_n"]
        print(f"  {r['ensemble']:>10}{r['noise']:>19}{r['rate']:>6}"
              f"{str(r['predicted_n']):>6}{str(r['measured_n_raw']):>6}"
              f"{str(r['measured_n_clipped']):>6}{str(r['measured_n_shrunk']):>8}")

    print("\nPER ENSEMBLE (raw -> clipped)")
    for e in ENSEMBLES:
        sel = [r for r in rows if r["ensemble"] == e]
        res_raw = [r for r in sel if r["measured_n_raw"] is not None]
        res_clip = [r for r in sel if r["measured_n_clipped"] is not None]
        both = [r for r in sel if r["measured_n_raw"] is not None
                and r["measured_n_clipped"] is not None]
        sh = [r["measured_n_clipped"] - r["measured_n_raw"] for r in both]
        lost = sum(1 for r in sel if r["measured_n_raw"] is not None
                   and r["measured_n_clipped"] is None)
        unchanged = sum(1 for d in sh if d == 0)
        print(f"  {e:>10}: resolve {len(res_raw)} -> {len(res_clip)} of {len(sel)}; "
              f"unchanged {unchanged}, shifted {len(sh)-unchanged} "
              f"(deltas {sorted(set(sh)) if sh else '-'}), lost {lost}")

    (R / "pass37_heldout_clipping.json").write_text(json.dumps({
        "description": "PASS 37.2: the 27 held-out crossover cells under RAW / CLIPPED / "
                       "SHRUNK, reproducing run_stress_test.py part 4 (RULE 3)",
        "validation_raw_reproduces_committed": f"{ok}/{tot}",
        "rows": rows,
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass37_heldout_clipping.json'}")


if __name__ == "__main__":
    main()
