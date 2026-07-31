"""PASS 37.2 (noisy-pure half): the 68 noisy-pure crossover cells under clipping.

Reproduces ``run_budget_sweep.py`` and ``run_moment_sweep.py`` on their committed
seeds, capturing the RAW per-trial estimates rather than only their squared errors,
then re-scores every cell under RAW / CLIPPED / SHRUNK through the identical
``_aggregate_cell`` / ``crossover_table`` pipeline (RULE 2, the paired z-test).

Together with ``run_heldout_clipping.py`` this covers all 83 resolved cells, so the
paper's aggregate accuracy figures can be recomputed exactly under each estimator.

VALIDATION GATE: under RAW every cell's crossover must reproduce the committed
tables.  Reported before anything else.

Writes ``results/pass37_noisypure_clipping.json``.
Run:  PYTHONPATH=. python -m experiments.run_noisypure_clipping
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.budget import moment_ustat_linear, sample_batched
from anrl.benchmark.budget_sweep import _collective_signal, budgets_for
from anrl.benchmark.constrained import clip_moment, shrink_moment
from anrl.benchmark.ensembles import noisy_pure
from anrl.benchmark.hardened import (
    _NOISE_ID, _aggregate_cell, _bootstrap_rmse_ci, _rate_key, crossover_table,
)
from anrl.benchmark.moments import collective_moment_estimate, moment

R = Path(__file__).resolve().parent.parent / "results"

SEED, Q = 0, 0.1
MAX_WORKERS = 6
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")

# run_budget_sweep.py
BS_SIZES_BY_K = {2: (2, 3, 4, 5, 6, 7, 8, 9), 3: (2, 3, 4, 5, 6, 7, 8), 4: (2, 3, 4, 5, 6)}
BS_RATES = (0.05, 0.1)
BS_STATES, BS_TRIALS = 48, 8
# run_moment_sweep.py (sweep_hardened)
MS_SIZES_BY_K = {2: tuple(range(2, 9)), 3: tuple(range(2, 9)), 4: tuple(range(2, 9))}
MS_RATES = (0.0, 0.02, 0.05, 0.1)
MS_STATES, MS_TRIALS, MS_BUDGET = 48, 10, 2000
MS_CAPS = {2: 10, 3: 8, 4: 8}


def _bs_unit(task):
    """budget_sweep: raw estimates per (k, budget) for one (n, state)."""
    n, s = task
    state = noisy_pure(n, Q, np.random.default_rng([SEED, n, s, 0]))
    density = state.density_matrix()
    out = {"n": n, "state": s, "truth": {}, "raw": {}, "coll_se": {}}
    for k in (2, 3, 4):
        if n not in BS_SIZES_BY_K[k]:
            continue
        budgets = budgets_for(n, k)
        m_max = max(budgets)
        tm = moment(density, k)
        out["truth"][str(k)] = float(tm)
        for b in budgets:
            out["raw"][f"{k}|{b}"] = []
        for t in range(BS_TRIALS):
            rng = np.random.default_rng([SEED, n, s, 1, k, t])
            snaps = sample_batched(state, m_max, rng)
            for b in budgets:
                out["raw"][f"{k}|{b}"].append(float(moment_ustat_linear(snaps[:b], k)))
        for nm in NOISE_MODELS:
            for rate in BS_RATES:
                signal = _collective_signal(density, n, k, nm, rate, tm)
                for b in budgets:
                    crng = np.random.default_rng(
                        [SEED, n, s, 2, k, b, _NOISE_ID[nm], _rate_key(rate)])
                    out["coll_se"][f"{k}|{b}|{nm}|{rate}"] = [
                        float((collective_moment_estimate(k, b // k, signal, crng) - tm) ** 2)
                        for _ in range(BS_TRIALS)]
    return out


def _ms_unit(task):
    """moment_sweep (sweep_hardened): raw estimates at M=2000 for one (n, state)."""
    n, s = task
    state = noisy_pure(n, Q, np.random.default_rng([SEED, n, s, 0]))
    density = state.density_matrix()
    out = {"n": n, "state": s, "truth": {}, "raw": {}, "coll_se": {}}
    for k in (2, 3, 4):
        if n > MS_CAPS[k]:
            continue
        tm = moment(density, k)
        out["truth"][str(k)] = float(tm)
        rng = np.random.default_rng([SEED, n, s, 1, k])
        from anrl.benchmark.moment_ustats import exact_moment_ustatistic
        from anrl.benchmark.scaling import snapshots_factored
        out["raw"][str(k)] = [
            float(exact_moment_ustatistic(snapshots_factored(state, MS_BUDGET, rng), k))
            for _ in range(MS_TRIALS)]
        for nm in NOISE_MODELS:
            for rate in MS_RATES:
                signal = _collective_signal(density, n, k, nm, rate, tm)
                crng = np.random.default_rng(
                    [SEED, n, s, 2, k, _NOISE_ID[nm], _rate_key(rate)])
                out["coll_se"][f"{k}|{nm}|{rate}"] = [
                    float((collective_moment_estimate(k, MS_BUDGET // k, signal, crng) - tm) ** 2)
                    for _ in range(MS_TRIALS)]
    return out


def _project(xs, truth, n, k, kind, sigma):
    xs = np.asarray(xs, dtype=float)
    y = (xs if kind == "raw" else clip_moment(xs, n, k) if kind == "clipped"
         else shrink_moment(xs, n, k, sigma))
    return float(np.mean((y - truth) ** 2))


def main() -> None:
    bs_tasks = [(n, s) for n in range(2, 10) for s in range(BS_STATES)]
    ms_tasks = [(n, s) for n in range(2, 9) for s in range(MS_STATES)]
    # Bounded pool: each worker peaks near 0.6 GB on the M=128000 cells, so an
    # unbounded pool exhausts memory and the run dies without writing.
    import sys
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        bs = []
        for i, u in enumerate(pool.map(_bs_unit, bs_tasks), 1):
            bs.append(u)
            if i % 24 == 0:
                print(f"  budget-sweep units {i}/{len(bs_tasks)}", flush=True)
        ms = []
        for i, u in enumerate(pool.map(_ms_unit, ms_tasks), 1):
            ms.append(u)
            if i % 48 == 0:
                print(f"  moment-sweep units {i}/{len(ms_tasks)}", flush=True)

    tables = {}
    for kind in ("raw", "clipped", "shrunk"):
        boot = np.random.default_rng([SEED, 999])
        # ---- budget sweep rows
        bs_rows = []
        for k in (2, 3, 4):
            for n in BS_SIZES_BY_K[k]:
                for b in budgets_for(n, k):
                    us = [u for u in bs if u["n"] == n]
                    key = f"{k}|{b}"
                    sig = float(np.sqrt(np.mean(
                        [(x - u["truth"][str(k)]) ** 2 for u in us for x in u["raw"][key]])))
                    smse = np.asarray([_project(u["raw"][key], u["truth"][str(k)], n, k, kind, sig)
                                       for u in us])
                    sci = _bootstrap_rmse_ci(smse, np.random.default_rng([SEED, 1, n, k, b]))
                    for nm in NOISE_MODELS:
                        for rate in BS_RATES:
                            cmse = np.asarray([float(np.mean(u["coll_se"][f"{k}|{b}|{nm}|{rate}"]))
                                               for u in us])
                            bs_rows.append({"n": n, "k": k, "budget": b, "noise_model": nm,
                                            "rate": rate,
                                            **_aggregate_cell(smse, cmse, sci, boot)})
        # ---- moment sweep rows
        ms_rows = []
        for k in (2, 3, 4):
            for n in MS_SIZES_BY_K[k]:
                if n > MS_CAPS[k]:
                    continue
                us = [u for u in ms if u["n"] == n]
                sig = float(np.sqrt(np.mean(
                    [(x - u["truth"][str(k)]) ** 2 for u in us for x in u["raw"][str(k)]])))
                smse = np.asarray([_project(u["raw"][str(k)], u["truth"][str(k)], n, k, kind, sig)
                                   for u in us])
                sci = _bootstrap_rmse_ci(smse, np.random.default_rng([SEED, 2, n, k]))
                for nm in NOISE_MODELS:
                    for rate in MS_RATES:
                        cmse = np.asarray([float(np.mean(u["coll_se"][f"{k}|{nm}|{rate}"]))
                                           for u in us])
                        ms_rows.append({"n": n, "k": k, "noise_model": nm, "rate": rate,
                                        **_aggregate_cell(smse, cmse, sci, boot)})
        tables[kind] = {
            "budget": crossover_table(bs_rows, group_keys=("k", "budget", "noise_model", "rate")),
            "moment": crossover_table(ms_rows, group_keys=("k", "noise_model", "rate")),
        }

    # ---- VALIDATION GATE
    def as_map(tab, with_budget):
        return {((e["k"], e["budget"], e["noise_model"], e["rate"]) if with_budget
                 else (e["k"], e["noise_model"], e["rate"])): e["crossover_n"] for e in tab}

    ok = tot = 0
    for name, path, wb in (("budget", "budget_scaling.json", True),
                           ("moment", "moment_sweep_corrected.json", False)):
        comm = json.loads((R / path).read_text())["crossover_table"]
        cm = as_map(comm, wb)
        mine = as_map(tables["raw"][name], wb)
        for key, v in cm.items():
            tot += 1
            ok += (mine.get(key) == v)
    print(f"VALIDATION: RAW crossover reproduces the committed noisy-pure tables "
          f"in {ok}/{tot} cells\n")

    for kind in ("raw", "clipped", "shrunk"):
        res = sum(1 for t in tables[kind].values() for e in t if e["crossover_n"] is not None)
        tot_c = sum(len(t) for t in tables[kind].values())
        print(f"  {kind:9s}: {res}/{tot_c} noisy-pure cells resolve a crossover")

    (R / "pass37_noisypure_clipping.json").write_text(json.dumps({
        "description": "PASS 37.2 (noisy-pure): the 96 noisy-pure crossover cells under "
                       "RAW / CLIPPED / SHRUNK via the committed paired-test pipeline",
        "validation_raw_reproduces_committed": f"{ok}/{tot}",
        "crossover_tables": tables,
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass37_noisypure_clipping.json'}")


if __name__ == "__main__":
    main()
