"""PASS 36: the k=2 hardened grid re-scored under RAW / CLIPPED / SHRINKAGE, exactly.

``experiments.run_clipping_crossover_grid`` re-scores the whole 96-cell noisy-pure
grid analytically, but it compares RMSEs pointwise, whereas the committed
``crossover_n`` comes from the PAIRED state-level z-test of
:func:`anrl.benchmark.hardened._aggregate_cell`.  The two rules disagree on a
handful of cells for reasons unrelated to clipping.

This script removes that ambiguity for the paper's headline family (k = 2,
noisy-pure, n = 2..10, three channels x four rates) by reproducing the committed
pipeline exactly: it reuses the per-trial single-copy realizations recorded by
``run_clipping_audit.py`` (same seeds as ``hardened.state_errors``), draws the
collective side on its own committed seeds, and then runs the identical
``_aggregate_cell`` / ``crossover_table`` aggregation once per estimator.

VALIDATION GATE: under RAW this must reproduce ``results/scaling_hardened.json``
row for row.  The script reports the agreement before reporting any shift.

Writes ``results/pass36_clipping_paired_grid.json``.
Run:  PYTHONPATH=. python -m experiments.run_clipping_paired_grid
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.benchmark.constrained import clip_moment, shrink_moment
from anrl.benchmark.hardened import (
    _NOISE_ID, _aggregate_cell, _bootstrap_rmse_ci, _rate_key, crossover_table,
)
from anrl.benchmark.moments import collective_moment_estimate
from anrl.benchmark.scaling import ENSEMBLES, _ENSEMBLE_ID, collective_purity_signal
from anrl.theory.single_copy_law import closed_form_zetas, hoeffding_rmse

R = Path(__file__).resolve().parent.parent / "results"

SEED, Q, BUDGET, N_STATES, N_TRIALS = 0, 0.1, 2000, 48, 10
SIZES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.0, 0.02, 0.05, 0.1)
ENSEMBLE, K = "noisy_pure", 2


def collective_se(n: int, state_idx: int) -> dict:
    """Collective squared errors per (noise, rate), on the committed seeds."""
    eid = _ENSEMBLE_ID[ENSEMBLE]
    state = ENSEMBLES[ENSEMBLE](n, Q, np.random.default_rng([SEED, eid, n, state_idx, 0]))
    true = state.purity()
    out = {}
    for nm in NOISE_MODELS:
        for rate in RATES:
            signal = collective_purity_signal(state, nm, rate)
            rng = np.random.default_rng(
                [SEED, eid, n, state_idx, 2, _NOISE_ID[nm], _rate_key(rate)])
            out[f"{nm}@{rate}"] = [
                (collective_moment_estimate(2, BUDGET // 2, signal, rng) - true) ** 2
                for _ in range(N_TRIALS)]
    return out


def main() -> None:
    audit = json.loads((R / "pass36_clipping_audit.json").read_text())
    units = {(u["n"], u["state"]): u for u in audit["raw_units"]}

    sigma = {}
    for n in SIZES:
        z1, z2 = closed_form_zetas(n, Q)
        sigma[n] = hoeffding_rmse(BUDGET, z1, z2)

    coll = {(n, s): collective_se(n, s) for n in SIZES for s in range(N_STATES)}

    rows_by_kind: dict[str, list[dict]] = {}
    boot = np.random.default_rng([SEED, 999])
    for kind in ("raw", "clipped", "shrunk"):
        rows = []
        for n in SIZES:
            single_mse = []
            for s in range(N_STATES):
                u = units[(n, s)]
                t = u["truth"]
                xs = np.asarray(u["raw"], dtype=float)
                if kind == "raw":
                    y = xs
                elif kind == "clipped":
                    y = clip_moment(xs, n, K)
                else:
                    y = shrink_moment(xs, n, K, sigma[n])
                single_mse.append(float(np.mean((y - t) ** 2)))
            single_mse = np.asarray(single_mse)
            s_ci = _bootstrap_rmse_ci(single_mse, np.random.default_rng([SEED, 1, n]))
            for nm in NOISE_MODELS:
                for rate in RATES:
                    cm = np.asarray([float(np.mean(coll[(n, s)][f"{nm}@{rate}"]))
                                     for s in range(N_STATES)])
                    cell = _aggregate_cell(single_mse, cm, s_ci, boot)
                    rows.append({"ensemble": ENSEMBLE, "n": n, "noise_model": nm,
                                 "rate": rate, **cell})
        rows_by_kind[kind] = rows

    tables = {k: crossover_table(v) for k, v in rows_by_kind.items()}

    # ---- validation gate against the committed file
    committed = json.loads((R / "scaling_hardened.json").read_text())
    comm_rows = {(r["n"], r["noise_model"], r["rate"]): r for r in committed["rows"]
                 if r.get("ensemble") == ENSEMBLE}
    n_ok = n_tot = 0
    worst = 0.0
    for r in rows_by_kind["raw"]:
        key = (r["n"], r["noise_model"], r["rate"])
        if key not in comm_rows:
            continue
        n_tot += 1
        d = abs(r["single_rmse"] - comm_rows[key]["single_rmse"])
        worst = max(worst, d)
        if d < 1e-9:
            n_ok += 1
    print(f"VALIDATION: RAW single_rmse matches committed scaling_hardened.json "
          f"in {n_ok}/{n_tot} rows (max abs diff {worst:.3e})")

    comm_tab = {(e["noise_model"], e["rate"]): e["crossover_n"]
                for e in committed["crossover_table"] if e.get("ensemble") == ENSEMBLE}
    raw_tab = {(e["noise_model"], e["rate"]): e["crossover_n"] for e in tables["raw"]}
    agree = sum(1 for k in comm_tab if raw_tab.get(k) == comm_tab[k])
    print(f"VALIDATION: RAW crossover reproduces committed in {agree}/{len(comm_tab)} cells")

    print("\nCROSSOVER n* by cell (paired test, k=2 noisy-pure)")
    print(f"  {'channel':<20}{'rate':>6}{'RAW':>7}{'CLIP':>7}{'SHRINK':>8}")
    shifts = []
    for nm in NOISE_MODELS:
        for rate in RATES:
            a = raw_tab.get((nm, rate))
            b = {(e["noise_model"], e["rate"]): e["crossover_n"] for e in tables["clipped"]}.get((nm, rate))
            c = {(e["noise_model"], e["rate"]): e["crossover_n"] for e in tables["shrunk"]}.get((nm, rate))
            print(f"  {nm:<20}{rate:>6}{str(a):>7}{str(b):>7}{str(c):>8}")
            if a is not None and b is not None:
                shifts.append(b - a)
    print(f"\n  clipped shift: {shifts}  (mean {np.mean(shifts):+.2f})" if shifts else "")

    (R / "pass36_clipping_paired_grid.json").write_text(json.dumps({
        "description": "PASS 36: k=2 noisy-pure hardened grid re-scored under RAW / "
                       "CLIPPED / SHRINKAGE using the committed paired-test pipeline",
        "validation": {"raw_single_rmse_matches_committed": f"{n_ok}/{n_tot}",
                       "max_abs_diff": worst,
                       "raw_crossover_matches_committed": f"{agree}/{len(comm_tab)}"},
        "rows": rows_by_kind, "crossover_tables": tables,
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass36_clipping_paired_grid.json'}")


if __name__ == "__main__":
    main()
