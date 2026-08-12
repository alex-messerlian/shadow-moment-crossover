"""PASS 47.2(c)/(d): does the EXACT statewise law predict what each INDIVIDUAL state does?

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/pass47_perstate_validation.py

The paper validates the criterion on ensemble aggregates: projection variances are averaged
over states before being compared with a state-aggregated RMSE.  A theory-first paper would
have to claim more -- that ``zeta_1(rho), zeta_2(rho)`` predict what THAT state does.  This
script tests exactly that, with the exact sampling-free evaluator
(:mod:`anrl.theory.statewise_zetas`) on the prediction side and fresh forward simulation of
the same exact U-statistic the paper uses on the measurement side.

Three per-state comparisons, reported in the paper's own form:

A. RMSE at fixed budget.  Per-state predicted vs measured single-copy RMSE, over six
   ensembles (four committed, two new) and ``n = 2..6``.
B. Budget-scaling exponent alpha.  Per-state predicted vs measured, within two bootstrap
   standard errors -- the same scoring rule as the paper's Table I.
C. Crossover size n*.  A state SEQUENCE is one seed evaluated at every ``n``, since a
   single state lives at one size and cannot cross over by itself.  Predicted vs measured
   n* per sequence, scored exactly / within one qubit, as in Section 5.2.

Writes ``results/pass47_perstate_validation.json``.  No paper edit.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.budget import moment_ustat_linear
from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank, noisy_pure
from anrl.benchmark.moments import moment
from anrl.theory.bias import collective_bias, collective_value
from anrl.theory.general import sample_batched_general
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights
from anrl.theory.variance import exact_fitted_alpha, exact_single_copy_rmse

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass47_perstate_validation.json"

SEED = 47
K = 2
BUDGETS = (2000, 8000, 32000)
RMSE_SIZES = (2, 3, 4, 5, 6)
CROSS_SIZES = (2, 3, 4, 5, 6, 7)
CROSS_BUDGET = 2000
N_STATES = 8            # independent states per (ensemble, n) for A and B
N_TRIALS = 24           # forward-simulation trials per state per budget
N_SEQUENCES = 8         # state sequences (one seed carried across n) for C
CHANNELS = (("depolarizing", 0.1), ("amplitude_damping", 0.05), ("dephasing", 0.05))
MAX_WORKERS = 4

_ENS_ID = {"noisy_pure_q0.1": 0, "haar_pure": 1, "low_rank_2": 2, "ghz_noisy_q0.15": 3,
           "variable_q": 4, "variable_rank": 5}
DETERMINISTIC = {"ghz_noisy_q0.15"}   # a fixed state: per-state spread is zero by construction


def make_state(ens: str, n: int, s: int):
    """Reproducible state ``s`` of ensemble ``ens`` at size ``n`` (value-based seed)."""
    rng = np.random.default_rng([SEED, _ENS_ID[ens], n, s])
    if ens == "noisy_pure_q0.1":
        return noisy_pure(n, 0.1, rng)
    if ens == "haar_pure":
        return haar_pure(n, rng)
    if ens == "low_rank_2":
        return low_rank(n, 2, rng)
    if ens == "ghz_noisy_q0.15":
        return ghz_noisy(n, 0.15, rng)
    if ens == "variable_q":
        return noisy_pure(n, float(rng.uniform(0.05, 0.45)), rng)
    if ens == "variable_rank":
        return low_rank(n, int(rng.integers(1, min(8, 2 ** n) + 1)), rng)
    raise ValueError(ens)


def _exact_zetas_state(state, weights) -> tuple[float, float]:
    m = pauli_expectations(state.density_matrix(), state.n)
    return exact_zeta1(m, state.n), exact_zeta2(m, state.n, weights)


def _bootstrap_alpha_se(sq_by_budget: dict, budgets, rng, n_boot: int = 400) -> tuple[float, float]:
    """Measured alpha (log-log slope of RMSE vs M) and its bootstrap standard error."""
    x = np.log(np.asarray(budgets, float))
    xm = x.mean()

    def slope(vals):
        y = np.log(vals)
        return -float(((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum())

    point = slope([np.sqrt(np.mean(sq_by_budget[b])) for b in budgets])
    draws = np.empty(n_boot)
    for i in range(n_boot):
        vals = []
        for b in budgets:
            arr = np.asarray(sq_by_budget[b])
            vals.append(np.sqrt(arr[rng.integers(0, arr.size, arr.size)].mean()))
        draws[i] = slope(vals)
    return point, float(draws.std(ddof=1))


# ------------------------------------------------------------------ workers (A and B)
def _rmse_worker(task):
    ens, n, s = task
    state = make_state(ens, n, s)
    weights = pauli_weights(n)
    z1, z2 = _exact_zetas_state(state, weights)
    truth = moment(state.density_matrix(), K)
    rng = np.random.default_rng([SEED, 101, _ENS_ID[ens], n, s])
    sq = {}
    for b in BUDGETS:
        sq[b] = [float((moment_ustat_linear(sample_batched_general(state, b, rng), K) - truth) ** 2)
                 for _ in range(N_TRIALS)]
    a_meas, a_se = _bootstrap_alpha_se(sq, BUDGETS, np.random.default_rng([SEED, 202, n, s]))
    return task, {
        "zeta1": z1, "zeta2": z2, "m_star": z2 / (2 * z1) if z1 > 0 else None,
        "true_moment": truth,
        "measured_rmse": {str(b): float(np.sqrt(np.mean(sq[b]))) for b in BUDGETS},
        "measured_rmse_se": {
            str(b): float(np.std(sq[b], ddof=1) / (2 * np.sqrt(np.mean(sq[b])) * np.sqrt(N_TRIALS)))
            for b in BUDGETS},
        "predicted_rmse": {str(b): exact_single_copy_rmse([z1, z2], K, b) for b in BUDGETS},
        "alpha_measured": a_meas, "alpha_measured_se": a_se,
        "alpha_predicted": exact_fitted_alpha(list(BUDGETS), [z1, z2], K),
    }


# ------------------------------------------------------------------- worker (C: n*)
def _sequence_worker(task):
    """One state sequence: exact zetas + measured single-copy RMSE at every n."""
    ens, s = task
    out = {}
    for n in CROSS_SIZES:
        state = make_state(ens, n, s)
        weights = pauli_weights(n)
        z1, z2 = _exact_zetas_state(state, weights)
        rho = state.density_matrix()
        truth = moment(rho, K)
        rng = np.random.default_rng([SEED, 303, _ENS_ID[ens], n, s])
        sq = [float((moment_ustat_linear(sample_batched_general(state, CROSS_BUDGET, rng), K) - truth) ** 2)
              for _ in range(N_TRIALS)]
        coll = {}
        for model, g in CHANNELS:
            bias = collective_bias(rho, K, model, g, n)
            signal = collective_value(rho, K, model, g, n)
            shot = max(0.0, 1.0 - signal * signal) / max(1, CROSS_BUDGET // K)
            coll[f"{model}|{g}"] = float(np.sqrt(bias * bias + shot))
        out[n] = {
            "zeta1": z1, "zeta2": z2, "m_star": z2 / (2 * z1) if z1 > 0 else None,
            "single_predicted": exact_single_copy_rmse([z1, z2], K, CROSS_BUDGET),
            "single_measured": float(np.sqrt(np.mean(sq))),
            "collective": coll,
        }
    return task, out


def _sustained_crossover(single_by_n: dict, coll_by_n: dict) -> int | None:
    """Smallest n from which single-copy error exceeds collective error at every larger n."""
    ns = sorted(single_by_n)
    wins = {n: single_by_n[n] > coll_by_n[n] for n in ns}
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None


def main() -> None:
    t0 = time.time()

    # ---- A and B -----------------------------------------------------------------
    grid = [(e, n, s) for e in _ENS_ID for n in RMSE_SIZES
            for s in range(1 if e in DETERMINISTIC else N_STATES)]
    print(f"A/B: {len(grid)} (ensemble, n, state) units, {N_TRIALS} trials x {len(BUDGETS)} budgets each")
    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, (task, res) in enumerate(pool.map(_rmse_worker, grid), 1):
            ens, n, s = task
            rows.append({"ensemble": ens, "n": n, "state": s, **res})
            if i % 25 == 0:
                print(f"    {i}/{len(grid)}", flush=True)

    per_state = []
    for r in rows:
        for b in BUDGETS:
            pred, meas = r["predicted_rmse"][str(b)], r["measured_rmse"][str(b)]
            se = r["measured_rmse_se"][str(b)]
            per_state.append({
                "ensemble": r["ensemble"], "n": r["n"], "state": r["state"], "budget": b,
                "predicted": pred, "measured": meas, "se": se,
                "rel_dev": (pred - meas) / meas,
                "within_2se": abs(pred - meas) <= 2 * se,
            })
    rel = np.abs([p["rel_dev"] for p in per_state])
    alpha_rows = [{"ensemble": r["ensemble"], "n": r["n"], "state": r["state"],
                   "predicted": r["alpha_predicted"], "measured": r["alpha_measured"],
                   "se": r["alpha_measured_se"],
                   "within_2se": abs(r["alpha_predicted"] - r["alpha_measured"]) <= 2 * r["alpha_measured_se"]}
                  for r in rows]
    a_hit = sum(a["within_2se"] for a in alpha_rows)

    print(f"\nA. per-state RMSE: median |rel dev| {np.median(rel)*100:.2f}%, "
          f"90th pct {np.percentile(rel,90)*100:.2f}%, max {rel.max()*100:.2f}%; "
          f"{sum(p['within_2se'] for p in per_state)}/{len(per_state)} within 2 SE")
    print(f"B. per-state alpha: {a_hit}/{len(alpha_rows)} within 2 SE")

    by_ens = {}
    for e in _ENS_ID:
        sub = [p for p in per_state if p["ensemble"] == e]
        suba = [a for a in alpha_rows if a["ensemble"] == e]
        by_ens[e] = {
            "n_rmse_points": len(sub),
            "median_abs_rel_dev": float(np.median(np.abs([p["rel_dev"] for p in sub]))),
            "max_abs_rel_dev": float(np.max(np.abs([p["rel_dev"] for p in sub]))),
            "rmse_within_2se": f"{sum(p['within_2se'] for p in sub)}/{len(sub)}",
            "alpha_within_2se": f"{sum(a['within_2se'] for a in suba)}/{len(suba)}",
        }
        print(f"   {e:18s} RMSE median |dev| {by_ens[e]['median_abs_rel_dev']*100:5.2f}%  "
              f"within-2SE {by_ens[e]['rmse_within_2se']:>7s}   alpha {by_ens[e]['alpha_within_2se']}")

    # ---- C -----------------------------------------------------------------------
    seq_grid = [(e, s) for e in _ENS_ID for s in range(1 if e in DETERMINISTIC else N_SEQUENCES)]
    print(f"\nC: {len(seq_grid)} state sequences x {len(CROSS_SIZES)} sizes x {len(CHANNELS)} channels")
    seqs = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for task, res in pool.map(_sequence_worker, seq_grid):
            seqs["|".join(map(str, task))] = {str(k): v for k, v in res.items()}

    cross_rows = []
    for key, per_n in seqs.items():
        ens, s = key.split("|")
        for model, g in CHANNELS:
            ch = f"{model}|{g}"
            pred = _sustained_crossover(
                {int(n): v["single_predicted"] for n, v in per_n.items()},
                {int(n): v["collective"][ch] for n, v in per_n.items()})
            meas = _sustained_crossover(
                {int(n): v["single_measured"] for n, v in per_n.items()},
                {int(n): v["collective"][ch] for n, v in per_n.items()})
            cross_rows.append({
                "ensemble": ens, "sequence": int(s), "channel": ch,
                "predicted_n": pred, "measured_n": meas,
                "resolves": pred is not None and meas is not None,
                "delta": (pred - meas) if (pred is not None and meas is not None) else None,
            })
    res_rows = [c for c in cross_rows if c["resolves"]]
    exact_hit = sum(1 for c in res_rows if c["delta"] == 0)
    within1 = sum(1 for c in res_rows if abs(c["delta"]) <= 1)
    agree_none = sum(1 for c in cross_rows if c["predicted_n"] is None and c["measured_n"] is None)
    print(f"C. per-sequence n*: {len(res_rows)}/{len(cross_rows)} resolve; "
          f"exact {exact_hit}/{len(res_rows)} ({100*exact_hit/max(1,len(res_rows)):.1f}%), "
          f"within one {within1}/{len(res_rows)} ({100*within1/max(1,len(res_rows)):.1f}%); "
          f"{agree_none} agreed no-crossover")

    payload = {
        "description": "PASS 47.2(c): per-state predictive accuracy of the exact statewise law",
        "config": {"seed": SEED, "k": K, "budgets": list(BUDGETS), "rmse_sizes": list(RMSE_SIZES),
                   "cross_sizes": list(CROSS_SIZES), "cross_budget": CROSS_BUDGET,
                   "n_states": N_STATES, "n_trials": N_TRIALS, "n_sequences": N_SEQUENCES,
                   "channels": [f"{m}|{g}" for m, g in CHANNELS]},
        "A_per_state_rmse": {
            "points": per_state,
            "median_abs_rel_dev": float(np.median(rel)),
            "p90_abs_rel_dev": float(np.percentile(rel, 90)),
            "max_abs_rel_dev": float(rel.max()),
            "within_2se": f"{sum(p['within_2se'] for p in per_state)}/{len(per_state)}",
        },
        "B_per_state_alpha": {"rows": alpha_rows, "within_2se": f"{a_hit}/{len(alpha_rows)}"},
        "by_ensemble": by_ens,
        "C_per_sequence_crossover": {
            "rows": cross_rows,
            "resolving": len(res_rows), "total": len(cross_rows),
            "exact": f"{exact_hit}/{len(res_rows)}",
            "within_one": f"{within1}/{len(res_rows)}",
            "agreed_no_crossover": agree_none,
        },
        "raw_units": rows,
        "sequences": seqs,
        "wall_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
