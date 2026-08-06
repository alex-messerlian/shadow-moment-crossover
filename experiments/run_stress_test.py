"""Adversarial stress-test of the crossover theory on unseen regimes.

    OMP_NUM_THREADS=1 .venv/bin/python experiments/run_stress_test.py

Tests the theory (anrl/theory, developed on noisy_pure) on three NEW ensembles
(haar_pure, low_rank, ghz_noisy), out-of-budget and out-of-noise-range:

  Part 2  continuous single-copy RMSE (predicted vs measured, with CIs);
  Part 3  bias-law exactness at extreme noise g in {0.2, 0.3};
  Part 4  crossover prediction on the new ensembles;
  plus three qualitative predictions that must hold.

The theory's shipped closed forms assume the noisy_pure spectrum, so the
state-agnostic estimators (anrl/theory/general.py: dense-rho Hoeffding components)
are used; this tests the THEORY, not its noisy_pure shortcut.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.budget import moment_ustat_linear
from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank
from anrl.benchmark.moments import moment
from anrl.theory.bias import brute_force_collective_value, collective_bias, collective_value
from anrl.theory.general import (
    estimate_hoeffding_components_general,
    predict_crossover_general,
    predicted_collective_rmse_general,
    sample_batched_general,
)
from anrl.theory.variance import exact_single_copy_rmse

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "stress_test.json"
COMP_CACHE = REPO / "results" / "stress_components.json"
MEAS_CACHE = REPO / "results" / "stress_measurements.json"

SEED = 0
N_SAMPLES = 60_000
MAX_WORKERS = 3       # bound peak RAM: each worker peaks ~0.7 GB -> ~2 GB total
BATCH = 12            # units per pool.map batch (save cache between batches -> resumable)
BUDGETS = (2000, 8000)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
# deterministic flag: ghz is a fixed state (average/measure over 1 state, many trials).
ENSEMBLES = {"haar_pure": False, "low_rank": False, "ghz_noisy": True}
_ENS_ID = {"haar_pure": 0, "low_rank": 1, "ghz_noisy": 2}  # value-based seeds (no salted hash)
N_COMP_STATES = {False: 4, True: 1}
N_MEAS_STATES = {False: 24, True: 1}
N_TRIALS = {False: 6, True: 36}


def _make_state(ens: str, n: int, s: int):
    rng = np.random.default_rng([SEED, _ENS_ID[ens], n, s])
    if ens == "haar_pure":
        return haar_pure(n, rng)
    if ens == "low_rank":
        return low_rank(n, 2, rng)
    return ghz_noisy(n, 0.15, rng)


# ---- parallel workers ----
def _component_worker(task):
    ens, n, k, s = task
    state = _make_state(ens, n, s)
    comps = estimate_hoeffding_components_general(state, k, N_SAMPLES, np.random.default_rng([SEED, n, k, s, 1]))
    return task, comps


def _measure_worker(task):
    ens, n, k, s = task
    state = _make_state(ens, n, s)
    truth = moment(state.density_matrix(), k)
    rng = np.random.default_rng([SEED, n, k, s, 2])
    se = {b: [(moment_ustat_linear(sample_batched_general(state, b, rng), k) - truth) ** 2
              for _ in range(N_TRIALS[ENSEMBLES[ens]])] for b in BUDGETS}
    return task, {"se": se, "true": truth}  # rho reconstructed from seed (not cached: complex)


def _bootstrap_rmse(per_unit_sq: list[list[float]], rng: np.random.Generator, n_boot: int = 2000):
    """68% bootstrap CI of RMSE, resampling the outer unit (states, or trials if 1 state)."""
    flat = np.array([e for u in per_unit_sq for e in u])
    point = float(np.sqrt(flat.mean()))
    units = per_unit_sq if len(per_unit_sq) > 1 else [[e] for e in per_unit_sq[0]]
    k = len(units)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        boot[b] = np.sqrt(np.concatenate([units[i] for i in idx]).mean())
    return point, float(np.percentile(boot, 16)), float(np.percentile(boot, 84))


def _run_cached(grid: list, worker, cache_path: Path, label: str) -> dict:
    """Compute ``worker(unit)`` for every unit, caching to disk in small batches.

    Resumable and memory-bounded: only uncached units are (re)computed, in
    ``BATCH``-sized ``pool.map`` chunks with ``MAX_WORKERS`` workers, saving the
    cache after each chunk so a crash loses at most one batch.
    """
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [u for u in grid if "|".join(map(str, u)) not in cache]
    print(f"  {label}: {len(cache)} cached, {len(todo)} to compute (batch {BATCH}, {MAX_WORKERS} workers)")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for unit, res in pool.map(worker, batch):
                cache["|".join(map(str, unit))] = res
        cache_path.write_text(json.dumps(cache))
        print(f"    {label}: {min(i + BATCH, len(todo))}/{len(todo)} done")
    return cache


def main() -> None:
    t0 = time.time()
    # grids: k=2 crossover scan n=2..6 + Part2 n=3,4,5; k=3 Part2 n=3,4,5
    comp_grid, meas_grid = [], []
    for ens, det in ENSEMBLES.items():
        for n in range(2, 7):
            for s in range(N_COMP_STATES[det]):
                comp_grid.append((ens, n, 2, s))
            for s in range(N_MEAS_STATES[det]):
                meas_grid.append((ens, n, 2, s))
        for n in (3, 4, 5):
            for s in range(N_COMP_STATES[det]):
                comp_grid.append((ens, n, 3, s))
            for s in range(N_MEAS_STATES[det]):
                meas_grid.append((ens, n, 3, s))

    print(f"Stress test: {len(comp_grid)} component units + {len(meas_grid)} measurement units ...")
    _run_cached(comp_grid, _component_worker, COMP_CACHE, "components")
    _run_cached(meas_grid, _measure_worker, MEAS_CACHE, "measurements")
    # Reload from disk so all keys are consistently JSON-stringified (fresh in-memory
    # worker returns have int budget keys; the disk round-trip makes them strings).
    comp_cache = json.loads(COMP_CACHE.read_text())
    meas_cache = json.loads(MEAS_CACHE.read_text())

    # aggregate components (mean over states) and measurements (RMSE + CI, rhos)
    comps_by = {}
    tmp = {}
    for unit in comp_grid:
        ens, n, k, s = unit
        tmp.setdefault((ens, n, k), []).append(comp_cache["|".join(map(str, unit))])
    for key, lst in tmp.items():
        comps_by[key] = [float(np.mean([c[i] for c in lst])) for i in range(key[2])]

    meas_res = [(unit, meas_cache["|".join(map(str, unit))]) for unit in meas_grid]
    meas_by, rhos_by = {}, {}
    boot_rng = np.random.default_rng([SEED, 999])
    tmpm = {}
    for (ens, n, k, s), d in meas_res:
        tmpm.setdefault((ens, n, k), []).append(d)
    for (ens, n, k), lst in tmpm.items():
        # rhos reconstructed from the deterministic seeds (not cached; complex not JSON-able)
        rhos_by[(ens, n, k)] = [_make_state(ens, n, s).density_matrix()
                                for s in range(N_MEAS_STATES[ENSEMBLES[ens]])]
        for b in BUDGETS:
            per_state = [d["se"][str(b)] for d in lst]  # JSON cache -> string budget keys
            meas_by[(ens, n, k, b)] = _bootstrap_rmse(per_state, boot_rng)

    results = {"meta": {"seed": SEED, "n_samples": N_SAMPLES, "budgets": list(BUDGETS)},
               "part2": [], "part3": [], "part4": [], "wall_seconds": None}

    # ================= PART 2: continuous RMSE =================
    print("\n" + "=" * 82)
    print("PART 2. Continuous single-copy RMSE: predicted vs measured (with 68% CI)")
    print(f"  {'ensemble':>10} {'n':>2} {'k':>2} {'M':>6} | {'measured [CI]':>26} {'predicted':>10} {'rel_err':>8} {'in_CI':>6}")
    hits, rel_errs = [], []
    for ens in ENSEMBLES:
        for n in (3, 4, 5):
            for k in (2, 3):
                comps = comps_by[(ens, n, k)]
                for b in BUDGETS:
                    pt, lo, hi = meas_by[(ens, n, k, b)]
                    pred = exact_single_copy_rmse(comps, k, b)
                    rel = abs(pred - pt) / pt
                    in_ci = lo <= pred <= hi
                    hits.append(in_ci)
                    rel_errs.append(rel)
                    results["part2"].append({"ensemble": ens, "n": n, "k": k, "budget": b,
                                             "measured": pt, "ci": [lo, hi], "predicted": pred,
                                             "rel_err": rel, "in_ci": in_ci})
                    print(f"  {ens:>10} {n:>2} {k:>2} {b:>6} | {pt:8.4f} [{lo:.4f},{hi:.4f}] "
                          f"{pred:10.4f} {rel:7.1%} {'Y' if in_ci else 'n':>6}")
    print(f"\n  Prediction inside measured 68% CI: {sum(hits)}/{len(hits)} = {np.mean(hits):.0%}"
          f"   |   median relative error: {np.median(rel_errs):.1%}")

    # ================= PART 3: extreme-noise bias exactness =================
    print("\n" + "=" * 82)
    print("PART 3. Bias-law exactness at extreme noise (must be exact identity, ~1e-9)")
    max_err = 0.0
    n_checks = 0
    for ens in ENSEMBLES:
        for nn in (2, 3):
            rho = _make_state(ens, nn, 0).density_matrix()
            for nm in NOISE_MODELS:
                for k in (2, 3, 4):
                    if nn == 3 and k == 4:
                        continue  # n=3,k=4 is a 4096-dim brute force; the identity is covered by n=2,k=4
                    for g in (0.2, 0.3):
                        law = collective_value(rho, k, nm, g, nn)
                        brute = brute_force_collective_value(rho, k, nm, g, nn)
                        err = abs(law - brute)
                        max_err = max(max_err, err)
                        n_checks += 1
                        results["part3"].append({"ensemble": ens, "n": nn, "k": k, "noise": nm,
                                                 "g": g, "error": err})
    print(f"  {n_checks} checks (3 ens x 3 chan x g=0.2,0.3; k=2,3,4 at n=2, k=2,3 at n=3): "
          f"max |law - brute| = {max_err:.2e}  ->  {'EXACT (< 1e-9)' if max_err < 1e-9 else 'FAILED'}")

    # ================= PART 4: crossover on new ensembles =================
    print("\n" + "=" * 82)
    print("PART 4. Crossover n* on new ensembles (k=2; predicted vs measured)")
    print(f"  {'ensemble':>10} {'noise':>18} {'rate':>5} | {'pred':>5} {'meas':>5} {'delta':>6}")
    sizes = list(range(2, 7))
    within1, deltas = [], []
    for ens in ENSEMBLES:
        comps_by_n = {n: comps_by[(ens, n, 2)] for n in sizes}
        rhos_by_n = {n: rhos_by[(ens, n, 2)] for n in sizes}
        for nm in NOISE_MODELS:
            for g in (0.05, 0.1, 0.3):
                pred = predict_crossover_general(2, 2000, sizes, comps_by_n, rhos_by_n, nm, g)
                # measured: sustained n where MEASURED single RMSE > exact collective floor
                wins = {n: meas_by[(ens, n, 2, 2000)][0]
                        > predicted_collective_rmse_general(rhos_by_n[n], 2, nm, g, 2000, n) for n in sizes}
                meas = next((n for i, n in enumerate(sizes) if wins[n] and all(wins[m] for m in sizes[i:])), None)
                delta = (pred - meas) if (pred is not None and meas is not None) else None
                if delta is not None:
                    within1.append(abs(delta) <= 1)
                    deltas.append(delta)
                results["part4"].append({"ensemble": ens, "noise": nm, "rate": g,
                                         "predicted_n": pred, "measured_n": meas, "delta": delta})
                ds = "" if delta is None else f"{delta:+d}"
                print(f"  {ens:>10} {nm:>18} {g:>5} | {str(pred):>5} {str(meas):>5} {ds:>6}")
    if within1:
        print(f"\n  Crossover within +-1 qubit: {sum(within1)}/{len(within1)} = {np.mean(within1):.0%}"
              f"   mean delta: {np.mean(deltas):+.2f}")

    # ================= Qualitative predictions =================
    print("\n" + "=" * 82)
    print("Three qualitative predictions:")
    # (1) haar_pure has the largest single-copy RMSE at fixed (n,k,M)
    q1_ok = q1_tot = 0
    for n in (3, 4, 5):
        for k in (2, 3):
            for b in BUDGETS:
                rmses = {e: meas_by[(e, n, k, b)][0] for e in ENSEMBLES}
                q1_tot += 1
                q1_ok += rmses["haar_pure"] == max(rmses.values())
    print(f"  (1) haar_pure has LARGEST single-copy RMSE: {q1_ok}/{q1_tot} cells "
          f"({'HOLDS' if q1_ok == q1_tot else 'VIOLATED'})")
    # (2) bias laws exact at g=0.3
    g3 = [r for r in results["part3"] if r["g"] == 0.3]
    print(f"  (2) bias laws exact at g=0.3: max err {max(r['error'] for r in g3):.2e} "
          f"({'HOLDS' if max(r['error'] for r in g3) < 1e-9 else 'VIOLATED'})")
    # (3) GHZ accuracy comparable to the others
    med = {e: np.median([r["rel_err"] for r in results["part2"] if r["ensemble"] == e]) for e in ENSEMBLES}
    print(f"  (3) GHZ accuracy comparable: median rel err  haar={med['haar_pure']:.1%} "
          f"low_rank={med['low_rank']:.1%} ghz={med['ghz_noisy']:.1%} "
          f"({'COMPARABLE' if med['ghz_noisy'] <= 2.5 * min(med.values()) else 'DEGRADED'})")

    results["wall_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWall: {results['wall_seconds']}s.  Saved -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
