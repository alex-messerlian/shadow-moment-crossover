"""PASS 48.3: the crossover sweep extended to the two varying-estimand ensembles.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/pass48_new_ensembles_crossover.py

Reproduces ``experiments/run_heldout_clipping.py`` -- which itself reproduces
``run_stress_test.py`` part 4 (RULE 3: measured single-copy RMSE against the exact collective
floor, sustained rule) -- on its committed seeds, and extends it from three held-out ensembles
to five by adding :func:`~anrl.benchmark.ensembles.variable_q` and
:func:`~anrl.benchmark.ensembles.variable_rank`.

Two changes from the committed run, both deliberate and both reported:

* Predictions use the EXACT statewise projection variances
  (:mod:`anrl.theory.statewise_zetas`, averaged over the cell's states) instead of the
  60k-sample Monte-Carlo components.  Whether that moves the three committed ensembles'
  predictions is a gate, not an assumption.
* Two ensembles are added.  The three committed ensembles' cells are recomputed on their
  committed seeds and must reproduce ``stress_test.json`` part 4 cell for cell.

VALIDATION GATES, reported before anything else:
  G1  RAW measured n* on the three committed ensembles == stress_test.json part 4 (27 cells)
  G2  RAW/CLIPPED resolving counts on those three == pass37_heldout_clipping.json
  G3  exact-input predicted n* on those three == the committed MC-input predictions

Writes ``results/pass48_new_ensembles_crossover.json``.  No paper edit.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.budget import moment_ustat_linear
from anrl.benchmark.constrained import clip_moment
from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank, variable_q, variable_rank
from anrl.benchmark.moments import moment
from anrl.theory.general import predicted_collective_rmse_general, sample_batched_general
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights
from anrl.theory.variance import exact_single_copy_rmse

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
OUT = R / "pass48_new_ensembles_crossover.json"

# Committed configuration of run_heldout_clipping.py / run_stress_test.py part 4.
SEED = 0
BUDGET = 2000
K = 2
SIZES = (2, 3, 4, 5, 6)
NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")
RATES = (0.05, 0.1, 0.3)
MAX_WORKERS = 2

# deterministic flag -> (n_measurement_states, n_trials), from the committed run
COMMITTED = {"haar_pure": False, "low_rank": False, "ghz_noisy": True}
NEW = {"variable_q": False, "variable_rank": False}
ENSEMBLES = {**COMMITTED, **NEW}
# Committed IDs must not change or the seeds move; new ensembles take fresh IDs.
_ENS_ID = {"haar_pure": 0, "low_rank": 1, "ghz_noisy": 2, "variable_q": 3, "variable_rank": 4}
N_MEAS_STATES = {False: 24, True: 1}
N_TRIALS = {False: 6, True: 36}
N_PRED_STATES = 24      # states averaged for the exact-input prediction side


def make_state(ens: str, n: int, s: int):
    """Committed seeding convention: rng = default_rng([SEED, ENS_ID[ens], n, s])."""
    rng = np.random.default_rng([SEED, _ENS_ID[ens], n, s])
    if ens == "haar_pure":
        return haar_pure(n, rng)
    if ens == "low_rank":
        return low_rank(n, 2, rng)
    if ens == "ghz_noisy":
        return ghz_noisy(n, 0.15, rng)
    if ens == "variable_q":
        return variable_q(n, rng)
    if ens == "variable_rank":
        return variable_rank(n, rng)
    raise ValueError(ens)


def _measure_unit(task):
    """Raw per-trial estimates at budget 2000 for one (ens, n, state), committed seeds."""
    ens, n, s = task
    state = make_state(ens, n, s)
    truth = moment(state.density_matrix(), K)
    rng = np.random.default_rng([SEED, n, K, s, 2])
    est = [moment_ustat_linear(sample_batched_general(state, BUDGET, rng), K)
           for _ in range(N_TRIALS[ENSEMBLES[ens]])]
    return {"ensemble": ens, "n": n, "state": s, "truth": float(truth),
            "raw": [float(e) for e in est]}


def _exact_components(task):
    """Ensemble-mean EXACT (zeta_1, zeta_2) at k=2 for one (ens, n), sampling-free."""
    ens, n = task
    weights = pauli_weights(n)
    ns = 1 if ENSEMBLES[ens] else N_PRED_STATES
    z1s, z2s, mss = [], [], []
    for s in range(ns):
        m = pauli_expectations(make_state(ens, n, s).density_matrix(), n)
        a, b = exact_zeta1(m, n), exact_zeta2(m, n, weights)
        z1s.append(a)
        z2s.append(b)
        mss.append(b / (2 * a) if a > 0 else np.nan)
    mss = np.array(mss)
    fin = np.isfinite(mss)
    return (ens, n), {
        "zeta1": float(np.mean(z1s)), "zeta2": float(np.mean(z2s)), "n_states": ns,
        "m_star_of_means": float(np.mean(z2s) / (2 * np.mean(z1s))),
        "m_star_statewise_min": float(mss[fin].min()) if fin.any() else None,
        "m_star_statewise_max": float(mss[fin].max()) if fin.any() else None,
        "m_star_spread_ratio": (float(mss[fin].max() / mss[fin].min()) if fin.any() else None),
        "m_star_rel_std": (float(np.std(mss[fin], ddof=1) / np.mean(mss[fin]))
                           if fin.sum() > 1 else 0.0),
        "purity_rel_std": None,
    }


def _sustained(sizes, wins):
    ns = sorted(sizes)
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None


def main() -> None:
    t0 = time.time()
    meas_tasks = [(e, n, s) for e in ENSEMBLES for n in SIZES
                  for s in range(N_MEAS_STATES[ENSEMBLES[e]])]
    comp_tasks = [(e, n) for e in ENSEMBLES for n in SIZES]
    print(f"48.3: {len(meas_tasks)} measurement units, {len(comp_tasks)} exact-component units")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        units = list(pool.map(_measure_unit, meas_tasks))
        comps = dict(pool.map(_exact_components, comp_tasks))
    print(f"  measurement + exact components done ({time.time()-t0:.0f} s)")

    # ---- single-copy RMSE per (ens, n), pooled exactly as the committed sweep does ----
    rmse = {"raw": {}, "clipped": {}}
    for e in ENSEMBLES:
        for n in SIZES:
            us = [u for u in units if u["ensemble"] == e and u["n"] == n]
            for kind in rmse:
                sq = []
                for u in us:
                    xs = np.asarray(u["raw"], dtype=float)
                    y = xs if kind == "raw" else clip_moment(xs, n, K)
                    sq.extend(((y - u["truth"]) ** 2).tolist())
                rmse[kind][(e, n)] = float(np.sqrt(np.mean(sq)))

    # ---- cells ----
    rows = []
    for e in ENSEMBLES:
        rho_by_n = {n: [make_state(e, n, s).density_matrix()
                        for s in range(N_MEAS_STATES[ENSEMBLES[e]])] for n in SIZES}
        for nm in NOISE_MODELS:
            for g in RATES:
                coll = {n: predicted_collective_rmse_general(rho_by_n[n], K, nm, g, BUDGET, n)
                        for n in SIZES}
                pred_single = {n: exact_single_copy_rmse(
                    [comps[(e, n)]["zeta1"], comps[(e, n)]["zeta2"]], K, BUDGET) for n in SIZES}
                cell = {
                    "ensemble": e, "noise": nm, "rate": g, "is_new": e in NEW,
                    "predicted_n_exact_inputs": _sustained(
                        SIZES, {n: pred_single[n] > coll[n] for n in SIZES}),
                    "collective": {str(n): coll[n] for n in SIZES},
                    "predicted_single": {str(n): pred_single[n] for n in SIZES},
                }
                for kind in ("raw", "clipped"):
                    cell[f"measured_n_{kind}"] = _sustained(
                        SIZES, {n: rmse[kind][(e, n)] > coll[n] for n in SIZES})
                    cell[f"single_{kind}"] = {str(n): rmse[kind][(e, n)] for n in SIZES}
                rows.append(cell)

    # ---- VALIDATION GATES ----
    committed_p4 = {(c["ensemble"], c["noise"], c["rate"]): c
                    for c in json.loads((R / "stress_test.json").read_text())["part4"]}
    committed_clip = {(c["ensemble"], c["noise"], c["rate"]): c
                      for c in json.loads((R / "pass37_heldout_clipping.json").read_text())["rows"]}
    g1 = g1t = g2 = g2t = g3 = g3t = 0
    g3_moves = []
    for r in rows:
        key = (r["ensemble"], r["noise"], r["rate"])
        c = committed_p4.get(key)
        if c is not None:
            g1t += 1
            g1 += (r["measured_n_raw"] == c["measured_n"])
            g3t += 1
            same = (r["predicted_n_exact_inputs"] == c["predicted_n"])
            g3 += same
            if not same:
                g3_moves.append({**{k: r[k] for k in ("ensemble", "noise", "rate")},
                                 "committed_mc_input": c["predicted_n"],
                                 "exact_input": r["predicted_n_exact_inputs"]})
            r["committed_predicted_n"] = c["predicted_n"]
            r["committed_measured_n"] = c["measured_n"]
        cc = committed_clip.get(key)
        if cc is not None:
            g2t += 1
            g2 += (r["measured_n_clipped"] == cc["measured_n_clipped"])

    print(f"\nG1 RAW measured n* == stress_test part 4              : {g1}/{g1t}")
    print(f"G2 CLIPPED measured n* == pass37_heldout_clipping      : {g2}/{g2t}")
    print(f"G3 exact-input predicted n* == committed MC-input      : {g3}/{g3t}"
          + ("" if not g3_moves else f"   MOVES: {g3_moves}"))

    # ---- accuracy, held-out cells only (RULE 3) ----
    def score(sel, kind):
        res = [r for r in sel if r[f"measured_n_{kind}"] is not None
               and r["predicted_n_exact_inputs"] is not None]
        both_none = [r for r in sel if r[f"measured_n_{kind}"] is None
                     and r["predicted_n_exact_inputs"] is None]
        ex = sum(1 for r in res if r["predicted_n_exact_inputs"] == r[f"measured_n_{kind}"])
        w1 = sum(1 for r in res if abs(r["predicted_n_exact_inputs"] - r[f"measured_n_{kind}"]) <= 1)
        return {"swept": len(sel), "resolving": len(res), "exact": ex, "within_one": w1,
                "agreed_no_crossover": len(both_none),
                "all_cells_within_one": w1 + len(both_none),
                "exact_pct": 100 * ex / max(1, len(res)),
                "within_one_pct": 100 * w1 / max(1, len(res)),
                "all_cells_within_one_pct": 100 * (w1 + len(both_none)) / len(sel)}

    groups = {"committed_three": [r for r in rows if not r["is_new"]],
              "new_two": [r for r in rows if r["is_new"]],
              "all_five": rows}
    accuracy = {gk: {kind: score(sel, kind) for kind in ("raw", "clipped")}
                for gk, sel in groups.items()}

    print(f"\n{'group':18s} {'est':8s} {'swept':>6s} {'resolv':>7s} {'exact':>12s} "
          f"{'within-1':>13s} {'no-x':>5s} {'all-cells w1':>14s}")
    for gk, sel in groups.items():
        for kind in ("raw", "clipped"):
            a = accuracy[gk][kind]
            print(f"{gk:18s} {kind:8s} {a['swept']:>6d} {a['resolving']:>7d} "
                  f"{a['exact']:>4d} ({a['exact_pct']:5.1f}%) "
                  f"{a['within_one']:>4d} ({a['within_one_pct']:5.1f}%) {a['agreed_no_crossover']:>5d} "
                  f"{a['all_cells_within_one']:>4d} ({a['all_cells_within_one_pct']:5.1f}%)")

    # ---- 48.3(c)/(d) per-ensemble breakdown and the spread/noise diagnostic ----
    print(f"\n{'ensemble':16s} {'estimand':>18s} {'M* max/min':>11s} {'M* rel-std':>11s} "
          f"{'pred spread':>12s} {'trial noise':>12s} {'ratio':>6s}  verdict")
    per_ens = {}
    for e in ENSEMBLES:
        n_ref = 4
        us = [u for u in units if u["ensemble"] == e and u["n"] == n_ref]
        truths = np.array([u["truth"] for u in us])
        est_rel_std = float(truths.std(ddof=1) / truths.mean()) if len(us) > 1 else 0.0
        c = comps[(e, n_ref)]
        # predicted per-state RMSE spread at this budget, and the trial noise of the run
        weights = pauli_weights(n_ref)
        preds = []
        for s in range(1 if ENSEMBLES[e] else N_PRED_STATES):
            m = pauli_expectations(make_state(e, n_ref, s).density_matrix(), n_ref)
            preds.append(exact_single_copy_rmse(
                [exact_zeta1(m, n_ref), exact_zeta2(m, n_ref, weights)], K, BUDGET))
        preds = np.array(preds)
        spread = float(preds.std(ddof=1) / preds.mean()) if preds.size > 1 else 0.0
        trials = N_TRIALS[ENSEMBLES[e]] * (1 if ENSEMBLES[e] else N_MEAS_STATES[False])
        noise = float(1.0 / np.sqrt(2 * trials))
        ratio = spread / noise
        verdict = ("cannot test statewise sensitivity" if ratio < 1.0
                   else "can test statewise sensitivity")
        per_ens[e] = {"is_new": e in NEW, "n_reference": n_ref,
                      "estimand_rel_std": est_rel_std,
                      "m_star_spread_ratio": c["m_star_spread_ratio"],
                      "m_star_rel_std": c["m_star_rel_std"],
                      "predicted_rmse_rel_spread": spread,
                      "trial_rel_noise": noise, "spread_over_noise": ratio,
                      "n_effective_trials": trials, "verdict": verdict}
        print(f"{e:16s} {est_rel_std*100:17.2f}% {c['m_star_spread_ratio'] or 1.0:11.2f} "
              f"{c['m_star_rel_std']*100:10.2f}% {spread*100:11.2f}% {noise*100:11.1f}% "
              f"{ratio:6.2f}  {verdict}")

    payload = {
        "description": "PASS 48.3: crossover sweep extended to variable_q and variable_rank "
                       "(RULE 3 held-out pipeline, exact statewise prediction inputs)",
        "config": {"seed": SEED, "budget": BUDGET, "k": K, "sizes": list(SIZES),
                   "noise_models": list(NOISE_MODELS), "rates": list(RATES),
                   "ensemble_ids": _ENS_ID, "n_meas_states": N_MEAS_STATES,
                   "n_trials": N_TRIALS, "n_pred_states": N_PRED_STATES},
        "validation_gates": {
            "G1_raw_vs_stress_test_part4": f"{g1}/{g1t}",
            "G2_clipped_vs_pass37_heldout": f"{g2}/{g2t}",
            "G3_exact_inputs_vs_committed_mc_predictions": f"{g3}/{g3t}",
            "G3_moves": g3_moves,
        },
        "accuracy": accuracy,
        "per_ensemble_diagnostic": per_ens,
        "components": {f"{e}|{n}": comps[(e, n)] for (e, n) in comps},
        "rows": rows,
        "wall_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.1f} s)")


if __name__ == "__main__":
    main()
