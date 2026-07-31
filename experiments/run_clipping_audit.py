"""PASS 36: how much of the crossover result survives a range-constrained estimator.

``Tr(rho^k)`` lies in ``[2^{n(1-k)}, 1]``, so an estimate outside that range can be
projected back into it, which weakly reduces squared error pointwise with no
assumptions (see :mod:`anrl.benchmark.constrained`).  The paper reports the RAW
unbiased U-statistic, whose RMSE reaches 11.98 at ``n = 10`` -- impossible for a
clipped estimator, since the feasible interval has width below one.  This script
measures the difference.

It re-runs the single-copy Monte Carlo with the SAME seeding as the committed
``results/scaling_hardened.json`` (``anrl.benchmark.hardened.state_errors``):

    state       rng = default_rng([seed, ensemble_id, n, state_idx, 0])
    single-copy rng = default_rng([seed, ensemble_id, n, state_idx, 1])

so the realized snapshots are bit-identical to the committed run.  The RAW column
reproduces the committed numbers; CLIPPED and SHRINKAGE are the same realizations
passed through the two projections.  The collective side is untouched and is read
from the committed file.

Writes ``results/pass36_clipping_audit.json``.
Run:  PYTHONPATH=. python -m experiments.run_clipping_audit
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.constrained import clip_moment, physical_range, shrink_moment
from anrl.benchmark.scaling import ENSEMBLES, _ENSEMBLE_ID, snapshots_factored
from anrl.benchmark.shadows import full_purity_ustatistic
from anrl.theory.single_copy_law import closed_form_zetas, hoeffding_rmse

R = Path(__file__).resolve().parent.parent / "results"

SEED = 0
Q = 0.1
BUDGET = 2000
N_STATES = 48
N_TRIALS = 10
SIZES = (2, 3, 4, 5, 6, 7, 8, 9, 10)
K = 2
ENSEMBLE = "noisy_pure"


def raw_estimates(n: int, state_idx: int) -> dict:
    """Per-trial RAW estimates for one (n, state) unit, on the committed seeds."""
    eid = _ENSEMBLE_ID[ENSEMBLE]
    state = ENSEMBLES[ENSEMBLE](n, Q, np.random.default_rng([SEED, eid, n, state_idx, 0]))
    truth = state.purity()
    rng = np.random.default_rng([SEED, eid, n, state_idx, 1])
    est = [full_purity_ustatistic(snapshots_factored(state, BUDGET, rng))
           for _ in range(N_TRIALS)]
    return {"n": n, "state": state_idx, "truth": float(truth), "raw": [float(e) for e in est]}


def _star(t):
    return raw_estimates(*t)


def rmse(errs) -> float:
    return float(np.sqrt(np.mean(np.asarray(errs, dtype=float) ** 2)))


def main() -> None:
    tasks = [(n, s) for n in SIZES for s in range(N_STATES)]
    with ProcessPoolExecutor() as pool:
        units = list(pool.map(_star, tasks))

    # exact Hoeffding sigma per n, from the closed-form projection variances
    sigma = {}
    for n in SIZES:
        z1, z2 = closed_form_zetas(n, Q)
        sigma[n] = hoeffding_rmse(BUDGET, z1, z2)

    by_n: dict[int, dict] = {}
    for n in SIZES:
        us = [u for u in units if u["n"] == n]
        lo, hi = physical_range(n, K)
        raw_e, clip_e, shr_e = [], [], []
        n_out, n_below, n_above, total = 0, 0, 0, 0
        clip_worse = 0
        shrink_worse = 0
        for u in us:
            t = u["truth"]
            for x in u["raw"]:
                total += 1
                c = float(clip_moment(x, n, K))
                s = float(shrink_moment(x, n, K, sigma[n]))
                raw_e.append(x - t)
                clip_e.append(c - t)
                shr_e.append(s - t)
                if x < lo:
                    n_out += 1
                    n_below += 1
                elif x > hi:
                    n_out += 1
                    n_above += 1
                if (c - t) ** 2 > (x - t) ** 2 + 1e-15:
                    clip_worse += 1
                if (s - t) ** 2 > (x - t) ** 2 + 1e-15:
                    shrink_worse += 1
        truth_mean = float(np.mean([u["truth"] for u in us]))
        by_n[n] = {
            "n": n, "true_purity": truth_mean,
            "physical_range": [lo, hi],
            "sigma_theory": sigma[n],
            "n_samples": total,
            "n_outside_range": n_out,
            "frac_outside_range": n_out / total,
            "n_below_floor": n_below, "n_above_one": n_above,
            "rmse_raw": rmse(raw_e),
            "rmse_clipped": rmse(clip_e),
            "rmse_shrunk": rmse(shr_e),
            "rel_err_raw": rmse(raw_e) / truth_mean,
            "rel_err_clipped": rmse(clip_e) / truth_mean,
            "rel_err_shrunk": rmse(shr_e) / truth_mean,
            "clip_increased_sq_error_count": clip_worse,
            "shrink_increased_sq_error_count": shrink_worse,
        }
        print(f"  n={n:2d}  raw {by_n[n]['rmse_raw']:9.4f}  clip {by_n[n]['rmse_clipped']:7.4f}  "
              f"shrink {by_n[n]['rmse_shrunk']:7.4f}   outside {n_out}/{total} "
              f"({100*n_out/total:.1f}%)  clip-worse {clip_worse}")

    out = {
        "description": "PASS 36 clipping audit: RAW vs CLIPPED vs SHRINKAGE single-copy "
                       "purity estimator on the committed seeds of scaling_hardened.json",
        "config": {"seed": SEED, "q": Q, "budget": BUDGET, "n_states": N_STATES,
                   "n_trials": N_TRIALS, "k": K, "sizes": list(SIZES),
                   "ensemble": "noisy_pure"},
        "shrinkage_rule": "a = sigma^2/(sigma^2 + (clip(theta)-floor)^2), sigma from the "
                          "exact Hoeffding formula at the closed-form projection variances; "
                          "shrunk toward floor = 2^{n(1-k)}, then clipped",
        "by_n": {str(n): by_n[n] for n in SIZES},
        "raw_units": units,
    }
    (R / "pass36_clipping_audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {R / 'pass36_clipping_audit.json'}")


if __name__ == "__main__":
    main()
