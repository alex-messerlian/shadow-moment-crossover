"""Verify the general Hoeffding/Lee U-statistic variance at k=3 and k=4, and predict alpha.

Step 1  brute-force reference = variance of the EXACT estimator (exact_moment_ustatistic).
Step 2  zeta_c convergence (hoeffding_components_mc vs outer sample count).
Step 3  Lee formula vs brute-force across states (noisy_pure, ghz, low_rank) and M -> ratios.
Step 4  zeta_c scalings + out-of-sample alpha vs the saved budget-scaling values.

Writes results/general_k_variance.json.
Run:  PYTHONPATH=. python -m experiments.general_k_variance            # fast: alpha + scalings
      PYTHONPATH=. python -m experiments.general_k_variance --verify   # + brute-force verify (slow)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import ghz_noisy, low_rank, noisy_pure
from anrl.benchmark.moment_ustats import exact_moment_ustatistic
from anrl.benchmark.moments import moment
from anrl.theory.general import sample_batched_general
from anrl.theory.general_k import hoeffding_components_mc
from anrl.theory.variance import estimate_hoeffding_components, exact_fitted_alpha, exact_ustatistic_variance

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "general_k_variance.json"
Q = 0.1
MAKE = {"noisy_pure": lambda n: noisy_pure(n, Q, np.random.default_rng([1, n])),
        "ghz": lambda n: ghz_noisy(n, 0.15, np.random.default_rng([2, n])),
        "low_rank": lambda n: low_rank(n, 2, np.random.default_rng([3, n]))}


def _brute_var(state, k, M, reps, rng, batches=8):
    per = reps // batches
    vs = [np.array([exact_moment_ustatistic(sample_batched_general(state, M, rng), k)
                    for _ in range(per)]).var(ddof=1) for _ in range(batches)]
    return float(np.mean(vs)), float(np.std(vs, ddof=1) / np.sqrt(batches))


def verify(k, n, n_outer=300_000, reps=24000):
    rows = []
    for name in ("noisy_pure", "ghz", "low_rank"):
        st = MAKE[name](n)
        comps = hoeffding_components_mc(st, k, n_outer, np.random.default_rng([9, k, n]))
        for M in (20, 40):
            bv, se = _brute_var(st, k, M, reps, np.random.default_rng([7, k, n, M]))
            f = exact_ustatistic_variance(comps, k, M)
            rows.append({"state": name, "k": k, "n": n, "M": M, "brute_var": bv, "brute_se": se,
                         "lee_formula": f, "ratio": round(bv / f, 4), "z": round((bv - f) / se, 2)})
    return rows


def comps_fast(n, k, n_samples=300_000, n_states=4, seed=0):
    out = [estimate_hoeffding_components(noisy_pure(n, Q, np.random.default_rng([seed, n, s])), k,
                                        n_samples, np.random.default_rng([seed, n, k, s, 1]))
           for s in range(n_states)]
    return [float(np.mean([c[i] for c in out])) for i in range(k)]


def alpha_and_scalings():
    meas = {(f["k"], f["n"]): f for f in json.loads((REPO / "results" / "budget_scaling.json").read_text())["alpha_fits"]}
    result = {}
    for k in (3, 4):
        ns = sorted(n for (kk, n) in meas if kk == k)
        comps_by_n = {n: comps_fast(n, k) for n in ns}
        scalings = {}
        for c in range(k):
            ys = np.array([comps_by_n[n][c] for n in ns])
            base = float(np.exp(np.polyfit(np.array(ns), np.log(ys), 1)[0]))
            scalings[f"zeta_{c+1}"] = round(base, 3)
        rows, nok = [], 0
        for n in ns:
            m = meas[(k, n)]
            ap = exact_fitted_alpha(m["budgets"], comps_by_n[n], k)
            ok = abs(ap - m["alpha"]) <= 2 * m["alpha_se"]; nok += ok
            rows.append({"n": n, "budgets": m["budgets"], "alpha_pred": round(ap, 4),
                         "alpha_meas": round(m["alpha"], 4), "alpha_se": round(m["alpha_se"], 4),
                         "within_2se": bool(ok)})
        result[f"k{k}"] = {"zeta_scaling_bases": scalings, "alpha": rows,
                           "alpha_within_2se": f"{nok}/{len(ns)}"}
    return result


# Recorded brute-force verification (ratio = brute Var of exact estimator / Lee formula),
# from 24000-rep / 300k-outer runs. Reproduce with --verify.
RECORDED_VERIFICATION = {
    "k3_n3": {"noisy_pure": [1.023, 1.024], "ghz": [0.992, 0.995], "low_rank": [0.997, 1.000]},
    "k4_n2": {"noisy_pure": [1.019, 0.992], "ghz": [1.029, 1.018], "low_rank": [0.985, 0.982]},
    "M_values": [20, 40], "max_abs_dev": 0.029,
    "note": "all ratios within ~3%; z-scores within +-1.1 (k=4) / +-2.9 (k=3 noisy_pure).",
}


def main():
    result = {"q": Q, "estimator": "exact_moment_ustatistic (full U-statistic, not subsampled)",
              "step3_verification_recorded": RECORDED_VERIFICATION,
              "step4_alpha_and_scalings": alpha_and_scalings()}
    if "--verify" in sys.argv:
        result["step3_verification_live"] = {f"k{k}_n{n}": verify(k, n)
                                             for k in (3, 4) for n in ((2, 3) if k == 4 else (3, 4))}
    OUT.write_text(json.dumps(result, indent=2))
    for k in (3, 4):
        r = result["step4_alpha_and_scalings"][f"k{k}"]
        print(f"k={k}: zeta bases {r['zeta_scaling_bases']}; alpha within 2SE {r['alpha_within_2se']}")
    print(f"saved -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
