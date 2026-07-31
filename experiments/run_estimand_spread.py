"""PASS 38.4: is the estimand actually random within each benchmark ensemble?

PASS 37 noticed that on the noisy-pure ensemble the per-state spread of
``Tr(rho^2)`` is 3e-16 -- numerically zero.  If the estimand does not vary with the
drawn state, then averaging over 48 states is not averaging over 48 different
estimation problems, and a "best constant" baseline is exactly right by
construction.  This quantifies that for every ensemble and every moment order, and
separately asks whether the ESTIMATOR's difficulty varies even when the estimand
does not.

Analytically, for ``rho = (1-q)|psi><psi| + q I/d`` the spectrum is

    (1-q) + q/d   once,      q/d   with multiplicity  d - 1,

independent of ``|psi>``, so ``Tr(rho^k)`` is a function of ``(n, q, k)`` alone:

    Tr(rho^k) = ((1-q) + q/d)^k + (d-1) (q/d)^k,

and at ``k = 2, q = 0.1`` this is ``0.81 + 0.19/2^n``.  The same is trivially true
of a Haar-pure state (every moment is 1) and of the fixed GHZ-noisy state.  Only
the rank-2 Ginibre family has a genuinely random estimand.

The projection variances are a different matter: ``zeta_2`` is a weighted sum over
the state's Pauli spectrum (Section 3.5), which does vary with ``|psi>``.  So the
benchmark averages over states that pose the SAME estimation target with DIFFERENT
estimator difficulty.

Writes ``results/pass38_estimand_spread.json``.
Run:  PYTHONPATH=. python -m experiments.run_estimand_spread
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank, noisy_pure
from anrl.benchmark.moments import moment
from anrl.theory.single_copy_law import closed_form_zetas

R = Path(__file__).resolve().parent.parent / "results"
Q = 0.1
N_STATES = 48
SIZES = (2, 3, 4, 5, 6)
KS = (2, 3, 4)


def analytic_noisy_pure_moment(n: int, k: int, q: float) -> float:
    d = 2 ** n
    return ((1 - q) + q / d) ** k + (d - 1) * (q / d) ** k


def _states(ens: str, n: int, count: int):
    out = []
    for s in range(count):
        rng = np.random.default_rng([38, n, s])
        if ens == "noisy_pure":
            out.append(noisy_pure(n, Q, rng))
        elif ens == "haar_pure":
            out.append(haar_pure(n, rng))
        elif ens == "low_rank":
            out.append(low_rank(n, 2, rng))
        else:
            out.append(ghz_noisy(n, 0.15, rng))
    return out


def pauli_spectral_sum(rho: np.ndarray, n: int) -> float:
    """sum_P <P>^2 14^{-|P|}, the functional that fixes base(zeta_2) in Sec 3.5."""
    import itertools
    P1 = [np.eye(2, dtype=complex),
          np.array([[0, 1], [1, 0]], dtype=complex),
          np.array([[0, -1j], [1j, 0]], dtype=complex),
          np.array([[1, 0], [0, -1]], dtype=complex)]
    tot = 0.0
    for ls in itertools.product(range(4), repeat=n):
        M = P1[ls[0]]
        for a in ls[1:]:
            M = np.kron(M, P1[a])
        w = sum(1 for a in ls if a)
        tot += float(np.real(np.trace(rho @ M))) ** 2 * 14.0 ** (-w)
    return tot


def main() -> None:
    print("=" * 96)
    print("38.4(a)  noisy-pure: Tr(rho^k) is a function of (n, q, k) alone")
    print("=" * 96)
    check = []
    for n in SIZES:
        sts = _states("noisy_pure", n, 12)
        for k in KS:
            vals = np.array([moment(s.density_matrix(), k) for s in sts])
            pred = analytic_noisy_pure_moment(n, k, Q)
            check.append({"n": n, "k": k, "analytic": pred, "measured_mean": float(vals.mean()),
                          "measured_std": float(vals.std()),
                          "max_abs_dev_from_analytic": float(np.abs(vals - pred).max())})
        r = [c for c in check if c["n"] == n and c["k"] == 2][0]
        print(f"  n={n}: k=2 analytic {r['analytic']:.12f}  0.81+0.19/2^n "
              f"{0.81 + 0.19 / 2 ** n:.12f}  spread {r['measured_std']:.2e}")
    worst = max(c["max_abs_dev_from_analytic"] for c in check)
    print(f"\n  max |measured - analytic| over all (n, k): {worst:.2e}")
    print(f"  max per-state spread over all (n, k):       "
          f"{max(c['measured_std'] for c in check):.2e}")

    print("\n" + "=" * 96)
    print("38.4(b)  per-state spread of Tr(rho^k), by ensemble  (48 states, n=4)")
    print("=" * 96)
    spread = {}
    n = 4
    print(f"  {'ensemble':>12}{'k':>4}{'mean':>14}{'std':>12}{'rel std':>11}   verdict")
    for ens in ("noisy_pure", "haar_pure", "low_rank", "ghz_noisy"):
        sts = _states(ens, n, N_STATES)
        for k in KS:
            v = np.array([moment(s.density_matrix(), k) for s in sts])
            rel = float(v.std() / v.mean()) if v.mean() else 0.0
            spread[f"{ens}|{k}"] = {"mean": float(v.mean()), "std": float(v.std()),
                                    "rel_std": rel}
            print(f"  {ens:>12}{k:>4}{v.mean():>14.9f}{v.std():>12.3e}{rel:>11.2e}"
                  f"   {'RANDOM' if rel > 1e-6 else 'deterministic'}")

    print("\n" + "=" * 96)
    print("38.4(c)  does the ESTIMATOR's difficulty vary with the drawn state?")
    print("=" * 96)
    print("  noisy-pure, n=4: Tr(rho^2) is constant, but zeta_2's spectral sum is not.")
    sts = _states("noisy_pure", 4, 24)
    sums = np.array([pauli_spectral_sum(s.density_matrix(), 4) for s in sts])
    z2 = 7.0 ** 4 * sums - np.array([moment(s.density_matrix(), 2) for s in sts]) ** 2
    cf1, cf2 = closed_form_zetas(4, Q)
    print(f"  spectral sum  mean {sums.mean():.8f}  std {sums.std():.3e}  "
          f"rel {sums.std()/sums.mean():.2e}")
    print(f"  zeta_2 per state  mean {z2.mean():.4f}  std {z2.std():.4f}  "
          f"rel {z2.std()/z2.mean():.2e}   (ensemble closed form {cf2:.4f})")
    print(f"  -> the estimand is fixed; the estimator's variance is not. The 48-state "
          f"average is over\n     estimator difficulty, not over estimation problems.")

    out = {
        "description": "PASS 38.4: is the estimand random within each benchmark ensemble?",
        "analytic_form": "Tr(rho^k) = ((1-q) + q/d)^k + (d-1)(q/d)^k for the noisy-pure "
                         "ensemble; independent of |psi>, so a function of (n,q,k) alone",
        "noisy_pure_checks": check,
        "max_abs_dev_from_analytic": worst,
        "per_ensemble_spread_n4": spread,
        "estimator_difficulty_varies": {
            "n": 4, "ensemble": "noisy_pure",
            "spectral_sum_mean": float(sums.mean()), "spectral_sum_std": float(sums.std()),
            "zeta2_per_state_mean": float(z2.mean()), "zeta2_per_state_std": float(z2.std()),
            "zeta2_closed_form_ensemble": cf2,
        },
    }
    (R / "pass38_estimand_spread.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {R / 'pass38_estimand_spread.json'}")


if __name__ == "__main__":
    main()
