"""CGK reconciliation experiment (arXiv:2512.10929, Cotler-Gong-Kannan).

Tests whether our exact per-qubit-channel bias law (Sec. 4.2) reproduces the
mechanism behind CGK Theorem 2.4: that the two-copy purity-*testing* advantage
(Haar-random pure vs. maximally mixed) collapses under per-qubit depolarizing
noise, and that the collapse is exponential in n.

CGK noise model: D_lambda^{ox n}[rho] ox D_lambda^{ox n}[rho], with the
single-qubit depolarizing channel D_lambda(rho) = (1-lambda) rho + lambda I/2.
This is exactly the k=2 case of our Sec. 4.2 per-qubit-channel setup, so the
noisy two-copy cyclic (SWAP) test returns Tr(sigma^2) with sigma = D_lambda^{ox n}(rho).

We compute, per (n, lambda):
  * Tr(sigma_pure^2), sigma_pure = D_lambda^{ox n}(|psi><psi|), averaged over Haar draws;
  * Tr(sigma_mixed^2), sigma_mixed = D_lambda^{ox n}(I/2^n) = I/2^n exactly (depolarizing
    fixes the maximally mixed state) -> Tr = 2^{-n};
  * the testing gap  Delta(n,lambda) = Tr(sigma_pure^2) - Tr(sigma_mixed^2);
  * the SWAP-test shot count 1/Delta^2 (the estimator has O(1) variance), and its
    exponential growth base, compared against CGK's lower-bound base
    b(lambda) = 4/(1 + 3(1-lambda)^4).

Everything routes through the committed bias-law code (perqubit_channel_value).
An independent analytic Haar average cross-checks the numerics.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from anrl.theory.bias import perqubit_channel_value

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)


def depolarizing_kraus(lam: float) -> list[np.ndarray]:
    """Single-qubit depolarizing Kraus ops for D_lambda(rho) = (1-lambda) rho + lambda I/2.

    Pauli form: (1 - 3 lambda/4) rho + (lambda/4)(X rho X + Y rho Y + Z rho Z).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [0,1], got {lam}")
    return [
        math.sqrt(1.0 - 3.0 * lam / 4.0) * _I,
        math.sqrt(lam / 4.0) * _X,
        math.sqrt(lam / 4.0) * _Y,
        math.sqrt(lam / 4.0) * _Z,
    ]


def haar_pure_density(n: int, rng: np.random.Generator) -> np.ndarray:
    """Density matrix of a Haar-random pure state on n qubits (normalized complex Gaussian)."""
    d = 2 ** n
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    v /= np.linalg.norm(v)
    return np.outer(v, v.conj())


def b_of_lambda(lam: float) -> float:
    return 4.0 / (1.0 + 3.0 * (1.0 - lam) ** 4)


def analytic_tr_sigma_pure2(n: int, lam: float) -> float:
    """Exact Haar average of Tr(sigma_pure^2) under per-qubit depolarizing.

    E[Tr(sigma^2)] = (1/2^n) [ 1 + ((1 + 3(1-lambda)^2)^n - 1) / (2^n + 1) ].
    """
    D = 2 ** n
    s = 1.0 + 3.0 * (1.0 - lam) ** 2
    return (1.0 / D) * (1.0 + (s ** n - 1.0) / (D + 1.0))


def fit_base(ns: list[int], ys: list[float]) -> float:
    """Per-unit-n exponential base from a log-linear fit of ys vs ns."""
    lx = np.array(ns, dtype=float)
    ly = np.log(np.array(ys, dtype=float))
    slope = np.polyfit(lx, ly, 1)[0]
    return float(math.exp(slope))


def main() -> dict:
    ns = [2, 4, 6, 8, 10]
    lams = [0.05, 0.1, 0.2]
    draws = {2: 400, 4: 400, 6: 200, 8: 100, 10: 40}
    rng = np.random.default_rng(20260714)

    # Confirm depolarizing fixes the maximally mixed state at the largest n.
    n_chk = max(ns)
    mixed = np.eye(2 ** n_chk, dtype=np.complex128) / (2 ** n_chk)
    tr_mixed2 = perqubit_channel_value(mixed, 2, depolarizing_kraus(0.1), n_chk)
    mixed_dev = abs(tr_mixed2 - 2.0 ** (-n_chk))

    out = {
        "description": "CGK Thm 2.4 mechanism via Sec 4.2 exact bias law (per-qubit depolarizing, k=2)",
        "seed": 20260714,
        "sigma_mixed_check": {
            "n": n_chk,
            "tr_sigma_mixed2_measured": tr_mixed2,
            "two_to_minus_n": 2.0 ** (-n_chk),
            "max_abs_deviation": mixed_dev,
            "is_maximally_mixed": bool(mixed_dev < 1e-12),
        },
        "per_lambda": {},
    }

    for lam in lams:
        kraus = depolarizing_kraus(lam)
        rec = {
            "n": ns,
            "tr_sigma_pure2_mean": [],
            "tr_sigma_pure2_sem": [],
            "tr_sigma_pure2_analytic": [],
            "tr_sigma_mixed2": [],
            "delta_numeric": [],
            "delta_analytic": [],
            "inv_delta_sq": [],
        }
        for n in ns:
            vals = [
                perqubit_channel_value(haar_pure_density(n, rng), 2, kraus, n)
                for _ in range(draws[n])
            ]
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
            trm = 2.0 ** (-n)
            delta = mean - trm
            rec["tr_sigma_pure2_mean"].append(mean)
            rec["tr_sigma_pure2_sem"].append(sem)
            rec["tr_sigma_pure2_analytic"].append(analytic_tr_sigma_pure2(n, lam))
            rec["tr_sigma_mixed2"].append(trm)
            rec["delta_numeric"].append(delta)
            rec["delta_analytic"].append(analytic_tr_sigma_pure2(n, lam) - trm)
            rec["inv_delta_sq"].append(1.0 / delta ** 2)

        x = 1.0 - lam
        rec["b_lambda"] = b_of_lambda(lam)
        rec["fitted_base_inv_delta_sq_numeric"] = fit_base(ns, rec["inv_delta_sq"])
        rec["fitted_base_inv_delta_sq_analytic"] = fit_base(
            ns, [1.0 / d ** 2 for d in rec["delta_analytic"]]
        )
        # asymptotic base of 1/Delta^2 = (4/(1+3(1-lambda)^2))^2
        rec["asymptotic_base_inv_delta_sq"] = (4.0 / (1.0 + 3.0 * x ** 2)) ** 2
        # asymptotic base of Delta itself = (1+3(1-lambda)^2)/4  (gap-closing rate)
        rec["asymptotic_base_delta"] = (1.0 + 3.0 * x ** 2) / 4.0
        # exact ordering: base(1/Delta^2) - ... ; 4(1+3x^2) - (1+3x^2 form) => 3(1-x)^2 >= 0
        rec["base_inv_delta_sq_minus_b"] = rec["asymptotic_base_inv_delta_sq"] - rec["b_lambda"]
        rec["base_ge_b"] = bool(rec["asymptotic_base_inv_delta_sq"] >= rec["b_lambda"] - 1e-12)
        out["per_lambda"][str(lam)] = rec

    return out


if __name__ == "__main__":
    result = main()
    os.makedirs("results", exist_ok=True)
    with open("results/cgk_mechanism.json", "w") as fh:
        json.dump(result, fh, indent=2)

    chk = result["sigma_mixed_check"]
    print(f"sigma_mixed = I/2^n at n={chk['n']}: measured Tr={chk['tr_sigma_mixed2_measured']:.3e}, "
          f"2^-n={chk['two_to_minus_n']:.3e}, dev={chk['max_abs_deviation']:.2e} "
          f"-> maximally mixed: {chk['is_maximally_mixed']}")
    print()
    for lam, rec in result["per_lambda"].items():
        print(f"=== lambda = {lam}  (b(λ) = {rec['b_lambda']:.5f}) ===")
        print(f"{'n':>3} {'Tr(σ_pure²)':>13} {'analytic':>11} {'Tr(σ_mix²)':>12} {'Δ':>11} {'1/Δ²':>12}")
        for i, n in enumerate(rec["n"]):
            print(f"{n:>3} {rec['tr_sigma_pure2_mean'][i]:>13.6f} "
                  f"{rec['tr_sigma_pure2_analytic'][i]:>11.6f} "
                  f"{rec['tr_sigma_mixed2'][i]:>12.6f} "
                  f"{rec['delta_numeric'][i]:>11.6f} {rec['inv_delta_sq'][i]:>12.3f}")
        print(f"  fitted base 1/Δ² (numeric)  = {rec['fitted_base_inv_delta_sq_numeric']:.5f}")
        print(f"  fitted base 1/Δ² (analytic) = {rec['fitted_base_inv_delta_sq_analytic']:.5f}")
        print(f"  asymptotic base 1/Δ²        = {rec['asymptotic_base_inv_delta_sq']:.5f}")
        print(f"  CGK b(λ)                    = {rec['b_lambda']:.5f}")
        print(f"  base(1/Δ²) - b(λ)           = {rec['base_inv_delta_sq_minus_b']:+.5f}  (>=0: {rec['base_ge_b']})")
        print()
    print("saved results/cgk_mechanism.json")
