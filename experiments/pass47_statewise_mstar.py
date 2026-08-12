"""PASS 47.2 / 47.3(a): is ``M*(rho)`` a usable statewise functional?

    OMP_NUM_THREADS=1 .venv/bin/python experiments/pass47_statewise_mstar.py

Answers the feasibility questions of PASS 47 with measurements rather than estimates.

47.2(a)  Times :mod:`anrl.theory.statewise_zetas`, the sampling-free evaluator of the
         cubic ``zeta_1`` identity and the spectral ``zeta_2`` identity for an ARBITRARY
         state, over ``n = 1..10``, and reports the observed scaling.
47.2(b)  What the evaluator needs as INPUT.  Measures how much of each functional a
         weight-limited subset of the Pauli spectrum recovers -- the subset local shadows
         estimate cheaply -- separately for ``zeta_2`` (kernel ``14^{-|P|}``, suppresses
         high weight) and for the ``zeta_1`` diagonal (kernel ``3^{|P|}``, amplifies it).
47.2(d)  Within-ensemble spread of ``M*(rho)`` on the four committed ensembles: if the
         statewise threshold barely varies inside an ensemble, per-state validation on it
         is not testing statewise sensitivity.  Then two new ensembles built to vary both
         the estimand and the threshold, at no extra cost.

Writes ``results/pass47_statewise_mstar.json``.  No paper edit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import NoisyState, ghz_noisy, haar_pure, low_rank, noisy_pure
from anrl.theory.single_copy_law import closed_form_zetas
from anrl.theory.statewise_zetas import (
    exact_zeta1,
    exact_zeta2,
    pauli_expectations,
    pauli_weights,
    purity_from_expectations,
    truncated_zeta2,
    zeta1_diagonal,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass47_statewise_mstar.json"

SEED = 47
TIMING_MAX_N = 10          # n=11 measured at 242 s, past the one-minute question
SPREAD_N = (3, 4, 5)
SPREAD_STATES = 64


# --------------------------------------------------------------------------- ensembles
def _variable_q(n: int, rng: np.random.Generator) -> NoisyState:
    """Noisy-pure with the depolarizing weight drawn PER STATE, ``q ~ U[0.05, 0.45]``.

    The committed noisy-pure ensemble fixes ``q``, which fixes ``Tr(rho^k)`` exactly
    (PASS 38); drawing ``q`` per state makes the estimand random at no extra cost.
    """
    q = float(rng.uniform(0.05, 0.45))
    base = noisy_pure(n, q, rng)
    return base


def _variable_rank(n: int, rng: np.random.Generator) -> NoisyState:
    """Ginibre with the rank drawn per state, ``r ~ U{1, .., min(8, 2^n)}``, no depolarizing.

    Varies the whole eigenspectrum, hence purity, ``zeta_1``, ``zeta_2`` and ``M*``
    together, and is exactly as cheap to sample as the committed rank-2 family.
    """
    r = int(rng.integers(1, min(8, 2 ** n) + 1))
    return low_rank(n, r, rng)


ENSEMBLES = {
    "noisy_pure_q0.1": lambda n, rng: noisy_pure(n, 0.1, rng),
    "haar_pure": lambda n, rng: haar_pure(n, rng),
    "low_rank_2": lambda n, rng: low_rank(n, 2, rng),
    "ghz_noisy_q0.15": lambda n, rng: ghz_noisy(n, 0.15, rng),
    "variable_q": _variable_q,
    "variable_rank": _variable_rank,
}
NEW_ENSEMBLES = ("variable_q", "variable_rank")


def _statewise(rho: np.ndarray, n: int, weights: np.ndarray) -> dict:
    m = pauli_expectations(rho, n)
    z1 = exact_zeta1(m, n)
    z2 = exact_zeta2(m, n, weights)
    return {
        "zeta1": z1,
        "zeta2": z2,
        "m_star": z2 / (2.0 * z1) if z1 > 0 else float("inf"),
        "purity": purity_from_expectations(m, n),
    }


# ------------------------------------------------------------------- 47.2(a) timing
def timing_scan() -> dict:
    rows = []
    for n in range(1, TIMING_MAX_N + 1):
        rho = noisy_pure(n, 0.1, np.random.default_rng([SEED, n])).density_matrix()
        t0 = time.perf_counter()
        m = pauli_expectations(rho, n)
        t_pauli = time.perf_counter() - t0
        w = pauli_weights(n)
        t0 = time.perf_counter()
        z2 = exact_zeta2(m, n, w)
        t_z2 = time.perf_counter() - t0
        t0 = time.perf_counter()
        z1 = exact_zeta1(m, n)
        t_z1 = time.perf_counter() - t0
        rows.append({
            "n": n, "pauli_terms": 4 ** n, "pair_terms": 10 ** n,
            "t_pauli_s": t_pauli, "t_zeta2_s": t_z2, "t_zeta1_s": t_z1,
            "t_total_s": t_pauli + t_z2 + t_z1,
            "zeta1": z1, "zeta2": z2, "m_star": z2 / (2.0 * z1),
        })
        print(f"  n={n:2d}  4^n={4**n:>9}  10^n={10**n:>13}   "
              f"pauli {t_pauli*1e3:8.2f} ms   zeta2 {t_z2*1e3:8.2f} ms   "
              f"zeta1 {t_z1:9.3f} s   M*={z2/(2*z1):.4g}", flush=True)
    ratios = [b["t_zeta1_s"] / a["t_zeta1_s"] for a, b in zip(rows, rows[1:]) if a["t_zeta1_s"] > 5e-3]
    under_60 = max(r["n"] for r in rows if r["t_total_s"] < 60.0)
    return {
        "rows": rows,
        "zeta1_time_ratio_per_qubit": ratios,
        "zeta1_time_ratio_median": float(np.median(ratios)) if ratios else None,
        "expected_ratio_if_theta_10n": 10.0,
        "largest_n_under_60s": under_60,
        "n11_measured_s": 242.07,
        "note": (
            "zeta_2 is Theta(4^n) and negligible; zeta_1 is Theta(10^n) and sets the ceiling. "
            "n=11 was measured separately at 242 s, so n=10 is the largest size evaluable "
            "in under a minute on one core."
        ),
    }


# ---------------------------------------------------- 47.2(b) what inputs are needed
def input_requirements() -> dict:
    """How much of each functional a weight-<= w subset of the Pauli spectrum recovers."""
    out = {}
    for ens in ("noisy_pure_q0.1", "haar_pure", "low_rank_2", "ghz_noisy_q0.15"):
        for n in (4, 6):
            rho = ENSEMBLES[ens](n, np.random.default_rng([SEED, 7, n])).density_matrix()
            m = pauli_expectations(rho, n)
            w = pauli_weights(n)
            z1_full, z2_full = exact_zeta1(m, n), exact_zeta2(m, n, w)
            diag_full = zeta1_diagonal(m, n, w)
            rows = []
            for wmax in range(0, n + 1):
                keep = w <= wmax
                n_terms = int(keep.sum())
                z2_t = truncated_zeta2(m, n, wmax, w)
                diag_t = float((np.float64(3.0) ** w[keep] * m[keep] ** 2).sum()) / 4.0 ** n
                rows.append({
                    "max_weight": wmax, "n_terms": n_terms,
                    "frac_of_4n": n_terms / 4 ** n,
                    "zeta2_trunc": z2_t,
                    "zeta2_rel_err": abs(z2_t - z2_full) / abs(z2_full),
                    "zeta1_diag_trunc": diag_t,
                    "zeta1_diag_rel_err": abs(diag_t - diag_full) / abs(diag_full),
                })
            out[f"{ens}|n{n}"] = {
                "zeta1_exact": z1_full, "zeta2_exact": z2_full,
                "zeta1_diagonal": diag_full,
                "zeta1_offdiag_fraction": (z1_full + purity_from_expectations(m, n) ** 2 - diag_full)
                / diag_full,
                "truncation": rows,
            }
            print(f"  {ens} n={n}: zeta1={z1_full:.4g} zeta2={z2_full:.5g}  "
                  f"weight<=1 gives zeta2 to {rows[1]['zeta2_rel_err']*100:.2f}% "
                  f"and the zeta1 diagonal to {rows[1]['zeta1_diag_rel_err']*100:.1f}%", flush=True)
    out["_inputs_needed"] = {
        "zeta2": "all 4^n squared Pauli expectations <P>^2, weighted 14^{-|P|}",
        "zeta1": "all 4^n SIGNED Pauli expectations <P>, in triple products over 10^n compatible pairs",
        "equivalent_to": "the full density matrix; 4^n real numbers is full state knowledge",
        "not_a_subset": (
            "no proper subset suffices exactly: the zeta_1 diagonal carries 3^{|P|}, which "
            "AMPLIFIES the high-weight strings that local shadows estimate worst"
        ),
    }
    return out


# ------------------------------------ 47.2(d) within-ensemble spread of the threshold
def ensemble_spread() -> dict:
    out = {}
    for ens, make in ENSEMBLES.items():
        for n in SPREAD_N:
            w = pauli_weights(n)
            rec = [_statewise(make(n, np.random.default_rng([SEED, 3, n, s])).density_matrix(), n, w)
                   for s in range(SPREAD_STATES)]
            ms = np.array([r["m_star"] for r in rec])
            pu = np.array([r["purity"] for r in rec])
            z1 = np.array([r["zeta1"] for r in rec])
            z2 = np.array([r["zeta2"] for r in rec])
            out[f"{ens}|n{n}"] = {
                "n_states": SPREAD_STATES,
                "m_star_mean": float(ms.mean()), "m_star_std": float(ms.std(ddof=1)),
                "m_star_rel_std": float(ms.std(ddof=1) / ms.mean()),
                "m_star_min": float(ms.min()), "m_star_max": float(ms.max()),
                "m_star_spread_ratio": float(ms.max() / ms.min()),
                "purity_mean": float(pu.mean()), "purity_rel_std": float(pu.std(ddof=1) / pu.mean()),
                "zeta1_rel_std": float(z1.std(ddof=1) / z1.mean()),
                "zeta2_rel_std": float(z2.std(ddof=1) / z2.mean()),
                "is_new_ensemble": ens in NEW_ENSEMBLES,
            }
            r = out[f"{ens}|n{n}"]
            print(f"  {ens:18s} n={n}: M* {r['m_star_mean']:10.4g}  rel-std {r['m_star_rel_std']*100:6.2f}%  "
                  f"max/min {r['m_star_spread_ratio']:6.3f}   purity rel-std {r['purity_rel_std']*100:6.2f}%",
                  flush=True)
    return out


# ------------------- cross-check: the statewise evaluator vs the ensemble closed form
def closed_form_crosscheck() -> dict:
    rows = []
    for n in (2, 3, 4, 5):
        w = pauli_weights(n)
        vals = [_statewise(noisy_pure(n, 0.1, np.random.default_rng([SEED, 11, n, s])).density_matrix(), n, w)
                for s in range(400)]
        z1 = np.array([v["zeta1"] for v in vals]); z2 = np.array([v["zeta2"] for v in vals])
        c1, c2 = closed_form_zetas(n, 0.1)
        rows.append({
            "n": n, "n_states": 400,
            "zeta1_statewise_mean": float(z1.mean()), "zeta1_sem": float(z1.std(ddof=1) / 20.0),
            "zeta1_closed_form": c1,
            "zeta1_dev_in_sem": float(abs(z1.mean() - c1) / (z1.std(ddof=1) / 20.0)),
            "zeta2_statewise_mean": float(z2.mean()), "zeta2_sem": float(z2.std(ddof=1) / 20.0),
            "zeta2_closed_form": c2,
            "zeta2_dev_in_sem": float(abs(z2.mean() - c2) / (z2.std(ddof=1) / 20.0)),
        })
        print(f"  n={n}: zeta1 {z1.mean():.6f} vs closed {c1:.6f} ({rows[-1]['zeta1_dev_in_sem']:.2f} sem) | "
              f"zeta2 {z2.mean():.3f} vs closed {c2:.3f} ({rows[-1]['zeta2_dev_in_sem']:.2f} sem)", flush=True)
    return {"rows": rows, "gate": "every deviation must be within 3 sem"}


def exact_limit_gates() -> dict:
    """Sampling-free gates: maximally mixed, single qubit, pure product, GHZ."""
    gates = []
    for n in (1, 2, 3, 4):
        d = 2 ** n
        m = pauli_expectations(np.eye(d) / d, n)
        gates.append({"gate": f"maximally_mixed_n{n}", "zeta1": exact_zeta1(m, n),
                      "zeta1_expected": 0.0, "zeta2": exact_zeta2(m, n),
                      "zeta2_expected": 7.0 ** n - 4.0 ** -n})
    for t in (0.0, 0.3, 0.7, 1.0):
        rho = 0.5 * np.array([[1 + t, 0], [0, 1 - t]], dtype=complex)
        gates.append({"gate": f"single_qubit_t{t}", "zeta1": exact_zeta1(pauli_expectations(rho, 1), 1),
                      "zeta1_expected": 0.75 * t * t - 0.25 * t ** 4})
    for n in (2, 3, 4, 5):
        psi = np.zeros(2 ** n); psi[0] = 1.0
        m = pauli_expectations(np.outer(psi, psi).astype(complex), n)
        gates.append({"gate": f"pure_product_n{n}", "zeta2": exact_zeta2(m, n),
                      "zeta2_expected": (15 / 2) ** n - 1})
        g = np.zeros(2 ** n); g[0] = g[-1] = 1 / np.sqrt(2)
        m = pauli_expectations(np.outer(g, g).astype(complex), n)
        gates.append({"gate": f"ghz_n{n}", "zeta2": exact_zeta2(m, n),
                      "zeta2_expected": 0.5 * ((15 / 2) ** n + (13 / 2) ** n) - 0.5})
    worst = max(abs(g[k] - g[f"{k}_expected"]) for g in gates for k in ("zeta1", "zeta2")
                if k in g and f"{k}_expected" in g)
    return {"gates": gates, "worst_absolute_deviation": float(worst)}


def main() -> None:
    t0 = time.time()
    print("47.2(a) timing the exact statewise evaluator")
    timing = timing_scan()
    print("\nexact-limit gates (sampling-free)")
    gates = exact_limit_gates()
    print(f"  worst absolute deviation across all gates: {gates['worst_absolute_deviation']:.3e}")
    print("\ncross-check: statewise ensemble mean vs the committed closed form")
    cross = closed_form_crosscheck()
    print("\n47.2(b) what inputs the evaluator needs")
    inputs = input_requirements()
    print("\n47.2(d) within-ensemble spread of M*(rho)")
    spread = ensemble_spread()

    payload = {
        "description": "PASS 47.2 / 47.3(a): exact statewise M*(rho) -- cost, inputs, ensemble spread",
        "seed": SEED,
        "timing_47_2a": timing,
        "exact_limit_gates": gates,
        "closed_form_crosscheck": cross,
        "input_requirements_47_2b": inputs,
        "ensemble_spread_47_2d": spread,
        "wall_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
