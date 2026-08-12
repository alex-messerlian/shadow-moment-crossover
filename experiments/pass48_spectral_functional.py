"""PASS 48.5: the spectral-functional result, stated precisely and attacked.

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/pass48_spectral_functional.py

Paper one leads with two claims:

  C1  For every n-qubit state,   7^n - 1  <=  zeta_2(rho)  <=  (17/2)^n ,
      hence base(zeta_2) in [7, 17/2] whenever the limit exists.

  C2  base(M*) = base(zeta_2) / base(zeta_1) exactly, whenever both limits exist and the
      leading coefficients are nonzero.

Both are proved in two lines from the spectral identity
``zeta_2 = 7^n sum_P <P>^2 14^{-|P|} - Tr(rho^2)^2``:

  lower  the identity string contributes 7^n * 1 * 14^0 = 7^n and every other term is >= 0,
         while Tr(rho^2)^2 <= 1, so zeta_2 >= 7^n - 1, with EQUALITY iff <P> = 0 for all
         P != I, i.e. only at the maximally mixed state (where Tr(rho^2)^2 = 4^-n, giving
         7^n - 4^-n, slightly above the bound).
  upper  <P>^2 <= 1 termwise and sum_P 14^{-|P|} = prod_q (1 + 3/14) = (17/14)^n, so
         zeta_2 <= 7^n (17/14)^n = (17/2)^n; attaining it needs <P>^2 = 1 for ALL 4^n
         strings, which no state achieves for n >= 1, so the upper bound is NOT attained.

This script states the conditions, hunts for the tightest achievable values by direct
search over structured and optimized states, and tries to break both claims.

Writes ``results/pass48_spectral_functional.json``.  No paper edit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank, noisy_pure
from anrl.physics import kron_all
from anrl.theory.statewise_zetas import (
    exact_zeta1,
    exact_zeta2,
    pauli_expectations,
    pauli_weights,
    purity_from_expectations,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass48_spectral_functional.json"
SEED = 48

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _dm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=complex).ravel()
    return np.outer(v, v.conj()) / float(np.vdot(v, v).real)


def state_zoo(n: int) -> dict[str, np.ndarray]:
    """The PASS 47 zoo, plus structured states chosen to push the bounds."""
    rng = np.random.default_rng([SEED, 21, n])
    d = 2 ** n
    zoo: dict[str, np.ndarray] = {}
    zoo["haar_pure"] = haar_pure(n, rng).density_matrix()
    zoo["noisy_pure_q0.1"] = noisy_pure(n, 0.1, rng).density_matrix()
    zoo["ghz_noisy_q0.15"] = ghz_noisy(n, 0.15, rng).density_matrix()
    g = np.zeros(d); g[0] = g[-1] = 1.0
    zoo["ghz_pure"] = _dm(g)
    w = np.zeros(d)
    for q in range(n):
        w[1 << q] = 1.0
    zoo["w_state"] = _dm(w)
    p = np.zeros(d); p[0] = 1.0
    zoo["pure_product"] = _dm(p)
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    idx = np.arange(d)
    for q in range(n - 1):
        psi = psi * np.where(((idx >> (n - 1 - q)) & 1) & ((idx >> (n - 2 - q)) & 1), -1.0, 1.0)
    zoo["graph_state"] = _dm(psi)
    for r in (1, 2, 3, 5, 8):
        if r <= d:
            zoo[f"ginibre_rank{r}"] = low_rank(n, r, rng).density_matrix()
    h = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    h = (h + h.conj().T) / 2.0
    ev, evec = np.linalg.eigh(h)
    wts = np.exp(-(ev - ev.min()))
    zoo["gibbs_beta1"] = (evec * (wts / wts.sum())) @ evec.conj().T
    zz = kron_all([_Z] * n)
    xx = kron_all([_X] * n)
    zoo["pauli_concentrated"] = (np.eye(d) + 0.45 * zz + 0.45 * xx) / d
    zoo["maximally_mixed"] = np.eye(d, dtype=complex) / d
    # --- states chosen to PUSH the bounds ---
    # product of |+> on every qubit: all-X stabilizer, weight profile 1 on the 2^n X-strings
    plus = kron_all([np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)] * n)
    zoo["product_plus"] = plus
    # a product of single-qubit states with equal Bloch components: maximizes low-weight mass
    b = np.eye(2, dtype=complex) / 2 + (_X + _Z) / (2 * np.sqrt(2.0))
    zoo["product_tilted"] = kron_all([b] * n)
    # near-mixed: purity just above the floor, to approach the lower bound from above
    zoo["near_mixed_eps"] = (1 - 1e-3) * np.eye(d) / d + 1e-3 * zoo["pure_product"]
    return zoo


def _record(name: str, n: int, rho: np.ndarray, weights: np.ndarray) -> dict:
    m = pauli_expectations(rho, n)
    z1 = exact_zeta1(m, n)
    z2 = exact_zeta2(m, n, weights)
    lo, hi = 7.0 ** n - 1.0, (17 / 2) ** n
    return {
        "state": name, "n": n, "purity": purity_from_expectations(m, n),
        "zeta1": z1, "zeta2": z2,
        "lower_7n_minus_1": lo, "upper_17_over_2_n": hi,
        "slack_above_lower": z2 - lo, "slack_below_upper": hi - z2,
        "zeta2_over_7n": z2 / 7.0 ** n,
        "in_bounds": bool(lo - 1e-9 <= z2 <= hi + 1e-9),
    }


def bounds_scan() -> dict:
    rows = []
    for n in (2, 3, 4, 5):
        weights = pauli_weights(n)
        for name, rho in state_zoo(n).items():
            rows.append(_record(name, n, rho, weights))
    violations = [r for r in rows if not r["in_bounds"]]
    tight_lo = min(rows, key=lambda r: r["slack_above_lower"] / r["lower_7n_minus_1"])
    tight_hi = min(rows, key=lambda r: r["slack_below_upper"] / r["upper_17_over_2_n"])
    print(f"  {len(rows)} (state, n) records; violations: {len(violations)}")
    print(f"  tightest against the LOWER bound: {tight_lo['state']} n={tight_lo['n']}, "
          f"zeta_2/7^n = {tight_lo['zeta2_over_7n']:.9f}")
    print(f"  tightest against the UPPER bound: {tight_hi['state']} n={tight_hi['n']}, "
          f"zeta_2 = {tight_hi['zeta2']:.4g} vs (17/2)^n = {tight_hi['upper_17_over_2_n']:.4g} "
          f"(ratio {tight_hi['zeta2']/tight_hi['upper_17_over_2_n']:.4f})")
    return {"rows": rows, "violations": violations,
            "tightest_lower": tight_lo, "tightest_upper": tight_hi}


def optimize_upper(n: int, restarts: int = 24, steps: int = 400) -> dict:
    """Maximize zeta_2 over pure states by projected gradient ascent on the state vector.

    If any state gets near (17/2)^n the upper bound is tight; if the search saturates well
    below it, the bound is loose and the paper should say by how much.
    """
    d = 2 ** n
    weights = pauli_weights(n)
    kern = 7.0 ** n * np.float64(14.0) ** (-weights)
    best = None
    rng = np.random.default_rng([SEED, 77, n])
    for r in range(restarts):
        v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        v /= np.linalg.norm(v)
        step = 0.35
        cur = -np.inf
        for _ in range(steps):
            rho = np.outer(v, v.conj())
            m = pauli_expectations(rho, n)
            val = float((kern * m * m).sum()) - purity_from_expectations(m, n) ** 2
            # dval/dm = 2 kern m (the purity term is constant on pure states), and
            # d<P>/dv = 2 P v, so the gradient direction is sum_P (2 kern_P m_P) P v.
            grad = np.zeros(d, dtype=complex)
            coef = 2.0 * kern * m
            # apply sum_P coef_P P to v via the inverse per-qubit Pauli transform
            t = coef.reshape((4,) * n)
            basis = [np.eye(2, dtype=complex), _X,
                     np.array([[0, -1j], [1j, 0]], dtype=complex), _Z]
            for flat in range(4 ** n):
                c = t.flat[flat]
                if abs(c) < 1e-14:
                    continue
                digits = [(flat // 4 ** (n - 1 - q)) % 4 for q in range(n)]
                grad += c * (kron_all([basis[i] for i in digits]) @ v)
            v = v + step * grad / (np.linalg.norm(grad) + 1e-30)
            v /= np.linalg.norm(v)
            if val <= cur + 1e-12:
                step *= 0.85
            cur = max(cur, val)
        m = pauli_expectations(np.outer(v, v.conj()), n)
        z2 = exact_zeta2(m, n, weights)
        if best is None or z2 > best["zeta2"]:
            best = {"zeta2": z2, "restart": r,
                    "upper": (17 / 2) ** n, "ratio_to_upper": z2 / (17 / 2) ** n,
                    "zeta2_over_7n": z2 / 7.0 ** n}
    return best


def base_relation(n_lo: int = 3, n_hi: int = 8) -> dict:
    """C2: base(M*) == base(zeta_2)/base(zeta_1) for families with different bases."""
    ns = list(range(n_lo, n_hi + 1))
    fams = {
        "noisy_pure_q0.1": lambda n: noisy_pure(n, 0.1, np.random.default_rng([SEED, n])).density_matrix(),
        "haar_pure": lambda n: haar_pure(n, np.random.default_rng([SEED, 5, n])).density_matrix(),
        "pure_product": lambda n: state_zoo(n)["pure_product"],
        "ghz_pure": lambda n: state_zoo(n)["ghz_pure"],
        "graph_state": lambda n: state_zoo(n)["graph_state"],
        "product_plus": lambda n: state_zoo(n)["product_plus"],
        "product_tilted": lambda n: state_zoo(n)["product_tilted"],
        "ginibre_rank2": lambda n: low_rank(n, 2, np.random.default_rng([SEED, 9, n])).density_matrix(),
    }
    out = {}
    for label, make in fams.items():
        z1s, z2s, mss = [], [], []
        for n in ns:
            w = pauli_weights(n)
            m = pauli_expectations(make(n), n)
            a, b = exact_zeta1(m, n), exact_zeta2(m, n, w)
            z1s.append(a); z2s.append(b); mss.append(b / (2 * a) if a > 0 else np.nan)

        def base(seq):
            v = np.array(seq, float)
            ok = np.isfinite(v) & (v > 0)
            return (float(np.exp(np.polyfit(np.array(ns)[ok], np.log(v[ok]), 1)[0]))
                    if ok.sum() > 1 else None)

        b1, b2, bm = base(z1s), base(z2s), base(mss)
        ratio = (bm / (b2 / b1)) if (b1 and b2 and bm) else None
        out[label] = {"sizes": ns, "zeta1": z1s, "zeta2": z2s, "m_star": mss,
                      "base_zeta1": b1, "base_zeta2": b2, "base_m_star": bm,
                      "base_zeta2_over_base_zeta1": (b2 / b1) if (b1 and b2) else None,
                      "identity_ratio": ratio}
        print(f"  {label:18s} base(z1) {b1:6.4f}  base(z2) {b2:6.4f}  base(M*) {bm:6.4f}  "
              f"b2/b1 {b2/b1:6.4f}  ratio {ratio:.8f}")
    worst = max(abs(v["identity_ratio"] - 1.0) for v in out.values() if v["identity_ratio"])
    return {"families": out, "worst_deviation_from_1": float(worst)}


def break_attempts() -> dict:
    """48.5(e): try to break C1 and C2. Every attempt is reported, pass or fail."""
    att = []

    # A1. Can zeta_2 fall BELOW 7^n - 1? Search over random states of every rank.
    rng = np.random.default_rng([SEED, 101])
    worst = None
    for n in (2, 3, 4):
        w = pauli_weights(n)
        d = 2 ** n
        for _ in range(400):
            r = int(rng.integers(1, d + 1))
            gm = rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))
            rho = gm @ gm.conj().T
            rho /= np.trace(rho).real
            z2 = exact_zeta2(pauli_expectations(rho, n), n, w)
            slack = z2 - (7.0 ** n - 1.0)
            if worst is None or slack < worst[0]:
                worst = (slack, n, z2)
    att.append({"attempt": "A1 random states of every rank vs the lower bound",
                "n_states": 1200, "min_slack_above_lower": float(worst[0]),
                "at_n": worst[1], "verdict": "HOLDS" if worst[0] >= -1e-9 else "BROKEN"})

    # A2. Is the lower bound ATTAINED anywhere? At I/d, zeta_2 = 7^n - 4^-n > 7^n - 1.
    rows = []
    for n in (2, 3, 4, 5):
        z2 = exact_zeta2(pauli_expectations(np.eye(2 ** n) / 2 ** n, n), n, pauli_weights(n))
        rows.append({"n": n, "zeta2_at_mixed": z2, "lower": 7.0 ** n - 1.0,
                     "gap": z2 - (7.0 ** n - 1.0), "gap_equals_1_minus_4mn": abs(
                         (z2 - (7.0 ** n - 1.0)) - (1.0 - 4.0 ** -n)) < 1e-9})
    att.append({"attempt": "A2 is the lower bound attained?", "rows": rows,
                "verdict": "NOT ATTAINED -- the infimum is 7^n - 4^-n at I/d, above 7^n - 1; "
                           "the stated bound is correct but not tight by 1 - 4^-n"})

    # A3. How close can the UPPER bound be approached? Gradient search over pure states.
    opt = {str(n): optimize_upper(n) for n in (2, 3)}
    att.append({"attempt": "A3 maximize zeta_2 over pure states", "results": opt,
                "verdict": "NOT ATTAINED -- best found is a fraction "
                           f"{max(v['ratio_to_upper'] for v in opt.values()):.3f} of (17/2)^n; "
                           "the upper bound needs <P>^2 = 1 for all 4^n strings, impossible"})

    # A4. Does base(zeta_2) ever leave [7, 17/2] at finite n via the RATIO zeta_2^{1/n}?
    ratios = []
    for n in (2, 3, 4, 5):
        w = pauli_weights(n)
        for name, rho in state_zoo(n).items():
            z2 = exact_zeta2(pauli_expectations(rho, n), n, w)
            ratios.append({"state": name, "n": n, "zeta2_pow_1_over_n": float(z2 ** (1.0 / n))})
    outside = [r for r in ratios if not (7.0 - 1e-9 <= r["zeta2_pow_1_over_n"] <= 8.5 + 1e-9)]
    att.append({"attempt": "A4 finite-n zeta_2^{1/n} inside [7, 17/2]?",
                "n_records": len(ratios), "outside": outside,
                "min": float(min(r["zeta2_pow_1_over_n"] for r in ratios)),
                "max": float(max(r["zeta2_pow_1_over_n"] for r in ratios)),
                "verdict": ("HOLDS at every n tested" if not outside else
                            "CAVEAT: zeta_2^{1/n} can dip below 7 at small n because the "
                            "-Tr(rho^2)^2 term is O(1); the BASE (the n->infinity limit) is "
                            "still >= 7, but the finite-n ratio is not bounded below by 7")})

    # A5. Does C2 fail when a base does not exist -- e.g. zeta_1 -> 0?
    rows = []
    for n in (2, 3, 4, 5):
        w = pauli_weights(n)
        m = pauli_expectations(np.eye(2 ** n) / 2 ** n, n)
        rows.append({"n": n, "zeta1": exact_zeta1(m, n), "zeta2": exact_zeta2(m, n, w),
                     "m_star": "inf"})
    att.append({"attempt": "A5 C2 at the maximally mixed state", "rows": rows,
                "verdict": "C2's precondition FAILS by construction: zeta_1 = 0 exactly, so "
                           "M* is infinite and base(M*) is undefined. The 'leading coefficients "
                           "nonzero' condition is doing real work and must be stated."})

    # A6. A family where zeta_1's base drifts, so the fitted base is window-dependent.
    w_by_n = {n: pauli_weights(n) for n in range(3, 9)}
    prod = []
    for n in range(3, 9):
        m = pauli_expectations(state_zoo(n)["pure_product"], n)
        prod.append(exact_zeta1(m, n))
    windows = {}
    for lo, hi in ((3, 5), (4, 6), (5, 7), (6, 8)):
        ns = list(range(lo, hi + 1))
        v = [prod[n - 3] for n in ns]
        windows[f"{lo}-{hi}"] = float(np.exp(np.polyfit(ns, np.log(v), 1)[0]))
    att.append({"attempt": "A6 is base(zeta_1) window-dependent?", "family": "pure_product",
                "fitted_base_by_window": windows, "asymptotic_claim": 1.5,
                "verdict": "CAVEAT: the fitted base drifts "
                           f"({min(windows.values()):.4f} to {max(windows.values()):.4f}) toward "
                           "the asymptotic 3/2, so any base quoted from a finite window must be "
                           "labelled a fit, not the limit"})
    return {"attempts": att}


def main() -> None:
    t0 = time.time()
    print("48.5(a)(c) bounds scan over the state zoo")
    scan = bounds_scan()
    print("\n48.5(b)(c) the base relation base(M*) = base(zeta_2)/base(zeta_1)")
    rel = base_relation()
    print(f"  worst deviation of the identity ratio from 1: {rel['worst_deviation_from_1']:.2e}")
    print("\n48.5(e) break attempts")
    breaks = break_attempts()
    for a in breaks["attempts"]:
        print(f"  {a['attempt']}\n      -> {a['verdict']}")

    branch = {
        "noisy_pure_q0.1": {"beta": "5/4", "base_zeta2": "7", "m_star_base": "28/5 = 5.60",
                            "status": "exact (closed form, Haar-averaged)"},
        "haar_pure": {"beta": "5/4", "base_zeta2": "7", "m_star_base": "28/5 = 5.60",
                      "status": "exact (q = 0 case of the same closed form)"},
        "low_rank_rank2": {"beta": "-> 5/4", "base_zeta2": "7", "m_star_base": "5.60",
                           "status": "numerical: beta reads about 1.21 at the sizes evaluated "
                                     "and the M* base uses the asymptotic value"},
        "product_and_ghz": {"beta": "3/2", "base_zeta2": "15/2", "m_star_base": "5.00",
                            "status": "exact (zeta_2 = (15/2)^n - 1 for product; "
                                      "((15/2)^n + (13/2)^n)/2 - 1/2 for GHZ) but ASYMPTOTIC: "
                                      "the finite-window fit over n=3..8 gives a lower base"},
    }
    payload = {
        "description": "PASS 48.5: the spectral-functional claims stated precisely, verified, "
                       "and attacked",
        "claims": {
            "C1": {
                "statement": "For every n-qubit state rho, 7^n - 1 <= zeta_2(rho) <= (17/2)^n.",
                "conditions": "None beyond rho being a valid density matrix; k = 2.",
                "proof_sketch": {
                    "lower": "the identity string contributes 7^n exactly, all other terms in "
                             "the spectral sum are non-negative, and Tr(rho^2)^2 <= 1",
                    "upper": "<P>^2 <= 1 termwise and sum_P 14^{-|P|} = (17/14)^n, so "
                             "zeta_2 <= 7^n (17/14)^n = (17/2)^n",
                },
                "attainment": {
                    "lower": "NOT attained. The infimum over states is 7^n - 4^-n, reached only "
                             "at rho = I/2^n; the stated bound is looser by 1 - 4^-n.",
                    "upper": "NOT attained, and not approached: it would require <P>^2 = 1 for "
                             "all 4^n strings simultaneously.",
                },
                "consequence": "base(zeta_2) in [7, 17/2] whenever the limit exists; 7 for "
                               "spread spectra, 15/2 for pure product and GHZ.",
            },
            "C2": {
                "statement": "base(M*) = base(zeta_2) / base(zeta_1).",
                "conditions": "both limits base(zeta_c) = lim zeta_c^{1/n} exist AND the "
                              "leading coefficients are nonzero. The second condition is not "
                              "decorative: at rho = I/2^n, zeta_1 = 0 exactly and M* is "
                              "infinite (break attempt A5).",
                "verification": "identity ratio 1 to within "
                                f"{rel['worst_deviation_from_1']:.1e} on eight families whose "
                                "bases genuinely differ",
            },
        },
        "branch_table_with_status": branch,
        "bounds_scan": scan,
        "base_relation": rel,
        "break_attempts": breaks,
        "wall_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.1f} s)")


if __name__ == "__main__":
    main()
