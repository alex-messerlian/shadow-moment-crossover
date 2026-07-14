"""Verify the single-copy variance law against data already in results/ (no tuning).

Claim 1  brute-force MC of the full purity U-statistic vs [4(M-2)z1+2 z2]/[M(M-1)].
Claim 3  predicted alpha (exact formula, recomputed zetas) vs measured alpha (budget_scaling.json).
Claim 4  M* = z2/(2 z1) base vs the earlier-phase empirical M* base.
Part E   single-qubit identity E[Tr(Gr)^2]=1/4+5/4 t^2, and the weight-only zeta1 ansatz.

Writes results/theory_derivation.json.  Uses the converged zetas in
results/theory_zetas_recomputed.json (produce with theory_single_copy_scaling.py).
Run:  PYTHONPATH=. python -m experiments.theory_single_copy_verify
"""

from __future__ import annotations

import json
from itertools import product as iproduct
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import NoisyState, ghz_noisy, low_rank, noisy_pure
from anrl.benchmark.shadows import _snapshots, full_purity_ustatistic
from anrl.physics import kron_all
from anrl.theory.general import sample_batched_general
from anrl.theory.single_copy_law import (
    crossover_budget, hoeffding_variance, predicted_alpha, single_qubit_second_moment,
)

R = Path(__file__).resolve().parent.parent / "results"
_P = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], complex),
      "Y": np.array([[0, -1j], [1j, 0]], complex), "Z": np.array([[1, 0], [0, -1]], complex)}


# ---------- Claim 1: brute force vs formula ----------
def _mc_zetas(rho, n, N, rng):
    snaps = _snapshots(rho, n, 2 * N, rng)
    tr = np.array([np.einsum("ij,ji->", kron_all([snaps[i, q] for q in range(n)]), rho).real for i in range(N)])
    kern = np.ones(N)
    for q in range(n):
        kern *= np.einsum("kij,kji->k", snaps[:N, q], snaps[N:2 * N, q]).real
    return tr.var(ddof=1), kern.var(ddof=1)


def claim1(reps=8000, N=200_000):
    rng = np.random.default_rng(0)
    rows = []
    cases = [("noisy_pure n=1 q=0.1", noisy_pure(1, 0.1, np.random.default_rng(1)).density_matrix(), 1),
             ("noisy_pure n=2 q=0.1", noisy_pure(2, 0.1, np.random.default_rng(2)).density_matrix(), 2),
             ("ghz n=2 q=0.15", ghz_noisy(2, 0.15).density_matrix(), 2)]
    for name, rho, n in cases:
        z1, z2 = _mc_zetas(rho, n, N, np.random.default_rng(hash(name) % 2 ** 31))
        for M in (4, 8):
            ests = np.array([full_purity_ustatistic(_snapshots(rho, n, M, rng)) for _ in range(reps)])
            brute = ests.var(ddof=1)
            ex = hoeffding_variance(M, z1, z2)
            rows.append({"case": name, "M": M, "brute_var": brute, "exact_formula": ex,
                         "ratio": round(brute / ex, 3)})
    ok = all(0.9 <= r["ratio"] <= 1.1 for r in rows)
    return {"verdict": "VERIFIED" if ok else "MISMATCH", "rows": rows}


# ---------- Claim 3: alpha ----------
def claim3(per_n):
    meas = {f["n"]: f for f in json.loads((R / "budget_scaling.json").read_text())["alpha_fits"] if f["k"] == 2}
    rows = []
    for n in sorted(meas):
        z1, z2 = per_n[str(n)]["zeta1"], per_n[str(n)]["zeta2"]
        ap = predicted_alpha(meas[n]["budgets"], z1, z2)
        am, ase = meas[n]["alpha"], meas[n]["alpha_se"]
        rows.append({"n": n, "budgets": meas[n]["budgets"], "alpha_pred": round(ap, 4),
                     "alpha_meas": round(am, 4), "alpha_meas_se": round(ase, 4),
                     "resid": round(ap - am, 4), "within_2se": bool(abs(ap - am) <= 2 * ase)})
    nok = sum(r["within_2se"] for r in rows)
    return {"rows": rows, "n_within_2se": nok, "n_total": len(rows),
            "verdict": f"exact-formula alpha within 2*SE of measured at {nok}/{len(rows)} n"}


# ---------- Claim 4: M* base ----------
def claim4(fits27, fits29):
    e = [z for z in json.loads((R / "theory_zetas.json").read_text())["zetas"] if z["k"] == 2]
    ns = np.array([z["n"] for z in e]); ms = np.array([z["M_star"] for z in e])
    emp = float(np.exp(np.polyfit(ns, np.log(ms), 1)[0]))
    d27, d29 = fits27["M_star_2z1"]["base"], fits29["M_star_2z1"]["base"]
    agree = abs(d29 - emp) / emp < 0.05
    return {"derived_base_n2_7": round(d27, 3), "derived_base_n2_9": round(d29, 3),
            "empirical_base_earlier_phase": round(emp, 3),
            "crossover": "M*=z2/(2 z1) (exact-formula crossover; corrects two-term z2/(4 z1))",
            "verdict": f"derived {d29:.3f} {'agrees with' if agree else 'DIFFERS from'} empirical {emp:.3f}"}


# ---------- Part E: single-qubit identity + weight-only ansatz ----------
def _zeta1_general(state, N, seed):
    rho, n = state.density_matrix(), state.n
    rng = np.random.default_rng(seed); s = s2 = 0.0; cnt = 0
    for c0 in range(0, N, 250_000):
        c = min(250_000, N - c0)
        sn = sample_batched_general(state, c, rng)
        A, B = sn[:, 0], sn[:, 1]
        G = (A[:, :, None, :, None] * B[:, None, :, None, :]).reshape(c, 4, 4)
        tr = np.einsum("mij,ji->m", G, rho).real
        s += tr.sum(); s2 += (tr ** 2).sum(); cnt += c
    return s2 / cnt - (s / cnt) ** 2


def _weight_sums(rho, n):
    out = {}
    for ls in iproduct("IXYZ", repeat=n):
        w = sum(1 for L in ls if L != "I")
        if w == 0:
            continue
        M = _P[ls[0]]
        for L in ls[1:]:
            M = np.kron(M, _P[L])
        out[w] = out.get(w, 0.0) + float(np.trace(rho @ M).real) ** 2
    return [out.get(w, 0.0) for w in range(1, n + 1)]


def partE():
    rng = np.random.default_rng(0)
    # single-qubit identity
    ident = []
    for t in (0.0, 0.5, 1.0):
        st = NoisyState(np.array([[1.0], [0.0]], complex), 1.0 - t, 1)
        r = st.density_matrix()
        sn = sample_batched_general(st, 2_000_000, rng)
        mc = float((np.einsum("mij,ji->m", sn[:, 0], r).real ** 2).mean())
        ident.append({"t": t, "mc": round(mc, 4), "pred": round(single_qubit_second_moment(t), 4)})
    id_ok = all(abs(d["mc"] - d["pred"]) < 3e-3 for d in ident)
    # weight-only ansatz at n=2 across DIVERSE states
    fam = ([(f"np q={q} s{s}", noisy_pure(2, q, np.random.default_rng([1, int(q * 100), s])))
            for q in (0.0, 0.1, 0.2, 0.3) for s in range(3)]
           + [(f"low_rank r={r} s{s}", low_rank(2, r, np.random.default_rng([2, r, s]))) for r in (2, 3) for s in range(2)]
           + [(f"ghz q={q}", ghz_noisy(2, q)) for q in (0.0, 0.15, 0.3)])
    A, y = [], []
    for i, (_, st) in enumerate(fam):
        y.append(_zeta1_general(st, 1_000_000, 500 + i)); A.append(_weight_sums(st.density_matrix(), 2))
    A, y = np.array(A), np.array(y)
    pred = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.abs(pred - y) / y
    return {"single_qubit_identity": {"verdict": "VERIFIED" if id_ok else "FAILS", "rows": ident},
            "single_qubit_zeta1_closed": "(3/4)t^2-(1/4)t^4",
            "weight_only_n2": {"max_resid": round(float(resid.max()), 3), "rms_resid": round(float(np.sqrt((resid ** 2).mean())), 3),
                               "verdict": "FAILS across diverse states" if resid.max() > 0.02 else "holds"},
            "closed_form_verdict": "no simple weight-only closed form for n>=2 (zeta1 numerical); single-qubit closed form only"}


def main():
    Z = json.loads((R / "theory_zetas_recomputed.json").read_text())
    out = {"claim1": claim1(), "claim2_scalings": {"fits_n2_7": Z["fits_n2_7"], "fits_n2_9": Z["fits_n2_9"], "per_n": Z["per_n"]},
           "claim3_alpha": claim3(Z["per_n"]), "claim4_mstar": claim4(Z["fits_n2_7"], Z["fits_n2_9"]),
           "partE": partE()}
    (R / "theory_derivation.json").write_text(json.dumps(out, indent=2))
    print("Claim 1:", out["claim1"]["verdict"], "| ratios", [r["ratio"] for r in out["claim1"]["rows"]])
    print("Claim 3:", out["claim3_alpha"]["verdict"])
    print("Claim 4:", out["claim4_mstar"]["verdict"])
    print("Part E identity:", out["partE"]["single_qubit_identity"]["verdict"],
          "| ansatz n=2:", out["partE"]["weight_only_n2"]["verdict"],
          f"(max resid {out['partE']['weight_only_n2']['max_resid']:.1%})")
    print("wrote results/theory_derivation.json")


if __name__ == "__main__":
    main()
