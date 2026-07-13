"""Analyze joint (multi-qubit) readout and test the cross-copy crosstalk hypothesis.

For the n=3 (and n=4) SWAP registers: quantify non-factorizability of the joint readout,
measure the cross-copy parity-pair (i, i+n) correlations, rebuild the readout confusion
WITH the measured correlation, re-predict the collective SWAP purity, and compare to the
measured 0.378 (n=3) / 0.420 (n=4).  Reports whether crosstalk explains the failure and
the non-monotonicity.  No credits.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from anrl.hardware import avg_gate_error_to_depol_param, swap_sign
from anrl.hardware.grid_predict import swap_gate_noisy_probs
from anrl.hardware.state_prep import ghz_state

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
PHYS = {3: [0, 1, 2, 9, 10, 11], 4: [0, 1, 2, 3, 9, 10, 11, 12]}
MEASURED_SWAP = {3: 0.3784, 4: 0.4204}
P1 = 0.001
CZ_MID = avg_gate_error_to_depol_param(0.009, 2)


def load_dist(state: str, n: int) -> np.ndarray:
    """Outcome distribution over 2^(2n) (index bit c = clbit c, LSB=clbit0)."""
    counts = {k.replace(" ", ""): int(v) for k, v in json.loads((HW / f"jr_{state}_counts.json").read_text()).items()}
    m = 2 * n
    v = np.zeros(2 ** m)
    for s, cnt in counts.items():
        idx = sum(int(s[m - 1 - c]) << c for c in range(m))
        v[idx] += cnt
    return v / v.sum(), sum(counts.values())


def marginal(dist: np.ndarray, c: int, val: int) -> float:
    """P(clbit c reads val)."""
    idxs = [i for i in range(len(dist)) if ((i >> c) & 1) == val]
    return float(dist[idxs].sum())


def joint2(dist: np.ndarray, ci: int, cj: int, vi: int, vj: int) -> float:
    idxs = [i for i in range(len(dist)) if ((i >> ci) & 1) == vi and ((i >> cj) & 1) == vj]
    return float(dist[idxs].sum())


def pair_confusion(states: dict, ci: int, cj: int) -> np.ndarray:
    """4x4 R[read_idx, prep_idx], prep 00/01/10/11 from s0/copyB/copyA/all; read_idx=2a+b.

    If a prep state is unavailable (n=4 has no copyB), that column is filled with the
    INDEPENDENT product of per-qubit marginals — so the missing prep is approximated as
    uncorrelated (flagged in the report).
    """
    prep_state = {0: "s0", 1: "copyB", 2: "copyA", 3: "all"}  # (a,b): a=clbit ci (copyA), b=clbit cj (copyB)
    R = np.zeros((4, 4))
    for prep_idx, st in prep_state.items():
        a_prep, b_prep = prep_idx >> 1, prep_idx & 1
        if st in states:
            d = states[st]
            for a in (0, 1):
                for b in (0, 1):
                    R[2 * a + b, prep_idx] = joint2(d, ci, cj, a, b)
        else:  # approximate missing prep column as independent (per-qubit marginals)
            pi1 = marginal(states["s0"], ci, 1) if a_prep == 0 else marginal(states["all"], ci, 1)
            pj1 = marginal(states["s0"], cj, 1) if b_prep == 0 else marginal(states["all"], cj, 1)
            for a in (0, 1):
                for b in (0, 1):
                    R[2 * a + b, prep_idx] = (pi1 if a else 1 - pi1) * (pj1 if b else 1 - pj1)
    return R


def build_R_joint(n: int, states: dict) -> np.ndarray:
    """Joint confusion = product over parity pairs of the measured 4x4 pair confusion."""
    m = 2 * n
    pairs = [(i, i + n) for i in range(n)]
    Rp = {p: pair_confusion(states, *p) for p in pairs}
    dim = 2 ** m
    R = np.ones((dim, dim))
    for read in range(dim):
        for prep in range(dim):
            v = 1.0
            for (ci, cj) in pairs:
                ra = (read >> ci) & 1; rb = (read >> cj) & 1
                pa = (prep >> ci) & 1; pb = (prep >> cj) & 1
                v *= Rp[(ci, cj)][2 * ra + rb, 2 * pa + pb]
            R[read, prep] = v
    return R, {f"{ci}-{cj}": Rp[(ci, cj)].tolist() for (ci, cj) in pairs}


def build_R_indep(n: int, states: dict) -> np.ndarray:
    """Independent confusion = product of per-qubit marginals (prep 0 from s0, prep 1 from all)."""
    m = 2 * n
    Mq = []
    for c in range(m):
        p10 = marginal(states["s0"], c, 1)          # prep 0 -> read 1
        p01 = 1 - marginal(states["all"], c, 1)     # prep 1 -> read 0
        Mq.append(np.array([[1 - p10, p01], [p10, 1 - p01]]))  # [read, prep]
    dim = 2 ** m
    R = np.ones((dim, dim))
    for read in range(dim):
        for prep in range(dim):
            v = 1.0
            for c in range(m):
                v *= Mq[c][(read >> c) & 1, (prep >> c) & 1]
            R[read, prep] = v
    return R


def swap_pred(n: int, R: np.ndarray) -> float:
    q, phys = swap_gate_noisy_probs(ghz_state(n), CZ_MID, P1)
    signs = np.array([swap_sign(format(b, f"0{2 * n}b"), n) for b in range(2 ** (2 * n))], dtype=float)
    return float(signs @ (R @ q))


def _frechet_column(col: np.ndarray, mode: str) -> np.ndarray:
    """Same 2-bit marginals as `col` (read_idx=2a+b), correlation pushed to a physical
    extreme: mode='max' Frechet upper bound, 'min' lower, 'indep' product."""
    pa1 = col[2] + col[3]
    pb1 = col[1] + col[3]
    both = {"max": min(pa1, pb1), "min": max(0.0, pa1 + pb1 - 1.0)}.get(mode, pa1 * pb1)
    return np.array([1 - pa1 - pb1 + both, pb1 - both, pa1 - both, both])


def physical_sensitivity(n: int, states: dict) -> dict:
    """Bracket the SWAP prediction over ALL physically-valid cross-copy correlations that
    PRESERVE the measured single-qubit flip rates (only correlation varies, to its Frechet
    extremes). This is the honest power test: max both-flip prob per pair <= min(marginals),
    so tiny measured flip rates cap how much any readout correlation can bend the parity."""
    m = 2 * n
    pairs = [(i, i + n) for i in range(n)]

    def pred(mode: str) -> float:
        Rp = {}
        for (ci, cj) in pairs:
            R = pair_confusion(states, ci, cj).copy()
            for prep in range(4):
                R[:, prep] = _frechet_column(R[:, prep], mode)
            Rp[(ci, cj)] = R
        dim = 2 ** m
        R = np.ones((dim, dim))
        for read in range(dim):
            for prep in range(dim):
                v = 1.0
                for (ci, cj) in pairs:
                    ra = (read >> ci) & 1; rb = (read >> cj) & 1
                    pa = (prep >> ci) & 1; pb = (prep >> cj) & 1
                    v *= Rp[(ci, cj)][2 * ra + rb, 2 * pa + pb]
                R[read, prep] = v
        return swap_pred(n, R)

    p_max, p_min = pred("max"), pred("min")
    max_both = [round(float(min(marginal(states["s0"], i, 1), marginal(states["s0"], i + n, 1))), 4)
                for i in range(n)]
    return {"max_corr_pred": round(p_max, 4), "min_corr_pred": round(p_min, 4),
            "bracket_lo": round(min(p_max, p_min), 4), "bracket_hi": round(max(p_max, p_min), 4),
            "bracket_width": round(abs(p_max - p_min), 4),
            "max_both_flip_prob_per_pair": max_both,
            "residual_at_max_correlation": round(MEASURED_SWAP[n] - min(p_max, p_min), 4)}


def cross_copy_corr(states: dict, n: int) -> dict:
    """Pearson correlation of readout flips for each qubit pair, from the all-idle state (both flips)."""
    d = states["s0"]; m = 2 * n
    pairs = [(i, i + n) for i in range(n)]
    def corr(ci, cj):
        pi = marginal(d, ci, 1); pj = marginal(d, cj, 1)
        both = joint2(d, ci, cj, 1, 1)
        cov = both - pi * pj
        den = np.sqrt(max(pi * (1 - pi), 1e-12) * max(pj * (1 - pj), 1e-12))
        cond = both / pi if pi > 0 else float("nan")
        return {"p_i": round(pi, 4), "p_j": round(pj, 4), "p_both": round(both, 4),
                "indep_both": round(pi * pj, 5), "cov": round(cov, 5),
                "pearson": round(cov / den, 4), "P(j|i)": round(cond, 4)}
    parity = {f"{PHYS[n][ci]}-{PHYS[n][cj]}": corr(ci, cj) for (ci, cj) in pairs}
    nonpair = [corr(ci, cj)["pearson"] for ci, cj in combinations(range(m), 2) if (ci, cj) not in pairs]
    return {"parity_pairs": parity, "mean_parity_pearson": round(float(np.mean([v["pearson"] for v in parity.values()])), 4),
            "mean_nonpair_pearson": round(float(np.mean(nonpair)), 4)}


def analyze_n(n: int) -> dict:
    names = {3: ["s0", "h0", "h1", "h2", "copyA", "copyB", "all"], 4: ["s0", "copyA", "all"]}[n]
    states = {}
    shots = {}
    for nm in names:
        d, s = load_dist(f"n{n}_{nm}", n)
        states[nm] = d; shots[nm] = s
    # non-factorizability: TVD between measured s0 and independent product of its marginals
    m = 2 * n
    p_marg = [marginal(states["s0"], c, 1) for c in range(m)]
    indep_s0 = np.ones(2 ** m)
    for i in range(2 ** m):
        for c in range(m):
            indep_s0[i] *= (p_marg[c] if (i >> c) & 1 else (1 - p_marg[c]))
    tvd_s0 = 0.5 * float(np.abs(states["s0"] - indep_s0).sum())

    corr = cross_copy_corr(states, n)
    R_joint, pair_confs = build_R_joint(n, states)
    R_indep = build_R_indep(n, states)
    pred_joint = swap_pred(n, R_joint)
    pred_indep = swap_pred(n, R_indep)
    phys_sens = physical_sensitivity(n, states)
    measured = MEASURED_SWAP[n]
    return {"n": n, "shots": shots, "tvd_s0_vs_independent": round(tvd_s0, 4),
            "cross_copy_correlation": corr,
            "swap_pred_independent": round(pred_indep, 4), "swap_pred_joint": round(pred_joint, 4),
            "physical_correlation_sensitivity": phys_sens,
            "measured": measured,
            "residual_independent": round(measured - pred_indep, 4),
            "residual_joint": round(measured - pred_joint, 4),
            "gap_closed_fraction": round((pred_indep - pred_joint) / (pred_indep - measured), 3)
            if abs(pred_indep - measured) > 1e-6 else None,
            "pair_confusions": pair_confs}


def main():
    result = {n: analyze_n(n) for n in (3, 4) if (HW / f"jr_n{n}_all_counts.json").exists()}
    (HW / "joint_readout_analysis.json").write_text(json.dumps(result, indent=2, default=float))
    for n, r in result.items():
        print(f"\n=== n={n} ===")
        print(f"  non-factorizability (TVD s0 vs independent): {r['tvd_s0_vs_independent']}")
        c = r["cross_copy_correlation"]
        print(f"  cross-copy parity-pair Pearson (mean): {c['mean_parity_pearson']}  vs non-pair: {c['mean_nonpair_pearson']}")
        for pk, pv in c["parity_pairs"].items():
            print(f"    pair {pk}: P(j|i)={pv['P(j|i)']} vs marginal P(j)={pv['p_j']}, pearson={pv['pearson']}")
        print(f"  SWAP pred independent={r['swap_pred_independent']}, joint={r['swap_pred_joint']}, measured={r['measured']}")
        print(f"  residual: indep {r['residual_independent']:+.3f} -> joint {r['residual_joint']:+.3f}; "
              f"gap closed {r['gap_closed_fraction']}")
        ps = r["physical_correlation_sensitivity"]
        print(f"  physical corr bracket (marginals fixed): [{ps['bracket_lo']}, {ps['bracket_hi']}] "
              f"width {ps['bracket_width']}; residual even at MAX corr = {ps['residual_at_max_correlation']:+.3f}")


if __name__ == "__main__":
    main()
