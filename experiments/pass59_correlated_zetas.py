r"""PASS 59.2: what a correlated readout channel does to the projection variances.

    PYTHONPATH=. .venv/bin/python experiments/pass59_correlated_zetas.py

The paper's identities for zeta_1 and zeta_2 are exact for any state, but they are derived by
taking a PRODUCT OVER QUBITS: the single-qubit second moment is evaluated once and raised to the
n-th power, because each qubit's basis choice and reported outcome are independent.

Readout error composes into that derivation as an extra per-qubit map ONLY IF the readout channel
factorizes.  A correlated channel does not, so the product step is where the derivation fails.
This computes the size of that failure exactly, by brute force over all basis choices and all
reported bitstrings, for:

  ideal        no readout error
  independent  per-qubit flips at the measured marginals
  correlated   the measured conditional structure (a spectator's flip rate depends on how many
               of its partners are excited in the reported register)

The shadow snapshot is built from the REPORTED bits with no noise-aware inverse, which is what
uncorrected hardware does.  zeta_1 = Var[Tr(G rho)] and zeta_2 = Var[Tr(G_i G_j)] are then exact
expectations over the (basis, outcome) distribution rather than Monte Carlo.

Writes ``results/pass59_correlated_zetas.json``.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass59_correlated_zetas.json"

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
# the three measurement bases as unitaries mapping the eigenbasis to the computational one
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SdH = H @ np.array([[1, 0], [0, -1j]], dtype=complex)
BASES = [H, SdH, I2]                      # X, Y, Z


def kron(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def snapshot(bits, basis_idx):
    """3 U^dag |b><b| U - I per qubit, from the REPORTED bits."""
    fac = []
    for b, k in zip(bits, basis_idx):
        U = BASES[k]
        proj = np.zeros((2, 2), dtype=complex)
        proj[b, b] = 1.0
        fac.append(3 * (U.conj().T @ proj @ U) - I2)
    return kron(fac)


def ideal_probs(rho, basis_idx, n):
    """P(b | basis) for the true state."""
    U = kron([BASES[k] for k in basis_idx])
    d = np.real(np.diag(U @ rho @ U.conj().T))
    return np.clip(d, 0, None) / d.sum()


def conf_independent(rates, n):
    """Product confusion matrix Lambda[b_reported, b_true]."""
    M = np.array([[1.0]])
    for (p10, p01) in rates:
        M = np.kron(M, np.array([[1 - p10, p01], [p10, 1 - p01]]))
    return M


def conf_correlated(rates, n, boost, partners):
    """Spectator flip rate rises with the number of excited partners in the TRUE bitstring.

    boost is the extra P(1|0) at full partner excitation; the dependence is taken linear in the
    excited fraction, which is what the pooled data supports (the saturating form fits worse).
    """
    dim = 2 ** n
    M = np.zeros((dim, dim))
    for t in range(dim):
        tb = [(t >> (n - 1 - i)) & 1 for i in range(n)]
        # per-qubit conditional rates given the true bitstring
        pq = []
        for i, (p10, p01) in enumerate(rates):
            part = partners[i]
            frac = (sum(tb[j] for j in part) / len(part)) if part else 0.0
            pq.append((min(0.99, p10 + boost * frac), p01))
        for r in range(dim):
            rb = [(r >> (n - 1 - i)) & 1 for i in range(n)]
            p = 1.0
            for i in range(n):
                p10, p01 = pq[i]
                p *= (p10 if rb[i] else 1 - p10) if tb[i] == 0 else ((1 - p01) if rb[i] else p01)
            M[r, t] = p
    return M


def zetas(rho, n, conf=None):
    """Exact zeta_1 and zeta_2 over all 3^n bases and 2^n reported outcomes."""
    dim = 2 ** n
    w = 1.0 / 3 ** n
    e1 = e1sq = 0.0
    snaps, probs = [], []
    for basis_idx in product(range(3), repeat=n):
        pi = ideal_probs(rho, basis_idx, n)
        pr = pi if conf is None else conf @ pi
        for r in range(dim):
            if pr[r] <= 0:
                continue
            bits = [(r >> (n - 1 - i)) & 1 for i in range(n)]
            G = snapshot(bits, basis_idx)
            p = w * pr[r]
            t = np.real(np.trace(G @ rho))
            e1 += p * t
            e1sq += p * t * t
            snaps.append((p, G))
    zeta1 = e1sq - e1 * e1
    # zeta_2 = Var[Tr(G_i G_j)] over two independent snapshots
    m1 = m2 = 0.0
    for pa, Ga in snaps:
        GaT = Ga
        for pb, Gb in snaps:
            v = np.real(np.trace(GaT @ Gb))
            m1 += pa * pb * v
            m2 += pa * pb * v * v
    zeta2 = m2 - m1 * m1
    return float(zeta1), float(zeta2), float(e1)


def main() -> None:
    rng = np.random.default_rng(59)
    # measured structure from results/pass59_saturation.json: idle ~2-6%, boosted to ~24%
    P10_IDLE, P01_IDLE, BOOST = 0.033, 0.065, 0.20
    res = []
    for n in (2, 3):
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
        rho = 0.9 * np.outer(psi, psi.conj()) + 0.1 * np.eye(2 ** n) / 2 ** n
        rates = [(P10_IDLE, P01_IDLE)] * n
        partners = [[j for j in range(n) if j != i] for i in range(n)]

        z_id = zetas(rho, n, None)
        z_in = zetas(rho, n, conf_independent(rates, n))
        z_co = zetas(rho, n, conf_correlated(rates, n, BOOST, partners))
        row = {"n": n,
               "ideal": {"zeta1": z_id[0], "zeta2": z_id[1], "mean": z_id[2]},
               "independent": {"zeta1": z_in[0], "zeta2": z_in[1], "mean": z_in[2]},
               "correlated": {"zeta1": z_co[0], "zeta2": z_co[1], "mean": z_co[2]},
               "purity": float(np.real(np.trace(rho @ rho)))}
        row["z1_corr_vs_indep_pct"] = 100 * (z_co[0] - z_in[0]) / z_in[0]
        row["z2_corr_vs_indep_pct"] = 100 * (z_co[1] - z_in[1]) / z_in[1]
        row["mstar_indep"] = z_in[1] / (2 * z_in[0])
        row["mstar_corr"] = z_co[1] / (2 * z_co[0])
        row["mstar_shift_pct"] = 100 * (row["mstar_corr"] - row["mstar_indep"]) / row["mstar_indep"]
        res.append(row)
        print(f"\n  n = {n}   true purity {row['purity']:.4f}")
        for k in ("ideal", "independent", "correlated"):
            r = row[k]
            print(f"    {k:12s} zeta1 = {r['zeta1']:.5f}   zeta2 = {r['zeta2']:9.4f}   "
                  f"E[Tr(G rho)] = {r['mean']:.5f}")
        print(f"    correlated vs independent: zeta1 {row['z1_corr_vs_indep_pct']:+.2f}%, "
              f"zeta2 {row['z2_corr_vs_indep_pct']:+.2f}%")
        print(f"    M* = zeta2/(2 zeta1): independent {row['mstar_indep']:.3f} -> "
              f"correlated {row['mstar_corr']:.3f}  ({row['mstar_shift_pct']:+.2f}%)")

    OUT.write_text(json.dumps({"params": {"p10_idle": P10_IDLE, "p01_idle": P01_IDLE,
                                          "boost_at_full_excitation": BOOST}, "rows": res}, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
