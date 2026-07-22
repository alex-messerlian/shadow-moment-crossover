"""Regenerate results/beta_law_test.json from the committed ensemble code.

``results/beta_law_test.json`` fixes the growth bases in the Section 3.5
four-family table -- in particular the low-rank ``M*`` base that PASS 28
corrected from 5.8 to 5.60.  The file had no generating script, so the number
deciding that correction could not be reproduced by a committed command.  This
module supplies one.

It recomputes, for every family, the exact k=2 projection variances

    V   = E[Tr(G rho)^2] = 4^-n sum_{compatible P,Q} 3^{|supp P cap supp Q|}
                            <P> <Q> <P (-) Q>            (HKP Lemma 4, trilinear)
    z1  = V - p2^2,   p2 = Tr(rho^2)
    z2  = 7^n sum_R <R>^2 14^{-|R|} - p2^2

and reports, per family, the local growth bases at the largest available size
(``beta_local`` = base of V, ``base_zeta2_local``, ``Mstar_base_local`` = base of
``M* = z2/(2 z1)``), matching the fields of the committed file.

The compatible-pair sum is enumerated once per size as flat index arrays
(exactly ``10^n`` terms) and reused across states, so each state costs one
vectorized pass.  Ensembles, seeds and state counts are taken from the committed
file's own metadata: Haar-pure ``[21,n,s]``, noisy-pure(q=0.1) ``[22,n,s]``,
low-rank rank-2 ``[23,n,s]``; product, GHZ and GHZ-noisy(q=0.15) are
deterministic.

Writes ``results/beta_law_test_regenerated.json``.  It does NOT overwrite the
original: the point is to diff against it.

Run:  PYTHONPATH=. python -m experiments.beta_law_regenerate [MAX_N]
      (default MAX_N = 7, the largest size the low-rank row reaches)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import NoisyState, ghz_noisy, haar_pure, low_rank, noisy_pure

R = Path(__file__).resolve().parent.parent / "results"

# single-qubit Paulis, indexed 0=I, 1=X, 2=Y, 3=Z
_P = [np.eye(2, dtype=complex),
      np.array([[0, 1], [1, 0]], dtype=complex),
      np.array([[0, -1j], [1j, 0]], dtype=complex),
      np.array([[1, 0], [0, -1]], dtype=complex)]
# a (-) b : the single-qubit product index (phases cancel in the trilinear form)
_XOR = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]


def pauli_expectations(rho: np.ndarray, n: int) -> np.ndarray:
    """``<P> = Tr(rho P)`` for all ``4^n`` strings, qubit 0 most significant.

    Fast basis change: contract each qubit's index pair against the 4x2x2 Pauli
    tensor, O(4^n n) rather than building 4^n dense matrices.
    """
    M = np.stack([p.T for p in _P])                      # M[a, i, j] = P_a[j, i]
    A = rho.reshape([2] * (2 * n))
    # After k contractions the layout is (i_{k+1}..i_n, j_{k+1}..j_n, a_1..a_k):
    # the next qubit's i-axis is 0 and its j-axis is n-k.  Each tensordot appends
    # the new a-axis at the end, so they accumulate in qubit order.
    for k in range(n):
        A = np.tensordot(A, M, axes=([0, n - k], [1, 2]))
    return np.real(A).reshape(-1)


def compatible_tables(n: int):
    """Flat ``(IP, IQ, IR, W)`` over the ``10^n`` compatible Pauli pairs."""
    a0, b0, c0, w0 = [], [], [], []
    for a in range(4):
        for b in range(4):
            if a and b and a != b:
                continue                                  # incompatible
            a0.append(a)
            b0.append(b)
            c0.append(_XOR[a][b])
            w0.append(3.0 if (a and b) else 1.0)
    ip = np.array(a0, dtype=np.int64)
    iq = np.array(b0, dtype=np.int64)
    ir = np.array(c0, dtype=np.int64)
    w = np.array(w0, dtype=np.float64)
    for _ in range(n - 1):
        ip = (ip[:, None] * 4 + np.array(a0)[None, :]).ravel()
        iq = (iq[:, None] * 4 + np.array(b0)[None, :]).ravel()
        ir = (ir[:, None] * 4 + np.array(c0)[None, :]).ravel()
        w = (w[:, None] * np.array(w0)[None, :]).ravel()
    return ip, iq, ir, w


def zetas_exact(rho: np.ndarray, n: int, tabs, weights) -> tuple[float, float, float]:
    """Exact ``(V, zeta1, zeta2)`` for one state."""
    e = pauli_expectations(rho, n)
    p2 = float(np.real(np.trace(rho @ rho)))
    ip, iq, ir, w = tabs
    V = float(4.0 ** (-n) * np.sum(w * e[ip] * e[iq] * e[ir]))
    z2 = float(7.0 ** n * np.sum(e ** 2 * 14.0 ** (-weights)) - p2 ** 2)
    return V, V - p2 ** 2, z2


def pauli_weights(n: int) -> np.ndarray:
    w = np.zeros(1, dtype=np.int64)
    for _ in range(n):
        w = (w[:, None] + np.array([0, 1, 1, 1])[None, :]).ravel()
    return w


def product_state(n: int) -> NoisyState:
    """Pure product state |+>^{otimes n} -- the stabilizer branch of the table."""
    v = np.ones((2 ** n, 1), dtype=complex) / np.sqrt(2 ** n)
    return NoisyState(v, 0.0, n)


def ghz_state(n: int) -> NoisyState:
    v = np.zeros((2 ** n, 1), dtype=complex)
    v[0, 0] = v[-1, 0] = 1 / np.sqrt(2)
    return NoisyState(v, 0.0, n)


FAMILIES = {
    "noisy-pure": lambda n, s: noisy_pure(n, 0.1, np.random.default_rng([22, n, s])),
    "Haar-pure": lambda n, s: haar_pure(n, np.random.default_rng([21, n, s])),
    "lowrank2": lambda n, s: low_rank(n, 2, np.random.default_rng([23, n, s])),
    "GHZ-noisy": lambda n, s: ghz_noisy(n, 0.15),
    "product": lambda n, s: product_state(n),
    "GHZ": lambda n, s: ghz_state(n),
}
DETERMINISTIC = {"GHZ-noisy", "product", "GHZ"}
#: state counts per size, matching the committed file so the averages compare
#: like for like.  n = 8 is out of reach for the exact sum (10^8 terms, and the
#: low-rank row -- the one the Section 3.5 table turns on -- stops at n = 7).
N_STATES = {2: 200, 3: 200, 4: 150, 5: 120, 6: 100, 7: 60}


def main(max_n: int = 7) -> None:
    sizes = [n for n in sorted(N_STATES) if n <= max_n]
    tabs = {n: compatible_tables(n) for n in sizes}
    wts = {n: pauli_weights(n) for n in sizes}
    out = {}
    for fam, make in FAMILIES.items():
        ns, Vs, z1s, z2s, Ns = [], [], [], [], []
        for n in sizes:
            reps = 1 if fam in DETERMINISTIC else N_STATES[n]
            acc = np.zeros(3)
            for s in range(reps):
                acc += np.array(zetas_exact(make(n, s).density_matrix(), n, tabs[n], wts[n]))
            acc /= reps
            ns.append(n)
            Vs.append(acc[0])
            z1s.append(acc[1])
            z2s.append(acc[2])
            Ns.append(reps)
        ms = [b / (2 * a) for a, b in zip(z1s, z2s)]
        out[fam] = {
            "n": ns, "N": Ns,
            "V": Vs, "zeta1": z1s, "zeta2": z2s,
            "zeta2_over_7n": [z / 7.0 ** n for z, n in zip(z2s, ns)],
            "beta_local": Vs[-1] / Vs[-2],
            "base_zeta2_local": z2s[-1] / z2s[-2],
            "Mstar_base_local": ms[-1] / ms[-2],
            "Mstar_successive": [ms[i + 1] / ms[i] for i in range(len(ms) - 1)],
        }
        print(f"{fam:>12}: beta_local {out[fam]['beta_local']:.5f}  "
              f"base(zeta2) {out[fam]['base_zeta2_local']:.5f}  "
              f"M* base {out[fam]['Mstar_base_local']:.5f}")

    payload = {
        "description": "regeneration of results/beta_law_test.json from the "
                       "committed ensemble code; does not overwrite the original",
        "max_n": max_n,
        "seeds": {"Haar-pure": "[21,n,s]", "noisy-pure q=0.1": "[22,n,s]",
                  "lowrank2 rank=2": "[23,n,s]"},
        "deterministic_families": sorted(DETERMINISTIC),
        "evaluator": "exact trilinear compatible-pair sum (HKP Lemma 4) for V; "
                     "spectral sum 7^n sum <P>^2 14^-|P| for zeta2",
        "families": out,
    }
    (R / "beta_law_test_regenerated.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {R / 'beta_law_test_regenerated.json'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
