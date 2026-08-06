"""State-agnostic Hoeffding components zeta_1..zeta_k for the moment U-statistic (k=2,3,4).

The single-copy estimator of Tr(rho^k) is a k-th order U-statistic with symmetric kernel
h = Re Tr(G_1 ... G_k).  Its EXACT variance is the Lee / Hoeffding formula
(:func:`anrl.theory.variance.exact_ustatistic_variance`)

    Var(U_M) = C(M,k)^{-1} sum_{c=1..k} C(k,c) C(M-k, k-c) zeta_c ,

with the c-th Hoeffding component zeta_c = Var[h_c(G_1..G_c)] and h_c the c-th order
projection; the kernel's conditional expectation given c fixed arguments, averaged over
the remaining k-c.  Since E[G]=rho and the trace is multilinear, the inner expectation is
EXACT (insert rho at each free slot):

    h_c(g_1..g_c) = mean over the k! orderings of  Re Tr( ordered product of the c fixed
                    dense G's and (k-c) copies of rho )

so only the outer c-tuple average is Monte Carlo.  IMPORTANT (the k>=3 trap): zeta_2 is the
TWO-argument projection, NOT the kernel variance; the kernel variance is zeta_k.  This
routine computes each projection correctly by the ordering-average above, for ANY dense rho
(including low-rank, where the noisy-pure closed forms in :mod:`anrl.theory.variance` do not
apply).  Verified at k=3 and k=4 against the brute-force variance of the exact estimator
(:func:`anrl.benchmark.moment_ustats.exact_moment_ustatistic`).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from anrl.benchmark.ensembles import NoisyState
from anrl.theory.general import sample_batched_general


def dense_snapshots(snaps: np.ndarray) -> np.ndarray:
    """``(M, n, 2, 2)`` per-qubit local shadows -> dense ``(M, 2^n, 2^n)`` (kron accumulation)."""
    m, n = snaps.shape[0], snaps.shape[1]
    acc = snaps[:, 0]
    for q in range(1, n):
        r = acc.shape[1]
        acc = (acc[:, :, None, :, None] * snaps[:, q][:, None, :, None, :]).reshape(m, 2 * r, 2 * r)
    return acc


def _arrangements(k: int, c: int):
    """Distinct orderings of the multiset {G_0..G_{c-1}, rho*(k-c)} with multiplicities.

    -1 denotes a rho slot.  Averaging Re Tr over the k! orderings equals the multiplicity-
    weighted average over these distinct arrangements (the (k-c) rho's are identical)."""
    seen = {}
    for perm in itertools.permutations(list(range(c)) + [-1] * (k - c)):
        seen[perm] = seen.get(perm, 0) + 1
    return list(seen.items())


def _projection(gd_list: list[np.ndarray], rho: np.ndarray, arrangements, total: int) -> np.ndarray:
    """h_c per outer sample: multiplicity-weighted mean over arrangements of Re Tr(chain)."""
    m, d = gd_list[0].shape[0], rho.shape[0]
    rho_b = np.broadcast_to(rho, (m, d, d))
    acc = np.zeros(m)
    for perm, mult in arrangements:
        chain = None
        for s in perm:
            mat = gd_list[s] if s >= 0 else rho_b
            chain = mat.copy() if chain is None else chain @ mat
        acc += mult * np.einsum("mii->m", chain).real
    return acc / total


def hoeffding_component_mc(state: NoisyState, k: int, c: int, n_outer: int,
                          rng: np.random.Generator, chunk: int = 20000) -> float:
    """zeta_c = Var[h_c] via exact-inner / MC-outer, for any state and k in {2,3,4}."""
    if k not in (2, 3, 4):
        raise ValueError(f"supported k in {{2,3,4}}, got {k}")
    if not 1 <= c <= k:
        raise ValueError(f"c must satisfy 1<=c<=k, got c={c}, k={k}")
    rho = state.density_matrix()
    arrangements = _arrangements(k, c)
    total = math.factorial(k)
    vals = np.empty(n_outer)
    done = 0
    while done < n_outer:
        b = min(chunk, n_outer - done)
        gd = [dense_snapshots(sample_batched_general(state, b, rng)) for _ in range(c)]
        vals[done:done + b] = _projection(gd, rho, arrangements, total)
        done += b
    return float(np.var(vals, ddof=1))


def hoeffding_components_mc(state: NoisyState, k: int, n_outer: int,
                           rng: np.random.Generator, chunk: int = 20000) -> list[float]:
    """State-agnostic Hoeffding components ``[zeta_1, ..., zeta_k]`` (k in {2,3,4}).

    Feed into :func:`anrl.theory.variance.exact_ustatistic_variance` for the exact
    single-copy U-statistic variance at any budget M.
    """
    return [hoeffding_component_mc(state, k, c, n_outer,
                                  np.random.default_rng(rng.integers(1 << 30)), chunk)
            for c in range(1, k + 1)]
