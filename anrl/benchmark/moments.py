"""Moment family Tr(rho^k) and its single-copy / collective estimators.

Generalizes the purity (k=2) benchmark to the moments Tr(rho^k), k = 2, 3, 4.

* Exact ground truth: ``moment(rho, k) = Tr(rho^k) = sum_i lambda_i^k``.
* Collective route: the k-copy cyclic-permutation test.  The cyclic permutation
  operator ``C_k`` on k registers of dimension ``d = 2^n`` satisfies
  ``Tr(C_k @ rho^{(x)k}) = Tr(rho^k)`` (a single k-cycle), and ``Tr(C_k) = d``.
  For k = 2 the cyclic permutation is exactly the SWAP.  The measurement is a
  Hadamard test of ``Re<C_k>`` (which equals the real ``Tr(rho^k)``), giving a
  binary +/-1 outcome exactly as for the SWAP test.
* Under *global depolarizing* at rate ``p`` on the k-copy register the signal is
  ``(1 - p) * Tr(rho^k) + p * d^(1-k)``  (the maximally-mixed contribution is
  ``p * Tr(C_k) / d^k = p * d / d^k = p * d^(1-k)``).
* Single-copy route: the k-th order classical-shadow U-statistic; the mean of
  ``Tr(snap_{i1} @ ... @ snap_{ik})`` over distinct index k-tuples, unbiased for
  ``Tr(rho^k)``.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from .shadows import _snapshots, full_purity_ustatistic

# k>=5 has no implemented closed-form full U-statistic, so the copy-fair
# estimator there falls back to a large random subsample of tuples (still zero
# extra copy cost, far more than M//k).  k=2,3,4 all have EXACT full U-statistics.
_HIGH_K_FAIR_TUPLES = 200_000

# The exact k=4 U-statistic builds the dense 2^n x 2^n snapshot matrices and a
# 2^{2n} x 2^{2n} "X" tensor (opposite-slot coincidences), so it is O(16^n) =
# O(2^{4n}) = O(d^4) memory (the X tensor) and O(16^n) time for the two X
# contractions (plus O(M 8^n) for the batched g@g matmuls).  The X tensor alone
# is ~1 MB at n=4, ~17 MB at n=5, ~268 MB at n=6 (the cap); it grows 16x per
# qubit, so fail fast beyond the cap rather than silently thrashing.  (The moment
# sweep runs at n <= 4, where the X tensor is ~1 MB.)
_K4_EXACT_MAX_N = 6


def moment(rho: np.ndarray, k: int) -> float:
    """Exact k-th moment ``Tr(rho^k)`` (real), via the eigenvalues of ``rho``."""
    if k < 1:
        raise ValueError(f"moment order k must be >= 1, got {k}")
    eigvals = np.linalg.eigvalsh(np.asarray(rho, dtype=np.complex128))
    return float(np.sum(eigvals.real ** k))


def cyclic_permutation_operator(d: int, k: int) -> np.ndarray:
    """Cyclic permutation operator ``C_k`` on ``k`` registers of dimension ``d``.

    ``C_k`` right-shifts the registers: ``|i_1 i_2 ... i_k> -> |i_k i_1 ... i_{k-1}>``.
    It is a single k-cycle, so ``Tr(C_k @ rho^{(x)k}) = Tr(rho^k)`` and
    ``Tr(C_k) = d``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    dim = d ** k
    operator = np.zeros((dim, dim), dtype=np.complex128)
    for index in range(dim):
        digits = []
        rest = index
        for _ in range(k):
            digits.append(rest % d)
            rest //= d
        digits.reverse()  # [i_1, ..., i_k], i_1 most significant
        shifted = [digits[-1]] + digits[:-1]  # right cyclic shift
        out = 0
        for digit in shifted:
            out = out * d + digit
        operator[out, index] = 1.0
    return operator


def kron_power(rho: np.ndarray, k: int) -> np.ndarray:
    """The k-fold tensor power ``rho^{(x)k}``."""
    return reduce(np.kron, [rho] * k)


def depolarizing_moment_signal(moment_value: float, k: int, p: float, n_qubits: int) -> float:
    """Noisy signal under global depolarizing at rate ``p`` on the k-copy register.

    ``signal = (1 - p) * Tr(rho^k) + p * d^(1-k)`` with ``d = 2^n``.
    """
    d = 2 ** n_qubits
    return (1.0 - p) * moment_value + p * d ** (1 - k)


def collective_moment_estimate(
    k: int, n_measurements: int, signal: float, rng: np.random.Generator
) -> float:
    """Sample the k-copy cyclic test: binary shots, ``estimate = 2*frac - 1``.

    ``P(+1) = (1 + signal) / 2``; unbiased for ``signal`` (hence for ``Tr(rho^k)``
    at zero noise).  ``signal`` is supplied by the caller (closed-form for
    depolarizing, explicit channel evaluation otherwise).
    """
    if n_measurements < 1:
        raise ValueError(f"n_measurements must be >= 1, got {n_measurements}")
    p_plus = float(np.clip((1.0 + signal) / 2.0, 0.0, 1.0))
    n_plus = int(rng.binomial(n_measurements, p_plus))
    return 2.0 * (n_plus / n_measurements) - 1.0


def _distinct_tuples(m: int, k: int, n_tuples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n_tuples`` index k-tuples with all k indices distinct (ordered)."""
    if m < k:
        raise ValueError(f"cannot form {k}-tuples of distinct indices from m={m} snapshots")
    collected = np.empty((0, k), dtype=np.int64)
    while len(collected) < n_tuples:
        candidates = rng.integers(0, m, size=(2 * n_tuples + k, k))
        ordered = np.sort(candidates, axis=1)
        distinct = (np.diff(ordered, axis=1) != 0).all(axis=1)
        collected = np.vstack([collected, candidates[distinct]])
    return collected[:n_tuples]


def shadow_moment_estimate(
    rho: np.ndarray,
    k: int,
    n_snapshots: int,
    rng: np.random.Generator,
    n_tuples: int | None = None,
) -> float:
    """Single-copy classical-shadow estimate of ``Tr(rho^k)``.

    The k-th order U-statistic: the mean of ``Tr(snap_{i1} @ ... @ snap_{ik})``
    over distinct index k-tuples, unbiased for ``Tr(rho^k)`` (per-qubit
    factorization ``prod_q Tr(snap_{i1}^q @ ... @ snap_{ik}^q)``).

    Copy accounting: forming tuples is classical post-processing (zero copy
    cost), so the copy-fair estimator uses as many tuples as possible.

    * ``n_tuples is None`` (default): the copy-fair estimator; the EXACT full
      U-statistic for k=2, k=3 and k=4, and a large random subsample
      (``_HIGH_K_FAIR_TUPLES``) for k>=5 (no closed form implemented there).
    * ``n_tuples`` given: a random subsample of that many tuples, variance
      inflating (saves no copies), kept only for comparison / the old
      ``n_snapshots // k`` convention.
    """
    if k < 2:
        raise ValueError(f"moment order k must be >= 2, got {k}")
    if n_snapshots < k:
        raise ValueError(f"need >= k={k} snapshots, got {n_snapshots}")
    rho = np.asarray(rho, dtype=np.complex128)
    n = int(round(np.log2(rho.shape[0])))

    snaps = _snapshots(rho, n, n_snapshots, rng)  # (M, n, 2, 2)
    if n_tuples is None:
        return fair_moment_ustatistic(snaps, k, rng)
    return moment_ustatistic_from_snapshots(snaps, k, n_tuples, rng)


def full_moment_ustatistic_k3(snaps: np.ndarray) -> float:
    """Exact full 3rd-order U-statistic of ``Tr(rho^3)`` over all distinct triples.

    Uses the non-commutative power-sum identity (all two-index coincidence
    patterns are cyclically equal to ``Tr(A^2 B)``):
    ``sum_{distinct i,j,l} Tr(G_i G_j G_l) = Tr(S^3) - 3 Tr(P2 S) + 2 Tr(P3)``
    with ``S = sum G_i``, ``P2 = sum G_i^2``, ``P3 = sum G_i^3`` and ``G_i`` the
    full 2^n-dim snapshot.  Divide by ``M(M-1)(M-2)``.
    """
    m, n = snaps.shape[0], snaps.shape[1]
    g = np.array([reduce(np.kron, list(snaps[i])) for i in range(m)])  # (M, d, d)
    s = g.sum(axis=0)
    p2 = np.einsum("mab,mbc->ac", g, g)
    p3 = np.einsum("mab,mbc,mcd->ad", g, g, g)
    val = np.trace(s @ s @ s) - 3.0 * np.trace(p2 @ s) + 2.0 * np.trace(p3)
    return float(val.real / (m * (m - 1) * (m - 2)))


def full_moment_ustatistic_k4(snaps: np.ndarray) -> float:
    """Exact full 4th-order U-statistic of ``Tr(rho^4)`` over all distinct 4-tuples.

    The all-distinct ordered sum over the 4 cyclic slots is a Mobius inversion
    over the 15 set partitions ``P`` of ``{0,1,2,3}``,
    ``sum_distinct = sum_P mu(P) T(P)`` with ``mu(P) = prod_blocks (-1)^(|b|-1)
    (|b|-1)!`` and ``T(P)`` the sum of ``Tr(G_a0 G_a1 G_a2 G_a3)`` over index
    assignments that share an index within each block.  All but the two
    opposite-slot partitions collapse to matrix power sums; grouping the 15
    partitions by value gives

        sum_distinct = Tr(S^4) - 4 Tr(P2 S^2) - 2 TrX2 + 2 Tr(P2^2)
                       + TrXalt + 8 Tr(P3 S) - 6 Tr(P4)

    where ``S = sum G_i``, ``P2 = sum G_i^2``, ``P3 = sum G_i^3``,
    ``P4 = sum G_i^4`` (``G_i`` the full 2^n-dim snapshot), and the two
    opposite-slot ("alternating") terms use the coincidence tensor
    ``Xr[a,b,c,d] = sum_i (G_i)_ab (G_i)_cd``:

        TrXalt = sum_{i,j} Tr(G_i G_j G_i G_j) = einsum('abcd,bcda->', Xr, Xr)
        TrX2   = sum_i     Tr(G_i S  G_i S)    = einsum('abcd,bc,da->', Xr, S, S)

    Divide by ``M(M-1)(M-2)(M-3)``.  Verified against brute-force enumeration of
    all distinct 4-tuples.  ``O(16^n) = O(2^{4n})`` memory (the ``Xr`` tensor is
    a ``2^{2n} x 2^{2n}`` matrix) and ``O(16^n)`` time for the two X
    contractions, so it is capped at ``n <= _K4_EXACT_MAX_N``.
    """
    m, n = snaps.shape[0], snaps.shape[1]
    if m < 4:
        raise ValueError(f"4th-order U-statistic needs >= 4 snapshots, got {m}")
    if n > _K4_EXACT_MAX_N:
        raise ValueError(
            f"exact k=4 U-statistic is O(16^n)=O(2^{{4n}}) memory; n={n} exceeds "
            f"_K4_EXACT_MAX_N={_K4_EXACT_MAX_N} (pass n_tuples for a subsample instead)"
        )
    d = 2 ** n
    g = np.array([reduce(np.kron, list(snaps[i])) for i in range(m)])  # (M, d, d)
    s = g.sum(axis=0)
    g2 = g @ g
    p2 = g2.sum(axis=0)
    p3 = (g2 @ g).sum(axis=0)
    p4 = (g2 @ g2).sum(axis=0)
    s2 = s @ s
    tr_s4 = np.trace(s2 @ s2)
    tr_p2_s2 = np.trace(p2 @ s2)
    tr_p2p2 = np.trace(p2 @ p2)
    tr_p3_s = np.trace(p3 @ s)
    tr_p4 = np.trace(p4)
    # Opposite-slot coincidence tensor Xr[a,b,c,d] = sum_i (G_i)_ab (G_i)_cd.
    v = g.reshape(m, d * d)  # rows = row-major vec(G_i)
    x = (v.T @ v).reshape(d, d, d, d)
    tr_xalt = np.einsum("abcd,bcda->", x, x)  # sum_{i,j} Tr(G_i G_j G_i G_j)
    tr_x2 = np.einsum("abcd,bc,da->", x, s, s)  # sum_i Tr(G_i S G_i S)
    sum_distinct = (
        tr_s4 - 4.0 * tr_p2_s2 - 2.0 * tr_x2 + 2.0 * tr_p2p2
        + tr_xalt + 8.0 * tr_p3_s - 6.0 * tr_p4
    )
    return float(sum_distinct.real / (m * (m - 1) * (m - 2) * (m - 3)))


def fair_moment_ustatistic(snaps: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """Copy-fair U-statistic of ``Tr(rho^k)``: EXACT for k=2,3,4; subsample k>=5."""
    if k == 2:
        return full_purity_ustatistic(snaps)
    if k == 3:
        return full_moment_ustatistic_k3(snaps)
    if k == 4:
        return full_moment_ustatistic_k4(snaps)
    total = snaps.shape[0]
    total_tuples = min(_HIGH_K_FAIR_TUPLES, total * (total - 1))
    return moment_ustatistic_from_snapshots(snaps, k, total_tuples, rng)


def moment_ustatistic_from_snapshots(
    snaps: np.ndarray, k: int, n_tuples: int, rng: np.random.Generator
) -> float:
    """k-th order shadow U-statistic of ``Tr(rho^k)`` from pre-drawn snapshots.

    ``snaps`` has shape ``(M, n, 2, 2)``.  Forming the ``n_tuples`` index k-tuples
    is pure classical post-processing; it consumes **no copies**; so the copy
    budget is ``M`` regardless of ``n_tuples``.  More tuples only lower the
    variance (toward the full U-statistic minimum).
    """
    m = snaps.shape[0]
    tuples = _distinct_tuples(m, k, n_tuples, rng)  # (T, k)
    gathered = snaps[tuples]  # (T, k, n, 2, 2)
    chain = gathered[:, 0]  # (T, n, 2, 2)
    for j in range(1, k):
        chain = chain @ gathered[:, j]  # batched 2x2 matmul over (T, n)
    per_qubit_trace = np.einsum("tnii->tn", chain).real  # (T, n)
    return float(per_qubit_trace.prod(axis=1).mean())
