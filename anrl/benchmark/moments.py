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
* Single-copy route: the k-th order classical-shadow U-statistic — the mean of
  ``Tr(snap_{i1} @ ... @ snap_{ik})`` over distinct index k-tuples, unbiased for
  ``Tr(rho^k)``.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from .shadows import _snapshots


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
    over ``n_tuples`` random distinct index k-tuples, computed via the exact
    per-qubit factorization ``prod_q Tr(snap_{i1}^q @ ... @ snap_{ik}^q)``.
    Unbiased for ``Tr(rho^k)``.  ``n_tuples`` defaults to ``n_snapshots // k`` —
    the same O(n_snapshots) subsampling convention as the k=2 purity estimator
    (each snapshot used about once).
    """
    if k < 2:
        raise ValueError(f"moment order k must be >= 2, got {k}")
    if n_snapshots < k:
        raise ValueError(f"need >= k={k} snapshots, got {n_snapshots}")
    rho = np.asarray(rho, dtype=np.complex128)
    n = int(round(np.log2(rho.shape[0])))

    snaps = _snapshots(rho, n, n_snapshots, rng)  # (M, n, 2, 2)
    t = max(1, n_snapshots // k) if n_tuples is None else n_tuples
    return moment_ustatistic_from_snapshots(snaps, k, t, rng)


def moment_ustatistic_from_snapshots(
    snaps: np.ndarray, k: int, n_tuples: int, rng: np.random.Generator
) -> float:
    """k-th order shadow U-statistic of ``Tr(rho^k)`` from pre-drawn snapshots.

    ``snaps`` has shape ``(M, n, 2, 2)``.  Forming the ``n_tuples`` index k-tuples
    is pure classical post-processing — it consumes **no copies** — so the copy
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
