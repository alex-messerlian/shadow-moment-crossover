"""Efficient exact single-copy moment U-statistics ``Tr(rho^k)`` for k=2,3,4.

These compute the *same* exact full U-statistics as
:mod:`anrl.benchmark.moments` (``full_purity_ustatistic``,
``full_moment_ustatistic_k3``, ``full_moment_ustatistic_k4``; all verified
against brute-force enumeration), but scale to large ``n`` so the moment sweep
can reach ``n = 8``.  Verified identical to those functions to ``~1e-13``.

Key fact: a local-shadow snapshot is a tensor product ``G_i = ox_q g_i^q``, so
``G_i^p = ox_q (g_i^q)^p``.  The power-sum matrices ``P_p = sum_i G_i^p`` are
therefore built by Kron-accumulating the per-qubit ``p``-th powers in
``O(M 2^{2n})``, never forming an ``(M, 2^n, 2^n)`` array or squaring a
``2^n x 2^n`` matrix (which is what makes the reference k=3/k=4 implementations
blow up at ``n >= 6-7``).  The k=4 "opposite-slot" terms use an ``O(M^2 n)``
per-qubit ``M x M`` contraction (``TrXalt``) and an ``O(M n 2^{2n})`` chunked
tensor-apply (``TrX2``), avoiding the ``O(2^{4n})`` X-tensor entirely.
"""

from __future__ import annotations

import numpy as np

from .shadows import full_purity_ustatistic

# Chunk over snapshots so the transient ``(chunk, 2^n, 2^n)`` work arrays stay
# bounded (matters at n=8, where one such array is ~270 MB at chunk 256, and
# several workers run in parallel).
_CHUNK = 96
# Row-chunk for the k=4 alternating term's M x M product, so its transient stays
# O(chunk * M) instead of O(M^2) (~1 GB at M=8000, which swaps under many workers).
_CHUNK_XALT = 1024


def _perqubit_powers(snaps: np.ndarray, max_power: int) -> list[np.ndarray]:
    """``[G^1, ..., G^max_power]`` as per-qubit ``(M, n, 2, 2)`` matrix powers."""
    powers = [snaps]
    for _ in range(2, max_power + 1):
        powers.append(powers[-1] @ snaps)  # batched 2x2 matmul over (M, n)
    return powers


def _dense_power_sum(gp: np.ndarray, chunk: int = _CHUNK) -> np.ndarray:
    """Dense ``sum_i ox_q gp[i, q]`` (``2^n x 2^n``) from per-qubit powers ``gp``.

    ``gp`` is ``(M, n, 2, 2)``.  Kron-accumulates over qubits within snapshot
    chunks (``O(M 2^{2n})`` time, ``O(chunk 2^{2n})`` peak memory).
    """
    m, n = gp.shape[0], gp.shape[1]
    d = 2 ** n
    out = np.zeros((d, d), dtype=np.complex128)
    for s in range(0, m, chunk):
        acc = gp[s:s + chunk, 0]  # (c, 2, 2)
        for q in range(1, n):
            r = acc.shape[1]
            acc = (acc[:, :, None, :, None] * gp[s:s + chunk, q][:, None, :, None, :]
                   ).reshape(acc.shape[0], 2 * r, 2 * r)
        out += acc.sum(axis=0)
    return out


def _tr_xalt(snaps: np.ndarray) -> complex:
    """``sum_{i,j} Tr(G_i G_j G_i G_j)`` via per-qubit ``M x M`` factors (O(M^2 n)).

    ``Tr(g_i g_j g_i g_j) = (A_q . B_q)[i, j]`` with the 16-vectors
    ``A_q[i] = vec(g_i (x) g_i)`` (index order abcd) and
    ``B_q[j] = g_j[b,c] g_j[d,a]``; the full alternating trace is the elementwise
    product over qubits of the ``M x M`` matrices ``A_q B_q^T``, summed.
    """
    m, n = snaps.shape[0], snaps.shape[1]
    # Per-qubit M x 16 factors (small); the M x M elementwise product is done in
    # row chunks so the transient is O(chunk * M), not O(M^2) (which is ~1 GB at
    # M=8000 and would swap under multiprocessing).
    a_facs = [np.einsum("jab,jcd->jabcd", snaps[:, q], snaps[:, q]).reshape(m, 16) for q in range(n)]
    b_facs = [np.einsum("jbc,jda->jabcd", snaps[:, q], snaps[:, q]).reshape(m, 16) for q in range(n)]
    total = 0.0 + 0.0j
    for s in range(0, m, _CHUNK_XALT):
        w = np.ones((min(_CHUNK_XALT, m - s), m), dtype=np.complex128)
        for q in range(n):
            w *= a_facs[q][s:s + _CHUNK_XALT] @ b_facs[q].T
        total += w.sum()
    return complex(total)


def _apply_left_2x2(gq: np.ndarray, y: np.ndarray, q: int) -> np.ndarray:
    """Left-multiply row-qubit axis ``q`` of ``y`` (``c, 2,..,2, d``) by ``gq[c]``."""
    y = np.moveaxis(y, q + 1, 1)  # bring row-axis q next to the chunk axis
    y = np.einsum("cxr,cr...->cx...", gq, y)  # contract the 2x2 on that axis
    return np.moveaxis(y, 1, q + 1)


def _tr_x2(snaps: np.ndarray, s_mat: np.ndarray, chunk: int = _CHUNK) -> complex:
    """``sum_i Tr(G_i S G_i S)`` by applying the tensor-product ``G_i`` to dense ``S``.

    For each snapshot, ``Y_i = G_i S`` is formed by contracting each 2x2
    ``g_i^q`` onto the matching row axis of ``S`` (``O(n 2^{2n})`` per snapshot),
    then ``Tr(G_i S G_i S) = Tr(Y_i^2) = sum_{ab} Y_i[a,b] Y_i[b,a]``.
    """
    m, n = snaps.shape[0], snaps.shape[1]
    d = 2 ** n
    total = 0.0 + 0.0j
    for s0 in range(0, m, chunk):
        g = snaps[s0:s0 + chunk]  # (c, n, 2, 2)
        c = g.shape[0]
        y = np.broadcast_to(s_mat, (c, d, d)).copy().reshape([c] + [2] * n + [d])
        for q in range(n):
            y = _apply_left_2x2(g[:, q], y, q)
        y = y.reshape(c, d, d)
        total += np.einsum("cab,cba->", y, y)  # sum_i Tr(Y_i^2)
    return complex(total)


def exact_moment_ustatistic(snaps: np.ndarray, k: int) -> float:
    """Exact full single-copy U-statistic of ``Tr(rho^k)`` for ``k in {2, 3, 4}``.

    Identical value to :mod:`anrl.benchmark.moments`' reference estimators but
    ``O(M 2^{2n})`` (k=2,3) / ``O(M^2 n + M n 2^{2n})`` (k=4) instead of forming
    dense ``(M, 2^n, 2^n)`` arrays or the ``2^{2n} x 2^{2n}`` X-tensor.
    """
    m = snaps.shape[0]
    if k == 2:
        return full_purity_ustatistic(snaps)  # already O(M^2 n)
    if m < k:
        raise ValueError(f"order-{k} U-statistic needs >= {k} snapshots, got {m}")
    if k == 3:
        s_mat, p2, p3 = (_dense_power_sum(g) for g in _perqubit_powers(snaps, 3))
        val = np.trace(s_mat @ s_mat @ s_mat) - 3.0 * np.trace(p2 @ s_mat) + 2.0 * np.trace(p3)
        return float(val.real / (m * (m - 1) * (m - 2)))
    if k == 4:
        s_mat, p2, p3, p4 = (_dense_power_sum(g) for g in _perqubit_powers(snaps, 4))
        s2 = s_mat @ s_mat
        val = (
            np.trace(s2 @ s2)
            - 4.0 * np.trace(p2 @ s2)
            - 2.0 * _tr_x2(snaps, s_mat)
            + 2.0 * np.trace(p2 @ p2)
            + _tr_xalt(snaps)
            + 8.0 * np.trace(p3 @ s_mat)
            - 6.0 * np.trace(p4)
        )
        return float(val.real / (m * (m - 1) * (m - 2) * (m - 3)))
    raise ValueError(f"exact_moment_ustatistic supports k in {{2, 3, 4}}, got {k}")
