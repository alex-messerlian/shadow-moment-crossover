"""Budget-scaling primitives: vectorized sampling and M-linear exact U-statistics.

Tests the crossover law across copy budgets ``M``, which requires the exact
single-copy U-statistic at large ``M`` (up to 64x the 2000 baseline).  Two
enabling pieces:

* :func:`sample_batched` — a vectorized local-shadow sampler for the
  :class:`~anrl.benchmark.ensembles.noisy_pure` ensemble (no Python loop over the
  ``M`` snapshots), so ``M ~ 10^5`` is cheap.
* :func:`moment_ustat_linear` — the EXACT single-copy moment U-statistic computed
  in ``O(M 2^{2n})`` (LINEAR in ``M``, no ``M x M`` pair matrix).  The key move:
  the power-sum matrix ``S = sum_i G_i = sum_i ox_q g_i^q`` is built by splitting
  the qubits in half, ``S = sum_i A_i ox B_i``, and evaluating ``sum_i A_i ox
  B_i = VA^T VB`` as a single BLAS matmul of the two half-vectorizations.  This
  gives the IDENTICAL value as :func:`~anrl.benchmark.shadows.full_purity_ustatistic`
  / :func:`~anrl.benchmark.moment_ustats.exact_moment_ustatistic` (verified to
  ~1e-13), but scales to large ``M`` and large ``n``.

k=2 and k=3 are fully M-linear here; k=4 delegates to the reference
``exact_moment_ustatistic`` (feasible only at the smaller budgets ``M <~ 8000``
because its opposite-slot terms are ``O(M^2)`` / ``O(2^{4n})``).
"""

from __future__ import annotations

import numpy as np

from .ensembles import NoisyState
from .moment_ustats import exact_moment_ustatistic
from .shadows import _I2

_K4_MAX_BUDGET = 8000  # exact k=4 reference is O(M^2); cap the budget it is used at
# Snapshot chunk for the split-Kron dense power sum, so its transient stays
# O(chunk * 2^{2 ceil(n/2)}) instead of O(M * ...) (~1.1 GB/worker at n=9, M=32000
# unchunked, which OOMs under many workers).  Mirrors moment_ustats._CHUNK.
_SPLIT_CHUNK = 4096


def sample_batched(
    state: NoisyState, n_snapshots: int, rng: np.random.Generator, chunk: int = 8192
) -> np.ndarray:
    """Vectorized ``(M, n, 2, 2)`` local-shadow snapshots from a noisy-pure state.

    Draws all ``M`` Haar single-qubit rotations at once (batched QR), rotates the
    pure component ``|psi>`` to get outcome probabilities ``p(b) = (1-q)|U psi|^2_b
    + q/2^n``, samples one outcome per snapshot by inverse-CDF, and forms the
    per-qubit shadow ``3 U_q^dag |b_q><b_q| U_q - I`` — all without a Python loop
    over snapshots.  Reproduces the distribution of
    :func:`~anrl.benchmark.scaling.snapshots_factored` (unbiased shadow).
    """
    n, d, q = state.n, state.dim, state.q
    psi = state.components[:, 0].reshape([2] * n)
    snaps = np.empty((n_snapshots, n, 2, 2), dtype=np.complex128)
    shifts = n - 1 - np.arange(n)
    for s0 in range(0, n_snapshots, chunk):
        c = min(chunk, n_snapshots - s0)
        z = (rng.standard_normal((c, n, 2, 2)) + 1j * rng.standard_normal((c, n, 2, 2))) / np.sqrt(2.0)
        qr_q, qr_r = np.linalg.qr(z)  # batched Mezzadri
        diag = np.diagonal(qr_r, axis1=-2, axis2=-1)
        U = qr_q * (diag / np.abs(diag))[:, :, None, :]  # (c, n, 2, 2) Haar
        amp = np.broadcast_to(psi, (c,) + psi.shape).astype(np.complex128).copy()
        for qb in range(n):
            amp = np.moveaxis(amp, qb + 1, 1)
            amp = np.einsum("cxr,cr...->cx...", U[:, qb], amp)
            amp = np.moveaxis(amp, 1, qb + 1)
        amp = amp.reshape(c, d)
        probs = (1.0 - q) * np.abs(amp) ** 2 + q / d
        probs /= probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(c)
        out = np.minimum((cdf < u[:, None]).sum(axis=1), d - 1)  # inverse-CDF sample
        bits = (out[:, None] >> shifts) & 1  # (c, n)
        for qb in range(n):
            v = np.conj(U[np.arange(c), qb, bits[:, qb], :])  # U_q^dag col = conj(U_q row)
            rho_meas = v[:, :, None] * np.conj(v[:, None, :])  # outer(v, v*)
            snaps[s0:s0 + c, qb] = 3.0 * rho_meas - _I2
    return snaps


def _vec_kron_half(gp: np.ndarray) -> np.ndarray:
    """``(M, h, 2, 2)`` per-qubit factors -> ``(M, (2^h)^2)`` rows ``vec(ox_q gp[:,q])``."""
    m, h = gp.shape[0], gp.shape[1]
    acc = gp[:, 0]
    for q in range(1, h):
        r = acc.shape[1]
        acc = (acc[:, :, None, :, None] * gp[:, q][:, None, :, None, :]).reshape(m, 2 * r, 2 * r)
    da = 2 ** h
    return acc.reshape(m, da * da)


def dense_power_sum_split(gp: np.ndarray, chunk: int = _SPLIT_CHUNK) -> np.ndarray:
    """Dense ``sum_i ox_q gp[i, q]`` (``2^n x 2^n``) via chunked BLAS matmuls (split in half).

    ``S = sum_i A_i (x) B_i`` with ``A_i`` over the first ``n//2`` qubits and
    ``B_i`` over the rest, accumulated over snapshot chunks as
    ``sum_i vec(A_i) vec(B_i)^T`` so the transient stays ``O(chunk * 2^{2 ceil(n/2)})``.
    """
    m, n = gp.shape[0], gp.shape[1]
    if n == 1:
        return gp[:, 0].sum(axis=0)
    h = n // 2
    da, db = 2 ** h, 2 ** (n - h)
    mab = np.zeros((da * da, db * db), dtype=np.complex128)
    for s in range(0, m, chunk):
        va = _vec_kron_half(gp[s:s + chunk, :h])  # (c, da^2)
        vb = _vec_kron_half(gp[s:s + chunk, h:])  # (c, db^2)
        mab += va.T @ vb  # sum_i A_i (x) B_i, accumulated
    return mab.reshape(da, da, db, db).transpose(0, 2, 1, 3).reshape(da * db, da * db)


def _powers(snaps: np.ndarray, max_power: int) -> list[np.ndarray]:
    powers = [snaps]
    for _ in range(2, max_power + 1):
        powers.append(powers[-1] @ snaps)
    return powers


def moment_ustat_linear(snaps: np.ndarray, k: int) -> float:
    """EXACT single-copy U-statistic of ``Tr(rho^k)``; M-linear for k=2,3.

    k=2: ``[Tr(S^2) - sum_i Tr(G_i^2)] / (M(M-1))``.
    k=3: ``[Tr(S^3) - 3 Tr(P2 S) + 2 Tr(P3)] / (M(M-1)(M-2))``.
    k=4: delegates to the reference ``exact_moment_ustatistic`` (O(M^2)/O(2^{4n}),
    so only used at small budgets).  Value identical to the reference estimators.
    """
    m, n = snaps.shape[0], snaps.shape[1]
    if m < k:
        raise ValueError(f"order-{k} U-statistic needs >= {k} snapshots, got {m}")
    if k == 2:
        s_mat = dense_power_sum_split(snaps)
        tr_s2 = float((np.abs(s_mat) ** 2).sum())  # ||S||_F^2 = Tr(S^2), S Hermitian
        g2 = snaps @ snaps
        c = float(np.einsum("mnii->mn", g2).real.prod(axis=1).sum())  # sum_i prod_q Tr((g_i^q)^2)
        return (tr_s2 - c) / (m * (m - 1))
    if k == 3:
        pw = _powers(snaps, 3)
        s_mat, p2, p3 = (dense_power_sum_split(g) for g in pw)
        val = np.trace(s_mat @ s_mat @ s_mat) - 3.0 * np.trace(p2 @ s_mat) + 2.0 * np.trace(p3)
        return float(val.real / (m * (m - 1) * (m - 2)))
    if k == 4:
        return exact_moment_ustatistic(snaps, 4)
    raise ValueError(f"moment_ustat_linear supports k in {{2, 3, 4}}, got {k}")
