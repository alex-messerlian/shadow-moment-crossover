"""Estimate the projection variances -- and hence ``M*`` -- from a PILOT shadow budget.

:mod:`anrl.theory.statewise_zetas` evaluates ``zeta_1(rho)`` and ``zeta_2(rho)`` exactly, but
needs all ``4^n`` Pauli expectations, i.e. the state itself.  An experimenter has snapshots.
This module estimates the same two quantities from the snapshots alone, without ever forming
the Pauli spectrum, so the high-weight strings local shadows estimate worst are never needed
individually.

Split the pilot into four disjoint blocks ``A, B, C, D`` and write
``K_ij = Tr(G_i G_j) = prod_q Tr(G_i^q G_j^q)``:

* ``zeta_2 = Var[K_ij]``  -- the sample variance of ``K`` over the disjoint pairs
  ``(A_i, B_i)``.  Each pair is independent, so this is unbiased directly.
* ``E[Tr(G rho)^2]`` -- ``mean_{i in A} Kbar_iB * Kbar_iC`` with
  ``Kbar_iB = mean_{j in B} K_ij``.  ``B`` and ``C`` are disjoint from each other and from
  ``A``, so both inner means are independent unbiased estimates of ``Tr(G_i rho)`` and their
  product is unbiased for ``Tr(G_i rho)^2``.
* ``Tr(rho^2)^2`` -- ``Kbar_AB * Kbar_CD`` over the four disjoint blocks, unbiased because
  the two factors are independent and each is unbiased for ``Tr(rho^2)``.
* ``zeta_1 = E[Tr(G rho)^2] - Tr(rho^2)^2``, the difference of the previous two.

``zeta_1`` is therefore a DIFFERENCE of two estimates sharing the scale ``Tr(rho^2)^2``, each
carrying noise of order ``zeta_2 / M``; since ``zeta_2`` grows as ``7^n``, ``zeta_1`` is the
hard factor and ``zeta_2`` the easy one.  That is the observed behaviour.

Cost.  Each snapshot is a product of Hermitian ``2 x 2`` factors, so writing
``G^q = sum_alpha a^q_alpha sigma_alpha`` with real ``a`` gives
``Tr(G_i^q G_j^q) = 2 <a_i^q, a_j^q>`` and ``K_ij = 2^n <x_i, x_j>`` for the Kronecker product
``x_i = ox_q a_i^q``.  Every block mean the estimators need is then a mean of these
``4^n``-vectors, so the whole computation is ``O(M 4^n)`` with no pairwise matrix.  Snapshots
are processed in chunks sized so the ``(chunk, 4^n)`` transient stays near
:data:`_TRANSIENT_BYTES` regardless of ``n`` or the pilot budget.
"""

from __future__ import annotations

import numpy as np

# Rows: I, X, Y, Z as flattened 2x2 matrices, halved so <sigma_a, vec(G)> reads off a_alpha.
_SIGMA_VEC = np.array(
    [[1, 0, 0, 1], [0, 1, 1, 0], [0, -1j, 1j, 0], [1, 0, 0, -1]], dtype=complex
) / 2.0

_TRANSIENT_BYTES = 1 << 28   # ~268 MB cap on the (chunk, 4^n) feature transient
_MIN_CHUNK = 256


def feature_chunk(n: int) -> int:
    """Snapshots per chunk so the ``(chunk, 4^n)`` float64 transient stays near the cap."""
    return max(_MIN_CHUNK, _TRANSIENT_BYTES // (8 * 4 ** n))


def pauli_coefficients(snaps: np.ndarray) -> np.ndarray:
    """``(M, n, 4)`` real Pauli coefficients of each per-qubit snapshot factor."""
    return np.einsum("ap,inp->ina", _SIGMA_VEC.conj(), snaps.reshape(*snaps.shape[:2], 4)).real


def snapshot_features(snaps: np.ndarray) -> np.ndarray:
    """``(M, 4^n)`` real features ``x_i`` with ``Tr(G_i G_j) = 2^n <x_i, x_j>``."""
    coeff = pauli_coefficients(snaps)
    x = coeff[:, 0, :]
    for q in range(1, snaps.shape[1]):
        x = (x[:, :, None] * coeff[:, q, None, :]).reshape(x.shape[0], -1)
    return x


def pair_traces(snaps_a: np.ndarray, snaps_b: np.ndarray) -> np.ndarray:
    """``Tr(G_i G_j)`` for the aligned pairs, by the per-qubit factorization -- ``O(M n)``."""
    return np.einsum("inuv,invu->in", snaps_a, snaps_b).real.prod(axis=1)


def _block_mean(snaps: np.ndarray, start: int, stop: int, dim: int, chunk: int) -> np.ndarray:
    total = np.zeros(dim)
    for s in range(start, stop, chunk):
        e = min(s + chunk, stop)
        total += snapshot_features(snaps[s:e]).sum(axis=0)
    return total / (stop - start)


def pilot_zetas(snaps: np.ndarray) -> tuple[float, float]:
    """Unbiased ``(zeta_1_hat, zeta_2_hat)`` from ``snaps`` alone.

    Uses the four-disjoint-block construction described in the module docstring.  Requires at
    least four snapshots; peak memory is independent of the pilot budget.
    """
    m, n = snaps.shape[0], snaps.shape[1]
    if m < 4:
        raise ValueError(f"pilot estimator needs at least 4 snapshots, got {m}")
    scale = 2.0 ** n
    dim = 4 ** n
    chunk = feature_chunk(n)
    q = m // 4

    mb = _block_mean(snaps, q, 2 * q, dim, chunk)
    mc = _block_mean(snaps, 2 * q, 3 * q, dim, chunk)
    md = _block_mean(snaps, 3 * q, 4 * q, dim, chunk)
    ma = _block_mean(snaps, 0, q, dim, chunk)

    # zeta_2 from the q disjoint pairs (A_i, B_i); the per-qubit form needs no features.
    z2 = float(np.var(pair_traces(snaps[:q], snaps[q:2 * q]), ddof=1))

    # E[Tr(G rho)^2]: A against the independent block means of B and C.
    prod = np.empty(q)
    for s in range(0, q, chunk):
        e = min(s + chunk, q)
        xa = snapshot_features(snaps[s:e])
        prod[s:e] = (scale * (xa @ mb)) * (scale * (xa @ mc))
    e_sq = float(prod.mean())

    p2_ab = scale * float(ma @ mb)
    p2_cd = scale * float(mc @ md)
    return e_sq - p2_ab * p2_cd, z2


def pilot_m_star(snaps: np.ndarray) -> float:
    """Pilot estimate of the threshold ``M* = zeta_2 / (2 zeta_1)``; ``nan`` if ``zeta_1 <= 0``.

    At small pilot budgets ``zeta_1_hat`` can come out non-positive -- it is a difference of
    two noisy estimates -- and the ratio is then undefined rather than large.
    """
    z1, z2 = pilot_zetas(snaps)
    return float("nan") if z1 <= 0 else z2 / (2.0 * z1)
