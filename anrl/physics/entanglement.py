"""Exact bipartite entanglement measures for n-qubit density matrices.

The A|B bipartition splits a ``dim = dA * dB`` system into subsystem ``A`` (the
most significant tensor factor / first qubits) and ``B`` (the rest).  When the
subsystem dimensions are omitted, a balanced qubit split is used: for an
``n``-qubit state the first ``ceil(n/2)`` qubits form ``A``
(``dA = 2**ceil(n/2)``, ``dB = 2**floor(n/2)``).

Key objects
-----------
* ``partial_transpose``, transpose only subsystem ``A``.
* ``negativity``, sum of |negative eigenvalues| of the partial transpose.
* ``pt_moment``, ``tr((rho^{T_A})^k)`` (the k-th PT spectral moment).
* ``purity``, ``tr(rho^2)``.
* ``negativity_from_moments``, reconstruct the negativity from PT moments alone
  (power sums) via Newton's identities and the characteristic polynomial.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _infer_balanced_bipartition(dim: int) -> tuple[int, int]:
    """Return ``(dA, dB)`` for the default balanced qubit bipartition.

    Assumes ``dim = 2**n``; the first ``ceil(n/2)`` qubits form ``A``.
    """
    n = int(round(math.log2(dim)))
    if 2 ** n != dim:
        raise ValueError(
            f"cannot infer a qubit bipartition: dim={dim} is not a power of two; "
            "pass dA and dB explicitly"
        )
    n_a = math.ceil(n / 2)
    return 2 ** n_a, 2 ** (n - n_a)


def _resolve_dims(rho: np.ndarray, dA: int | None, dB: int | None) -> tuple[int, int]:
    dim = rho.shape[0]
    if dA is None and dB is None:
        return _infer_balanced_bipartition(dim)
    if dA is None:
        dA = dim // dB
    if dB is None:
        dB = dim // dA
    if dA * dB != dim:
        raise ValueError(f"dA*dB={dA * dB} does not match rho dimension {dim}")
    return dA, dB


def partial_transpose(rho: np.ndarray, dA: int | None = None, dB: int | None = None) -> np.ndarray:
    """Partial transpose over subsystem ``A``.

    Uses the reshape convention specified for this project: view ``rho`` as a
    ``(dA, dB, dA, dB)`` tensor with axes ``(a, b, a', b')``, transpose axes to
    ``(2, 1, 0, 3)``, i.e. swap the two ``A`` indices ``a <-> a'`` while leaving
    the ``B`` indices fixed, then reshape back to ``(dA*dB, dA*dB)``.

    For a Hermitian ``rho`` the result is Hermitian.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    dA, dB = _resolve_dims(rho, dA, dB)
    tensor = rho.reshape(dA, dB, dA, dB)
    transposed = tensor.transpose(2, 1, 0, 3)
    return transposed.reshape(dA * dB, dA * dB)


def purity(rho: np.ndarray) -> float:
    """Return ``tr(rho^2)`` as a real float."""
    rho = np.asarray(rho, dtype=np.complex128)
    return float(np.trace(rho @ rho).real)


def negativity(rho: np.ndarray, dA: int | None = None, dB: int | None = None) -> float:
    """Entanglement negativity ``N(rho) = sum_i |lambda_i^-|``.

    ``lambda_i^-`` are the negative eigenvalues of the partial transpose
    ``rho^{T_A}``.  Equivalently ``N = (||rho^{T_A}||_1 - 1) / 2``.  Zero for
    states with a positive partial transpose (all separable states, and bound
    entangled PPT states).
    """
    pt = partial_transpose(rho, dA, dB)
    # PT of a Hermitian matrix is Hermitian -> use the symmetric eigensolver.
    eigvals = np.linalg.eigvalsh(pt)
    negative = eigvals[eigvals < 0.0]
    return float(-negative.sum())


def pt_moment(rho: np.ndarray, k: int, dA: int | None = None, dB: int | None = None) -> float:
    """The k-th partial-transpose moment ``tr((rho^{T_A})^k)``.

    For ``k = 2`` this equals the purity ``tr(rho^2)`` (partial transpose
    preserves the Hilbert-Schmidt norm).
    """
    if k < 1:
        raise ValueError(f"pt_moment requires k >= 1, got {k}")
    pt = partial_transpose(rho, dA, dB)
    return float(np.trace(np.linalg.matrix_power(pt, k)).real)


def negativity_from_moments(moments: Sequence[float]) -> float:
    """Reconstruct the negativity from the partial-transpose spectral moments.

    ``moments`` are the power sums ``[p_1, p_2, ..., p_K]`` with
    ``p_k = tr((rho^{T_A})^k) = sum_i lambda_i^k`` for the ``K`` eigenvalues of
    the partial transpose (so ``K = len(moments)`` and, for a ``d``-dimensional
    partial transpose, ``K = d``; ``p_1 = 1``).

    Method
    ------
    1. Newton's identities turn power sums into elementary symmetric
       polynomials: ``e_k = (1/k) * sum_{i=1..k} (-1)^{i-1} * e_{k-i} * p_i``,
       with ``e_0 = 1``.
    2. The monic characteristic polynomial of the (implicit) partial transpose
       has coefficients ``[(-1)^i * e_i]`` for ``i = 0..K`` (highest degree
       first): ``x^K - e_1 x^{K-1} + e_2 x^{K-2} - ...``.
    3. Its roots are the eigenvalues.  Take their real parts and return the sum
       of the absolute values of the negative ones.
    """
    p = [float(np.real(m)) for m in moments]
    k_max = len(p)
    if k_max == 0:
        return 0.0

    # Newton's identities.  e[0] = 1, e[1..k_max] computed from power sums p_1..p_k.
    e = [0.0] * (k_max + 1)
    e[0] = 1.0
    for k in range(1, k_max + 1):
        acc = 0.0
        for i in range(1, k + 1):
            acc += ((-1.0) ** (i - 1)) * e[k - i] * p[i - 1]
        e[k] = acc / k

    # Monic characteristic polynomial coefficients, highest degree first.
    coeffs = [((-1.0) ** i) * e[i] for i in range(k_max + 1)]

    roots = np.roots(coeffs).real
    negative = roots[roots < 0.0]
    return float(-negative.sum())
