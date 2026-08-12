"""EXACT per-state ``k = 2`` projection variances for an ARBITRARY density matrix.

:mod:`anrl.theory.single_copy_law` gives ``closed_form_zetas(n, q)``, the *ensemble*
average ``E_psi[zeta_c]`` over the noisy-pure family.  This module evaluates the same
two identities pointwise, ``zeta_1(rho)`` and ``zeta_2(rho)`` for any ``rho``, with no
sampling anywhere.  Both are contractions of the Huang--Kueng--Preskill second-moment
identity (arXiv:2002.08953, Lemma 4) with the Pauli expansion of the state:

    zeta_1 = 4^-n sum_{P ~ Q} 3^{|supp P cap supp Q|} <P> <Q> <P (-) Q>  -  Tr(rho^2)^2,
    zeta_2 = 7^n  sum_P <P>^2 (1/14)^{|P|}                              -  Tr(rho^2)^2,

where ``P ~ Q`` means the two strings agree on every qubit where both are non-identity
and ``P (-) Q`` is what is left once the shared non-identity factors cancel.

Cost, given the ``4^n`` Pauli expectations:

* ``zeta_2`` is a single weighted sum over the ``4^n`` strings: ``Theta(4^n)``.
* ``zeta_1`` runs over compatible PAIRS.  Per qubit exactly ten local patterns are
  compatible -- ``(I,I)``, ``(I,a)``, ``(a,I)`` and ``(a,a)`` for the three
  non-identity letters ``a`` -- so the pair set has size ``10^n`` and the cost is
  ``Theta(10^n)``, i.e. ``N^{log 10 / log 4} = N^{1.661}`` in the input size
  ``N = 4^n``.  :func:`exact_zeta1` enumerates the tail vectorized and loops over a
  head of ``head`` qubits so peak memory stays at ``O(10^{n - head})``.

The input is the FULL Pauli spectrum -- all ``4^n`` signed expectations for
``zeta_1``, all ``4^n`` squares for ``zeta_2``.  That is the same information as the
density matrix, so these are functionals of a fully known state, not of anything an
experiment measures directly; :func:`truncated_zeta2` quantifies what a weight-limited
subset buys.
"""

from __future__ import annotations

import numpy as np

# Local compatible patterns (s_q, s'_q, t_q, weight): t is the string left after the
# shared non-identity factors cancel, weight is 3^{[both non-identity]}.
_LOCAL_PATTERNS: tuple[tuple[int, int, int, float], ...] = (
    (0, 0, 0, 1.0),
    *((0, a, a, 1.0) for a in (1, 2, 3)),
    *((a, 0, a, 1.0) for a in (1, 2, 3)),
    *((a, a, 0, 3.0) for a in (1, 2, 3)),
)

_PAULI_TO_DIAG = np.array(
    [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]], dtype=complex
)


def pauli_expectations(rho: np.ndarray, n: int) -> np.ndarray:
    """All ``4^n`` expectations ``Tr(rho P_s)``, flat base-4 order (0=I, 1=X, 2=Y, 3=Z).

    Computed by ``n`` successive ``4 x 4`` per-qubit transforms in ``O(4^n n)``, never
    by building the ``4^n`` Pauli strings.
    """
    v = np.asarray(rho, dtype=complex).reshape((2,) * (2 * n))
    perm: list[int] = []
    for q in range(n):
        perm += [q, n + q]
    v = np.transpose(v, perm).reshape((4,) * n)
    for q in range(n):
        v = np.moveaxis(v, q, 0)
        shape = v.shape
        v = (_PAULI_TO_DIAG @ v.reshape(4, -1)).reshape(shape)
        v = np.moveaxis(v, 0, q)
    return v.real.reshape(-1).copy()


def pauli_weights(n: int) -> np.ndarray:
    """Weight ``|P_s|`` (number of non-identity factors) for every string, base-4 order."""
    w = np.zeros(1, dtype=np.int64)
    for _ in range(n):
        w = np.concatenate([w, w + 1, w + 1, w + 1])
    return w


def purity_from_expectations(m: np.ndarray, n: int) -> float:
    """``Tr(rho^2) = 2^-n sum_s <P_s>^2``."""
    return float((m * m).sum() / 2 ** n)


def exact_zeta2(m: np.ndarray, n: int, weights: np.ndarray | None = None) -> float:
    """EXACT ``zeta_2(rho) = 7^n sum_P <P>^2 14^{-|P|} - Tr(rho^2)^2``.  ``Theta(4^n)``."""
    w = pauli_weights(n) if weights is None else weights
    spectral = float((m * m * np.float64(14.0) ** (-w)).sum())
    p2 = purity_from_expectations(m, n)
    return 7.0 ** n * spectral - p2 * p2


def _tail_patterns(n_tail: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Base-4 offsets and weights for every compatible pattern on ``n_tail`` qubits."""
    s = np.zeros(1, dtype=np.int64)
    sp = np.zeros(1, dtype=np.int64)
    t = np.zeros(1, dtype=np.int64)
    wt = np.ones(1, dtype=np.float64)
    for _ in range(n_tail):
        s_new, sp_new, t_new, wt_new = [], [], [], []
        for a, b, c, w in _LOCAL_PATTERNS:
            s_new.append(s * 4 + a)
            sp_new.append(sp * 4 + b)
            t_new.append(t * 4 + c)
            wt_new.append(wt * w)
        s = np.concatenate(s_new)
        sp = np.concatenate(sp_new)
        t = np.concatenate(t_new)
        wt = np.concatenate(wt_new)
    return s, sp, t, wt


def _head_patterns(n_head: int) -> list[tuple[int, int, int, float]]:
    """Every compatible pattern on the leading ``n_head`` qubits, as base-4 prefixes."""
    out = [(0, 0, 0, 1.0)]
    for _ in range(n_head):
        out = [
            (s * 4 + a, sp * 4 + b, t * 4 + c, w * lw)
            for (s, sp, t, w) in out
            for (a, b, c, lw) in _LOCAL_PATTERNS
        ]
    return out


def exact_zeta1(m: np.ndarray, n: int, max_tail: int = 6) -> float:
    """EXACT ``zeta_1(rho)`` by the compatible-pair sum.  ``Theta(10^n)`` time.

    ``max_tail`` qubits are enumerated vectorized (``10^{max_tail}`` entries held in
    memory); the remaining ``n - max_tail`` are looped over in Python, so peak memory is
    ``O(10^{max_tail})`` regardless of ``n``.
    """
    n_tail = min(n, max_tail)
    n_head = n - n_tail
    s_t, sp_t, t_t, wt_t = _tail_patterns(n_tail)
    stride = 4 ** n_tail
    total = 0.0
    for s_h, sp_h, t_h, w_h in _head_patterns(n_head):
        total += w_h * float(
            (wt_t * m[s_h * stride + s_t] * m[sp_h * stride + sp_t] * m[t_h * stride + t_t]).sum()
        )
    p2 = purity_from_expectations(m, n)
    return total / 4.0 ** n - p2 * p2


def exact_zetas(rho: np.ndarray, n: int, max_tail: int = 6) -> tuple[float, float]:
    """EXACT ``(zeta_1, zeta_2)`` for an arbitrary ``rho``, sampling-free."""
    m = pauli_expectations(rho, n)
    return exact_zeta1(m, n, max_tail=max_tail), exact_zeta2(m, n)


def exact_m_star(rho: np.ndarray, n: int, max_tail: int = 6) -> float:
    """EXACT statewise threshold ``M*(rho) = zeta_2(rho) / (2 zeta_1(rho))``."""
    z1, z2 = exact_zetas(rho, n, max_tail=max_tail)
    return float("inf") if z1 <= 0 else z2 / (2.0 * z1)


def zeta1_diagonal(m: np.ndarray, n: int, weights: np.ndarray | None = None) -> float:
    """The ``P = Q`` part of the ``zeta_1`` sum, ``4^-n sum_P 3^{|P|} <P>^2``.

    This is the weight-only quadratic ansatz of Section 3.4; the difference from
    :func:`exact_zeta1` (before subtracting ``Tr(rho^2)^2``) is the cubic off-diagonal
    remainder.  ``Theta(4^n)``.
    """
    w = pauli_weights(n) if weights is None else weights
    return float((np.float64(3.0) ** w * m * m).sum()) / 4.0 ** n


def truncated_zeta2(m: np.ndarray, n: int, max_weight: int, weights: np.ndarray | None = None) -> float:
    """``zeta_2`` with the spectral sum truncated to Pauli weight ``<= max_weight``.

    The kernel ``14^{-|P|}`` suppresses high weight, so a weight-limited subset of the
    spectrum -- the part local shadows estimate cheaply -- may already determine
    ``zeta_2``.  The purity term is also recomputed from the same truncation, so this is
    what an experimenter limited to low-weight observables would actually form.
    """
    w = pauli_weights(n) if weights is None else weights
    keep = w <= max_weight
    spectral = float((m[keep] ** 2 * np.float64(14.0) ** (-w[keep])).sum())
    p2 = float((m[keep] ** 2).sum() / 2 ** n)
    return 7.0 ** n * spectral - p2 * p2
