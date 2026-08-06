"""Witness-based negativity estimation.

The negativity of a state equals the expectation of a *state-dependent* witness
operator built from the negative eigenspace of the partial transpose.  Writing
``rho^{T_A}`` for the partial transpose over subsystem ``A`` with eigenpairs
``{(lambda_i, v_i)}``,

    negativity(rho) = - sum_{lambda_i < 0} lambda_i
                    = - sum_{lambda_i < 0} <v_i| rho^{T_A} |v_i>.

Expanding ``rho`` in the normalized n-qubit Pauli basis
``rho = (1/2^n) sum_P <P> P`` with ``<P> = tr(rho P)`` turns this into a linear
functional of the Pauli expectations,

    negativity(rho) = sum_P w_P <P>,   w_P = -(1/2^n) sum_{lambda_i < 0} <v_i| P^{T_A} |v_i>.

Equivalently the witness operator is ``W = -R^{T_A}`` where ``R`` projects onto
the negative eigenspace of ``rho^{T_A}``, and ``w_P = (1/2^n) tr(W P)``; the
sum over the negative subspace is computed here as ``tr(R P^{T_A})``.

This module provides:

* :func:`witness_weights`; the exact weights ``w_P(rho)``.
* :func:`estimate_negativity_witness`; the linear functional ``sum_P w_P <P>``.
* :func:`negativity_witness_estimator`; the realizable estimator: build the
  witness from the reconstructed state, then evaluate it on the measured Pauli
  expectations.

These entanglement-negativity utilities are exercised only by the test suite,
not by the paper's moment-estimation code.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Dict, Mapping, Tuple

import numpy as np

from .entanglement import _resolve_dims, partial_transpose
from .measurement import reconstruct
from .pauli import pauli_string

PauliTerm = Tuple[str, ...]


@lru_cache(maxsize=None)
def _pt_pauli_basis(n: int, dA: int, dB: int) -> Tuple[Tuple[PauliTerm, np.ndarray], ...]:
    """Partial-transposed Pauli-string matrices ``P^{T_A}`` for all ``4^n`` Paulis.

    Cached on ``(n, dA, dB)`` since the basis is state-independent; callers must
    treat the returned matrices as read-only.
    """
    basis = []
    for labels in itertools.product("IXYZ", repeat=n):
        pt_pauli = partial_transpose(pauli_string(labels), dA, dB)
        basis.append((tuple(labels), pt_pauli))
    return tuple(basis)


def witness_weights(
    rho: np.ndarray, dA: int | None = None, dB: int | None = None
) -> Dict[PauliTerm, float]:
    """Exact witness weights ``w_P(rho)`` for every ``n``-qubit Pauli string.

    ``w_P = -(1/2^n) * sum_{lambda_i < 0} <v_i| P^{T_A} |v_i>`` where
    ``(lambda_i, v_i)`` are the eigenpairs of the partial transpose
    ``rho^{T_A}``.  Returns a dict keyed by Pauli term (an ``n``-tuple over
    ``I/X/Y/Z``); the same keying used by
    :func:`~anrl.physics.measurement.estimate_pauli_expectations` and
    :func:`~anrl.physics.measurement.reconstruct`.

    For a state with a positive partial transpose (zero negativity) the witness
    functional ``sum_P w_P <P>`` evaluates to zero.  (Eigensolver round-off can
    still leave individual weights nonzero, but they cancel in the functional,
    which uses the same ``< 0`` threshold as :func:`~anrl.physics.entanglement.negativity`.)
    """
    rho = np.asarray(rho, dtype=np.complex128)
    dA, dB = _resolve_dims(rho, dA, dB)
    dim = rho.shape[0]
    n = int(round(np.log2(dim)))

    pt = partial_transpose(rho, dA, dB)
    eigvals, eigvecs = np.linalg.eigh(pt)  # PT of Hermitian rho is Hermitian
    negative = eigvals < 0.0

    basis = _pt_pauli_basis(n, dA, dB)
    coeff = 1.0 / dim  # 1 / 2^n

    if not np.any(negative):
        return {labels: 0.0 for labels, _ in basis}

    v = eigvecs[:, negative]          # columns are the negative eigenvectors v_i
    r = v @ v.conj().T                # projector onto the negative eigenspace

    # tr(R P^{T_A}) = sum_i <v_i| P^{T_A} |v_i>  (sum over the negative subspace).
    weights: Dict[PauliTerm, float] = {}
    for labels, pt_pauli in basis:
        weights[labels] = float(-coeff * np.trace(r @ pt_pauli).real)
    return weights


def estimate_negativity_witness(
    pauli_expectations: Mapping[PauliTerm, float], weights: Mapping[PauliTerm, float]
) -> float:
    """Evaluate the witness functional ``sum_P weights[P] * pauli_expectations[P]``.

    Paulis present in ``weights`` but absent from ``pauli_expectations``
    contribute zero.
    """
    total = 0.0
    for labels, w in weights.items():
        total += w * float(pauli_expectations.get(labels, 0.0))
    return float(total)


def negativity_witness_estimator(
    pauli_expectations: Mapping[PauliTerm, float],
    n: int,
    dA: int | None = None,
    dB: int | None = None,
) -> float:
    """Realizable witness estimator from finite-shot measured Pauli expectations.

    Reconstruct ``rho_hat`` from the measured expectations, build the *estimated*
    witness ``w_P(rho_hat)`` from it, then estimate the negativity as
    ``sum_P w_P(rho_hat) * <P>_measured``.  Negativity is non-negative, so a
    negative estimate is clipped to zero.
    """
    rho_hat = reconstruct(pauli_expectations, n)
    weights = witness_weights(rho_hat, dA, dB)
    estimate = estimate_negativity_witness(pauli_expectations, weights)
    return max(0.0, estimate)
