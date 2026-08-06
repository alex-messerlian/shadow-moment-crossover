"""Pauli operators and n-qubit Pauli-string matrices.

Conventions
-----------
* Single-qubit Paulis are the standard 2x2 matrices ``I, X, Y, Z``.
* An n-qubit Pauli string is the Kronecker (tensor) product of single-qubit
  Paulis in qubit order: qubit 0 is the *most significant* factor, i.e.
  ``pauli_string("XZ") == kron(X, Z)``.  This matches the usual computational
  basis ordering where the basis state ``|b_0 b_1 ... b_{n-1}>`` has row index
  ``sum_i b_i * 2**(n-1-i)`` and qubit 0 is the leftmost bit.

All matrices are returned as ``complex128`` ``numpy`` arrays and are never
mutated in place by callers of this module.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable

import numpy as np

# Single-qubit Pauli matrices (immutable module-level references, copy before
# mutating).  Kept read-only-by-convention; callers receive fresh arrays from
# the builder functions below.
I2: np.ndarray = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y: np.ndarray = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Lookup by label.  'I' is the 2x2 identity.
PAULI: dict[str, np.ndarray] = {"I": I2, "X": X, "Y": Y, "Z": Z}

PAULI_LABELS: tuple[str, ...] = ("I", "X", "Y", "Z")


def pauli_matrix(label: str) -> np.ndarray:
    """Return a fresh copy of the single-qubit Pauli matrix for ``label``.

    Parameters
    ----------
    label:
        One of ``'I', 'X', 'Y', 'Z'``.
    """
    try:
        return PAULI[label].copy()
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown Pauli label {label!r}; expected one of {PAULI_LABELS}") from exc


def pauli_string(labels: Iterable[str]) -> np.ndarray:
    """Build the n-qubit Pauli-string matrix for a sequence of labels.

    ``labels`` may be a string such as ``"XIZ"`` or any iterable of single-qubit
    labels.  The result is ``kron(P_0, P_1, ..., P_{n-1})``, a
    ``2**n x 2**n`` complex matrix.  The empty string yields the ``1x1``
    scalar identity ``[[1]]``.
    """
    mats = [pauli_matrix(lbl) for lbl in labels]
    if not mats:
        return np.array([[1.0 + 0.0j]], dtype=np.complex128)
    return reduce(np.kron, mats)


def kron_all(matrices: Iterable[np.ndarray]) -> np.ndarray:
    """Kronecker product of an ordered iterable of matrices.

    Order is preserved: the first matrix is the most significant factor.
    """
    mats = list(matrices)
    if not mats:
        return np.array([[1.0 + 0.0j]], dtype=np.complex128)
    return reduce(np.kron, mats)
