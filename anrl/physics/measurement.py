"""Single-copy local-Pauli measurement simulation and state reconstruction.

A *setting* is an ``n``-tuple over ``{'X', 'Y', 'Z'}`` naming the Pauli measured
on each qubit.  Measuring is simulated by rotating each qubit into the measured
Pauli's eigenbasis, reading outcome probabilities off the diagonal of the
rotated state, and multinomially sampling ``shots`` bitstrings.  Per-outcome bit
``0`` maps to eigenvalue ``+1`` and bit ``1`` to ``-1``.

Single-qubit rotations (rows are the +/- eigenstate bras, so the rotated
diagonal gives ``<+|rho|+>`` then ``<-|rho|->``):

* ``U_Z = I``                              (Z eigenbasis = computational)
* ``U_X = Hadamard``                       = [[1, 1], [1, -1]] / sqrt(2)
* ``U_Y = [[1, -i], [1, i]] / sqrt(2)``

From accumulated per-setting counts we form every compatible Pauli-term
expectation (each qubit is ``I`` or the setting's Pauli) and reconstruct
``rho_hat = (1/2^n) * sum_P <P> P``, projected to the nearest physical density
matrix.

Kronecker / bit ordering matches :mod:`anrl.physics.pauli`: qubit 0 is the most
significant factor and the most significant bit of the outcome index.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .pauli import kron_all, pauli_string

Setting = Tuple[str, ...]
PauliTerm = Tuple[str, ...]

_INV_SQRT2 = 1.0 / np.sqrt(2.0)

# Single-qubit basis-change unitaries: rows are the +1 then -1 eigenstate bras.
_SINGLE_QUBIT_ROTATION: Dict[str, np.ndarray] = {
    "Z": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) * _INV_SQRT2,
    "Y": np.array([[1.0, -1j], [1.0, 1j]], dtype=np.complex128) * _INV_SQRT2,
}


def single_qubit_rotation(pauli: str) -> np.ndarray:
    """Return the 2x2 rotation into the eigenbasis of ``pauli`` ('X', 'Y', 'Z')."""
    try:
        return _SINGLE_QUBIT_ROTATION[pauli].copy()
    except KeyError as exc:
        raise ValueError(f"measurement setting must be one of X, Y, Z; got {pauli!r}") from exc


def measurement_unitary(setting: Sequence[str]) -> np.ndarray:
    """Full ``2^n x 2^n`` rotation ``U = U_{q0} (x) ... (x) U_{q_{n-1}}``."""
    return kron_all(single_qubit_rotation(p) for p in setting)


def outcome_probabilities(rho: np.ndarray, setting: Sequence[str]) -> np.ndarray:
    """Probability of each of the ``2^n`` outcomes for ``setting``.

    Rotate ``rho`` into the measured eigenbasis and read the real diagonal of
    ``U rho U^dagger``.  Small negative numerical noise is clipped and the
    vector is renormalized so it is a valid probability distribution.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    u = measurement_unitary(setting)
    rotated = u @ rho @ u.conj().T
    probs = np.real(np.diag(rotated))
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0.0:  # pragma: no cover - defensive; a valid rho has trace 1
        raise ValueError("outcome probabilities summed to zero")
    return probs / total


def sample_counts(
    rho: np.ndarray, setting: Sequence[str], shots: int, rng: np.random.Generator
) -> np.ndarray:
    """Multinomially sample ``shots`` outcomes; return integer counts of length ``2^n``."""
    probs = outcome_probabilities(rho, setting)
    return rng.multinomial(shots, probs)


def simulate_settings(
    rho: np.ndarray,
    settings: Iterable[Sequence[str]],
    shots: int,
    rng: np.random.Generator,
) -> Dict[Setting, np.ndarray]:
    """Sample every setting once and return ``{setting_tuple: counts}``."""
    counts_by_setting: Dict[Setting, np.ndarray] = {}
    for setting in settings:
        key = tuple(setting)
        counts_by_setting[key] = sample_counts(rho, key, shots, rng)
    return counts_by_setting


def all_local_pauli_settings(n: int) -> list[Setting]:
    """The full set of ``3^n`` local Pauli settings (each qubit in X/Y/Z)."""
    return [tuple(s) for s in itertools.product("XYZ", repeat=n)]


def _eigenvalue_signs(n: int) -> np.ndarray:
    """``(2^n, n)`` array of +/-1 eigenvalues: entry ``[b, i] = (-1)^{bit_i(b)}``.

    Bit ``i`` is qubit ``i`` with qubit 0 the most significant bit of ``b``.
    """
    dim = 2 ** n
    outcomes = np.arange(dim)
    bits = ((outcomes[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int64)
    return 1 - 2 * bits  # 0 -> +1, 1 -> -1


def estimate_pauli_expectations(
    counts_by_setting: Mapping[Setting, np.ndarray], n: int
) -> Dict[PauliTerm, float]:
    """Estimate ``<P> = tr(rho P)`` for every Pauli term the settings can resolve.

    For each setting and each subset of qubits (each qubit either ``I`` or the
    setting's Pauli) the estimate of that term is the shot-weighted average over
    sampled outcomes of the product of ``+/-1`` eigenvalues on the term's
    non-identity qubits.  A term compatible with several settings is aggregated
    by pooling counts (shot-weighted), which is the natural combined estimator.

    Returns a dict mapping each Pauli term (an ``n``-tuple over ``I/X/Y/Z``,
    including the all-``I`` identity with value ``1``) to its estimate.
    """
    signs = _eigenvalue_signs(n)  # (2^n, n)
    numerator: Dict[PauliTerm, float] = {}
    denominator: Dict[PauliTerm, float] = {}

    for setting, counts in counts_by_setting.items():
        counts = np.asarray(counts, dtype=np.float64)
        total = counts.sum()
        if total <= 0:
            continue
        # Enumerate every subset of qubits via a bitmask; bit i selects qubit i.
        for mask in range(2 ** n):
            support = [i for i in range(n) if (mask >> i) & 1]
            term: PauliTerm = tuple(
                setting[i] if i in support else "I" for i in range(n)
            )
            if support:
                sign = np.prod(signs[:, support], axis=1)
            else:
                sign = np.ones(counts.shape[0], dtype=np.float64)
            contribution = float(counts @ sign)
            numerator[term] = numerator.get(term, 0.0) + contribution
            denominator[term] = denominator.get(term, 0.0) + total

    return {term: numerator[term] / denominator[term] for term in numerator}


def _project_to_simplex(y: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``y`` onto the probability simplex ``{x>=0, sum x = 1}``.

    Standard sort-based algorithm (Held-Karp / Wang & Carreira-Perpinan).
    """
    y = np.asarray(y, dtype=np.float64)
    d = y.size
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, d + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[rho - 1] / rho
    return np.maximum(y - theta, 0.0)


def project_to_density_matrix(mat: np.ndarray) -> np.ndarray:
    """Project a Hermitian-ish matrix to the nearest unit-trace density matrix.

    Symmetrize, eigendecompose, project the eigenvalues onto the probability
    simplex (Euclidean, sort-based), then rebuild.  The result is Hermitian,
    positive semidefinite, and has unit trace.
    """
    mat = np.asarray(mat, dtype=np.complex128)
    herm = 0.5 * (mat + mat.conj().T)
    eigvals, eigvecs = np.linalg.eigh(herm)
    projected = _project_to_simplex(eigvals.real)
    rho = (eigvecs * projected) @ eigvecs.conj().T
    return 0.5 * (rho + rho.conj().T)  # scrub residual asymmetry


def reconstruct(expectations: Mapping[PauliTerm, float], n: int) -> np.ndarray:
    """Rebuild ``rho_hat = (1/2^n) * sum_P <P> P`` and project it to a density matrix.

    ``expectations`` maps Pauli terms (``n``-tuples over ``I/X/Y/Z``) to their
    estimated expectation values.  Terms absent from the mapping contribute
    zero.  The raw estimator is Hermitian but generally not positive
    semidefinite; ``project_to_density_matrix`` returns the closest physical
    state.
    """
    dim = 2 ** n
    rho_hat = np.zeros((dim, dim), dtype=np.complex128)
    for term, coeff in expectations.items():
        rho_hat += coeff * pauli_string(term)
    rho_hat /= dim
    return project_to_density_matrix(rho_hat)
