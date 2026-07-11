"""Reference quantum states and noise channels used throughout the physics core.

Every function returns a density matrix as a fresh ``complex128`` array of shape
``(dim, dim)`` with unit trace and Hermitian symmetry (up to floating point).
Nothing is mutated in place.

State-vector / Kronecker ordering follows :mod:`anrl.physics.pauli`: qubit 0 is
the most significant tensor factor, so ``|b_0 ... b_{n-1}>`` maps to index
``sum_i b_i * 2**(n-1-i)``.
"""

from __future__ import annotations

import numpy as np


def _density_from_statevector(psi: np.ndarray) -> np.ndarray:
    """Outer product ``|psi><psi|`` for a (normalized) column state vector."""
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    return np.outer(psi, psi.conj())


def bell_phi_plus() -> np.ndarray:
    """The 2-qubit Bell state |Phi+> = (|00> + |11>)/sqrt(2) as a density matrix."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0b00] = 1.0
    psi[0b11] = 1.0
    psi /= np.sqrt(2.0)
    return _density_from_statevector(psi)


def werner(p: float) -> np.ndarray:
    """2-qubit Werner state ``p * |psi-><psi-| + (1 - p) * I/4``.

    ``|psi-> = (|01> - |10>)/sqrt(2)`` is the singlet.  This state is entangled
    (non-zero negativity) precisely for ``p > 1/3``.
    """
    singlet = np.zeros(4, dtype=np.complex128)
    singlet[0b01] = 1.0
    singlet[0b10] = -1.0
    singlet /= np.sqrt(2.0)
    rho_singlet = _density_from_statevector(singlet)
    identity = np.eye(4, dtype=np.complex128) / 4.0
    return p * rho_singlet + (1.0 - p) * identity


def ghz(n: int) -> np.ndarray:
    """n-qubit GHZ state (|0...0> + |1...1>)/sqrt(2) as a density matrix."""
    if n < 1:
        raise ValueError(f"ghz requires n >= 1, got {n}")
    dim = 2 ** n
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    psi[dim - 1] = 1.0
    psi /= np.sqrt(2.0)
    return _density_from_statevector(psi)


def random_density(dim: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Random density matrix from the Ginibre / Hilbert-Schmidt ensemble.

    Draw ``G`` as a ``dim x rank`` complex Gaussian matrix and set
    ``rho = G G^dagger / tr(G G^dagger)``.  The result is Hermitian, positive
    semidefinite, unit trace, and has rank at most ``rank``.

    Parameters
    ----------
    dim:
        Hilbert-space dimension.
    rank:
        Number of Ginibre columns; caps the rank of ``rho``.
    rng:
        A ``numpy`` random generator (for reproducibility).
    """
    if dim < 1 or rank < 1:
        raise ValueError(f"dim and rank must be >= 1, got dim={dim}, rank={rank}")
    real = rng.standard_normal((dim, rank))
    imag = rng.standard_normal((dim, rank))
    g = real + 1j * imag
    rho = g @ g.conj().T
    rho /= np.trace(rho).real
    return rho.astype(np.complex128)


def depolarize(rho: np.ndarray, q: float) -> np.ndarray:
    """Global depolarizing channel: ``(1 - q) * rho + q * I/dim``.

    ``q = 0`` returns ``rho`` unchanged; ``q = 1`` returns the maximally mixed
    state.  ``dim`` is inferred from ``rho``.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    dim = rho.shape[0]
    identity = np.eye(dim, dtype=np.complex128) / dim
    return (1.0 - q) * rho + q * identity


def maximally_mixed(dim: int) -> np.ndarray:
    """The maximally mixed state ``I / dim`` (handy reference / test fixture)."""
    return np.eye(dim, dtype=np.complex128) / dim
