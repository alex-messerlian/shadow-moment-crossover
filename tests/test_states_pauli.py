"""Unit tests for the Pauli and state primitives (properties + conventions)."""

from __future__ import annotations

import numpy as np
import pytest

from anrl.physics import (
    PAULI,
    bell_phi_plus,
    depolarize,
    ghz,
    maximally_mixed,
    pauli_matrix,
    pauli_string,
    random_density,
    werner,
)


def _is_valid_density(rho: np.ndarray, dim: int) -> None:
    assert rho.shape == (dim, dim)
    assert np.allclose(rho, rho.conj().T, atol=1e-12)  # Hermitian
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)  # unit trace
    assert np.linalg.eigvalsh(rho).min() >= -1e-12  # positive semidefinite


def test_pauli_algebra() -> None:
    x, y, z, i2 = PAULI["X"], PAULI["Y"], PAULI["Z"], PAULI["I"]
    # X Y = i Z and each Pauli squares to the identity.
    assert np.allclose(x @ y, 1j * z)
    for p in (x, y, z):
        assert np.allclose(p @ p, i2)


def test_pauli_matrix_returns_copy() -> None:
    m = pauli_matrix("X")
    m[0, 0] = 99.0
    assert PAULI["X"][0, 0] == 0.0  # module constant untouched


def test_pauli_string_kron_order() -> None:
    # pauli_string("XZ") must equal kron(X, Z) with qubit 0 most significant.
    expected = np.kron(PAULI["X"], PAULI["Z"])
    assert np.allclose(pauli_string("XZ"), expected)
    assert pauli_string("III").shape == (8, 8)


def test_reference_states_are_valid_densities() -> None:
    _is_valid_density(bell_phi_plus(), 4)
    _is_valid_density(werner(0.7), 4)
    _is_valid_density(ghz(3), 8)
    _is_valid_density(maximally_mixed(4), 4)


def test_random_density_properties() -> None:
    rng = np.random.default_rng(123)
    for dim, rank in [(4, 1), (4, 2), (8, 3)]:
        rho = random_density(dim, rank, rng)
        _is_valid_density(rho, dim)
        # rank at most `rank` (count non-negligible eigenvalues).
        nnz = int(np.sum(np.linalg.eigvalsh(rho) > 1e-9))
        assert nnz <= rank


def test_depolarize_limits() -> None:
    rho = bell_phi_plus()
    assert np.allclose(depolarize(rho, 0.0), rho)  # q=0 identity map
    assert np.allclose(depolarize(rho, 1.0), maximally_mixed(4))  # q=1 -> I/dim


def test_werner_reduces_to_maximally_mixed_at_zero() -> None:
    assert np.allclose(werner(0.0), maximally_mixed(4))
