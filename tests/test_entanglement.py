"""Exact-value tests for the entanglement core.

Each test group is pinned to a known analytic value (tolerances noted inline).
Covers six of the seven required groups; the single-copy tomography group lives
in :mod:`tests.test_measurement`.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.physics import (
    bell_phi_plus,
    depolarize,
    ghz,
    maximally_mixed,
    negativity,
    negativity_from_moments,
    pt_moment,
    purity,
    random_density,
    werner,
)


# ---------------------------------------------------------------------------
# Group 1 — reference negativity values (Bell entangled, maximally mixed sep.)
# ---------------------------------------------------------------------------
def test_negativity_reference_values() -> None:
    # |Phi+> is maximally entangled: negativity = 1/2.
    assert negativity(bell_phi_plus()) == pytest.approx(0.5, abs=1e-9)
    # The maximally mixed 2-qubit state I/4 is separable: negativity = 0.
    assert negativity(maximally_mixed(4)) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Group 2 — the second PT moment equals the purity
# ---------------------------------------------------------------------------
def test_pt_moment_two_equals_purity() -> None:
    rng = np.random.default_rng(20260711)
    # A spread of random states (2- and 3-qubit, various ranks).
    for dim, rank in [(4, 1), (4, 2), (4, 4), (8, 3), (8, 8)]:
        for _ in range(5):
            rho = random_density(dim, rank, rng)
            assert pt_moment(rho, 2) == pytest.approx(purity(rho), abs=1e-9)


# ---------------------------------------------------------------------------
# Group 3 — Werner threshold at p = 1/3
# ---------------------------------------------------------------------------
def test_werner_threshold() -> None:
    assert negativity(werner(0.30)) == pytest.approx(0.0, abs=1e-9)
    assert negativity(werner(0.40)) > 0.0
    # Pin the SINGLET convention (incl. the minus sign): at p=1 the Werner state
    # must be exactly |psi-><psi-| with psi- = (|01> - |10>)/sqrt(2).  This
    # distinguishes it from the triplet / Phi+, which share the same negativity.
    singlet = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2.0)
    singlet_proj = np.outer(singlet, singlet.conj())
    assert np.allclose(werner(1.0), singlet_proj, atol=1e-12)


# ---------------------------------------------------------------------------
# Group 4 — four PT moments reconstruct the exact two-qubit negativity
# ---------------------------------------------------------------------------
def test_negativity_from_four_moments() -> None:
    rng = np.random.default_rng(11072026)
    for _ in range(40):
        # Noisy full-rank 2-qubit states (mix a random state with depolarizing
        # noise) so the PT spectrum is well conditioned for root finding.
        rho = random_density(4, 4, rng)
        rho = depolarize(rho, rng.uniform(0.0, 0.5))
        moments = [1.0, pt_moment(rho, 2), pt_moment(rho, 3), pt_moment(rho, 4)]
        assert negativity_from_moments(moments) == pytest.approx(
            negativity(rho), abs=1e-6
        )


# ---------------------------------------------------------------------------
# Group 5 — GHZ negativity is 1/2 for the balanced bipartition
# ---------------------------------------------------------------------------
def test_ghz_negativity_balanced_bipartition() -> None:
    assert negativity(ghz(3)) == pytest.approx(0.5, abs=1e-9)  # dA=4, dB=2
    assert negativity(ghz(4)) == pytest.approx(0.5, abs=1e-9)  # dA=4, dB=4


def test_default_bipartition_is_first_ceil_half() -> None:
    # GHZ negativity is 0.5 across *any* cut, so it cannot pin *which* qubits
    # form A.  Use a 3-qubit state with a Bell pair on qubits {0,1} and |0> on
    # qubit 2.  The default split (A = first ceil(3/2)=2 qubits = {0,1}) cuts
    # between the (product) Bell pair and |0>, giving negativity 0.  A wrong
    # floor-based split (A = {0}) would cut *through* the Bell pair -> 0.5.
    ket0_proj = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho = np.kron(bell_phi_plus(), ket0_proj)  # qubits {0,1}=Bell, {2}=|0>
    assert negativity(rho) == pytest.approx(0.0, abs=1e-9)  # default: dA=4,dB=2
    assert negativity(rho, dA=2, dB=4) == pytest.approx(0.5, abs=1e-9)  # cut through Bell


# ---------------------------------------------------------------------------
# Group 6 — depolarized Bell state: N(q) = max(0, 0.5 - 0.75 q)
# ---------------------------------------------------------------------------
def test_depolarized_bell_negativity() -> None:
    bell = bell_phi_plus()
    for q in [0.0, 0.1, 0.25, 0.4, 0.5, 2.0 / 3.0, 0.8, 1.0]:
        expected = max(0.0, 0.5 - 0.75 * q)
        assert negativity(depolarize(bell, q)) == pytest.approx(expected, abs=1e-9)
