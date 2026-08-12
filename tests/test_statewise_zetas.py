"""Tests locking the EXACT per-state projection variances (:mod:`anrl.theory.statewise_zetas`).

The ensemble closed form ``closed_form_zetas(n, q)`` is an average over the noisy-pure
family.  This module evaluates the same two identities pointwise for an arbitrary state, so
the checks here are the statewise limits where the answer is known independently:

* the maximally mixed state, where ``zeta_1 = 0`` exactly and ``zeta_2 = 7^n - 4^-n``;
* one qubit, where ``zeta_1 = 3/4 t^2 - 1/4 t^4`` in closed form (Section 3.3);
* the pure product and GHZ states, whose ``zeta_2`` the paper gives in closed form;
* the general bounds ``7^n - 1 <= zeta_2 <= (17/2)^n``;
* Haar-averaging back onto ``closed_form_zetas``, which ties the statewise evaluator to the
  committed ensemble result.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.ensembles import ghz_noisy, haar_pure, low_rank, noisy_pure
from anrl.theory.single_copy_law import closed_form_zetas, single_qubit_zeta1
from anrl.theory.statewise_zetas import (
    exact_m_star,
    exact_zeta1,
    exact_zeta2,
    exact_zetas,
    pauli_expectations,
    pauli_weights,
    purity_from_expectations,
    truncated_zeta2,
    zeta1_diagonal,
)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_maximally_mixed_limits(n):
    """zeta_1 vanishes identically and zeta_2 = 7^n - 4^-n (Section 2.5)."""
    z1, z2 = exact_zetas(np.eye(2 ** n) / 2 ** n, n)
    assert z1 == pytest.approx(0.0, abs=1e-10)
    assert z2 == pytest.approx(7.0 ** n - 4.0 ** -n, rel=1e-12)


@pytest.mark.parametrize("t", [0.0, 0.3, 0.7, 1.0])
def test_single_qubit_closed_form(t):
    """At n = 1 the evaluator reproduces zeta_1 = 3/4 t^2 - 1/4 t^4 exactly."""
    rho = 0.5 * np.array([[1 + t, 0], [0, 1 - t]], dtype=complex)
    z1, _ = exact_zetas(rho, 1)
    assert z1 == pytest.approx(single_qubit_zeta1(t), abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_product_and_ghz_zeta2_closed_forms(n):
    """zeta_2 = (15/2)^n - 1 for a pure product state and the GHZ form of Section 3.5."""
    d = 2 ** n
    psi = np.zeros(d)
    psi[0] = 1.0
    _, z2_prod = exact_zetas(np.outer(psi, psi).astype(complex), n)
    assert z2_prod == pytest.approx((15 / 2) ** n - 1.0, rel=1e-11)

    ghz = np.zeros(d)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    _, z2_ghz = exact_zetas(np.outer(ghz, ghz).astype(complex), n)
    assert z2_ghz == pytest.approx(0.5 * ((15 / 2) ** n + (13 / 2) ** n) - 0.5, rel=1e-11)


@pytest.mark.parametrize("n", [2, 3])
def test_zeta2_general_bounds(n):
    """7^n - 1 <= zeta_2 <= (17/2)^n for every state, the Section 3.5 base bounds."""
    rng = np.random.default_rng([7, n])
    states = [haar_pure(n, rng), noisy_pure(n, 0.1, rng), noisy_pure(n, 0.4, rng),
              ghz_noisy(n, 0.15, rng), low_rank(n, 2, rng), low_rank(n, 3, rng)]
    states.append(None)  # sentinel for the maximally mixed state, handled below
    for st in states:
        rho = np.eye(2 ** n) / 2 ** n if st is None else st.density_matrix()
        z2 = exact_zeta2(pauli_expectations(rho, n), n)
        assert 7.0 ** n - 1.0 - 1e-9 <= z2 <= (17 / 2) ** n + 1e-9


@pytest.mark.parametrize("n,q", [(2, 0.1), (3, 0.1), (4, 0.1), (3, 0.3)])
def test_haar_average_reproduces_ensemble_closed_form(n, q):
    """Averaging the statewise evaluator over the noisy-pure family returns closed_form_zetas."""
    weights = pauli_weights(n)
    N = 300
    z1 = np.empty(N)
    z2 = np.empty(N)
    for s in range(N):
        m = pauli_expectations(noisy_pure(n, q, np.random.default_rng([88, n, s])).density_matrix(), n)
        z1[s] = exact_zeta1(m, n)
        z2[s] = exact_zeta2(m, n, weights)
    cf1, cf2 = closed_form_zetas(n, q)
    assert abs(z1.mean() - cf1) <= 3 * z1.std(ddof=1) / np.sqrt(N)
    assert abs(z2.mean() - cf2) <= 3 * z2.std(ddof=1) / np.sqrt(N)


def test_purity_and_diagonal_consistency():
    """The Pauli transform is unitary-consistent and the diagonal bounds nothing it should not."""
    rng = np.random.default_rng(5)
    st = noisy_pure(3, 0.15, rng)
    rho = st.density_matrix()
    m = pauli_expectations(rho, 3)
    assert purity_from_expectations(m, 3) == pytest.approx(st.purity(), rel=1e-12)
    # The weight-only diagonal is a partial sum of a positive-term series, so it is positive
    # and, for this state, differs from the full cubic sum -- the Section 3.4 obstruction.
    diag = zeta1_diagonal(m, 3)
    full = exact_zeta1(m, 3) + purity_from_expectations(m, 3) ** 2
    assert diag > 0
    assert abs(full - diag) / diag > 1e-3


def test_truncated_zeta2_converges_at_full_weight():
    """Truncating the spectral sum at max_weight = n is the untruncated value."""
    m = pauli_expectations(low_rank(3, 2, np.random.default_rng(9)).density_matrix(), 3)
    assert truncated_zeta2(m, 3, 3) == pytest.approx(exact_zeta2(m, 3), rel=1e-12)


def test_max_tail_chunking_is_exact():
    """The head/tail split is a pure reorganization: every max_tail gives the same value."""
    m = pauli_expectations(noisy_pure(4, 0.1, np.random.default_rng(3)).density_matrix(), 4)
    ref = exact_zeta1(m, 4, max_tail=4)
    for tail in (1, 2, 3, 6):
        assert exact_zeta1(m, 4, max_tail=tail) == pytest.approx(ref, rel=1e-12)


def test_exact_m_star_matches_ratio():
    rho = noisy_pure(3, 0.1, np.random.default_rng(11)).density_matrix()
    z1, z2 = exact_zetas(rho, 3)
    assert exact_m_star(rho, 3) == pytest.approx(z2 / (2 * z1), rel=1e-12)
    assert exact_m_star(np.eye(8) / 8, 3) == float("inf")
