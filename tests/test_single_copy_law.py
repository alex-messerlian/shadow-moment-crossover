"""Tests locking the first-principles single-copy variance law (Part 2, derived).

These encode the derivation as executable checks: the exact k=2 Hoeffding formula
(vs the general Lee formula and vs brute-force Monte Carlo), the corrected crossover
M* = zeta2/(2 zeta1), the alpha transition, and the single-qubit closed form.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.ensembles import noisy_pure
from anrl.benchmark.shadows import _snapshots, full_purity_ustatistic
from anrl.theory.single_copy_law import (
    crossover_budget,
    hoeffding_rmse,
    hoeffding_variance,
    predicted_alpha,
    single_qubit_second_moment,
    single_qubit_zeta1,
)
from anrl.theory.variance import exact_ustatistic_variance


@pytest.mark.parametrize("m", [2, 3, 4, 6, 10, 50, 2000])
def test_hoeffding_equals_general_lee_formula(m):
    """The explicit k=2 form equals the general Hoeffding/Lee U-statistic variance."""
    z1, z2 = 1.3, 47.0
    assert hoeffding_variance(m, z1, z2) == pytest.approx(
        exact_ustatistic_variance([z1, z2], 2, m), rel=1e-12
    )


def test_hoeffding_matches_brute_force_mc():
    """(*) matches the variance of the actual full U-statistic (small-N MC, n=1)."""
    rho = noisy_pure(1, 0.1, np.random.default_rng(1)).density_matrix()
    rng = np.random.default_rng(0)
    # zeta1, zeta2 from an independent large sample
    from anrl.physics import kron_all

    snaps = _snapshots(rho, 1, 200_000, rng)
    tr_grho = np.einsum("mij,ji->m", snaps[:, 0], rho).real
    z1 = tr_grho.var(ddof=1)
    sa = _snapshots(rho, 1, 200_000, rng)
    sb = _snapshots(rho, 1, 200_000, rng)
    z2 = (np.einsum("mij,mji->m", sa[:, 0], sb[:, 0]).real).var(ddof=1)
    # brute-force estimator variance at M=6
    M, reps = 6, 6000
    ests = np.array([full_purity_ustatistic(_snapshots(rho, 1, M, rng)) for _ in range(reps)])
    brute = ests.var(ddof=1)
    assert brute == pytest.approx(hoeffding_variance(M, z1, z2), rel=0.10)


def test_crossover_is_exact_formula_balance():
    """M* = zeta2/(2 zeta1); at M = M* the two asymptotic terms (4 z1/M, 2 z2/M^2) balance."""
    z1, z2 = 1.5, 300.0
    ms = crossover_budget(z1, z2)
    assert ms == pytest.approx(z2 / (2 * z1))
    # asymptotic linear term 4 z1/M vs higher-order 2 z2/M^2 are equal at M*
    assert (4 * z1 / ms) == pytest.approx(2 * z2 / (ms * ms), rel=1e-12)
    # and it is DOUBLE the two-term model's zeta2/(4 zeta1)
    assert ms == pytest.approx(2.0 * (z2 / (4 * z1)))


def test_alpha_transition_limits():
    """alpha -> 1/2 for M >> M*, -> 1 for M << M* (M still large so M(M-1)~=M^2),
    and interpolates in between. Budgets are large in absolute terms, matching the
    real experiment where finite-M curvature is negligible."""
    z1, z2 = 1.0, 1.0e8  # M* = 5e7
    a_big = predicted_alpha([10_000_000_000, 40_000_000_000, 160_000_000_000], z1, z2)
    a_small = predicted_alpha([2000, 8000, 32000], z1, z2)
    a_mid = predicted_alpha([25_000_000, 50_000_000, 100_000_000], z1, z2)
    assert a_big == pytest.approx(0.5, abs=0.03)
    assert a_small == pytest.approx(1.0, abs=0.02)
    assert 0.5 < a_mid < 1.0


def test_single_qubit_identity_closed_form():
    """single_qubit_second_moment / zeta1 closed forms are self-consistent and MC-accurate."""
    for t in (0.0, 0.4, 0.7, 1.0):
        p = (1.0 + t * t) / 2.0
        assert single_qubit_second_moment(t) == pytest.approx(2.5 * p - 1.0)
        assert single_qubit_zeta1(t) == pytest.approx(single_qubit_second_moment(t) - p * p)
        assert single_qubit_zeta1(t) == pytest.approx(0.75 * t * t - 0.25 * t ** 4)


def test_single_qubit_identity_vs_mc():
    """E[Tr(G r)^2] = 1/4 + 5/4 t^2 against Monte-Carlo shadows (t = 1, pure |0>)."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)  # t = 1, p = 1
    snaps = _snapshots(rho, 1, 400_000, np.random.default_rng(3))
    mc = float((np.einsum("mij,ji->m", snaps[:, 0], rho).real ** 2).mean())
    assert mc == pytest.approx(single_qubit_second_moment(1.0), abs=5e-3)
