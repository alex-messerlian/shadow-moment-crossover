"""Tests: the general Hoeffding/Lee U-statistic variance holds at k=3 and k=4.

Locks (a) the state-agnostic Hoeffding components (nested projections, incl. the
zeta_2 != kernel-variance distinction at k>=3), and (b) that the Lee formula fed by
them matches the brute-force variance of the EXACT estimator. Modest MC (generous
tolerances) so the suite stays fast but would catch a wrong projection or formula.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from anrl.benchmark.ensembles import low_rank, noisy_pure
from anrl.benchmark.moment_ustats import exact_moment_ustatistic
from anrl.theory.general import sample_batched_general
from anrl.theory.general_k import _arrangements, hoeffding_components_mc
from anrl.theory.variance import estimate_hoeffding_components, exact_ustatistic_variance


def _brute_var(state, k, M, reps, rng):
    ests = np.array([exact_moment_ustatistic(sample_batched_general(state, M, rng), k) for _ in range(reps)])
    return ests.var(ddof=1)


@pytest.mark.parametrize("k,make,tol", [
    (3, lambda: noisy_pure(2, 0.1, np.random.default_rng(1)), 0.07),
    (3, lambda: low_rank(2, 2, np.random.default_rng(2)), 0.07),
    (4, lambda: noisy_pure(2, 0.1, np.random.default_rng(3)), 0.10),
])
def test_lee_formula_matches_brute_force(k, make, tol):
    """exact_ustatistic_variance(zeta_c, k, M) == Var of the exact estimator (within MC error).

    A gross error (wrong projection / zeta_2-vs-kernel-variance conflation) would be off by
    tens of percent; the per-k tolerance only absorbs the variance-of-variance MC scatter of
    the heavy-tailed estimator at these rep counts."""
    state = make()
    comps = hoeffding_components_mc(state, k, 150_000, np.random.default_rng([7, k]))
    M, reps = 20, 12000
    brute = _brute_var(state, k, M, reps, np.random.default_rng([8, k]))
    formula = exact_ustatistic_variance(comps, k, M)
    assert brute == pytest.approx(formula, rel=tol)


def test_mc_components_match_noisy_pure_closed_form_k3():
    """State-agnostic nested-MC zeta_c == the noisy-pure closed-form estimator (k=3)."""
    st = noisy_pure(3, 0.1, np.random.default_rng(4))
    mc = hoeffding_components_mc(st, 3, 200_000, np.random.default_rng([9, 3]))
    closed = estimate_hoeffding_components(st, 3, 300_000, np.random.default_rng([10, 3]))
    for a, b in zip(mc, closed):
        assert a == pytest.approx(b, rel=0.05)


def test_zeta1_is_first_projection():
    """zeta_1 = Var[Re Tr(G rho^{k-1})]; the first-order projection, not the kernel var."""
    st = noisy_pure(2, 0.1, np.random.default_rng(5))
    rho = st.density_matrix()
    rng = np.random.default_rng(6)
    from anrl.theory.general_k import dense_snapshots
    G = dense_snapshots(sample_batched_general(st, 150_000, rng))
    rho2 = rho @ rho
    direct = np.var(np.einsum("mij,ji->m", G, rho2).real, ddof=1)  # Re Tr(G rho^2), k=3
    comp1 = hoeffding_components_mc(st, 3, 150_000, np.random.default_rng([11, 3]))[0]
    assert comp1 == pytest.approx(direct, rel=0.05)


def test_arrangement_counts():
    """Distinct multiset arrangements number k!/(k-c)! with multiplicities summing to k!."""
    for k in (2, 3, 4):
        for c in range(1, k + 1):
            arr = _arrangements(k, c)
            assert len(arr) == math.factorial(k) // math.factorial(k - c)
            assert sum(m for _, m in arr) == math.factorial(k)
