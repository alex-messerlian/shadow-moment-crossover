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


# ---- PASS 11: closed-form ensemble-averaged k=2 zetas ----
import itertools as _it
import numpy as _np
from anrl.theory.single_copy_law import closed_form_zetas as _cfz
from anrl.benchmark.ensembles import noisy_pure as _noisy_pure


def _pauli_exps(rho, n):
    """Tr(rho P_s) for all Pauli strings s, flat base-4 order (0=I,1=X,2=Y,3=Z)."""
    Mp = _np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]], complex)
    v = rho.astype(complex).reshape((2,) * (2 * n))
    perm = []
    for q in range(n):
        perm += [q, n + q]
    v = _np.transpose(v, perm).reshape((4,) * n)
    for q in range(n):
        v = _np.moveaxis(v, q, 0); sh = v.shape
        v = (Mp @ v.reshape(4, -1)).reshape(sh); v = _np.moveaxis(v, 0, q)
    return v.real.reshape(-1)


def _compat_pairs(n):
    strs = list(_it.product(range(4), repeat=n)); idx = {s: i for i, s in enumerate(strs)}
    S, SP, TRI, OV = [], [], [], []
    for s in strs:
        for sp in strs:
            ov = [i for i in range(n) if s[i] != 0 and sp[i] != 0]
            if all(s[i] == sp[i] for i in ov):
                tri = tuple(s[i] if (s[i] != 0 and sp[i] == 0) else sp[i] if (sp[i] != 0 and s[i] == 0) else 0
                            for i in range(n))
                S.append(idx[s]); SP.append(idx[sp]); TRI.append(idx[tri]); OV.append(len(ov))
    return _np.array(S), _np.array(SP), _np.array(TRI), _np.array(OV)


def _exact_zetas_per_state(state, pairs):
    """Sampling-free per-state (zeta1, zeta2) via HKP Lemma 4 compatible-pair sum."""
    n = state.n; S, SP, TRI, OV = pairs; m = _pauli_exps(state.density_matrix(), n)
    trr2 = float((m ** 2).sum() / 2 ** n)
    z1 = float((3.0 ** OV * m[S] * m[SP] * m[TRI]).sum()) / 4 ** n - trr2 ** 2
    z2 = float((9.0 ** OV * m[TRI] ** 2).sum()) / 4 ** n - trr2 ** 2
    return z1, z2


def test_closed_form_four_structural_counts():
    for n in (2, 3, 4):
        strs = list(_it.product("IXYZ", repeat=n))
        def sup(s): return set(i for i, c in enumerate(s) if c != "I")
        c16 = c10 = c34 = c28 = 0
        for s in strs:
            c10 += 3 ** len(sup(s)); c28 += 9 ** len(sup(s))
            for sp in strs:
                ov = sup(s) & sup(sp)
                if all(s[i] == sp[i] for i in ov):
                    c16 += 3 ** len(ov); c34 += 9 ** len(ov)
        assert c16 == 16 ** n and c10 == 10 ** n and c34 == 34 ** n and c28 == 28 ** n


def test_closed_form_matches_exact_evaluator_ensemble():
    q = 0.1
    for n in (3, 4):
        pairs = _compat_pairs(n)
        N = 250
        z1s = _np.empty(N); z2s = _np.empty(N)
        for s in range(N):
            z1s[s], z2s[s] = _exact_zetas_per_state(_noisy_pure(n, q, _np.random.default_rng([4242, n, s])), pairs)
        cf1, cf2 = _cfz(n, q)
        sem1 = z1s.std(ddof=1) / _np.sqrt(N); sem2 = z2s.std(ddof=1) / _np.sqrt(N)
        assert abs(z1s.mean() - cf1) <= 2 * sem1, (n, z1s.mean(), cf1, sem1)
        assert abs(z2s.mean() - cf2) <= 2 * sem2, (n, z2s.mean(), cf2, sem2)


def test_closed_form_asymptotic_limits():
    q = 0.1
    z1_20, z2_20 = _cfz(20, q)
    assert abs(z2_20 / 7 ** 20 - 1.0) < 1e-3
    mstar = z2_20 / (2 * z1_20)
    assert abs(mstar / 5.6 ** 20 - 1.0 / (2 * (1 - q) ** 2)) < 0.02
