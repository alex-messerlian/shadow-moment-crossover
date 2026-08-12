"""Tests locking the pilot estimator of the projection variances (:mod:`anrl.theory.pilot_zetas`).

The estimator's whole point is that it reaches ``M*`` from snapshots alone, without forming the
``4^n`` Pauli spectrum the exact functional needs.  The properties that make it trustworthy are:

* the feature map is an exact rewriting of ``Tr(G_i G_j)``, not an approximation;
* both estimators are unbiased, which is what lets a pilot be believed at small budgets;
* the chunking that bounds memory is a pure re-association and changes no value;
* ``zeta_1`` can come out non-positive at small budgets, and the ratio must then be ``nan``
  rather than a huge number.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.ensembles import low_rank, noisy_pure
from anrl.physics import kron_all
from anrl.theory import pilot_zetas as pz
from anrl.theory.general import sample_batched_general
from anrl.theory.pilot_zetas import (
    feature_chunk,
    pair_traces,
    pilot_m_star,
    pilot_zetas,
    snapshot_features,
)
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights


@pytest.mark.parametrize("n", [1, 2, 3])
def test_feature_map_reproduces_dense_pair_trace(n):
    """Tr(G_i G_j) = 2^n <x_i, x_j> exactly, against dense matrix products."""
    state = noisy_pure(n, 0.1, np.random.default_rng([1, n]))
    snaps = sample_batched_general(state, 5, np.random.default_rng([2, n]))
    x = snapshot_features(snaps)
    k_feat = (2.0 ** n) * (x @ x.T)
    dense = [kron_all([snaps[i, q] for q in range(n)]) for i in range(5)]
    k_ref = np.array([[np.trace(dense[i] @ dense[j]).real for j in range(5)] for i in range(5)])
    assert np.abs(k_feat - k_ref).max() == pytest.approx(0.0, abs=1e-11)


@pytest.mark.parametrize("n", [2, 3])
def test_pair_traces_matches_feature_map(n):
    """The O(M n) per-qubit pair product agrees with the feature inner product."""
    state = low_rank(n, 2, np.random.default_rng([3, n]))
    snaps = sample_batched_general(state, 12, np.random.default_rng([4, n]))
    a, b = snaps[:6], snaps[6:]
    xa, xb = snapshot_features(a), snapshot_features(b)
    assert pair_traces(a, b) == pytest.approx((2.0 ** n) * np.einsum("ij,ij->i", xa, xb), rel=1e-11)


def test_chunking_does_not_change_the_value(monkeypatch):
    """The memory cap is a pure re-association: every chunk size gives the same estimate."""
    state = noisy_pure(4, 0.1, np.random.default_rng(7))
    snaps = sample_batched_general(state, 2000, np.random.default_rng(8))
    ref = pilot_zetas(snaps)
    for cap in (1 << 18, 1 << 22, 1 << 30):
        monkeypatch.setattr(pz, "_TRANSIENT_BYTES", cap)
        got = pilot_zetas(snaps)
        assert got[0] == pytest.approx(ref[0], rel=1e-12)
        assert got[1] == pytest.approx(ref[1], rel=1e-12)


def test_feature_chunk_respects_the_cap_and_the_floor():
    for n in range(1, 9):
        c = feature_chunk(n)
        assert c >= 256
        assert c == 256 or 8 * c * 4 ** n <= pz._TRANSIENT_BYTES


@pytest.mark.parametrize("n", [2, 3])
def test_estimators_are_unbiased(n):
    """Averaged over independent pilots, both estimates converge on the exact values."""
    state = noisy_pure(n, 0.1, np.random.default_rng([5, n]))
    m = pauli_expectations(state.density_matrix(), n)
    z1_ex, z2_ex = exact_zeta1(m, n), exact_zeta2(m, n, pauli_weights(n))
    rng = np.random.default_rng([6, n])
    reps = 120
    z1s, z2s = np.empty(reps), np.empty(reps)
    for i in range(reps):
        z1s[i], z2s[i] = pilot_zetas(sample_batched_general(state, 4000, rng))
    for got, exact in ((z1s, z1_ex), (z2s, z2_ex)):
        sem = got.std(ddof=1) / np.sqrt(reps)
        assert abs(got.mean() - exact) <= 3 * sem, (got.mean(), exact, sem)


def test_m_star_is_nan_when_zeta1_is_nonpositive(monkeypatch):
    """A non-positive zeta_1 estimate makes the ratio undefined, not enormous."""
    monkeypatch.setattr(pz, "pilot_zetas", lambda snaps: (-0.5, 40.0))
    snaps = sample_batched_general(noisy_pure(2, 0.1, np.random.default_rng(9)), 8,
                                  np.random.default_rng(10))
    assert np.isnan(pz.pilot_m_star(snaps))


def test_m_star_matches_the_ratio_of_the_two_estimates():
    state = noisy_pure(3, 0.1, np.random.default_rng(11))
    snaps = sample_batched_general(state, 4000, np.random.default_rng(12))
    z1, z2 = pilot_zetas(snaps)
    assert pilot_m_star(snaps) == pytest.approx(z2 / (2 * z1), rel=1e-12)


def test_too_few_snapshots_raises():
    state = noisy_pure(2, 0.1, np.random.default_rng(13))
    with pytest.raises(ValueError, match="at least 4"):
        pilot_zetas(sample_batched_general(state, 3, np.random.default_rng(14)))
