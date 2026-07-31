"""Tests for the range-constrained moment estimators of :mod:`anrl.benchmark.constrained`."""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.constrained import (
    clip_moment,
    physical_range,
    shrink_moment,
    shrinkage_coefficient,
)
from anrl.benchmark.scaling import ENSEMBLES, _ENSEMBLE_ID, snapshots_factored
from anrl.benchmark.shadows import full_purity_ustatistic


@pytest.mark.parametrize("n,k,lo", [(1, 2, 0.5), (2, 2, 0.25), (10, 2, 2.0 ** -10),
                                    (2, 3, 2.0 ** -4), (3, 4, 2.0 ** -9)])
def test_physical_range_endpoints(n, k, lo):
    """The range is [2^{n(1-k)}, 1] -- the maximally mixed and pure values."""
    a, b = physical_range(n, k)
    assert a == pytest.approx(lo)
    assert b == 1.0


def test_physical_range_rejects_bad_arguments():
    with pytest.raises(ValueError):
        physical_range(0, 2)
    with pytest.raises(ValueError):
        physical_range(2, 1)


def test_clip_is_identity_inside_the_range():
    """CLIPPED reduces to the raw estimator whenever the estimate is feasible."""
    n, k = 4, 2
    lo, hi = physical_range(n, k)
    inside = np.linspace(lo, hi, 101)
    assert np.allclose(clip_moment(inside, n, k), inside)


def test_clip_projects_outside_values_to_the_boundary():
    n, k = 4, 2
    lo, hi = physical_range(n, k)
    assert clip_moment(-3.0, n, k) == pytest.approx(lo)
    assert clip_moment(17.0, n, k) == pytest.approx(hi)


def test_shrink_is_identity_when_sigma_is_zero():
    """With no estimator noise the shrinkage weight vanishes and nothing moves."""
    n, k = 4, 2
    inside = np.linspace(*physical_range(n, k), 51)
    assert np.allclose(shrink_moment(inside, n, k, 0.0), inside)


def test_shrinkage_coefficient_is_a_weight():
    n, k = 5, 2
    a = shrinkage_coefficient(np.linspace(-5, 5, 201), n, k, 0.7)
    assert np.all((a >= 0.0) & (a <= 1.0))


def test_shrinkage_pulls_toward_the_floor():
    """Large sigma relative to the gap sends the estimate to the floor."""
    n, k = 6, 2
    lo, _ = physical_range(n, k)
    assert shrink_moment(0.9, n, k, 1e6) == pytest.approx(lo, abs=1e-6)


def test_clipping_never_increases_squared_error_on_real_snapshots():
    """The contraction property, checked on genuine shadow realizations.

    Projection onto a convex set containing the truth cannot move an estimate
    away from it, so the squared error is weakly reduced on EVERY sample.  This
    is the property that makes clipping free, and it is what makes the raw
    RMSE at large n a statement about an estimator nobody would deploy.
    """
    rng_seed, q, budget = 0, 0.1, 2000
    worse = 0
    total = 0
    for n in (4, 7):
        eid = _ENSEMBLE_ID["noisy_pure"]
        state = ENSEMBLES["noisy_pure"](n, q, np.random.default_rng([rng_seed, eid, n, 0, 0]))
        truth = state.purity()
        rng = np.random.default_rng([rng_seed, eid, n, 0, 1])
        for _ in range(6):
            raw = full_purity_ustatistic(snapshots_factored(state, budget, rng))
            clipped = float(clip_moment(raw, n, 2))
            total += 1
            if (clipped - truth) ** 2 > (raw - truth) ** 2 + 1e-15:
                worse += 1
    assert total > 0
    assert worse == 0


def test_clipped_rmse_cannot_exceed_the_range_width():
    """A clipped estimator's RMSE is bounded by the interval width, which is < 1.

    This is why the paper's raw RMSE of ~12 at n = 10 is unreachable for any
    pipeline that projects into the physical range.
    """
    n, k = 10, 2
    lo, hi = physical_range(n, k)
    rng = np.random.default_rng(0)
    raw = rng.normal(0.81, 12.0, size=20000)  # the n=10 noise scale
    err = clip_moment(raw, n, k) - 0.81
    assert np.sqrt(np.mean(err ** 2)) < (hi - lo)
