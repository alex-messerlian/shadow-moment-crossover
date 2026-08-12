"""Tests locking the two varying-estimand ensembles and paper one's theory figures.

:func:`~anrl.benchmark.ensembles.variable_q` and
:func:`~anrl.benchmark.ensembles.variable_rank` exist for one reason: on the four committed
families the statewise threshold barely varies, so a per-state claim cannot be tested on them.
The properties that have to hold are therefore quantitative, not merely structural -- the new
families must actually spread the estimand and the threshold -- and the tests check that rather
than only that the constructors return valid states.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark.ensembles import (
    ghz_noisy,
    haar_pure,
    low_rank,
    noisy_pure,
    variable_q,
    variable_rank,
)
from anrl.benchmark.moments import moment
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights


def _m_stars_and_moments(make, n: int, count: int, salt: int):
    w = pauli_weights(n)
    ms, mom = [], []
    for s in range(count):
        st = make(n, np.random.default_rng([99, salt, n, s]))
        rho = st.density_matrix()
        m = pauli_expectations(rho, n)
        z1, z2 = exact_zeta1(m, n), exact_zeta2(m, n, w)
        ms.append(z2 / (2 * z1))
        mom.append(moment(rho, 2))
    return np.array(ms), np.array(mom)


@pytest.mark.parametrize("make,salt", [(variable_q, 1), (variable_rank, 2)])
def test_new_ensembles_produce_valid_states(make, salt):
    for n in (2, 3, 4):
        for s in range(6):
            rho = make(n, np.random.default_rng([99, salt, n, s])).density_matrix()
            ev = np.linalg.eigvalsh(rho)
            assert ev.min() > -1e-10
            assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
            assert np.abs(rho - rho.conj().T).max() < 1e-12


@pytest.mark.parametrize("make,salt", [(variable_q, 1), (variable_rank, 2)])
def test_new_ensembles_are_reproducible_from_the_generator(make, salt):
    """Same seed, same state: the drawn parameter must come from the passed generator."""
    for n in (3, 4):
        a = make(n, np.random.default_rng([99, salt, n, 0])).density_matrix()
        b = make(n, np.random.default_rng([99, salt, n, 0])).density_matrix()
        assert np.allclose(a, b)


def test_committed_families_have_a_fixed_estimand():
    """noisy_pure, haar_pure and ghz_noisy have zero within-family estimand spread."""
    n = 4
    for make, salt in ((lambda k, r: noisy_pure(k, 0.1, r), 3),
                       (lambda k, r: haar_pure(k, r), 4),
                       (lambda k, r: ghz_noisy(k, 0.15, r), 5)):
        _, mom = _m_stars_and_moments(make, n, 8, salt)
        assert mom.std(ddof=1) / mom.mean() < 1e-12


def test_new_ensembles_actually_vary_the_estimand():
    """Both new families give a genuinely random moment, unlike the committed three."""
    for make, salt, floor in ((variable_q, 1, 0.05), (variable_rank, 2, 0.20)):
        _, mom = _m_stars_and_moments(make, 4, 16, salt)
        assert mom.std(ddof=1) / mom.mean() > floor


def test_variable_rank_spreads_the_threshold_far_more_than_rank_two():
    """The point of the family: an order-of-magnitude statewise M* range."""
    n = 4
    vr, _ = _m_stars_and_moments(variable_rank, n, 24, 2)
    lr, _ = _m_stars_and_moments(lambda k, r: low_rank(k, 2, r), n, 24, 6)
    np_, _ = _m_stars_and_moments(lambda k, r: noisy_pure(k, 0.1, r), n, 24, 3)
    assert vr.max() / vr.min() > 4.0
    assert np_.max() / np_.min() < 1.35
    assert vr.max() / vr.min() > 2.0 * (lr.max() / lr.min())


def test_variable_q_respects_its_bounds_and_validates_them():
    for s in range(20):
        st = variable_q(3, np.random.default_rng([99, 7, s]), low=0.2, high=0.3)
        assert 0.2 <= st.q <= 0.3
    with pytest.raises(ValueError, match="0 <= low <= high <= 1"):
        variable_q(3, np.random.default_rng(0), low=0.6, high=0.4)


def test_variable_rank_stays_within_the_dimension_and_validates_max_rank():
    for n in (1, 2, 3):
        for s in range(8):
            st = variable_rank(n, np.random.default_rng([99, 8, n, s]))
            assert 1 <= st.components.shape[1] <= min(8, 2 ** n)
    with pytest.raises(ValueError, match="max_rank must be >= 1"):
        variable_rank(3, np.random.default_rng(0), max_rank=0)


# ------------------------------------------------------------------ paper-one figures
_FIG_INPUTS = ("pass47_statewise_ranking.json", "pass47_pilot_estimator.json",
               "pass47_statewise_mstar.json")


def test_theory_figures_build_with_data_and_captions():
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    if not all((repo / "results" / f).exists() for f in _FIG_INPUTS):
        pytest.skip("needs the generated PASS 47 results/*.json")
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from anrl.figures import apply_style
    from anrl.figures.figures_theory import THEORY_FIGURES

    apply_style()
    assert len(THEORY_FIGURES) == 4
    for name, make in THEORY_FIGURES.items():
        fig, header, rows, caption = make()
        assert rows, f"{name} produced no data rows"
        for r in rows:
            assert len(r) == len(header), f"{name} row width != header width"
        assert caption.startswith("Figure "), name
        # Draft captions become LaTeX \caption{} bodies, so a bare percent would break the build.
        assert "\\%" in caption or "%" not in caption, f"{name} has an unescaped percent"
        plt.close(fig)
