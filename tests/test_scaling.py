"""Tests for the scaling study (single-copy vs collective vs system size).

Pins the noisy-pure purity against its closed form, checks both ensembles'
``purity()`` against the dense ``Tr(rho^2)``, confirms the factored shadow
sampler is unbiased and its exact U-statistic matches a brute-force dense-feature
U-statistic at small n, verifies the collective channel signal matches the
reference construction, and that the scaling grid runs end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.benchmark import (
    NoisyState,
    channel_collective_signal,
    collective_purity_signal,
    dephasing_kraus,
    noisy_pure,
    random_mixed,
    run_scaling,
    save_scaling,
    snapshots_factored,
)
from anrl.benchmark.shadows import _snapshots, full_purity_ustatistic


def _noisy_pure_purity_formula(q: float, n: int) -> float:
    """Closed form for ``(1-q)|psi><psi| + q I/2^n`` (a pure component, Tr P^2=1)."""
    d = 2 ** n
    return (1.0 - q) ** 2 + 2.0 * (1.0 - q) * q / d + q ** 2 / d


# ---------------------------------------------------------------------------
# 1, noisy_pure purity matches the closed form and stays O(1) as n grows
# ---------------------------------------------------------------------------
def test_noisy_pure_purity_matches_formula() -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 5):
        for q in (0.05, 0.1, 0.2, 0.3):
            state = noisy_pure(n, q, rng)
            assert state.purity() == pytest.approx(_noisy_pure_purity_formula(q, n), abs=1e-12)

    # The whole point of the ensemble: purity stays O(1) (does NOT collapse to 2^-n).
    rng = np.random.default_rng(1)
    for n in (2, 4, 6, 8):
        p = noisy_pure(n, 0.1, rng).purity()
        assert p > 0.75  # ~ (1-q)^2 = 0.81, essentially flat in n


# ---------------------------------------------------------------------------
# 2, both ensembles' analytic purity() equals the dense Tr(rho^2)
# ---------------------------------------------------------------------------
def test_purity_matches_dense_trace() -> None:
    rng = np.random.default_rng(2)
    for maker in (noisy_pure, random_mixed):
        for n in (1, 2, 3):
            for q in (0.0, 0.1, 0.4):
                state = maker(n, q, rng)
                rho = state.density_matrix()
                dense = float(np.trace(rho @ rho).real)
                assert state.purity() == pytest.approx(dense, abs=1e-12)
                assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3, random_mixed (full rank) collapses toward 2^-n; noisy_pure does not
# ---------------------------------------------------------------------------
def test_random_mixed_purity_collapses() -> None:
    rng = np.random.default_rng(3)
    prev = 1.0
    for n in (2, 4, 6, 8):
        p = random_mixed(n, 0.3, rng).purity()
        # Monotonically shrinking and bounded below by the maximally-mixed 2^-n.
        assert p < prev
        assert p >= 2.0 ** (-n) - 1e-12
        prev = p
    # By n=8 it is within a small factor of the maximally-mixed floor.
    assert random_mixed(8, 0.3, rng).purity() < 0.05


# ---------------------------------------------------------------------------
# 4, factored sampler + O(M^2 n) U-statistic == dense sampler + brute U-stat
# ---------------------------------------------------------------------------
def _brute_force_ustatistic(snaps: np.ndarray) -> float:
    """Reference: full pairwise U-statistic via explicit per-qubit trace products."""
    m, n = snaps.shape[0], snaps.shape[1]
    total = 0.0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            prod = 1.0
            for q in range(n):
                prod *= float(np.trace(snaps[i, q] @ snaps[j, q]).real)
            total += prod
    return total / (m * (m - 1))


def test_ustatistic_matches_brute_force() -> None:
    rng = np.random.default_rng(4)
    for n in (1, 2, 3):
        state = noisy_pure(n, 0.15, rng)
        snaps = snapshots_factored(state, 40, rng)
        assert full_purity_ustatistic(snaps) == pytest.approx(_brute_force_ustatistic(snaps), abs=1e-9)


def test_factored_sampler_unbiased() -> None:
    # E[shadow U-statistic] = Tr(rho^2); the factored sampler must reproduce the
    # true purity in the large-sample mean (checked against the dense sampler too).
    rng = np.random.default_rng(5)
    state = noisy_pure(2, 0.2, rng)
    truth = state.purity()
    factored = [full_purity_ustatistic(snapshots_factored(state, 400, rng)) for _ in range(40)]
    assert abs(float(np.mean(factored)) - truth) < 0.03

    # The dense reference sampler (diag of U rho U^dag) must agree in the mean.
    rho = state.density_matrix()
    dense = [full_purity_ustatistic(_snapshots(rho, 2, 400, rng)) for _ in range(40)]
    assert abs(float(np.mean(dense)) - truth) < 0.03


# ---------------------------------------------------------------------------
# 5, collective channel signal matches the reference channel construction
# ---------------------------------------------------------------------------
def test_collective_signal_matches_reference() -> None:
    rng = np.random.default_rng(6)
    for n in (1, 2, 3):
        state = random_mixed(n, 0.25, rng)
        rho = state.density_matrix()
        for g in (0.02, 0.05, 0.1):
            got = collective_purity_signal(state, "dephasing", g)
            ref = channel_collective_signal(rho, 2, dephasing_kraus(g), n)
            assert got == pytest.approx(ref, abs=1e-10)
        # Depolarizing closed form.
        p = 0.1
        assert collective_purity_signal(state, "depolarizing", p) == pytest.approx(
            (1.0 - p) * state.purity() + p / state.dim, abs=1e-12
        )


# ---------------------------------------------------------------------------
# 6; the scaling grid runs end to end and saves a results file
# ---------------------------------------------------------------------------
def test_scaling_runs_and_saves(tmp_path) -> None:
    rows = run_scaling(
        ensembles=("noisy_pure", "random_mixed"),
        sizes=(2, 3),
        noise_models=("depolarizing", "dephasing"),
        rates=(0.0, 0.1),
        budget=400,
        n_states=4,
        seed=0,
    )
    # 2 ensembles x 2 sizes x 2 noise x 2 rates = 16 rows.
    assert len(rows) == 16
    for row in rows:
        assert row["ensemble"] in ("noisy_pure", "random_mixed")
        assert row["winner"] in ("collective", "single-copy")
        assert row["single_rmse"] >= 0.0
        assert row["collective_rmse"] >= 0.0
        assert row["budget"] == 400
        assert 0.0 <= row["mean_true_purity"] <= 1.0 + 1e-9

    # Single-copy RMSE is noise-independent for a fixed (ensemble, n): same value
    # across every noise/rate cell of that (ensemble, n).
    by_key: dict[tuple[str, int], set[float]] = {}
    for row in rows:
        by_key.setdefault((row["ensemble"], row["n"]), set()).add(round(row["single_rmse"], 12))
    for key, values in by_key.items():
        assert len(values) == 1, f"single_rmse varies within {key}: {values}"

    # Determinism: a second run with the same seed reproduces the grid exactly.
    rows2 = run_scaling(
        ensembles=("noisy_pure", "random_mixed"),
        sizes=(2, 3),
        noise_models=("depolarizing", "dephasing"),
        rates=(0.0, 0.1),
        budget=400,
        n_states=4,
        seed=0,
    )
    assert [r["single_rmse"] for r in rows] == [r["single_rmse"] for r in rows2]
    assert [r["collective_rmse"] for r in rows] == [r["collective_rmse"] for r in rows2]

    out = tmp_path / "scaling.json"
    save_scaling(rows, out, {"budget": 400})
    assert out.exists()
    import json

    payload = json.loads(out.read_text())
    assert len(payload["rows"]) == 16
    assert payload["metadata"]["budget"] == 400


# ---------------------------------------------------------------------------
# 7, boundary validation: fail fast on unphysical / unsupported inputs
# ---------------------------------------------------------------------------
def test_input_validation() -> None:
    rng = np.random.default_rng(7)

    # q outside [0, 1] is unphysical (would give purity > 1).
    for bad_q in (-0.1, 1.5):
        with pytest.raises(ValueError):
            noisy_pure(2, bad_q, rng)
        with pytest.raises(ValueError):
            random_mixed(2, bad_q, rng)

    # rank must be >= 1.
    with pytest.raises(ValueError):
        random_mixed(2, 0.1, rng, rank=0)

    # The purity U-statistic needs >= 2 snapshots (else 0/0 -> NaN).
    snaps = snapshots_factored(noisy_pure(2, 0.1, rng), 1, rng)
    with pytest.raises(ValueError):
        full_purity_ustatistic(snaps)

    # Collective signal rejects out-of-range rate and oversized dense-channel n.
    state = noisy_pure(2, 0.1, rng)
    with pytest.raises(ValueError):
        collective_purity_signal(state, "dephasing", 1.5)
    big = noisy_pure(1, 0.1, rng)
    object.__setattr__(big, "n", 14)  # pretend it's large; guard must trip before alloc
    with pytest.raises(ValueError):
        collective_purity_signal(big, "amplitude_damping", 0.05)


def test_components_are_read_only() -> None:
    # frozen=True does not stop in-place array mutation; __post_init__ must.
    state = noisy_pure(2, 0.1, np.random.default_rng(8))
    with pytest.raises(ValueError):  # numpy raises on write to a read-only array
        state.components[0, 0] = 999.0 + 0j
