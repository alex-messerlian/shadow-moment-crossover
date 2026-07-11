"""Tests for the moment-family / multi-noise robustness sweep.

Pins the cyclic-permutation identity, the depolarizing signal formula against an
explicit construction, the Kraus-channel anchor values, shadow-moment
unbiasedness for k=2,3,4, and that the sweep runs end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from anrl.physics import depolarize, random_density
from anrl.benchmark import (
    amplitude_damping_kraus,
    channel_collective_signal,
    cyclic_permutation_operator,
    dephasing_kraus,
    depolarizing_moment_signal,
    explicit_channel_collective_signal,
    kron_power,
    moment,
    run_sweep,
    save_sweep,
    shadow_moment_estimate,
)


# ---------------------------------------------------------------------------
# 1 — cyclic permutation identity Tr(C_k rho^{(x)k}) == Tr(rho^k)
# ---------------------------------------------------------------------------
def test_cyclic_permutation_identity() -> None:
    rng = np.random.default_rng(0)
    for n in (2, 3):
        d = 2 ** n
        rho = depolarize(random_density(d, 3, rng), 0.3)
        for k in (2, 3, 4):
            c_k = cyclic_permutation_operator(d, k)
            lhs = float(np.trace(c_k @ kron_power(rho, k)).real)
            assert lhs == pytest.approx(moment(rho, k), abs=1e-9)
            assert float(np.trace(c_k).real) == pytest.approx(d, abs=1e-9)  # single k-cycle


# ---------------------------------------------------------------------------
# 2 — depolarizing signal formula matches an explicit construction
# ---------------------------------------------------------------------------
def test_depolarizing_signal_matches_explicit() -> None:
    rng = np.random.default_rng(1)
    for n in (2, 3):
        d = 2 ** n
        rho = depolarize(random_density(d, 3, rng), 0.3)
        for k in (2, 3, 4):
            c_k = cyclic_permutation_operator(d, k)
            big_dim = d ** k
            state = kron_power(rho, k)
            for p in (0.03, 0.1):
                noisy = (1.0 - p) * state + p * np.eye(big_dim) / big_dim
                explicit = float(np.trace(c_k @ noisy).real)
                formula = depolarizing_moment_signal(moment(rho, k), k, p, n)
                assert formula == pytest.approx(explicit, abs=1e-9)


# ---------------------------------------------------------------------------
# 3 — Kraus channels reproduce the independently computed anchor values
# ---------------------------------------------------------------------------
def test_kraus_channels_reproduce_anchor() -> None:
    # rank-3 random state depolarized at q=0.25 with purity 0.373313 (seed 11).
    rho = depolarize(random_density(4, 3, np.random.default_rng(11)), 0.25)
    assert moment(rho, 2) == pytest.approx(0.373313, abs=1e-6)

    g = 0.05
    # Explicit k-copy construction (build C_k and the noisy two-copy state).
    amp_explicit = explicit_channel_collective_signal(rho, 2, amplitude_damping_kraus(g), 2)
    deph_explicit = explicit_channel_collective_signal(rho, 2, dephasing_kraus(g), 2)
    assert amp_explicit == pytest.approx(0.343096, abs=1e-5)
    assert deph_explicit == pytest.approx(0.366884, abs=1e-5)

    # The factorized signal used by the sweep must equal the explicit construction.
    assert channel_collective_signal(rho, 2, amplitude_damping_kraus(g), 2) == pytest.approx(
        amp_explicit, abs=1e-12
    )
    assert channel_collective_signal(rho, 2, dephasing_kraus(g), 2) == pytest.approx(
        deph_explicit, abs=1e-12
    )


# ---------------------------------------------------------------------------
# 4 — shadow moment estimators are unbiased for k = 2, 3, 4
# ---------------------------------------------------------------------------
def test_shadow_moment_estimators_unbiased() -> None:
    # The k-th order U-statistic is unbiased for Tr(rho^k) by construction; verify
    # the mean converges.  Single-qubit (n=1) has low variance, so it pins all of
    # k=2,3,4 cheaply; n=2 additionally exercises the multi-qubit factorization
    # (k=4 at n=2 has astronomically high variance and is checked in the sweep).
    rng = np.random.default_rng(7)
    rho1 = depolarize(random_density(2, 2, rng), 0.2)  # n=1
    for k in (2, 3, 4):
        estimates = [shadow_moment_estimate(rho1, k, 2000, rng, n_tuples=10000) for _ in range(35)]
        assert abs(float(np.mean(estimates)) - moment(rho1, k)) < 0.04

    rho2 = depolarize(random_density(4, 4, rng), 0.2)  # n=2
    for k in (2, 3):
        estimates = [shadow_moment_estimate(rho2, k, 2500, rng, n_tuples=10000) for _ in range(35)]
        assert abs(float(np.mean(estimates)) - moment(rho2, k)) < 0.04


# ---------------------------------------------------------------------------
# 5 — the sweep runs end to end and produces a results file
# ---------------------------------------------------------------------------
def test_sweep_runs_and_saves(tmp_path) -> None:
    rows = run_sweep(
        sizes=(2,), ks=(2, 3), noise_models=("depolarizing", "amplitude_damping"),
        rates=(0.0, 0.1), budget=400, n_states=4, n_tuples_fair=5000, seed=0,
    )
    # 1 size x 2 k x 2 noise x 2 rates = 8 cells.
    assert len(rows) == 8
    for row in rows:
        assert row["winner_subsampled"] in ("collective", "single-copy")
        assert row["winner_fair"] in ("collective", "single-copy")
        # The fair (copy-optimal) estimator uses no extra copies, so it is never
        # worse than the subsampled one at the same budget.
        assert row["single_rmse_fair"] <= row["single_rmse_subsampled"] + 1e-9
        assert row["collective_rmse"] >= 0.0
        assert row["single_copies"] == 400
        assert row["collective_measurements"] == 400 // row["k"]  # k copies per measurement

    out = tmp_path / "sweep.json"
    save_sweep(rows, out, {"budget": 400})
    assert out.exists()
    import json

    payload = json.loads(out.read_text())
    assert len(payload["rows"]) == 8
