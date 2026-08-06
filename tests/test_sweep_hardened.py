"""Tests for the corrected, hardened moment-family sweep.

Pins that the efficient exact estimators equal the brute-force-verified
reference estimators for every k, that the efficient collective signal matches
the reference channel signal, that the noisy-pure true moments stay O(1), and
that the sweep runs end to end (reproducibly, across worker counts) with error
bars and z-scores.
"""

from __future__ import annotations

import itertools
import json
from functools import reduce

import numpy as np
import pytest

from anrl.benchmark import (
    crossover_table,
    exact_moment_ustatistic,
    moment,
    moment_state_errors,
    run_moment_sweep,
    save_moment_sweep,
    skipped_cells,
)
from anrl.benchmark.channels import collective_moment_signal
from anrl.benchmark.ensembles import noisy_pure
from anrl.benchmark.moments import full_moment_ustatistic_k3, full_moment_ustatistic_k4
from anrl.benchmark.shadows import _snapshots, full_purity_ustatistic
from anrl.benchmark.sweep_hardened import _collective_signal


# ---------------------------------------------------------------------------
# 1, efficient exact estimators == brute-force-verified reference (all k)
# ---------------------------------------------------------------------------
def test_exact_moment_ustatistic_matches_reference() -> None:
    rng = np.random.default_rng(0)
    for n in (1, 2, 3, 4, 5, 6):  # existing k=4 reference is capped at n<=6
        state = noisy_pure(n, 0.1, rng)
        density = state.density_matrix()
        snaps = _snapshots(density, n, 60, rng)
        assert exact_moment_ustatistic(snaps, 2) == pytest.approx(full_purity_ustatistic(snaps), abs=1e-9)
        assert exact_moment_ustatistic(snaps, 3) == pytest.approx(full_moment_ustatistic_k3(snaps), abs=1e-9)
        assert exact_moment_ustatistic(snaps, 4) == pytest.approx(full_moment_ustatistic_k4(snaps), abs=1e-9)


def _brute_moment(snaps: np.ndarray, k: int) -> float:
    """Reference: mean of Tr(prod of k snapshots) over all distinct ordered k-tuples."""
    m = snaps.shape[0]
    g = np.array([reduce(np.kron, list(snaps[i])) for i in range(m)])
    total = 0.0 + 0.0j
    denom = 0
    for tup in itertools.permutations(range(m), k):
        prod = g[tup[0]]
        for idx in tup[1:]:
            prod = prod @ g[idx]
        total += np.trace(prod)
        denom += 1
    return float((total / denom).real)


def test_exact_moment_ustatistic_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    for n in (1, 2):
        state = noisy_pure(n, 0.15, rng)
        snaps = _snapshots(state.density_matrix(), n, 7, rng)
        for k in (2, 3, 4):
            assert exact_moment_ustatistic(snaps, k) == pytest.approx(_brute_moment(snaps, k), abs=1e-9)


def test_exact_moment_ustatistic_rejects_too_few_snapshots() -> None:
    rng = np.random.default_rng(2)
    snaps = _snapshots(noisy_pure(1, 0.1, rng).density_matrix(), 1, 3, rng)  # m=3
    with pytest.raises(ValueError):
        exact_moment_ustatistic(snaps, 4)  # needs >= 4


# ---------------------------------------------------------------------------
# 2, efficient collective signal == reference channel signal (all k)
# ---------------------------------------------------------------------------
def test_collective_signal_matches_reference() -> None:
    rng = np.random.default_rng(3)
    for n in (2, 3):
        density = noisy_pure(n, 0.1, rng).density_matrix()
        for k in (2, 3, 4):
            tm = moment(density, k)
            for nm, rate in [("depolarizing", 0.05), ("amplitude_damping", 0.05), ("dephasing", 0.1)]:
                got = _collective_signal(density, n, k, nm, rate, tm)
                ref = collective_moment_signal(density, k, nm, rate, n)
                assert got == pytest.approx(ref, abs=1e-10)


# ---------------------------------------------------------------------------
# 3, noisy-pure true moments stay O(1) (do NOT collapse toward zero)
# ---------------------------------------------------------------------------
def test_noisy_pure_moments_stay_order_one() -> None:
    rng = np.random.default_rng(4)
    for n in (2, 4, 6, 8):
        density = noisy_pure(n, 0.1, rng).density_matrix()
        for k in (2, 3, 4):
            # Closed form for (1-q)|psi><psi| + q I/d: top eigenvalue (1-q)+q/d.
            top = (1.0 - 0.1) + 0.1 / (2 ** n)
            assert moment(density, k) > top ** k  # >= dominant term, comfortably O(1)
            assert moment(density, k) > 0.5  # concretely order-one at q=0.1, all k, all n


# ---------------------------------------------------------------------------
# 4, per-state error function: deterministic, shaped, budget-guarded
# ---------------------------------------------------------------------------
def test_moment_state_errors_deterministic_and_shaped() -> None:
    kw = dict(
        n=3, state_idx=2, ks=(2, 3, 4), noise_models=("depolarizing", "dephasing"),
        rates=(0.0, 0.1), budget=400, n_trials=5, ensemble_q=0.1, seed=0,
        max_n_by_k={2: 10, 3: 8, 4: 8},
    )
    a = moment_state_errors(**kw)
    b = moment_state_errors(**kw)
    assert a["single_se"] == b["single_se"] and a["coll_se"] == b["coll_se"]
    for k in (2, 3, 4):
        assert len(a["single_se"][k]) == 5 and all(e >= 0 for e in a["single_se"][k])
        assert a["true_moment"][k] > 0.5
    assert len(a["coll_se"]) == 3 * 2 * 2  # k x noise x rate
    assert all(len(v) == 5 for v in a["coll_se"].values())

    with pytest.raises(ValueError):
        moment_state_errors(**{**kw, "budget": 3})  # budget < max(k)=4


def test_moment_state_errors_slice_independent() -> None:
    # Collective draws are seeded by VALUE, so a cell's squared errors do not
    # depend on the order of noise_models/rates or which others are swept.
    base = dict(n=3, state_idx=1, ks=(2, 3), budget=400, n_trials=4, ensemble_q=0.1,
                seed=0, max_n_by_k={2: 10, 3: 8})
    a = moment_state_errors(noise_models=("depolarizing", "dephasing"), rates=(0.0, 0.1), **base)
    b = moment_state_errors(noise_models=("dephasing", "depolarizing"), rates=(0.1, 0.0), **base)
    # A shared cell must be identical despite the reordering.
    assert a["coll_se"]["3|dephasing@0.1"] == b["coll_se"]["3|dephasing@0.1"]
    assert a["coll_se"]["2|depolarizing@0.0"] == b["coll_se"]["2|depolarizing@0.0"]
    # Front-truncating the noise set leaves the surviving cell unchanged.
    c = moment_state_errors(noise_models=("dephasing",), rates=(0.0, 0.1), **base)
    assert c["coll_se"]["3|dephasing@0.1"] == a["coll_se"]["3|dephasing@0.1"]


def test_incomplete_caps_fail_fast() -> None:
    # A max_n_by_k missing a k must raise a clear ValueError, not a bare KeyError.
    with pytest.raises(ValueError, match="missing caps"):
        skipped_cells((2, 9), (2, 3, 4), max_n_by_k={2: 10})
    with pytest.raises(ValueError, match="missing caps"):
        moment_state_errors(n=3, state_idx=0, ks=(2, 3, 4), noise_models=("depolarizing",),
                            rates=(0.0,), budget=400, n_trials=2, ensemble_q=0.1, seed=0,
                            max_n_by_k={2: 10})


def test_moment_state_errors_skips_infeasible() -> None:
    # Cap k=4 at n<=3 so an n=4 unit reports k=4 as skipped (None), others fine.
    res = moment_state_errors(
        n=4, state_idx=0, ks=(2, 3, 4), noise_models=("depolarizing",), rates=(0.0,),
        budget=400, n_trials=3, ensemble_q=0.1, seed=0, max_n_by_k={2: 10, 3: 8, 4: 3},
    )
    assert res["single_se"][4] is None  # infeasible
    assert res["single_se"][2] is not None and res["single_se"][3] is not None


# ---------------------------------------------------------------------------
# 5; the sweep runs end to end (reproducibly) and saves error bars + z-scores
# ---------------------------------------------------------------------------
def test_run_moment_sweep_end_to_end(tmp_path) -> None:
    kw = dict(
        sizes=(2, 3), ks=(2, 3, 4), noise_models=("depolarizing", "dephasing"),
        rates=(0.0, 0.1), budget=400, n_states=6, n_trials=4, seed=0,
    )
    rows = run_moment_sweep(**kw, max_workers=4)
    # 2 sizes x 3 k x 2 noise x 2 rates = 24 rows.
    assert len(rows) == 24
    for r in rows:
        assert r["winner"] in ("collective", "single-copy", "tie")
        assert r["single_rmse"] >= 0 and r["collective_rmse"] >= 0
        lo, hi = r["single_rmse_ci68"]
        assert lo <= r["single_rmse"] <= hi
        assert r["mean_true_moment"] > 0.5  # O(1) target
        assert r["single_copies"] == 400
        assert r["collective_measurements"] == 400 // r["k"]

    # Single-copy RMSE is noise-independent for a fixed (n, k).
    by_key: dict[tuple[int, int], set[float]] = {}
    for r in rows:
        by_key.setdefault((r["n"], r["k"]), set()).add(round(r["single_rmse"], 12))
    for key, vals in by_key.items():
        assert len(vals) == 1, f"single_rmse varies within {key}: {vals}"

    # Reproducible across worker counts.
    rows2 = run_moment_sweep(**kw, max_workers=2)
    assert [r["single_rmse"] for r in rows] == [r["single_rmse"] for r in rows2]
    assert [r["paired_z"] for r in rows] == [r["paired_z"] for r in rows2]

    table = crossover_table(rows, group_keys=("k", "noise_model", "rate"))
    assert {e["k"] for e in table} == {2, 3, 4}
    out = tmp_path / "moment_sweep.json"
    save_moment_sweep(rows, table, out, {"budget": 400})
    payload = json.loads(out.read_text())
    assert len(payload["rows"]) == 24 and "crossover_table" in payload


def test_skipped_cells() -> None:
    skipped = skipped_cells((2, 6, 9), (2, 3, 4), max_n_by_k={2: 10, 3: 8, 4: 8})
    combos = {(c["n"], c["k"]) for c in skipped}
    assert (9, 3) in combos and (9, 4) in combos  # n=9 > 8 for k=3,4
    assert (9, 2) not in combos  # k=2 cap is 10
    assert (6, 4) not in combos  # n=6 within all caps
