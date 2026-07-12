"""Tests for the statistically hardened scaling study.

Checks the per-state error function is deterministic and shaped right, the
paired-cell aggregation classifies clear/tie cases correctly, the crossover
table flags ambiguous boundaries, and the grid runs end to end (with
multiprocessing) and saves results carrying error bars.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anrl.benchmark import crossover_table, run_hardened, save_hardened, state_errors
from anrl.benchmark.hardened import _aggregate_cell


# ---------------------------------------------------------------------------
# 1 — state_errors is deterministic and correctly shaped
# ---------------------------------------------------------------------------
def test_state_errors_deterministic_and_shaped() -> None:
    kw = dict(
        ensemble="noisy_pure", n=2, state_idx=3,
        noise_models=("depolarizing", "dephasing"), rates=(0.0, 0.1),
        budget=400, n_trials=5, ensemble_q=0.1, seed=0,
    )
    a = state_errors(**kw)
    b = state_errors(**kw)
    assert a["single_se"] == b["single_se"]  # same seed -> identical
    assert a["coll_se"] == b["coll_se"]
    assert len(a["single_se"]) == 5
    assert set(a["coll_se"]) == {"depolarizing@0.0", "depolarizing@0.1", "dephasing@0.0", "dephasing@0.1"}
    for errs in a["coll_se"].values():
        assert len(errs) == 5
        assert all(e >= 0.0 for e in errs)  # squared errors
    assert 0.0 <= a["true_purity"] <= 1.0
    # A different state index gives different draws.
    c = state_errors(**{**kw, "state_idx": 4})
    assert c["single_se"] != a["single_se"]


# ---------------------------------------------------------------------------
# 2 — paired-cell aggregation classifies clear wins and ties
# ---------------------------------------------------------------------------
def test_aggregate_cell_classification() -> None:
    rng = np.random.default_rng(0)
    n = 40
    ci = (0.0, 1.0)  # precomputed single-copy CI (passed in, not recomputed here)
    # Collective clearly better: single MSE ~0.10, collective MSE ~0.02 (paired).
    single = np.full(n, 0.10) + rng.normal(0, 0.005, n)
    coll = np.full(n, 0.02) + rng.normal(0, 0.005, n)
    cell = _aggregate_cell(single, coll, ci, rng)
    assert cell["winner"] == "collective"
    assert cell["paired_mse_diff"] > 0 and cell["paired_z"] > 2
    assert cell["single_rmse"] > cell["collective_rmse"]
    assert cell["single_rmse_ci68"] == [0.0, 1.0]  # uses the passed-in CI verbatim

    # Single clearly better: reversed.
    cell2 = _aggregate_cell(coll, single, ci, rng)
    assert cell2["winner"] == "single-copy"
    assert cell2["paired_z"] < -2

    # Statistically indistinguishable: same distribution -> tie.
    x = np.full(n, 0.05) + rng.normal(0, 0.02, n)
    y = np.full(n, 0.05) + rng.normal(0, 0.02, n)
    cell3 = _aggregate_cell(x, y, ci, rng)
    assert cell3["winner"] == "tie"
    assert abs(cell3["paired_z"]) <= 2

    # n_states == 1 -> SE undefined -> None (JSON-portable), tie, collective CI degenerate.
    cell4 = _aggregate_cell(np.array([0.1]), np.array([0.05]), ci, rng)
    assert cell4["paired_mse_diff_se"] is None and cell4["winner"] == "tie"
    lo, hi = cell4["collective_rmse_ci68"]
    assert lo == pytest.approx(hi)  # single-state bootstrap is degenerate


# ---------------------------------------------------------------------------
# 3 — crossover table locates the boundary and flags ambiguity
# ---------------------------------------------------------------------------
def test_crossover_table_flags() -> None:
    def row(ensemble, n, nm, rate, winner, z=10.0):
        return {"ensemble": ensemble, "n": n, "noise_model": nm, "rate": rate,
                "winner": winner, "paired_z": z}

    # Clean crossover at n=4 (n=3 is single-copy, not a tie), strong z -> resolved.
    clean = [row("e", 2, "x", 0.1, "single-copy"), row("e", 3, "x", 0.1, "single-copy"),
             row("e", 4, "x", 0.1, "collective"), row("e", 5, "x", 0.1, "collective")]
    got = {t["noise_model"]: t for t in crossover_table(clean)}["x"]
    assert got["crossover_n"] == 4 and got["ambiguous"] is False and got["crossover_z"] == 10.0

    # Boundary just below crossover is a tie -> ambiguous.
    amb = [row("e", 2, "y", 0.1, "single-copy"), row("e", 3, "y", 0.1, "tie"),
           row("e", 4, "y", 0.1, "collective")]
    got2 = {t["noise_model"]: t for t in crossover_table(amb)}["y"]
    assert got2["crossover_n"] == 4 and got2["ambiguous"] is True

    # Non-monotone (single-copy win after the crossover) -> ambiguous.
    non_mono = [row("e", 2, "z", 0.1, "collective"), row("e", 3, "z", 0.1, "single-copy"),
                row("e", 4, "z", 0.1, "collective")]
    got3 = {t["noise_model"]: t for t in crossover_table(non_mono)}["z"]
    assert got3["crossover_n"] == 2 and got3["ambiguous"] is True

    # A tie two steps below the crossover must also be flagged (broadened coverage).
    tie_low = [row("e", 2, "w", 0.1, "tie"), row("e", 3, "w", 0.1, "single-copy"),
               row("e", 4, "w", 0.1, "collective"), row("e", 5, "w", 0.1, "collective")]
    got4 = {t["noise_model"]: t for t in crossover_table(tie_low)}["w"]
    assert got4["crossover_n"] == 4 and got4["ambiguous"] is True

    # Marginal boundary z (clears Z_CRIT but |z| < MARGINAL_Z) -> ambiguous.
    marginal = [row("e", 2, "m", 0.1, "single-copy"), row("e", 3, "m", 0.1, "single-copy"),
                row("e", 4, "m", 0.1, "collective", z=2.3), row("e", 5, "m", 0.1, "collective")]
    got5 = {t["noise_model"]: t for t in crossover_table(marginal)}["m"]
    assert got5["crossover_n"] == 4 and got5["ambiguous"] is True and got5["crossover_z"] == 2.3


# ---------------------------------------------------------------------------
# 4 — the hardened grid runs end to end (multiprocessing) and saves error bars
# ---------------------------------------------------------------------------
def test_hardened_runs_and_saves(tmp_path) -> None:
    rows = run_hardened(
        ensembles=("noisy_pure",), sizes=(2, 3),
        noise_models=("depolarizing", "dephasing"), rates=(0.0, 0.1),
        budget=400, n_states=8, n_trials=4, seed=0, max_workers=2,
    )
    # 1 ensemble x 2 sizes x 2 noise x 2 rates = 8 rows.
    assert len(rows) == 8
    for r in rows:
        assert r["winner"] in ("collective", "single-copy", "tie")
        assert r["single_rmse"] >= 0.0 and r["collective_rmse"] >= 0.0
        lo, hi = r["single_rmse_ci68"]
        assert lo <= r["single_rmse"] <= hi
        assert r["n_states"] == 8 and r["n_trials"] == 4
        assert np.isfinite(r["paired_mse_diff"]) and r["paired_mse_diff_se"] >= 0.0

    # Single-copy RMSE is noise-independent for a fixed (ensemble, n).
    by_key: dict[tuple[str, int], set[float]] = {}
    for r in rows:
        by_key.setdefault((r["ensemble"], r["n"]), set()).add(round(r["single_rmse"], 12))
    for key, vals in by_key.items():
        assert len(vals) == 1, f"single_rmse varies within {key}: {vals}"

    table = crossover_table(rows)
    out = tmp_path / "hardened.json"
    save_hardened(rows, table, out, {"budget": 400})
    payload = json.loads(out.read_text())
    assert len(payload["rows"]) == 8
    assert "crossover_table" in payload and payload["metadata"]["budget"] == 400


def test_run_hardened_rejects_tiny_budget() -> None:
    with pytest.raises(ValueError):
        run_hardened(ensembles=("noisy_pure",), sizes=(2,), noise_models=("depolarizing",),
                     rates=(0.0,), budget=1, n_states=2, n_trials=2)
