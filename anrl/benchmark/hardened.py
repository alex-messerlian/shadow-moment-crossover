"""Statistically hardened scaling study with error bars and paired crossover tests.

Extends :mod:`anrl.benchmark.scaling` from point RMSE estimates to a hardened
comparison: many states per cell, many measurement trials per state, bootstrap
error bars on every RMSE, and a **paired** single-vs-collective test at the
state level so a crossover boundary is called only when it is statistically
resolved (not sampling noise).

Design
------
The independent statistical unit is the *state* (states are drawn iid from the
ensemble).  For each state ``s`` we form the per-state mean squared error of
each estimator over ``n_trials`` independent measurement realizations:

* single-copy MSE ``u(s)`` — ``n_trials`` fresh factored-shadow draws, each
  reduced by the exact copy-fair U-statistic.  Noise-independent, so ``u(s)`` is
  reused against every ``(noise_model, rate)`` collective cell.
* collective MSE ``c(s)`` — the deterministic noisy SWAP signal plus
  ``n_trials`` fresh binomial shot draws.

The paired difference ``delta(s) = u(s) - c(s)`` has mean over states
``mean_delta = MSE_single - MSE_collective`` with ``SE = std(delta)/sqrt(n_states)``.
The 48 ``delta(s)`` are iid draws, so ``z = mean_delta / SE`` is a valid
statistic.  (Note: because noise is modelled *only* on the collective route and
every noisy-pure state has the same true purity, ``u(s)`` and ``c(s)`` are
essentially independent — ``corr ~= 0`` — so pairing gives no variance reduction
over an unpaired two-sample test; it is used only as a simple, valid framing,
not for added power.)  The winner is:

* ``collective``   if ``mean_delta > z_crit * SE`` (single genuinely worse),
* ``single-copy``  if ``mean_delta < -z_crit * SE``,
* ``tie``          otherwise (statistically indistinguishable at this budget).

RMSE point values carry a bootstrap 68% interval resampled over states.

Caveats on the crossover map.  The verdicts are per-cell at nominal ``|z| > 2``
with no family-wise correction across cells, and the 12 ``(noise, rate)`` cells
at a fixed ``(ensemble, n)`` reuse the *same* single-copy sample, so their
verdicts are positively correlated (one unlucky single-copy draw tilts all 12
together).  Effect sizes at the resolved boundaries are large (``|z| ~ 5-16``),
so the headline crossovers are robust; but boundaries resting on a marginal
``|z|`` are flagged ``ambiguous`` by :func:`crossover_table` and should not be
read as sharply resolved.

Parallelism: each ``(ensemble, n, state_idx)`` unit is computed by
:func:`state_errors` — a pure function seeded deterministically from
``(seed, ensemble_id, n, state_idx)`` — so results are identical regardless of
worker count or scheduling.  The units are fanned out over processes.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .moments import collective_moment_estimate
from .scaling import (
    _ENSEMBLE_ID,
    ENSEMBLES,
    collective_purity_signal,
    snapshots_factored,
)
from .shadows import full_purity_ustatistic

Z_CRIT = 2.0  # ~95% paired-test threshold for calling a winner
# A crossover boundary resting on a |z| below this is flagged 'ambiguous': it
# clears Z_CRIT but not by enough to survive multiplicity across the ~168 cells.
MARGINAL_Z = 3.0
_WINNER_ORDER = {"single-copy": 0, "tie": 1, "collective": 2}
_BOOTSTRAP_RESAMPLES = 2000

# Deterministic integer id per noise model, and a stable integer key per rate, so
# collective substreams are seeded by VALUE (not loop position) — a cell's draws
# then depend only on its (noise_model, rate), not on how the grid is sliced.
_NOISE_ID = {"depolarizing": 0, "amplitude_damping": 1, "dephasing": 2}


def _rate_key(rate: float) -> int:
    """Stable integer seed component for a noise rate (value-based, not positional)."""
    return int(round(float(rate) * 1_000_000_000))


def state_errors(
    ensemble: str,
    n: int,
    state_idx: int,
    noise_models: tuple[str, ...],
    rates: tuple[float, ...],
    budget: int,
    n_trials: int,
    ensemble_q: float,
    seed: int,
) -> dict:
    """Per-state squared errors for one ``(ensemble, n, state_idx)`` unit (pure).

    Deterministic in ``(seed, ensemble_id, n, state_idx)``.  Each source of
    randomness draws from its OWN seeded substream (state construction,
    single-copy shadows, and every ``(noise_model, rate)`` collective cell), so a
    cell's draws are independent of loop ordering and of which other noise models
    are swept — the numbers do not depend on how the grid is sliced.  Returns the
    ``n_trials`` single-copy squared errors (noise-independent) and, per
    ``(noise_model, rate)``, the ``n_trials`` collective squared errors.
    """
    if budget < 2:
        raise ValueError(f"budget must be >= 2 (single-copy U-statistic needs >= 2 snapshots), got {budget}")
    eid = _ENSEMBLE_ID[ensemble]
    state_rng = np.random.default_rng([seed, eid, n, state_idx, 0])
    state = ENSEMBLES[ensemble](n, ensemble_q, state_rng)
    true = state.purity()

    single_rng = np.random.default_rng([seed, eid, n, state_idx, 1])
    single_se = [
        (full_purity_ustatistic(snapshots_factored(state, budget, single_rng)) - true) ** 2
        for _ in range(n_trials)
    ]

    coll_se: dict[str, list[float]] = {}
    for noise_model in noise_models:
        for rate in rates:
            signal = collective_purity_signal(state, noise_model, rate)
            # Seed by VALUE (noise id + rate key), so a cell's draws are the same
            # regardless of loop order / which other cells are swept.
            cell_rng = np.random.default_rng(
                [seed, eid, n, state_idx, 2, _NOISE_ID[noise_model], _rate_key(rate)]
            )
            key = f"{noise_model}@{rate}"
            coll_se[key] = [
                (collective_moment_estimate(2, budget // 2, signal, cell_rng) - true) ** 2
                for _ in range(n_trials)
            ]
    return {"true_purity": float(true), "single_se": single_se, "coll_se": coll_se}


def _bootstrap_rmse_ci(
    per_state_mse: np.ndarray, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    """68% bootstrap interval for ``RMSE = sqrt(mean_s MSE(s))``, resampling states.

    Returns ``(None, None)`` for an empty sample (JSON-portable — no NaN token).
    """
    k = per_state_mse.shape[0]
    if k == 0:
        return (None, None)
    idx = rng.integers(0, k, size=(_BOOTSTRAP_RESAMPLES, k))
    boot_rmse = np.sqrt(per_state_mse[idx].mean(axis=1))
    return (float(np.percentile(boot_rmse, 16.0)), float(np.percentile(boot_rmse, 84.0)))


def _aggregate_cell(
    single_mse_per_state: np.ndarray,  # (n_states,)
    coll_mse_per_state: np.ndarray,  # (n_states,)
    single_ci: tuple[float | None, float | None],  # precomputed once per (ensemble, n)
    boot_rng: np.random.Generator,
) -> dict:
    """Paired single-vs-collective comparison for one cell, with error bars.

    ``single_ci`` is passed in (not recomputed here) so the noise-independent
    single-copy estimand carries ONE consistent bootstrap interval across all the
    ``(noise, rate)`` cells of its ``(ensemble, n)`` group.
    """
    n_states = single_mse_per_state.shape[0]
    rmse_single = float(np.sqrt(single_mse_per_state.mean()))
    rmse_coll = float(np.sqrt(coll_mse_per_state.mean()))

    delta = single_mse_per_state - coll_mse_per_state  # per-state MSE difference
    mean_delta = float(delta.mean())
    # None (not inf) for a single state -> JSON-portable and reads as "undefined SE".
    se_delta = float(delta.std(ddof=1) / np.sqrt(n_states)) if n_states > 1 else None
    z = mean_delta / se_delta if (se_delta is not None and se_delta > 0) else 0.0

    if z > Z_CRIT:
        winner = "collective"
    elif z < -Z_CRIT:
        winner = "single-copy"
    else:
        winner = "tie"

    coll_lo, coll_hi = _bootstrap_rmse_ci(coll_mse_per_state, boot_rng)
    return {
        "single_rmse": rmse_single,
        "single_rmse_ci68": [single_ci[0], single_ci[1]],
        "collective_rmse": rmse_coll,
        "collective_rmse_ci68": [coll_lo, coll_hi],
        "paired_mse_diff": mean_delta,  # MSE_single - MSE_collective
        "paired_mse_diff_se": se_delta,
        "paired_z": float(z),
        "winner": winner,
    }


def run_hardened(
    ensembles: tuple[str, ...] = ("noisy_pure", "random_mixed"),
    sizes: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10),
    noise_models: tuple[str, ...] = ("depolarizing", "amplitude_damping", "dephasing"),
    rates: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1),
    ensemble_q: float = 0.1,
    budget: int = 2000,
    n_states: int = 40,
    n_trials: int = 10,
    seed: int = 0,
    max_random_mixed_n: int = 6,
    max_workers: int | None = None,
) -> list[dict]:
    """Hardened scaling grid; one row per ``(ensemble, n, noise_model, rate)``.

    Each row carries bootstrap RMSE intervals and a paired-test verdict
    (``collective`` / ``single-copy`` / ``tie``) with ``paired_mse_diff +- se``.
    """
    if budget < 2:
        raise ValueError(f"budget must be >= 2, got {budget}")
    tasks = []
    for ensemble in ensembles:
        max_n = max(sizes) if ensemble == "noisy_pure" else max_random_mixed_n
        for n in sizes:
            if n > max_n:
                continue
            for state_idx in range(n_states):
                tasks.append((ensemble, n, state_idx))

    args = (noise_models, rates, budget, n_trials, ensemble_q, seed)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_state_errors_star, [(t, args) for t in tasks]))

    # Group per-state results by (ensemble, n).
    grouped: dict[tuple[str, int], list[dict]] = {}
    for (ensemble, n, _), res in zip(tasks, results):
        grouped.setdefault((ensemble, n), []).append(res)

    boot_rng = np.random.default_rng([seed, 999])
    rows: list[dict] = []
    for (ensemble, n), states in grouped.items():
        single_mse = np.array([np.mean(s["single_se"]) for s in states])  # (n_states,)
        # Bootstrap the noise-independent single-copy CI ONCE per (ensemble, n).
        single_ci = _bootstrap_rmse_ci(single_mse, boot_rng)
        mean_true = float(np.mean([s["true_purity"] for s in states]))
        for noise_model in noise_models:
            for rate in rates:
                key = f"{noise_model}@{rate}"
                coll_mse = np.array([np.mean(s["coll_se"][key]) for s in states])
                cell = _aggregate_cell(single_mse, coll_mse, single_ci, boot_rng)
                rows.append(
                    {
                        "ensemble": ensemble,
                        "n": int(n),
                        "noise_model": noise_model,
                        "rate": float(rate),
                        "budget": int(budget),
                        "n_states": int(len(states)),
                        "n_trials": int(n_trials),
                        "mean_true_purity": mean_true,
                        **cell,
                    }
                )
    rows.sort(key=lambda r: (r["ensemble"], r["n"], r["noise_model"], r["rate"]))
    return rows


def _state_errors_star(packed: tuple) -> dict:
    """Top-level unpacker so ProcessPoolExecutor can pickle the work unit."""
    (ensemble, n, state_idx), (noise_models, rates, budget, n_trials, ensemble_q, seed) = packed
    return state_errors(
        ensemble, n, state_idx, noise_models, rates, budget, n_trials, ensemble_q, seed
    )


def crossover_table(
    rows: list[dict],
    group_keys: tuple[str, ...] = ("ensemble", "noise_model", "rate"),
    marginal_z: float = MARGINAL_Z,
) -> list[dict]:
    """Per group (default ``(ensemble, noise_model, rate)``): crossover n and flags.

    ``group_keys`` chooses the fields that define a single curve-in-``n`` (e.g.
    ``("k", "noise_model", "rate")`` for the moment sweep).  ``crossover_n`` is
    the smallest n whose paired verdict is ``collective``.  ``ambiguous`` is set
    when the boundary is not cleanly resolved, i.e. any of:

    * the winner sequence is non-monotone in ``single-copy < tie < collective``
      order anywhere (a tie/single win sandwiched between collective wins, or a
      dip below the crossover);
    * any ``tie`` sits strictly below ``crossover_n`` (an unresolved cell below
      the boundary);
    * the crossover cell itself clears ``Z_CRIT`` only marginally
      (``|paired_z| < marginal_z``) — too weak to survive multiplicity across the
      full grid.

    ``crossover_z`` records the boundary cell's ``paired_z`` (``None`` if the
    rows carry no ``paired_z``).
    """
    table: list[dict] = []
    keys = sorted({tuple(r[g] for g in group_keys) for r in rows})
    for keyvals in keys:
        sel = dict(zip(group_keys, keyvals))
        cells = sorted(
            (r for r in rows if all(r[g] == sel[g] for g in group_keys)),
            key=lambda r: r["n"],
        )
        winners = {c["n"]: c["winner"] for c in cells}
        zscores = {c["n"]: c.get("paired_z") for c in cells}
        ns = sorted(winners)
        crossover_n = next((n for n in ns if winners[n] == "collective"), None)

        ambiguous = False
        crossover_z = None
        if crossover_n is not None:
            ordinals = [_WINNER_ORDER[winners[n]] for n in ns]
            # Clean crossover => winners non-decreasing in single<tie<collective.
            if any(ordinals[i] > ordinals[i + 1] for i in range(len(ordinals) - 1)):
                ambiguous = True
            # Any unresolved (tie) cell strictly below the boundary.
            if any(winners[n] == "tie" for n in ns if n < crossover_n):
                ambiguous = True
            # Boundary cell only marginally significant.
            crossover_z = zscores.get(crossover_n)
            if crossover_z is not None and abs(crossover_z) < marginal_z:
                ambiguous = True
        table.append(
            {
                **sel,
                "crossover_n": crossover_n,
                "crossover_z": crossover_z,
                "ambiguous": ambiguous,
                "winners_by_n": {int(n): winners[n] for n in ns},
            }
        )
    return table


def save_hardened(
    rows: list[dict], table: list[dict], path: str | Path, metadata: dict | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {"metadata": metadata or {}, "rows": rows, "crossover_table": table},
            handle,
            indent=2,
        )
    return path
