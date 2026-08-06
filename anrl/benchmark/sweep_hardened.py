"""Corrected, statistically hardened moment-family sweep (k = 2, 3, 4).

Fixes the two defects of the original :mod:`anrl.benchmark.sweep`:

1. **Ensemble.**  The old sweep used random Ginibre states depolarized at
   ``q in [0, 0.5]``, whose purity collapses toward ``2^-n`` so ``Tr(rho^k)`` for
   ``k = 3, 4`` sits near zero, single-copy then "wins" by estimating almost
   nothing.  This sweep uses :func:`~anrl.benchmark.ensembles.noisy_pure`
   (``(1-q)|psi><psi| + q I/2^n``, Haar ``|psi>``), the realistic NISQ model,
   whose ``Tr(rho^k)`` stays ``O(1)``.
2. **System size.**  The old sweep capped at ``n = 4``, below the ``n >= 5`` where
   the single-copy exponential becomes visible.  This one runs ``n = 2..8``.

Statistics (per cell): ``n_states`` iid states x ``n_trials`` measurement trials,
bootstrap 68% RMSE intervals, and a state-level paired MSE test
(:func:`~anrl.benchmark.hardened._aggregate_cell`) that declares ``collective`` /
``single-copy`` / ``tie`` at ``|z| > 2``.  Each ``(k, noise, rate)`` cell and the
single-copy draws use independent seeded substreams; the whole run is
reproducible from one top-level ``seed``.  Caveats carried forward: no family-wise
correction across cells, and the single-copy sample is shared across the noise
cells at a given ``(n, k)`` (so those verdicts are positively correlated).

Single-copy uses the EXACT full U-statistic for every k
(:func:`~anrl.benchmark.moment_ustats.exact_moment_ustatistic`, identical to the
brute-force-verified reference estimators); collective uses the k-copy cyclic
test at ``budget // k`` measurements.  Copy budget is equal: single spends
``budget`` snapshots, collective spends ``k * (budget // k)`` copies.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .channels import amplitude_damping_kraus, dephasing_kraus
from .ensembles import noisy_pure
from .hardened import _NOISE_ID, _aggregate_cell, _bootstrap_rmse_ci, _rate_key
from .moment_ustats import exact_moment_ustatistic
from .moments import collective_moment_estimate, depolarizing_moment_signal, moment
from .scaling import _apply_channel_dense, snapshots_factored

# Per-k feasibility cap for the single-copy exact U-statistic (efficient
# estimators): k=2 is O(M^2 n); k=3 is O(M 2^{2n}); k=4 adds an O(M n 2^{2n})
# term.  n=8 costs ~1.7 s (k=3) / ~7.5 s (k=4) per draw; beyond that, skip.
_MAX_N_BY_K = {2: 10, 3: 8, 4: 8}


def _resolve_caps(max_n_by_k: dict[int, int] | None, ks: tuple[int, ...]) -> dict[int, int]:
    """Resolve the per-k feasibility caps, failing fast if any ``k`` lacks a cap."""
    caps = dict(_MAX_N_BY_K if max_n_by_k is None else max_n_by_k)
    missing = [k for k in ks if k not in caps]
    if missing:
        raise ValueError(f"max_n_by_k is missing caps for k={missing}; provided keys={sorted(caps)}")
    return caps


def _true_moment(density: np.ndarray, k: int) -> float:
    return moment(density, k)


def _collective_signal(
    density: np.ndarray, n: int, k: int, noise_model: str, rate: float, true_moment: float
) -> float:
    """Noisy k-copy cyclic-test signal for ``Tr(rho^k)`` under a named noise model."""
    d = 2 ** n
    if noise_model == "depolarizing":
        return depolarizing_moment_signal(true_moment, k, rate, n)
    if noise_model == "amplitude_damping":
        kraus = amplitude_damping_kraus(rate)
    elif noise_model == "dephasing":
        kraus = dephasing_kraus(rate)
    else:
        raise ValueError(f"unknown noise_model {noise_model!r}")
    sigma = _apply_channel_dense(density, kraus, n)
    return moment(sigma, k)  # Tr(sigma^k), sigma = N^{ox n}(rho)


def moment_state_errors(
    n: int,
    state_idx: int,
    ks: tuple[int, ...],
    noise_models: tuple[str, ...],
    rates: tuple[float, ...],
    budget: int,
    n_trials: int,
    ensemble_q: float,
    seed: int,
    max_n_by_k: dict[int, int],
) -> dict:
    """Per-state squared errors for one ``(n, state_idx)`` unit (pure, deterministic).

    Every source of randomness (state, single-copy draws per k, and each
    ``(k, noise, rate)`` collective cell) draws from its own seeded substream, so
    numbers do not depend on loop ordering or which slices are swept.
    """
    if budget < max(ks):
        raise ValueError(f"budget must be >= max(k)={max(ks)}, got {budget}")
    caps = _resolve_caps(max_n_by_k, ks)
    state_rng = np.random.default_rng([seed, n, state_idx, 0])
    state = noisy_pure(n, ensemble_q, state_rng)
    density = state.density_matrix()

    true_moment = {k: _true_moment(density, k) for k in ks}
    single_se: dict[int, list[float] | None] = {}
    for k in ks:
        if n > caps[k]:
            single_se[k] = None  # infeasible at this (n, k); reported as skipped
            continue
        rng = np.random.default_rng([seed, n, state_idx, 1, k])
        tm = true_moment[k]
        single_se[k] = [
            (exact_moment_ustatistic(snapshots_factored(state, budget, rng), k) - tm) ** 2
            for _ in range(n_trials)
        ]

    coll_se: dict[str, list[float]] = {}
    for k in ks:
        tm = true_moment[k]
        n_uses = budget // k
        for noise_model in noise_models:
            for rate in rates:
                signal = _collective_signal(density, n, k, noise_model, rate, tm)
                # Seed collective substream by VALUE (k, noise id, rate key) so a
                # cell's draws are independent of loop order / grid slicing.
                cell_rng = np.random.default_rng(
                    [seed, n, state_idx, 2, k, _NOISE_ID[noise_model], _rate_key(rate)]
                )
                coll_se[f"{k}|{noise_model}@{rate}"] = [
                    (collective_moment_estimate(k, n_uses, signal, cell_rng) - tm) ** 2
                    for _ in range(n_trials)
                ]
    return {"n": n, "true_moment": true_moment, "single_se": single_se, "coll_se": coll_se}


def _moment_state_errors_star(packed: tuple) -> dict:
    """Top-level unpacker so ProcessPoolExecutor can pickle the work unit."""
    (n, state_idx), rest = packed
    return moment_state_errors(n, state_idx, *rest)


def run_moment_sweep(
    sizes: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    ks: tuple[int, ...] = (2, 3, 4),
    noise_models: tuple[str, ...] = ("depolarizing", "amplitude_damping", "dephasing"),
    rates: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1),
    ensemble_q: float = 0.1,
    budget: int = 2000,
    n_states: int = 48,
    n_trials: int = 10,
    seed: int = 0,
    max_n_by_k: dict[int, int] | None = None,
    max_workers: int | None = None,
) -> list[dict]:
    """Run the corrected moment sweep; one row per feasible ``(n, k, noise, rate)``.

    Each row carries the mean true ``Tr(rho^k)``, bootstrap RMSE intervals, and a
    paired-test verdict with ``paired_mse_diff +- se``.  Cells with
    ``n > max_n_by_k[k]`` are skipped (reported separately by the caller).
    """
    if budget < max(ks):
        raise ValueError(f"budget must be >= max(k)={max(ks)}, got {budget}")
    caps = _resolve_caps(max_n_by_k, ks)

    tasks = [(n, s) for n in sizes for s in range(n_states)]
    rest = (ks, noise_models, rates, budget, n_trials, ensemble_q, seed, caps)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_moment_state_errors_star, [(t, rest) for t in tasks]))

    grouped: dict[int, list[dict]] = {}
    for (n, _), res in zip(tasks, results):
        grouped.setdefault(n, []).append(res)

    boot_rng = np.random.default_rng([seed, 424242])
    rows: list[dict] = []
    for n in sorted(grouped):
        states = grouped[n]
        for k in ks:
            if n > caps[k]:
                continue  # infeasible single-copy at this (n, k)
            single_mse = np.array([np.mean(s["single_se"][k]) for s in states])
            single_ci = _bootstrap_rmse_ci(single_mse, boot_rng)
            mean_true = float(np.mean([s["true_moment"][k] for s in states]))
            for noise_model in noise_models:
                for rate in rates:
                    key = f"{k}|{noise_model}@{rate}"
                    coll_mse = np.array([np.mean(s["coll_se"][key]) for s in states])
                    cell = _aggregate_cell(single_mse, coll_mse, single_ci, boot_rng)
                    rows.append(
                        {
                            "n": int(n),
                            "k": int(k),
                            "noise_model": noise_model,
                            "rate": float(rate),
                            "budget": int(budget),
                            "single_copies": int(budget),
                            "collective_measurements": int(budget // k),
                            "n_states": int(len(states)),
                            "n_trials": int(n_trials),
                            "mean_true_moment": mean_true,
                            **cell,
                        }
                    )
    rows.sort(key=lambda r: (r["k"], r["n"], r["noise_model"], r["rate"]))
    return rows


def skipped_cells(
    sizes: tuple[int, ...], ks: tuple[int, ...], max_n_by_k: dict[int, int] | None = None
) -> list[dict]:
    """List ``(n, k)`` combinations skipped as infeasible for the single-copy exact."""
    caps = _resolve_caps(max_n_by_k, ks)
    return [{"n": int(n), "k": int(k), "reason": f"n > max_n_by_k[{k}]={caps[k]}"}
            for k in ks for n in sizes if n > caps[k]]


def save_moment_sweep(
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
