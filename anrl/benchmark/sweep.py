"""Multi-task, multi-noise robustness sweep: single-copy vs collective.

For each moment order ``k``, noise model, noise rate, and system size ``n`` at a
fixed copy budget, compares the RMSE of the single-copy classical-shadow
estimate of ``Tr(rho^k)`` against the k-copy collective (cyclic-test) estimate
over a shared set of random noisy states.

The gate noise lives on the *collective measurement apparatus* (the k-copy
register), not on the state, so the single-copy shadow RMSE is
**noise-independent** for a given ``(n, k)``, only the collective estimator is
degraded by noise.

Two single-copy estimators are reported from the SAME snapshots (same copy
budget), differing only in classical post-processing:

* ``subsampled``; the ``n_snapshots // k`` U-statistic tuple convention (the
  existing O(M) convention; variance-inflating).
* ``fair``; the copy-fair estimator (forming tuples costs no copies): the EXACT
  full U-statistic for k=2, k=3 AND k=4 (the k=4 exact U-statistic is the Mobius
  inversion over the 15 set partitions of the 4 cyclic slots, verified against
  brute force; so no subsampling artifact remains at any k here).  The gap
  between the two conventions is pure post-processing, not measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.physics import depolarize, random_density

from .channels import NOISE_MODELS, collective_moment_signal
from .moments import (
    collective_moment_estimate,
    fair_moment_ustatistic,
    moment,
    moment_ustatistic_from_snapshots,
)
from .shadows import _snapshots


def _make_states(n: int, n_states: int, rng: np.random.Generator) -> list[np.ndarray]:
    dim = 2 ** n
    return [
        depolarize(random_density(dim, int(rng.integers(1, dim + 1)), rng), rng.uniform(0.0, 0.3))
        for _ in range(n_states)
    ]


def _rmse(errors: list[float]) -> float:
    arr = np.asarray(errors, dtype=np.float64)
    return float(np.sqrt(np.mean(arr ** 2)))


def _single_copy_rmse(states, n, k, budget, rng):
    """RMSE of the single-copy shadow estimate of Tr(rho^k), both conventions.

    Both estimates are formed from the SAME ``budget`` snapshots per state; only
    the classical post-processing differs.  ``fair`` is the copy-optimal
    estimator (EXACT full U-statistic for k=2,3,4; large subsample only for
    k>=5); ``subsampled`` is the old variance-inflating ``budget // k`` convention.
    """
    subsampled, fair = [], []
    m_sub = max(1, budget // k)
    for state in states:
        snaps = _snapshots(state, n, budget, rng)
        true = moment(state, k)
        subsampled.append(abs(moment_ustatistic_from_snapshots(snaps, k, m_sub, rng) - true))
        fair.append(abs(fair_moment_ustatistic(snaps, k, rng) - true))
    return _rmse(subsampled), _rmse(fair)


def _collective_rmse(states, k, noise_model, rate, budget, rng):
    n_uses = budget // k
    errors = []
    for state in states:
        n = int(round(np.log2(state.shape[0])))
        signal = collective_moment_signal(state, k, noise_model, rate, n)
        errors.append(abs(collective_moment_estimate(k, n_uses, signal, rng) - moment(state, k)))
    return _rmse(errors), n_uses


def run_sweep(
    sizes: tuple[int, ...] = (2, 3, 4),
    ks: tuple[int, ...] = (2, 3, 4),
    noise_models: tuple[str, ...] = NOISE_MODELS,
    rates: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1),
    budget: int = 2000,
    n_states: int = 12,
    seed: int = 0,
) -> list[dict]:
    """Run the full sweep; one row per (n, k, noise_model, rate) cell.

    Each row reports the collective RMSE and BOTH single-copy RMSEs (the O(M)
    ``subsampled`` convention and the copy-optimal ``fair`` estimator), and the
    winner under each.
    """
    rows: list[dict] = []
    for n in sizes:
        states = _make_states(n, n_states, np.random.default_rng([seed, n, 0]))
        for k in ks:
            single_sub, single_fair = _single_copy_rmse(
                states, n, k, budget, np.random.default_rng([seed, n, k, 1])
            )
            for noise_model in noise_models:
                for rate in rates:
                    collective_rmse, n_uses = _collective_rmse(
                        states, k, noise_model, rate, budget,
                        np.random.default_rng(
                            [seed, n, k, noise_models.index(noise_model), int(round(rate * 1000))]
                        ),
                    )
                    rows.append(
                        {
                            "n": int(n),
                            "k": int(k),
                            "noise_model": noise_model,
                            "rate": float(rate),
                            "budget": int(budget),
                            "single_copies": int(budget),
                            "collective_measurements": int(n_uses),
                            "single_rmse_subsampled": float(single_sub),
                            "single_rmse_fair": float(single_fair),
                            "collective_rmse": float(collective_rmse),
                            "winner_subsampled": "collective" if collective_rmse < single_sub else "single-copy",
                            "winner_fair": "collective" if collective_rmse < single_fair else "single-copy",
                            "factor_subsampled": float(single_sub / collective_rmse) if collective_rmse > 0 else float("inf"),
                            "factor_fair": float(single_fair / collective_rmse) if collective_rmse > 0 else float("inf"),
                        }
                    )
    return rows


def save_sweep(rows: list[dict], path: str | Path, metadata: dict | None = None) -> Path:
    """Write the sweep rows (and optional metadata) to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"metadata": metadata or {}, "rows": rows}, handle, indent=2)
    return path
