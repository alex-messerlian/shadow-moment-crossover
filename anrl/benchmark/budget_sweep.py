"""Copy-budget sweep: out-of-sample test of the crossover law.

Sweeps the copy budget ``M`` over multipliers of a baseline and measures, at each
``(n, k, budget, noise, rate)`` cell, the single-copy RMSE and the collective
RMSE, to test three locked-in predictions:

* **P1** single-copy RMSE ``prop M^{-alpha}`` with ``alpha = 0.5`` (variance,
  ``1/sqrt(M)``);
* **P2** the collective error saturates at a budget-independent bias floor
  ``Bias = [1 - (1-g)^{k n}] Tr(rho^k)`` (increasing ``M`` cannot lower it);
* **P3** the crossover ``n*`` moves to larger ``n`` as the budget grows
  (single-copy improves with ``M``, the collective floor does not).

Feasibility of the EXACT single-copy U-statistic sets the reachable budgets per
``(n, k)`` (:func:`budgets_for`): k=2,3 are M-linear
(:func:`~anrl.benchmark.budget.moment_ustat_linear`) so reach large ``M``; k=4's
reference estimator is ``O(M^2)`` so is capped at small budgets.  Within a trial
the budgets are NESTED; one ``M_max`` sample is drawn and the U-statistic is read
off each prefix; so the budget axis is nearly free and its points are paired.

Statistics reuse the hardened harness: bootstrap 68% RMSE intervals, a paired
state-level MSE test (``|z|>2`` winner else 'tie', boundaries at ``|z|<3`` flagged
ambiguous), value-seeded per-cell substreams, one top-level seed.  Caveats
unchanged: no family-wise correction; the single-copy sample is shared across the
noise cells at a fixed ``(n, k, budget)``.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .budget import _K4_MAX_BUDGET, moment_ustat_linear, sample_batched
from .ensembles import noisy_pure
from .hardened import _NOISE_ID, _aggregate_cell, _bootstrap_rmse_ci, _rate_key
from .moments import collective_moment_estimate, moment
from .sweep_hardened import _collective_signal

BASELINE = 2000
# Budget multipliers reachable per k, before the n-dependent feasibility cap.
_BUDGETS_BY_K = {
    2: (2000, 8000, 32000, 128000),  # 1x, 4x, 16x, 64x
    3: (2000, 8000, 32000),          # 1x, 4x, 16x
    4: (500, 2000, 8000),            # 0.25x, 1x, 4x (exact k=4 is O(M^2))
}


def _budget_cap(n: int, k: int) -> int:
    """Largest exact-U-statistic budget that is feasible at this ``(n, k)``."""
    if k == 4:
        return _K4_MAX_BUDGET
    if k == 3:
        return 32000
    # k=2: 64x only at small n.  The dense split-Kron build is O(M 2^{2 ceil(n/2)}),
    # so the heaviest k=2 cell is max-n at 16x (n=9,M=32000), not n=6 at 64x.
    return 128000 if n <= 6 else 32000


def budgets_for(n: int, k: int) -> tuple[int, ...]:
    """Feasible budgets (ascending) for ``(n, k)`` = the plan filtered by the cap."""
    cap = _budget_cap(n, k)
    return tuple(b for b in _BUDGETS_BY_K[k] if b <= cap)


def predicted_bias_floor(n: int, k: int, rate: float, true_moment: float) -> float:
    """The law's collective bias floor ``[1 - (1-g)^{k n}] * Tr(rho^k)``."""
    return (1.0 - (1.0 - rate) ** (k * n)) * true_moment


def budget_state_errors(
    n: int,
    state_idx: int,
    ks: tuple[int, ...],
    noise_models: tuple[str, ...],
    rates: tuple[float, ...],
    n_trials: int,
    ensemble_q: float,
    seed: int,
) -> dict:
    """Per-state squared errors across budgets for one ``(n, state_idx)`` unit (pure).

    Nested budgets: for each trial one ``M_max`` sample is drawn and the exact
    single-copy U-statistic is evaluated on every budget prefix.  Collective draws
    ``budget // k`` binomial measurements of the noisy signal.  All substreams are
    value-seeded (k, budget, noise id, rate key).
    """
    state_rng = np.random.default_rng([seed, n, state_idx, 0])
    state = noisy_pure(n, ensemble_q, state_rng)
    density = state.density_matrix()

    true_moment = {k: moment(density, k) for k in ks}
    single_se: dict[tuple[int, int], list[float]] = {}
    coll_se: dict[tuple[int, int, str, float], list[float]] = {}
    signal_info: dict[tuple[int, str, float], dict] = {}

    for k in ks:
        budgets = budgets_for(n, k)
        m_max = max(budgets)
        tm = true_moment[k]
        # Single-copy: nested budgets from one M_max sample per trial.
        for k_budget in budgets:
            single_se[(k, k_budget)] = []
        for t in range(n_trials):
            rng = np.random.default_rng([seed, n, state_idx, 1, k, t])
            snaps = sample_batched(state, m_max, rng)
            for k_budget in budgets:
                est = moment_ustat_linear(snaps[:k_budget], k)
                single_se[(k, k_budget)].append((est - tm) ** 2)
        # Collective: signal is budget-independent; sample budget//k measurements.
        for noise_model in noise_models:
            for rate in rates:
                signal = _collective_signal(density, n, k, noise_model, rate, tm)
                signal_info[(k, noise_model, rate)] = {
                    "signal": float(signal),
                    "bias": float(abs(signal - tm)),
                    "pred_floor": predicted_bias_floor(n, k, rate, tm),
                }
                for k_budget in budgets:
                    crng = np.random.default_rng(
                        [seed, n, state_idx, 2, k, k_budget, _NOISE_ID[noise_model], _rate_key(rate)]
                    )
                    coll_se[(k, k_budget, noise_model, rate)] = [
                        (collective_moment_estimate(k, k_budget // k, signal, crng) - tm) ** 2
                        for _ in range(n_trials)
                    ]
    return {
        "n": n,
        "true_moment": true_moment,
        "single_se": single_se,
        "coll_se": coll_se,
        "signal_info": signal_info,
    }


def _budget_state_errors_star(packed: tuple) -> dict:
    (n, state_idx), rest = packed
    return budget_state_errors(n, state_idx, *rest)


def run_budget_sweep(
    sizes_by_k: dict[int, tuple[int, ...]] | None = None,
    noise_models: tuple[str, ...] = ("depolarizing", "amplitude_damping", "dephasing"),
    rates: tuple[float, ...] = (0.05, 0.1),
    ensemble_q: float = 0.1,
    n_states: int = 48,
    n_trials: int = 8,
    seed: int = 0,
    max_workers: int | None = None,
    return_alpha_fits: bool = False,
) -> list[dict] | tuple[list[dict], list[dict]]:
    """Run the budget sweep; one row per ``(n, k, budget, noise_model, rate)``.

    ``sizes_by_k`` maps ``k -> sizes``; default reaches n=9 (k=2), n=8 (k=3),
    n=6 (k=4).  Each row carries the single/collective RMSE with bootstrap
    intervals, the paired verdict, and the predicted vs measured bias floor.  With
    ``return_alpha_fits=True`` also returns the per-``(n, k)`` P1 exponent fits
    (``alpha`` with a state-bootstrap SE).
    """
    plan = sizes_by_k or {2: tuple(range(2, 10)), 3: tuple(range(2, 9)), 4: tuple(range(2, 7))}
    ks = tuple(sorted(plan))
    all_ns = sorted({n for ns in plan.values() for n in ns})

    # A work unit computes every k feasible at its n; filter per k inside via `ks_here`.
    tasks = [(n, s) for n in all_ns for s in range(n_states)]
    # Each unit is told which ks apply to its n.
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        packed = []
        for (n, s) in tasks:
            ks_here = tuple(k for k in ks if n in plan[k])
            packed.append(((n, s), (ks_here, noise_models, rates, n_trials, ensemble_q, seed)))
        results = list(pool.map(_budget_state_errors_star, packed))

    grouped: dict[int, list[dict]] = {}
    for (n, _), res in zip(tasks, results):
        grouped.setdefault(n, []).append(res)

    boot_rng = np.random.default_rng([seed, 707070])
    rows: list[dict] = []
    alpha_fits: list[dict] = []
    for n in sorted(grouped):
        states = grouped[n]
        ks_here = tuple(k for k in ks if n in plan[k])
        for k in ks_here:
            budgets = budgets_for(n, k)
            # P1: fit the budget-scaling exponent alpha with a state-level bootstrap
            # SE (nested budgets share a draw, so residual SE would be optimistic).
            mse_by_budget = np.array(
                [[np.mean(s["single_se"][(k, b)]) for s in states] for b in budgets]
            )  # (n_budgets, n_states)
            alpha, alpha_se = fit_budget_exponent_bootstrap(list(budgets), mse_by_budget, boot_rng)
            alpha_fits.append({
                "n": int(n), "k": int(k), "budgets": list(budgets),
                "single_rmse": [float(np.sqrt(mse_by_budget[i].mean())) for i in range(len(budgets))],
                "alpha": float(alpha), "alpha_se": float(alpha_se),
            })
            for k_budget in budgets:
                single_mse = np.array([np.mean(s["single_se"][(k, k_budget)]) for s in states])
                single_ci = _bootstrap_rmse_ci(single_mse, boot_rng)
                mean_true = float(np.mean([s["true_moment"][k] for s in states]))
                for noise_model in noise_models:
                    for rate in rates:
                        coll_mse = np.array(
                            [np.mean(s["coll_se"][(k, k_budget, noise_model, rate)]) for s in states]
                        )
                        cell = _aggregate_cell(single_mse, coll_mse, single_ci, boot_rng)
                        infos = [s["signal_info"][(k, noise_model, rate)] for s in states]
                        rows.append(
                            {
                                "n": int(n),
                                "k": int(k),
                                "budget": int(k_budget),
                                "budget_mult": round(k_budget / BASELINE, 4),
                                "noise_model": noise_model,
                                "rate": float(rate),
                                "single_copies": int(k_budget),
                                "collective_measurements": int(k_budget // k),
                                "n_states": int(len(states)),
                                "n_trials": int(n_trials),
                                "mean_true_moment": mean_true,
                                "measured_bias": float(np.mean([i["bias"] for i in infos])),
                                "predicted_floor": float(np.mean([i["pred_floor"] for i in infos])),
                                **cell,
                            }
                        )
    rows.sort(key=lambda r: (r["k"], r["noise_model"], r["rate"], r["budget"], r["n"]))
    alpha_fits.sort(key=lambda a: (a["k"], a["n"]))
    if return_alpha_fits:
        return rows, alpha_fits
    return rows


def _ols_neg_slope(logm: np.ndarray, logr: np.ndarray) -> float:
    xm = logm.mean()
    sxx = float(((logm - xm) ** 2).sum())
    return -float(((logm - xm) * (logr - logr.mean())).sum() / sxx)


def fit_budget_exponent(budgets: list[int], rmses: list[float]) -> tuple[float, float]:
    """Fit ``log RMSE = -alpha log M + c``; return ``(alpha, residual stderr(alpha))``.

    The law predicts ``alpha = 0.5``.  OLS on log-log; the returned stderr is the
    residual-scatter estimate, which ASSUMES independent budget points.  When the
    budgets are nested (share one M_max draw, as in this sweep) that assumption is
    violated and this stderr is optimistic; prefer
    :func:`fit_budget_exponent_bootstrap`, which resamples the iid state unit.
    """
    x = np.log(np.asarray(budgets, dtype=np.float64))
    y = np.log(np.clip(np.asarray(rmses, dtype=np.float64), 1e-12, None))
    nfit = len(x)
    if nfit < 2:
        return (float("nan"), float("nan"))
    slope = _ols_neg_slope(x, y)
    if nfit == 2:
        return (slope, float("nan"))
    xm = x.mean()
    sxx = float(((x - xm) ** 2).sum())
    resid = y - (-slope * (x - xm) + y.mean())
    s2 = float((resid ** 2).sum() / (nfit - 2))
    return (slope, float(np.sqrt(s2 / sxx)))


def fit_budget_exponent_bootstrap(
    budgets: list[int],
    mse_by_budget: np.ndarray,  # (n_budgets, n_states)
    rng: np.random.Generator,
    n_resamples: int = 2000,
) -> tuple[float, float]:
    """``(alpha, bootstrap_se)`` for ``RMSE prop M^{-alpha}``, resampling STATES.

    The point ``alpha`` is the OLS slope of ``log RMSE`` vs ``log M`` with
    ``RMSE(M) = sqrt(mean_states MSE)``.  The standard error resamples the iid
    state unit CONSISTENTLY across budgets (each resampled state carries its whole
    correlated budget vector), so the nested-sample cross-budget correlation is
    respected, unlike the residual stderr, which ignores it and is optimistic.
    """
    n_states = mse_by_budget.shape[1]
    rmse = np.sqrt(np.clip(mse_by_budget.mean(axis=1), 1e-24, None))
    if len(budgets) < 2:
        return (float("nan"), float("nan"))
    logm = np.log(np.asarray(budgets, dtype=np.float64))
    alpha = _ols_neg_slope(logm, np.log(rmse))
    if n_states < 2:
        return (alpha, float("nan"))
    idx = rng.integers(0, n_states, size=(n_resamples, n_states))
    boot = np.empty(n_resamples)
    for b in range(n_resamples):
        rr = np.sqrt(np.clip(mse_by_budget[:, idx[b]].mean(axis=1), 1e-24, None))
        boot[b] = _ols_neg_slope(logm, np.log(rr))
    return (alpha, float(np.std(boot, ddof=1)))


def save_budget_sweep(
    rows: list[dict],
    table: list[dict],
    path: str | Path,
    metadata: dict | None = None,
    alpha_fits: list[dict] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "metadata": metadata or {},
                "alpha_fits": alpha_fits or [],
                "rows": rows,
                "crossover_table": table,
            },
            handle,
            indent=2,
        )
    return path
