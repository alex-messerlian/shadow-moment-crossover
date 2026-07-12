"""Part 3 — the crossover, re-solved with the corrected laws.

The crossover ``n*`` is the smallest system size where the single-copy RMSE rises
above the collective error.  We predict both sides analytically and locate the
crossover, then compare cell-by-cell against the measured crossovers saved in
``results/``.

* Single-copy RMSE: :func:`~anrl.theory.variance.single_copy_rmse` from the
  Hoeffding zetas (Part 2).
* Collective RMSE: the exact bias floor (Part 1) plus the binomial sampling
  variance ``(1 - signal^2) / (M // k)`` of the ``M // k`` cyclic-test shots.

``n*`` is the smallest ``n`` in the swept range where ``single_rmse >
collective_rmse``; ``None`` if single-copy never overtakes (a predicted no-cross).
"""

from __future__ import annotations

import json
from pathlib import Path

from .bias import collective_bias, collective_value
from .variance import exact_single_copy_rmse, single_copy_rmse


_N_BIAS_STATES = 8  # states to average the (state-dependent) per-qubit bias over


def noisy_pure_moment(n: int, k: int, q: float) -> float:
    """Exact ``Tr(rho^k)`` for the noisy-pure state ``(1-q)|psi><psi| + q I/2^n``."""
    d = 2 ** n
    lam1, lam0 = (1.0 - q) + q / d, q / d
    return lam1 ** k + (d - 1) * lam0 ** k


def _rho(n: int, q: float, state_idx: int = 0):
    """Dense noisy-pure ``rho`` from a fixed ``|psi>`` (seed-stable per state_idx)."""
    import numpy as np

    from anrl.benchmark.ensembles import noisy_pure
    return noisy_pure(n, q, np.random.default_rng([0, n, state_idx, 0])).density_matrix()


def predicted_collective_rmse(n: int, k: int, noise_model: str, g: float, budget: int, q: float) -> float:
    """Bias floor (Part 1) plus binomial shot noise of the ``budget // k`` cyclic shots.

    The per-qubit bias is state-dependent, so ``E_s[bias^2]`` and ``E_s[signal]``
    are averaged over a few states (matching the ensemble RMSE the sweep measured).
    Depolarizing is state-independent (a closed form in ``Tr(rho^k)``).
    """
    import numpy as np

    n_states = 1 if noise_model == "depolarizing" else _N_BIAS_STATES
    biases, signals = [], []
    for s in range(n_states):
        rho = _rho(n, q, s)
        biases.append(collective_bias(rho, k, noise_model, g, n))
        signals.append(collective_value(rho, k, noise_model, g, n))
    mean_bias_sq = float(np.mean(np.square(biases)))
    mean_signal = float(np.mean(signals))
    n_meas = max(1, budget // k)
    var_shot = max(0.0, 1.0 - mean_signal * mean_signal) / n_meas
    return float((mean_bias_sq + var_shot) ** 0.5)


def predicted_single_rmse(n: int, k: int, budget: int, zetas: dict, q: float, model: str = "two_term") -> float:
    """Theory single-copy RMSE for ``(n, k)`` under ``model`` in {two_term, exact}.

    ``two_term`` is the task's specified ``sqrt(k^2 zeta1/M + zeta2/M^2)``; ``exact``
    is the full Hoeffding-decomposition RMSE from the component vector.
    """
    z = zetas[(n, k)]
    if model == "exact":
        return exact_single_copy_rmse(z["components"], k, budget)
    return single_copy_rmse(k, z["zeta1"], z["zeta2"], budget)


def predict_crossover(
    k: int, noise_model: str, g: float, budget: int, sizes: list[int], zetas: dict, q: float,
    model: str = "two_term",
) -> int | None:
    """The SUSTAINED crossover: smallest ``n`` from which single-copy RMSE exceeds
    collective for the rest of the swept range.

    Single-copy RMSE grows ~exponentially while the collective floor saturates, so
    physically there is a single crossover; requiring it to be sustained matches
    the measured ``crossover_n`` (which, for the resolved cells, has a monotone
    ``single...single collective...collective`` winner sequence) and avoids a
    spurious near-tie flip at small ``n``.
    """
    ns = [n for n in sorted(sizes) if (n, k) in zetas]
    wins = {n: predicted_single_rmse(n, k, budget, zetas, q, model)
            > predicted_collective_rmse(n, k, noise_model, g, budget, q) for n in ns}
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None


# ---------------------------------------------------------------------------
# Loading measured crossovers from the saved sweeps
# ---------------------------------------------------------------------------
def load_measured_crossovers(path: str | Path, default_budget: int) -> list[dict]:
    """Measured crossover cells from a saved sweep's ``crossover_table``.

    ``budget_scaling.json`` entries carry ``budget``; the corrected sweep does not
    (fixed baseline), so ``default_budget`` is attached there.  Each returned cell
    has (k, noise_model, rate, budget, crossover_n, crossover_z, ambiguous, sizes).
    """
    payload = json.loads(Path(path).read_text())
    cells = []
    for e in payload["crossover_table"]:
        sizes = sorted(int(x) for x in e["winners_by_n"])
        cells.append({
            "k": int(e["k"]), "noise_model": e["noise_model"], "rate": float(e["rate"]),
            "budget": int(e.get("budget", default_budget)),
            "measured_n": e["crossover_n"], "z": e.get("crossover_z"),
            "ambiguous": bool(e.get("ambiguous", False)), "sizes": sizes,
        })
    return cells


def build_comparison(measured_cells: list[dict], zetas: dict, q: float) -> list[dict]:
    """Attach the predicted crossover (both models) to each measured cell."""
    out = []
    for c in measured_cells:
        m = c["measured_n"]
        row = {**c}
        for model, suffix in (("two_term", ""), ("exact", "_exact")):
            pred = predict_crossover(c["k"], c["noise_model"], c["rate"], c["budget"],
                                     c["sizes"], zetas, q, model)
            row[f"predicted_n{suffix}"] = pred
            row[f"delta{suffix}"] = (pred - m) if (pred is not None and m is not None) else None
        out.append(row)
    return out
