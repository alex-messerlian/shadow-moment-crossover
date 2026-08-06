"""Loaders for the saved results JSON and the (cheap, deterministic) theory curves.

Figures read measured numbers straight from ``results/*.json`` (no experiment is
re-run).  Theory curves are recomputed from :mod:`anrl.theory`, parameter-free
and deterministic, using the Hoeffding components already saved in
``theory_zetas.json`` (so even the components are not re-estimated).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from anrl.theory.crossover import predicted_collective_rmse
from anrl.theory.variance import exact_fitted_alpha, exact_single_copy_rmse

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
Q = 0.1  # noisy-pure depolarizing weight used throughout the study

# Which zeta source the theory curves read.  Default: the exact closed-form file
# (k=2 ensemble-averaged closed forms, k>=3 carried from the sampled file unchanged).
# Set ANRL_ZETAS_FILE=theory_zetas.json to fall back to the pure Monte Carlo file.
ZETAS_FILE = os.environ.get("ANRL_ZETAS_FILE", "theory_zetas_closedform.json")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def zetas_raw() -> list:
    """The saved zeta entries from the currently selected source (:data:`ZETAS_FILE`)."""
    return load(ZETAS_FILE)["zetas"]


def zeta_components() -> dict:
    """``(n, k) -> [zeta_1, ..., zeta_k]`` from the currently selected zeta source."""
    return {(z["n"], z["k"]): z["components"] for z in zetas_raw()}


def theory_single_rmse(n: int, k: int, budget: int, comps: dict) -> float | None:
    """Parameter-free single-copy RMSE from the exact Hoeffding decomposition."""
    if (n, k) not in comps:
        return None
    return exact_single_copy_rmse(comps[(n, k)], k, budget)


def theory_collective_rmse(n: int, k: int, noise_model: str, g: float, budget: int) -> float:
    """Parameter-free collective RMSE = exact bias floor + binomial shot noise."""
    return predicted_collective_rmse(n, k, noise_model, g, budget, Q)


def theory_alpha(budgets: list[int], k: int, comps: dict, n: int) -> float | None:
    """Parameter-free budget exponent alpha from the exact-variance RMSE curve."""
    if (n, k) not in comps:
        return None
    return exact_fitted_alpha(budgets, comps[(n, k)], k)
