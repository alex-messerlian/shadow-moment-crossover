"""Loaders for the saved results JSON and the (cheap, deterministic) theory curves.

Figures read measured numbers straight from ``results/*.json`` (no experiment is
re-run).  Theory curves are recomputed from :mod:`anrl.theory` — parameter-free
and deterministic — using the Hoeffding components already saved in
``theory_zetas.json`` (so even the components are not re-estimated).
"""

from __future__ import annotations

import json
from pathlib import Path

from anrl.theory.crossover import predicted_collective_rmse
from anrl.theory.variance import exact_fitted_alpha, exact_single_copy_rmse

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
Q = 0.1  # noisy-pure depolarizing weight used throughout the study


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def zeta_components() -> dict:
    """``(n, k) -> [zeta_1, ..., zeta_k]`` from the saved Hoeffding estimates."""
    return {(z["n"], z["k"]): z["components"] for z in load("theory_zetas.json")["zetas"]}


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
