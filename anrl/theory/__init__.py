"""Analytic theory of the single-copy vs collective crossover.

* :mod:`~anrl.theory.bias` — Part 1: the exact, parameter-free collective bias
  laws (global depolarizing; per-qubit channel).
* :mod:`~anrl.theory.variance` — Part 2: the single-copy variance law (Hoeffding
  decomposition, the budget threshold ``M*``, the effective exponent).
* :mod:`~anrl.theory.crossover` — Part 3: the crossover predictor and the
  comparison against the measured sweeps in ``results/``.
"""

from __future__ import annotations

from .bias import (
    brute_force_collective_value,
    collective_bias,
    collective_value,
    depolarizing_bias,
    depolarizing_collective_value,
    perqubit_channel_bias,
    perqubit_channel_value,
)
from .crossover import (
    build_comparison,
    load_measured_crossovers,
    noisy_pure_moment,
    predict_crossover,
    predicted_collective_rmse,
    predicted_single_rmse,
)
from .variance import (
    alpha_eff,
    estimate_hoeffding_components,
    estimate_zeta1,
    estimate_zeta2,
    estimate_zetas,
    exact_fitted_alpha,
    exact_single_copy_rmse,
    exact_ustatistic_variance,
    fitted_alpha,
    single_copy_rmse,
    single_copy_variance,
)

__all__ = [
    # bias laws (Part 1)
    "depolarizing_bias",
    "depolarizing_collective_value",
    "perqubit_channel_bias",
    "perqubit_channel_value",
    "collective_bias",
    "collective_value",
    "brute_force_collective_value",
    # variance law (Part 2)
    "estimate_zeta1",
    "estimate_zeta2",
    "estimate_zetas",
    "estimate_hoeffding_components",
    "single_copy_variance",
    "single_copy_rmse",
    "exact_ustatistic_variance",
    "exact_single_copy_rmse",
    "alpha_eff",
    "fitted_alpha",
    "exact_fitted_alpha",
    # crossover (Part 3)
    "noisy_pure_moment",
    "predicted_single_rmse",
    "predicted_collective_rmse",
    "predict_crossover",
    "load_measured_crossovers",
    "build_comparison",
]
