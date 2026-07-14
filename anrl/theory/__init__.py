"""Analytic theory of the single-copy vs collective crossover.

* :mod:`~anrl.theory.bias` — Part 1: the exact, parameter-free collective bias
  laws (global depolarizing; per-qubit channel).
* :mod:`~anrl.theory.variance` — Part 2: the single-copy variance law (Hoeffding
  decomposition, the budget threshold ``M*``, the effective exponent).
* :mod:`~anrl.theory.single_copy_law` — Part 2 (derived + independently verified):
  the exact k=2 Hoeffding variance ``[4(M-2)zeta1 + 2 zeta2]/[M(M-1)]``, the
  corrected crossover ``M* = zeta2/(2 zeta1)``, the alpha predictor, and the
  single-qubit closed form.
* :mod:`~anrl.theory.crossover` — Part 3: the crossover predictor and the
  comparison against the measured sweeps in ``results/``.
* :mod:`~anrl.theory.clipping` — closed-form RMSE of a Gaussian purity estimate
  clipped to a physical range [a, b], for pipelines that clip (anrl's own does not).
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
from .general import (
    estimate_hoeffding_components_general,
    predict_crossover_general,
    predicted_collective_rmse_general,
    sample_batched_general,
)
from .clipping import clipped_mse, clipped_rmse
from .single_copy_law import (
    REFERENCE_SCALINGS_Q0_1,
    crossover_budget,
    hoeffding_rmse,
    hoeffding_variance,
    predicted_alpha,
    single_qubit_second_moment,
    single_qubit_zeta1,
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
    # first-principles single-copy law (derived + verified)
    "hoeffding_variance",
    "hoeffding_rmse",
    "crossover_budget",
    "predicted_alpha",
    "single_qubit_second_moment",
    "single_qubit_zeta1",
    "REFERENCE_SCALINGS_Q0_1",
    # clipped-estimator RMSE (for pipelines that clip to a physical range)
    "clipped_mse",
    "clipped_rmse",
    # crossover (Part 3)
    "noisy_pure_moment",
    "predicted_single_rmse",
    "predicted_collective_rmse",
    "predict_crossover",
    "load_measured_crossovers",
    "build_comparison",
    # state-agnostic estimators (stress test: any ensemble)
    "sample_batched_general",
    "estimate_hoeffding_components_general",
    "predicted_collective_rmse_general",
    "predict_crossover_general",
]
