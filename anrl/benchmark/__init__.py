"""Single-copy vs collective measurement benchmark for nonlinear functionals.

Estimators and an evaluation harness for comparing single-copy classical shadows
against the noisy 2-copy SWAP test when estimating ``Tr(rho^2)`` under gate
noise.  Estimators and harness only; no multi-task sweep, plots, or RL.

Built on the Phase 1-2 physics core (states, Pauli machinery, purity).
"""

from __future__ import annotations

from .channels import (
    NOISE_MODELS,
    amplitude_damping_kraus,
    apply_channel_per_qubit,
    channel_collective_signal,
    collective_moment_signal,
    dephasing_kraus,
    explicit_channel_collective_signal,
)
from .collective import (
    collective_purity_estimate,
    collective_signal,
    gates_all_to_all,
    gates_linear_1d,
)
from .ensembles import NoisyState, ghz_noisy, haar_pure, low_rank, noisy_pure, random_mixed
from .evaluation import (
    evaluate_estimator,
    make_collective_estimator,
    make_collective_moment_estimator,
    make_shadow_estimator,
    make_shadow_moment_estimator,
)
from .functionals import purity
from .budget import moment_ustat_linear, sample_batched
from .budget_sweep import (
    budgets_for,
    fit_budget_exponent,
    fit_budget_exponent_bootstrap,
    predicted_bias_floor,
    run_budget_sweep,
    save_budget_sweep,
)
from .moment_ustats import exact_moment_ustatistic
from .moments import (
    collective_moment_estimate,
    cyclic_permutation_operator,
    depolarizing_moment_signal,
    fair_moment_ustatistic,
    full_moment_ustatistic_k3,
    full_moment_ustatistic_k4,
    kron_power,
    moment,
    moment_ustatistic_from_snapshots,
    shadow_moment_estimate,
)
from .hardened import (
    crossover_table,
    run_hardened,
    save_hardened,
    state_errors,
)
from .scaling import (
    collective_purity_signal,
    run_scaling,
    save_scaling,
    snapshots_factored,
)
from .sweep_hardened import (
    moment_state_errors,
    run_moment_sweep,
    save_moment_sweep,
    skipped_cells,
)
from .shadows import full_purity_ustatistic, haar_unitary, shadow_purity_estimate
from .sweep import run_sweep, save_sweep

__all__ = [
    # functionals / moments
    "purity",
    "moment",
    # single-copy shadows
    "haar_unitary",
    "shadow_purity_estimate",
    "full_purity_ustatistic",
    "shadow_moment_estimate",
    "fair_moment_ustatistic",
    "full_moment_ustatistic_k3",
    "full_moment_ustatistic_k4",
    # collective cyclic test
    "collective_signal",
    "collective_purity_estimate",
    "cyclic_permutation_operator",
    "kron_power",
    "depolarizing_moment_signal",
    "collective_moment_estimate",
    "moment_ustatistic_from_snapshots",
    "gates_all_to_all",
    "gates_linear_1d",
    # noise channels
    "NOISE_MODELS",
    "amplitude_damping_kraus",
    "dephasing_kraus",
    "apply_channel_per_qubit",
    "channel_collective_signal",
    "explicit_channel_collective_signal",
    "collective_moment_signal",
    # harness
    "evaluate_estimator",
    "make_shadow_estimator",
    "make_collective_estimator",
    "make_shadow_moment_estimator",
    "make_collective_moment_estimator",
    # sweep
    "run_sweep",
    "save_sweep",
    # scaling study (single-copy vs collective vs system size)
    "NoisyState",
    "noisy_pure",
    "random_mixed",
    "haar_pure",
    "low_rank",
    "ghz_noisy",
    "snapshots_factored",
    "collective_purity_signal",
    "run_scaling",
    "save_scaling",
    # hardened scaling study (error bars + paired crossover test)
    "run_hardened",
    "save_hardened",
    "crossover_table",
    "state_errors",
    # efficient exact moment U-statistics (k=2,3,4, scale to large n)
    "exact_moment_ustatistic",
    # budget-scaling primitives + sweep (M-linear exact U-statistics)
    "sample_batched",
    "moment_ustat_linear",
    "run_budget_sweep",
    "save_budget_sweep",
    "budgets_for",
    "predicted_bias_floor",
    "fit_budget_exponent",
    "fit_budget_exponent_bootstrap",
    # corrected hardened moment sweep
    "run_moment_sweep",
    "save_moment_sweep",
    "moment_state_errors",
    "skipped_cells",
]
