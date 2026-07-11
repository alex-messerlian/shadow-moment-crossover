"""Single-copy vs collective measurement benchmark for nonlinear functionals.

Estimators and an evaluation harness for comparing single-copy classical shadows
against the noisy 2-copy SWAP test when estimating ``Tr(rho^2)`` under gate
noise.  Estimators and harness only — no multi-task sweep, plots, or RL.

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
from .evaluation import (
    evaluate_estimator,
    make_collective_estimator,
    make_collective_moment_estimator,
    make_shadow_estimator,
    make_shadow_moment_estimator,
)
from .functionals import purity
from .moments import (
    collective_moment_estimate,
    cyclic_permutation_operator,
    depolarizing_moment_signal,
    fair_moment_ustatistic,
    full_moment_ustatistic_k3,
    kron_power,
    moment,
    moment_ustatistic_from_snapshots,
    shadow_moment_estimate,
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
]
