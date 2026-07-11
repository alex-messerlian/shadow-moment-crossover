"""Single-copy vs collective measurement benchmark for nonlinear functionals.

Estimators and an evaluation harness for comparing single-copy classical shadows
against the noisy 2-copy SWAP test when estimating ``Tr(rho^2)`` under gate
noise.  Estimators and harness only — no multi-task sweep, plots, or RL.

Built on the Phase 1-2 physics core (states, Pauli machinery, purity).
"""

from __future__ import annotations

from .collective import (
    collective_purity_estimate,
    collective_signal,
    gates_all_to_all,
    gates_linear_1d,
)
from .evaluation import (
    evaluate_estimator,
    make_collective_estimator,
    make_shadow_estimator,
)
from .functionals import purity
from .shadows import haar_unitary, shadow_purity_estimate

__all__ = [
    # functional
    "purity",
    # single-copy shadows
    "haar_unitary",
    "shadow_purity_estimate",
    # collective SWAP test
    "collective_signal",
    "collective_purity_estimate",
    "gates_all_to_all",
    "gates_linear_1d",
    # harness
    "evaluate_estimator",
    "make_shadow_estimator",
    "make_collective_estimator",
]
