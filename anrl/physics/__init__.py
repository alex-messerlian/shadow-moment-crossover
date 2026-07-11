"""Physics core for adaptive-negativity-rl.

Exact bipartite-entanglement computation and single-copy local-Pauli
measurement simulation for n-qubit density matrices.  No reinforcement-learning,
environment, or agent logic lives here — this package is pure quantum-state
math and measurement.

Submodules
----------
* :mod:`~anrl.physics.pauli`        — Pauli operators and n-qubit Pauli strings.
* :mod:`~anrl.physics.states`       — reference states and noise channels.
* :mod:`~anrl.physics.entanglement` — partial transpose, negativity, PT moments.
* :mod:`~anrl.physics.measurement`  — local-Pauli sampling and reconstruction.
"""

from __future__ import annotations

from .entanglement import (
    negativity,
    negativity_from_moments,
    partial_transpose,
    pt_moment,
    purity,
)
from .measurement import (
    all_local_pauli_settings,
    estimate_pauli_expectations,
    measurement_unitary,
    outcome_probabilities,
    project_to_density_matrix,
    reconstruct,
    sample_counts,
    simulate_settings,
    single_qubit_rotation,
)
from .pauli import PAULI, PAULI_LABELS, kron_all, pauli_matrix, pauli_string
from .states import (
    bell_phi_plus,
    depolarize,
    ghz,
    maximally_mixed,
    random_density,
    werner,
)

__all__ = [
    # pauli
    "PAULI",
    "PAULI_LABELS",
    "pauli_matrix",
    "pauli_string",
    "kron_all",
    # states
    "bell_phi_plus",
    "werner",
    "ghz",
    "random_density",
    "depolarize",
    "maximally_mixed",
    # entanglement
    "partial_transpose",
    "negativity",
    "pt_moment",
    "purity",
    "negativity_from_moments",
    # measurement
    "single_qubit_rotation",
    "measurement_unitary",
    "outcome_probabilities",
    "sample_counts",
    "simulate_settings",
    "all_local_pauli_settings",
    "estimate_pauli_expectations",
    "reconstruct",
    "project_to_density_matrix",
]
