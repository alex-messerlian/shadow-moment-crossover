"""Real-hardware purity experiment circuits for Rigetti Cepheus (Open Quantum).

* :mod:`~anrl.hardware.backend` — auth, device metadata, coupling map (listing is
  free; no job is ever submitted here).
* :mod:`~anrl.hardware.state_prep` — reproducible state prep with exact recorded rho.
* :mod:`~anrl.hardware.swap_test` — the destructive (Bell-basis) SWAP test (collective).
* :mod:`~anrl.hardware.shadows` — local Pauli classical shadows (single-copy).
"""

from __future__ import annotations

from .backend import (
    CEPHEUS_BASIS_GATES,
    CEPHEUS_SHORT_CODE,
    cepheus_coupling_map,
    fetch_cepheus_metadata,
    list_backend_classes,
    load_cepheus_metadata,
)
from .shadows import pauli_shadow_circuits, shadow_purity, snapshots_from_outcomes
from .state_prep import PreparedState, bell_state, haar_pure, random_mixed
from .swap_test import destructive_swap_test, exact_swap_purity, purity_from_counts, swap_sign

__all__ = [
    "CEPHEUS_SHORT_CODE",
    "CEPHEUS_BASIS_GATES",
    "list_backend_classes",
    "fetch_cepheus_metadata",
    "load_cepheus_metadata",
    "cepheus_coupling_map",
    "PreparedState",
    "bell_state",
    "haar_pure",
    "random_mixed",
    "destructive_swap_test",
    "swap_sign",
    "purity_from_counts",
    "exact_swap_purity",
    "pauli_shadow_circuits",
    "shadow_purity",
    "snapshots_from_outcomes",
]
