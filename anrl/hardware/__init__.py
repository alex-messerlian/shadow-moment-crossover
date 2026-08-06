"""Real-hardware purity experiment circuits for Rigetti Cepheus (Open Quantum).

* :mod:`~anrl.hardware.backend`, auth, device metadata, coupling map (listing is
  free; no job is ever submitted here).
* :mod:`~anrl.hardware.state_prep`, reproducible state prep with exact recorded rho.
* :mod:`~anrl.hardware.swap_test`; the destructive (Bell-basis) SWAP test (collective).
* :mod:`~anrl.hardware.shadows`, local Pauli classical shadows (single-copy).
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
from .calibration import (
    CEPHEUS_SQUARE,
    bell_calibration_surface,
    consistent_error_rates,
    effective_g_from_purity,
    gate_noisy_probs,
    invert_measured_to_p2,
    measured_swap_purity,
    measured_swap_purity_ensemble,
    purity_from_g,
    swap_purity_from_probs,
)
from .noise_model import (
    REF_P1,
    REF_P2,
    REF_P_RO,
    avg_gate_error_to_depol_param,
    cepheus_noise_model,
    depol_param_to_avg_gate_error,
    readout_confusion_matrix,
)
from .shadows import pauli_shadow_circuits, shadow_purity, snapshots_from_outcomes
from .state_prep import (
    MixedEnsemble,
    PreparedState,
    bell_state,
    haar_pure,
    mixed_ensemble,
    random_mixed,
)
from .swap_test import destructive_swap_test, exact_swap_purity, purity_from_counts, swap_sign

__all__ = [
    # backend / device
    "CEPHEUS_SHORT_CODE",
    "CEPHEUS_BASIS_GATES",
    "list_backend_classes",
    "fetch_cepheus_metadata",
    "load_cepheus_metadata",
    "cepheus_coupling_map",
    # state prep
    "PreparedState",
    "MixedEnsemble",
    "bell_state",
    "haar_pure",
    "random_mixed",
    "mixed_ensemble",
    # swap test (collective route)
    "destructive_swap_test",
    "swap_sign",
    "purity_from_counts",
    "exact_swap_purity",
    # shadows (single-copy route)
    "pauli_shadow_circuits",
    "shadow_purity",
    "snapshots_from_outcomes",
    # noise model
    "REF_P2",
    "REF_P1",
    "REF_P_RO",
    "cepheus_noise_model",
    "readout_confusion_matrix",
    "avg_gate_error_to_depol_param",
    "depol_param_to_avg_gate_error",
    # calibration + inversion
    "CEPHEUS_SQUARE",
    "gate_noisy_probs",
    "swap_purity_from_probs",
    "measured_swap_purity",
    "measured_swap_purity_ensemble",
    "bell_calibration_surface",
    "effective_g_from_purity",
    "purity_from_g",
    "invert_measured_to_p2",
    "consistent_error_rates",
]
