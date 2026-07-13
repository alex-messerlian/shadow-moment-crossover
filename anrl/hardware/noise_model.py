"""Parameterized Aer noise model for Rigetti Cepheus-1-108Q.

Open Quantum's metadata API returns an empty noise dict, so we have no per-device
error rates.  Instead we build a *parameterized* device model and, in
:mod:`~anrl.hardware.calibration`, the machinery to invert measured hardware data
back into effective rates.  This module supplies the model.

Three knobs, applied to the transpiled native circuit (basis ``cz, rx, rz``):

* ``p2`` — two-qubit depolarizing parameter on every ``cz`` gate;
* ``p1`` — single-qubit depolarizing parameter on every ``rx`` gate;
* ``p_ro`` — symmetric readout (bit-flip) error on every measurement.

Modeling choices (documented, not silent):

* **``rz`` is virtual.**  On Rigetti's superconducting transmons ``rz`` is a
  frame change implemented in software (zero duration, no physical pulse), so it
  is noiseless.  We therefore attach single-qubit depolarizing to ``rx`` only —
  the sole physical single-qubit gate.  Putting ``p1`` on the 16 ``rz`` gates of
  the transpiled Bell SWAP test would fabricate ~1.6% of spurious error.  The
  ``include_rz_error`` flag exists only to demonstrate that effect in a test.

* **``p2``/``p1`` are the Qiskit depolarizing *parameters*** (the argument to
  ``depolarizing_error``), not average gate infidelities.  A depolarizing channel
  with parameter ``lambda`` on ``d`` dimensions has average gate error
  ``r = lambda (d-1)/d``.  So a datasheet "median CZ fidelity 99.1%"
  (``r = 0.009``) corresponds to ``lambda = 4r/3 = 0.012``; a "median 1q fidelity
  99.9%" (``r = 0.001``) to ``lambda = 2r = 0.002``.  Both conventions are exposed
  via :func:`avg_gate_error_to_depol_param`, and the locked predictions report the
  reference at the user-anchored ``p2 = 0.009`` and at the datasheet-faithful
  ``p2 = 0.012`` so the two readings are explicit.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# Reference values (Rigetti GA announcement / AWS Braket, Cepheus-1-108Q).
# Medians across 108 qubits with real spread; drift between calibrations.
REF_P2 = 0.009  # user-anchored two-qubit depolarizing parameter (~99.1% CZ fidelity headline)
REF_P1 = 0.001  # single-qubit depolarizing parameter (~99.9% 1q fidelity headline)
REF_P_RO = 0.02  # assumed per-qubit readout error — see justification in docs/report

CEPHEUS_BASIS_GATES = ["cz", "rx", "rz"]


def avg_gate_error_to_depol_param(avg_error: float, num_qubits: int) -> float:
    """Depolarizing parameter ``lambda`` reproducing an average gate error ``r``.

    ``r = lambda (d-1)/d`` with ``d = 2**num_qubits``  =>  ``lambda = r d/(d-1)``.
    """
    d = 2 ** num_qubits
    return avg_error * d / (d - 1)


def depol_param_to_avg_gate_error(depol_param: float, num_qubits: int) -> float:
    """Average gate error ``r = lambda (d-1)/d`` for a depolarizing parameter."""
    d = 2 ** num_qubits
    return depol_param * (d - 1) / d


def cepheus_noise_model(
    p2: float = REF_P2,
    p1: float = REF_P1,
    p_ro: float = REF_P_RO,
    *,
    include_rz_error: bool = False,
) -> NoiseModel:
    """All-qubit Cepheus noise model (median device; independent of physical index).

    Depolarizing ``p2`` on ``cz``, ``p1`` on ``rx`` (and ``rz`` iff
    ``include_rz_error``), symmetric readout ``p_ro`` on measurement.  Zero rates
    are skipped so ``p2 = p1 = p_ro = 0`` returns a noiseless model.
    """
    for name, val in (("p2", p2), ("p1", p1), ("p_ro", p_ro)):
        if not (0.0 <= val < 1.0):
            raise ValueError(f"{name} must be in [0, 1), got {val}")
    nm = NoiseModel(basis_gates=CEPHEUS_BASIS_GATES)
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    if p1 > 0:
        oneq = ["rx"] + (["rz"] if include_rz_error else [])
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), oneq)
    if p_ro > 0:
        nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def readout_confusion_matrix(n_meas: int, p_ro: float) -> np.ndarray:
    """``2^n_meas x 2^n_meas`` symmetric readout confusion (tensor of per-qubit bit flips).

    ``R[b, b'] = P(read b | prepared b')``.  Applied to a noiseless-readout outcome
    probability vector, it yields the readout-damaged distribution, so readout can
    be swept cheaply without re-simulating the gate noise.
    """
    single = np.array([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]], dtype=np.float64)
    return reduce(np.kron, [single] * n_meas)
