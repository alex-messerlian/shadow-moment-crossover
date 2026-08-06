"""Measured Cepheus readout model, including correlated (context-dependent) readout.

Per-physical-qubit asymmetric readout was measured on {0,1,9,10}
(device-characterization phase).  The dominant non-ideality is CORRELATED readout on
$0: its false-1 rate P(1|0) rises with the number of *other* measured qubits that are
truly excited (crosstalk during simultaneous readout), from 1.6% (all others idle)
to 16.9% (two others excited).  We model each qubit's P(1|0) as linear in that count
``w`` (the number of OTHER measured qubits in |1>):

    P(1|0)_q(w) = p10_idle_q + (p10_excited_q - p10_idle_q) / 2 * w        (calibrated
                                                                          at w=0, w=2)

P(0|1) is treated as context-independent (no correlation was measured for it).  Qubits
beyond the four characterized ones (needed for the n=3,4 SWAP tests) have no readout
data, so they take the MEAN measured rates with no correlation, an explicit
assumption, flagged wherever it is used.

The joint 2^m x 2^m confusion matrix is NOT a tensor product (a qubit's flip
probability depends on the whole true bitstring via ``w``), but it factorizes *within a
column* (the context ``w`` is fixed by the true outcome), so it is built column by
column in ``O(4^m)``, feasible for the m = 2n <= 8 qubits we need.
"""

from __future__ import annotations

import numpy as np

# physical qubit -> (p10_idle, p10_excited_at_w2, p01)   [measured, device-characterization]
MEASURED_READOUT = {
    0: (0.0165, 0.1685, 0.0892),   # $0, strong P(1|0) correlation
    1: (0.0055, 0.0080, 0.0780),   # $1
    9: (0.0615, 0.0730, 0.0978),   # $9
    10: (0.0295, 0.0340, 0.0595),  # $10
}
# mean measured rates for UNCHARACTERIZED qubits (n=3,4 extras): aggregate p10, p01; no correlation.
_MEAN_P10 = float(np.mean([0.5 * (v[0] + v[1]) for v in MEASURED_READOUT.values()]))  # ~0.0496
_MEAN_P01 = float(np.mean([v[2] for v in MEASURED_READOUT.values()]))                 # ~0.0811


def qubit_rates(phys: int) -> tuple[float, float, float]:
    """(p10_idle, p10_excited_at_w2, p01) for a physical qubit; mean+uncorrelated if unknown."""
    if phys in MEASURED_READOUT:
        return MEASURED_READOUT[phys]
    return (_MEAN_P10, _MEAN_P10, _MEAN_P01)


def _p10(idle: float, excited: float, w: int) -> float:
    """Linear-in-w false-1 rate, calibrated at w=0 (idle) and w=2 (excited)."""
    return idle + 0.5 * (excited - idle) * w


def correlated_confusion(phys_qubits: list[int], correlated: bool = True) -> np.ndarray:
    """Joint readout confusion ``R[measured, true]`` for ``phys_qubits`` (clbit order).

    Index convention: bit ``c`` of an index is clbit ``c`` = ``phys_qubits[c]`` (LSB =
    clbit 0), matching Qiskit ``probabilities()``.  With ``correlated=False`` every
    qubit uses its aggregate p10 (mean of idle/excited); the independent-qubit model,
    kept for the with/without-correlation comparison.
    """
    m = len(phys_qubits)
    dim = 2 ** m
    rates = [qubit_rates(q) for q in phys_qubits]
    R = np.zeros((dim, dim))
    for t in range(dim):  # true outcome (column); context w fixed here
        pop = bin(t).count("1")
        for mo in range(dim):  # measured outcome (row)
            prob = 1.0
            for c in range(m):
                idle, exc, p01 = rates[c]
                t_c = (t >> c) & 1
                m_c = (mo >> c) & 1
                if correlated:
                    w = pop - t_c  # number of OTHER measured qubits excited
                    p1_0 = _p10(idle, exc, w)
                else:
                    p1_0 = 0.5 * (idle + exc)  # aggregate
                if t_c == 0:
                    prob *= (1 - p1_0) if m_c == 0 else p1_0
                else:
                    prob *= p01 if m_c == 0 else (1 - p01)
            R[mo, t] = prob
    return R
