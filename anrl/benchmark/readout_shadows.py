"""Uncorrected per-qubit readout error on the single-copy shadow route (PASS 14).

Sec 5.2 treats the single-copy route as the noiseless statistical baseline: noise
is modelled on the collective route only.  This module measures what readout error
-- the dominant hardware error (Sec 6.3, mean 6.5% per qubit) -- does to the
single-copy shadow estimator, so the comparison can be made symmetric.

Readout hits every shadow snapshot: after the random single-qubit rotation the
qubit is measured in the computational basis, and the reported bit is flipped with
the device's asymmetric probabilities ``P(1|0)`` (false excitation) and ``P(0|1)``
(missed excitation).  The snapshot ``rho_hat_q = 3 U_q^dag |b><b| U_q - I`` is then
built from the *flipped* bit ``b`` with **no** noise-aware inverse channel -- exactly
what uncorrected hardware does.  This biases the estimator: the noiseless
U-statistic is exactly unbiased, but ``E[3 U^dag |b_flipped><b_flipped| U - I] != rho``,
so ``E[U_M] != Tr(rho^2)`` and the RMSE gains a bias term.

The outcome-sampling RNG stream is byte-identical to
:func:`~anrl.benchmark.scaling.snapshots_factored`; the bit flips draw from a
*separate* ``readout_rng``.  So at zero readout rates the snapshots -- and hence
every downstream estimate -- reproduce the noiseless baseline exactly (verified in
the tests and in :func:`validate_zero_readout`).
"""

from __future__ import annotations

import numpy as np

from anrl.hardware.readout_model import qubit_rates

from .ensembles import NoisyState
from .scaling import _apply_local_unitaries
from .shadows import _I2, _KET_BRA, haar_unitary

# The four physically characterized Cepheus readout qubits (Sec 6.3), cycled across
# the shadow qubits.  Rate per qubit is (mean P(1|0), P(0|1)): P(1|0) is the mean of
# the idle/excited values (the correlated model's aggregate), P(0|1) as measured.
_READOUT_CYCLE = (0, 1, 9, 10)
UNIFORM_READOUT = 0.065  # Sec 6.3 mean per-qubit readout, symmetric comparator


def cyclic_readout_rates(n: int) -> list[tuple[float, float]]:
    """Per-shadow-qubit ``(p10, p01)`` from the measured device rates, cycled over
    physical qubits ``{0, 1, 9, 10}``.  ``p10`` is the aggregate (mean of the
    idle/excited P(1|0)); ``p01`` is the measured missed-excitation rate."""
    rates = []
    for qb in range(n):
        idle, excited, p01 = qubit_rates(_READOUT_CYCLE[qb % len(_READOUT_CYCLE)])
        rates.append((0.5 * (idle + excited), float(p01)))
    return rates


def uniform_readout_rates(n: int, f: float = UNIFORM_READOUT) -> list[tuple[float, float]]:
    """Uniform symmetric readout ``(f, f)`` on every qubit -- the simplified comparator."""
    return [(float(f), float(f)) for _ in range(n)]


def snapshots_factored_readout(
    state: NoisyState,
    n_snapshots: int,
    rng: np.random.Generator,
    readout_rng: np.random.Generator,
    rates: list[tuple[float, float]],
) -> np.ndarray:
    """``(M, n, 2, 2)`` local-shadow snapshots with uncorrected per-qubit readout.

    Identical to :func:`~anrl.benchmark.scaling.snapshots_factored` except each
    measured bit is flipped with its ``(p10, p01)`` before the snapshot is formed.
    The outcome-sampling draws from ``rng`` in the *same* order/quantity as the
    noiseless routine; the flips draw from ``readout_rng``.  With all-zero ``rates``
    (or a ``readout_rng`` whose draws never trip a flip) the returned array equals
    the noiseless snapshots exactly.
    """
    n, dim, g, q = state.n, state.dim, state.components, state.q
    if len(rates) != n:
        raise ValueError(f"need one (p10,p01) per qubit: got {len(rates)} for n={n}")
    snaps = np.empty((n_snapshots, n, 2, 2), dtype=np.complex128)
    shifts = n - 1 - np.arange(n)
    for s in range(n_snapshots):
        unitaries = [haar_unitary(2, rng) for _ in range(n)]
        ug = _apply_local_unitaries(g, unitaries, n)
        p_pure = (np.abs(ug) ** 2).sum(axis=1)
        probs = np.clip((1.0 - q) * p_pure + q / dim, 0.0, None)
        probs /= probs.sum()
        outcome = int(rng.choice(dim, p=probs))
        bits = (outcome >> shifts) & 1
        flips = readout_rng.random(n)  # one uniform per qubit, from the SEPARATE stream
        for qb in range(n):
            b = int(bits[qb])
            p10, p01 = rates[qb]
            if b == 0 and flips[qb] < p10:
                b = 1
            elif b == 1 and flips[qb] < p01:
                b = 0
            u_q = unitaries[qb]
            rho_meas = u_q.conj().T @ _KET_BRA[b] @ u_q
            snaps[s, qb] = 3.0 * rho_meas - _I2
    return snaps


def collective_parity_contraction(rates: list[tuple[float, float]]) -> float:
    """Readout contraction of a destructive-SWAP / cyclic-test parity signal.

    A k-copy cyclic (SWAP for k=2) test estimates ``Tr(rho^k)`` from the expectation
    of a +/-1 parity over its measured computational-basis qubits.  Under uncorrected
    readout each measured qubit's contribution to that parity contracts by
    ``E[(-1)^flip] = 1 - P(1|0) - P(0|1)`` (uniform outcome approximation), so the
    signal is multiplied by the product over all measured qubits.  For symmetric
    rate ``f`` this is ``(1 - 2f)^(len(rates))``.

    MODELING CHOICE, not forced by the existing code: the committed collective model
    (``collective_moment_estimate``) carries the *channel* but no readout.  This
    applies readout to the collective side in expectation, using the same per-qubit
    rates as the single-copy side, so the two routes are treated consistently.
    """
    c = 1.0
    for p10, p01 in rates:
        c *= (1.0 - p10 - p01)
    return c
