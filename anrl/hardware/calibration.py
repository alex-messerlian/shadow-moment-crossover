"""The calibration curve, measured purity vs noise, and its inversion.

The destructive SWAP test on a state of true purity ``Tr(rho^2)`` returns a
*biased* purity under noise.  Here we compute that biased value exactly for the
parameterized Cepheus model, sweep it into a calibration surface over
``(p2, p_ro)``, and invert it: given a purity actually measured on hardware,
recover the effective error rates consistent with it, extracting the device's
noise from our own data instead of trusting the datasheet.

Exactness and speed.  We never rely on shot sampling for the *prediction*:

1. Build the SWAP-test circuit WITHOUT measurement, transpile it to the real
   4-qubit Cepheus square (native ``cz, rx, rz``; physical qubits {0,1,9,10}).
2. Simulate it in Aer's density-matrix method with the GATE noise only
   (depolarizing on ``cz``/``rx``).  The diagonal of the final density matrix is
   the ideal-readout outcome distribution ``q(b)``.
3. Apply the readout confusion matrix ``R(p_ro)`` analytically: ``p = R q``.
   Because gate noise is independent of ``p_ro``, one density-matrix solve covers
   the whole ``p_ro`` axis of the surface.
4. The measured purity is the sign-rule sum ``sum_b sign(b) p(b)``.

Shot noise is added separately (see :mod:`~anrl.hardware.shot_budget`) as an
analytic standard error, and cross-checked against a direct shot-based Aer run
with the full ``ReadoutError`` model in the tests.
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from .noise_model import CEPHEUS_BASIS_GATES, cepheus_noise_model, readout_confusion_matrix
from .state_prep import PreparedState
from .swap_test import destructive_swap_test, swap_sign

# The real Cepheus square {0,1,9,10} relabeled to logical 0..3 (edges form the
# 4-cycle the n=2 two-copy SWAP test needs; zero routing overhead, prior phase).
CEPHEUS_SQUARE = CouplingMap(couplinglist=[(0, 1), (0, 2), (1, 3), (2, 3)])

_DM_SIM = AerSimulator(method="density_matrix")


def _transpiled_swap_no_meas(prep: PreparedState, prep_b: PreparedState | None = None):
    """Transpiled (native, on-square) SWAP-test circuit with a density-matrix save, no measure."""
    logical = destructive_swap_test(prep, prep_b).remove_final_measurements(inplace=False)
    tqc = transpile(logical, coupling_map=CEPHEUS_SQUARE, basis_gates=CEPHEUS_BASIS_GATES,
                    optimization_level=3, seed_transpiler=0)
    tqc.save_density_matrix()
    return tqc


def gate_noisy_probs(prep: PreparedState, p2: float, p1: float,
                     prep_b: PreparedState | None = None) -> np.ndarray:
    """Ideal-readout SWAP-test outcome distribution ``q(b)`` under gate noise only.

    Density-matrix solve of the transpiled circuit with depolarizing on cz/rx;
    the diagonal is the outcome distribution before any readout error.
    """
    tqc = _transpiled_swap_no_meas(prep, prep_b)
    nm = cepheus_noise_model(p2=p2, p1=p1, p_ro=0.0)
    rho = _DM_SIM.run(tqc, noise_model=nm).result().data(0)["density_matrix"]
    return np.clip(np.real(rho.probabilities()), 0.0, None)


def swap_purity_from_probs(probs: np.ndarray, n: int) -> float:
    """Sign-rule purity estimate ``sum_b (-1)^{#pairs both-1} p(b)`` from a prob vector."""
    m = 2 * n
    signs = np.array([swap_sign(format(b, f"0{m}b"), n) for b in range(2 ** m)], dtype=np.float64)
    return float(signs @ probs)


def measured_swap_purity(prep: PreparedState, p2: float, p1: float, p_ro: float,
                         prep_b: PreparedState | None = None) -> float:
    """Exact noisy measured purity for a pure-state SWAP test (gate noise + readout)."""
    q = gate_noisy_probs(prep, p2, p1, prep_b)
    n = prep.n
    p = readout_confusion_matrix(2 * n, p_ro) @ q
    return swap_purity_from_probs(p, n)


def measured_swap_purity_ensemble(ensemble, p2: float, p1: float, p_ro: float) -> float:
    """Noisy measured purity for a classically-mixed state (sum over eigenpair terms).

    A mixture ``rho = sum_i w_i |v_i><v_i|`` realized as a per-shot classical
    ensemble measures ``sum_{i,j} w_i w_j <SWAP>_{v_i, v_j}``; each term is one
    noisy density-matrix solve.  ``ensemble`` is an :class:`MixedEnsemble`.
    """
    n = ensemble.n
    r = readout_confusion_matrix(2 * n, p_ro)
    total = 0.0
    for w_i, comp_i in ensemble.components:
        for w_j, comp_j in ensemble.components:
            q = gate_noisy_probs(comp_i, p2, p1, prep_b=comp_j)
            total += w_i * w_j * swap_purity_from_probs(r @ q, n)
    return float(total)


# --------------------------------------------------------------------------- #
# Calibration surface + effective-g mapping                                   #
# --------------------------------------------------------------------------- #
def bell_calibration_surface(p2_grid: np.ndarray, p_ro_grid: np.ndarray, p1: float):
    """Measured Bell purity over the ``(p2, p_ro)`` grid (true purity 1.0).

    Returns ``surface`` of shape ``(len(p2_grid), len(p_ro_grid))``.  One
    density-matrix solve per ``p2`` (readout applied analytically across the
    ``p_ro`` axis), so the cost is ``O(len(p2_grid))`` solves.
    """
    from .state_prep import bell_state

    bell = bell_state()
    n = bell.n
    signs = np.array([swap_sign(format(b, f"0{2 * n}b"), n) for b in range(2 ** (2 * n))], dtype=np.float64)
    confusions = [readout_confusion_matrix(2 * n, float(pr)) for pr in p_ro_grid]
    surface = np.empty((len(p2_grid), len(p_ro_grid)), dtype=np.float64)
    for i, p2 in enumerate(p2_grid):
        q = gate_noisy_probs(bell, float(p2), p1)
        for j, r in enumerate(confusions):
            surface[i, j] = float(signs @ (r @ q))
    return surface


def effective_g_from_purity(measured: float, moment_k: float, n: int = 2, k: int = 2) -> float:
    """Effective global-depolarizing ``g`` implied by a measured purity.

    Inverts the bias law ``measured = (1-g) Tr(rho^k) + g 2^{n(1-k)}``:
    ``g = (Tr(rho^k) - measured) / (Tr(rho^k) - 2^{n(1-k)})``.  For Bell (n=2,k=2)
    this is ``g = (1 - measured) / 0.75``.
    """
    floor = 2.0 ** (n * (1 - k))
    denom = moment_k - floor
    if abs(denom) < 1e-15:
        raise ValueError("degenerate: Tr(rho^k) equals the depolarizing floor")
    return (moment_k - measured) / denom


def purity_from_g(g: float, moment_k: float, n: int = 2, k: int = 2) -> float:
    """Bias-law measured purity ``(1-g) Tr(rho^k) + g 2^{n(1-k)}``."""
    return (1.0 - g) * moment_k + g * 2.0 ** (n * (1 - k))


# --------------------------------------------------------------------------- #
# Inversion: measured purity -> effective error rates                         #
# --------------------------------------------------------------------------- #
def invert_measured_to_p2(measured: float, p_ro: float, p1: float,
                          p2_bracket: tuple[float, float] = (0.0, 0.2),
                          tol: float = 1e-6, max_iter: int = 60) -> float:
    """Recover the two-qubit error ``p2`` from a measured Bell purity at known ``p_ro``.

    Bisection on the (monotone-decreasing) map ``p2 -> measured_swap_purity``.
    Returns the ``p2`` reproducing ``measured`` (clamped to the bracket if the
    measured value lies outside the achievable range).
    """
    from .state_prep import bell_state

    bell = bell_state()

    def f(p2: float) -> float:
        return measured_swap_purity(bell, p2, p1, p_ro)

    lo, hi = p2_bracket
    f_lo, f_hi = f(lo), f(hi)
    if measured >= f_lo:
        return lo
    if measured <= f_hi:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid - measured) < tol:
            return mid
        if f_mid > measured:  # too little error -> raise p2
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def consistent_error_rates(measured: float, p1: float, p_ro_grid: np.ndarray,
                           p2_bracket: tuple[float, float] = (0.0, 0.2)) -> list[tuple[float, float]]:
    """The level set of ``(p_ro, p2)`` pairs consistent with a measured Bell purity.

    A single scalar purity cannot pin both rates, so we return the trade-off
    curve: for each assumed readout error ``p_ro`` on the grid, the ``p2`` that
    reproduces ``measured``.  Independent readout characterization then selects a
    point on this curve.
    """
    return [(float(pr), invert_measured_to_p2(measured, float(pr), p1, p2_bracket)) for pr in p_ro_grid]
