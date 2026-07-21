"""Corrected readout models for the collective and single-copy routes (PASS 15).

PASS 14 applied readout to the collective (destructive-SWAP) route as a blanket
contraction ``prod(1 - p10 - p01)`` over the 2n measured qubits, flagged as a harsh
upper bound.  The destructive-SWAP estimator is ``prod_i (-1)^{a_i b_i}`` over the n
measured Bell pairs, whose readout effect is *outcome-dependent*: conditioned on the
true pair outcome ``(a,b)`` the expected measured parity is

    f(a,b) = 1 - 2 P(a'=1|a) P(b'=1|b)
           = { 1 - 2 p10^2            (0,0)
               1 - 2 p10 (1-p01)      (0,1),(1,0)
               1 - 2 (1-p01)^2        (1,1) }

(verified against direct enumeration).  The exact readout-degraded mean parity is
therefore ``S' = Tr[(sigma x sigma) prod_i O_i]`` with the per-pair operator
``O_i = U_Bell,i^dag diag(f00,f01,f10,f11) U_Bell,i`` (``U_Bell = (H x I) CNOT`` is
the Bell-measurement circuit).  At zero readout ``O_i = SWAP_i`` and ``S' = Tr(sigma^2)``
-- the model reduces to the noiseless SWAP test exactly.  This is computed by an exact
contraction (no sampling), tractable for ``n <= 6`` (2n <= 12 qubits).

Readout mitigation (both routes) uses the known per-qubit confusion matrix:

* single-copy: rescale each per-qubit snapshot's traceless (Bloch) part by
  ``1/(1 - p10 - p01)`` -- the noise-aware inverse (restores unbiasedness exactly for
  symmetric readout);
* collective: replace the ideal per-pair observable ``v = (+1,+1,+1,-1)`` by the
  mitigated ``w = (R^{-1})^T v`` (``R`` the 4x4 pair confusion), which is unbiased at
  the cost of inflated per-shot variance.
"""

from __future__ import annotations

import numpy as np

from .scaling import _I2

# Bell-measurement circuit U_Bell = (H on A) . CNOT(A->B), 4x4 on (A,B); |ab> order.
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
_U_BELL = np.kron(_H, _I2) @ _CNOT  # measure-basis change for the destructive SWAP


def confusion_1q(p10: float, p01: float) -> np.ndarray:
    """2x2 readout confusion ``R[measured, true]``: cols true 0/1, rows measured 0/1."""
    return np.array([[1.0 - p10, p01], [p10, 1.0 - p01]], dtype=np.float64)


def per_pair_factor(a: int, b: int, p10a: float, p01a: float, p10b: float, p01b: float) -> float:
    """``E[(-1)^{a' b'}]`` given the true pair outcome ``(a,b)`` under independent flips."""
    pa1 = (1.0 - p01a) if a == 1 else p10a
    pb1 = (1.0 - p01b) if b == 1 else p10b
    return 1.0 - 2.0 * pa1 * pb1


def _pair_diag(p10a, p01a, p10b, p01b):
    """diag over true outcomes (00,01,10,11) of the readout-averaged parity factor."""
    return np.array([per_pair_factor(a, b, p10a, p01a, p10b, p01b)
                     for a in (0, 1) for b in (0, 1)], dtype=np.complex128)


def _pair_diag_mitigated(p10a, p01a, p10b, p01b):
    """diag h(x) = sum_y R(y|x) w(y)^2 for the mitigated observable w = (R^-1)^T v.

    Gives the per-pair second moment of the unbiased mitigated estimator (variance
    ingredient).  Its plain first moment is the ideal parity v (unbiased by construction)."""
    Ra = confusion_1q(p10a, p01a)
    Rb = confusion_1q(p10b, p01b)
    R = np.kron(Ra, Rb)  # 4x4 pair confusion R[y|x], y,x in (00,01,10,11)
    v = np.array([1.0, 1.0, 1.0, -1.0])  # ideal parity (-1)^{ab}
    w = np.linalg.solve(R.T, v)  # (R^T) w = v  ->  unbiased mitigated observable
    # h(x) = sum_y R[y,x] w[y]^2
    h = (R * (w[:, None] ** 2)).sum(axis=0)
    return h.astype(np.complex128)


def _pair_operator(diag: np.ndarray) -> np.ndarray:
    """O = U_Bell^dag diag(.) U_Bell -- a 4x4 observable on (A,B) in the computational basis."""
    return _U_BELL.conj().T @ np.diag(diag) @ _U_BELL


def two_copy_noisy_state(state, noise_model: str, rate: float) -> np.ndarray:
    """The physical noisy 2-copy state ``tau`` the destructive-SWAP measures (2^{2n} dense).

    Per-qubit channels (amplitude_damping, dephasing) act on each copy independently, so
    ``tau = sigma x sigma`` with ``sigma`` the noisy single-copy state.  Depolarizing is
    modelled on the 2-copy register (matching :mod:`anrl.benchmark.moments`), so
    ``tau = (1-p)(rho x rho) + p I/d^2``.  At zero readout ``Tr[tau SWAP] = Tr(sigma^2)``
    for the per-qubit channels and ``(1-p)Tr(rho^2)+p/d`` for depolarizing -- exactly the
    committed ``collective_purity_signal``."""
    from .channels import amplitude_damping_kraus, dephasing_kraus
    from .scaling import _apply_channel_dense
    n, d = state.n, state.dim
    rho = state.density_matrix()
    if noise_model == "depolarizing":
        rr = np.kron(rho, rho)
        return (1.0 - rate) * rr + rate * np.eye(d * d) / (d * d)
    kraus = amplitude_damping_kraus(rate) if noise_model == "amplitude_damping" else dephasing_kraus(rate)
    sigma = _apply_channel_dense(rho, kraus, n)
    return np.kron(sigma, sigma)


def _contract_two_copy(tau: np.ndarray, n: int, pair_ops: list[np.ndarray]) -> float:
    """``Tr[tau prod_i O_i]`` with the 2-copy state ``tau`` (2^{2n}) and ``O_i`` on qubits
    ``(A_i, B_i)``; qubit order [A_1..A_n, B_1..B_n], pair i acts on (i, n+i).

    Applied by left-multiplying each O_i onto the 2n-qubit operator, then tracing.
    O(4^n) memory -- tractable for n <= 6."""
    d = 2 ** n
    op = tau.reshape([2] * (2 * n) + [d * d])  # rows split into 2n qubit axes, cols flat
    for i in range(n):
        O = pair_ops[i].reshape(2, 2, 2, 2)  # (a'_i, b'_i, a_i, b_i)
        op = np.moveaxis(op, (i, n + i), (0, 1))
        op = np.tensordot(O, op, axes=([2, 3], [0, 1]))  # contract (a_i,b_i)
        op = np.moveaxis(op, (0, 1), (i, n + i))
    op = op.reshape(d * d, d * d)
    return float(np.trace(op).real)


import string as _string


def _contract_factorized(sigma: np.ndarray, n: int, pair_ops: list[np.ndarray]) -> float:
    """``Tr[(sigma x sigma) prod_i O_i]`` WITHOUT materialising the 2^{2n} operator.

    Uses the per-qubit structure (each O_i couples copy-A qubit i to copy-B qubit i), an
    einsum over sigma_A, sigma_B and the O_i whose intermediates stay at O(2^n) -- so it
    reaches larger n than :func:`_contract_two_copy`.  Valid only when the 2-copy state
    factorises as ``sigma x sigma`` (per-qubit channels; depolarizing is handled by
    :func:`collective_readout_signal_state` via a split)."""
    L = _string.ascii_letters
    a = [L[i] for i in range(n)]; ap = [L[n + i] for i in range(n)]
    b = [L[2 * n + i] for i in range(n)]; bp = [L[3 * n + i] for i in range(n)]
    sig = sigma.reshape([2] * (2 * n))
    subs = "".join(a) + "".join(ap) + "," + "".join(b) + "".join(bp)
    ops = [sig, sig.reshape([2] * (2 * n))]
    for i in range(n):
        subs += "," + ap[i] + bp[i] + a[i] + b[i]
        ops.append(pair_ops[i].reshape(2, 2, 2, 2))
    subs += "->"
    return float(np.einsum(subs, *ops, optimize="greedy").real)


def collective_readout_signal_state(state, noise_model, rate, n, pair_rates, mitigate=False):
    """Readout-degraded SWAP quantity via the factorised contraction (reaches n<=8).

    Per-qubit channels: ``tau = sigma x sigma`` -> factorised contraction directly.
    Depolarizing (2-copy register): split ``tau = (1-p)(rho x rho) + p I/d^2`` so
    ``Tr[tau prod O_i] = (1-p) Tr[(rho x rho) prod O_i] + p prod_i Tr(O_i) / d^2``."""
    from .channels import amplitude_damping_kraus, dephasing_kraus
    from .scaling import _apply_channel_dense
    d = state.dim
    rho = state.density_matrix()
    ops = []
    for (ra, rb) in pair_rates:
        diag = _pair_diag_mitigated(*ra, *rb) if mitigate else _pair_diag(*ra, *rb)
        ops.append(_pair_operator(diag))
    if noise_model == "depolarizing":
        pure = _contract_factorized(rho, n, ops)
        ident = float(np.prod([np.trace(O).real for O in ops])) / (d * d)
        return (1.0 - rate) * pure + rate * ident
    kraus = amplitude_damping_kraus(rate) if noise_model == "amplitude_damping" else dephasing_kraus(rate)
    sigma = _apply_channel_dense(rho, kraus, n)
    return _contract_factorized(sigma, n, ops)


def collective_readout_signal(tau: np.ndarray, n: int, pair_rates, mitigate: bool = False) -> float:
    """Exact readout-degraded destructive-SWAP quantity for the 2-copy state ``tau``.

    ``pair_rates[i] = ((p10a,p01a),(p10b,p01b))`` for pair i (qubits A_i, B_i).
    ``mitigate=False``: E[measured parity] = readout-biased signal S' (equals the noiseless
    SWAP signal at zero readout).  ``mitigate=True``: E[w^2] over measured outcomes -- the
    per-shot second moment of the unbiased mitigated estimator."""
    ops = []
    for (ra, rb) in pair_rates:
        diag = _pair_diag_mitigated(*ra, *rb) if mitigate else _pair_diag(*ra, *rb)
        ops.append(_pair_operator(diag))
    return _contract_two_copy(tau, n, ops)


def collective_rmse_analytic(signal_or_secondmoment: float, true: float, n_shots: int,
                             unbiased: bool = False, mean_signal: float | None = None) -> float:
    """Exact RMSE of the binary collective estimator.

    Uncorrected: per-shot value in {+1,-1} with mean S'=signal_or_secondmoment, so
    Var/shot = 1 - S'^2 and RMSE = sqrt((S'-true)^2 + (1-S'^2)/n_shots).
    Corrected (unbiased): mean = mean_signal (= Tr(sigma^2)), per-shot 2nd moment =
    signal_or_secondmoment (=E[w^2]); Var/shot = E[w^2]-mean^2; RMSE = sqrt((mean-true)^2 +
    (E[w^2]-mean^2)/n_shots)."""
    if unbiased:
        m = mean_signal
        var = max(0.0, signal_or_secondmoment - m * m)
        return float(np.sqrt((m - true) ** 2 + var / n_shots))
    s = signal_or_secondmoment
    var = max(0.0, 1.0 - s * s)
    return float(np.sqrt((s - true) ** 2 + var / n_shots))


def corrected_snapshots(snaps: np.ndarray, rates: list[tuple[float, float]]) -> np.ndarray:
    """Noise-aware inverse on single-copy snapshots: rescale each qubit's traceless
    (Bloch) part by ``1/(1 - p10 - p01)`` using the SAME rates the readout applied.

    ``snaps`` is ``(M, n, 2, 2)`` (already carrying readout).  Restores unbiasedness for
    symmetric readout; inflates variance by ~``1/(1-p10-p01)^2`` per qubit."""
    n = snaps.shape[1]
    out = snaps.copy()
    half = 0.5 * _I2
    for q in range(n):
        g = 1.0 / (1.0 - rates[q][0] - rates[q][1])
        out[:, q] = half + (snaps[:, q] - half) * g
    return out
