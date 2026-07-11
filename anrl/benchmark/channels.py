"""Realistic single-qubit noise channels for the collective-measurement test.

Amplitude damping and dephasing are applied as *explicit Kraus operators on
every qubit* of the k-copy register — no closed-form signal formula (they are
not depolarizing).  Depolarizing is kept as the global-channel closed form from
:mod:`anrl.benchmark.moments`.

Key structural fact (used to make the sweep tractable and validated against the
explicit construction in the tests): a per-qubit channel ``N`` applied to every
qubit of ``rho^{(x)k}`` factorizes across the disjoint copies,
``N^{(x)kn}(rho^{(x)k}) = (N^{(x)n}(rho))^{(x)k} = sigma^{(x)k}``, so the noisy
cyclic-test signal is exactly ``Tr(C_k sigma^{(x)k}) = Tr(sigma^k)`` where
``sigma = N^{(x)n}(rho)`` is the single-copy noisy state.  This is not the
depolarizing formula — it genuinely applies the Kraus channel — it just uses the
tensor structure so we never build the ``2^(kn)``-dim k-copy operator.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from .moments import (
    cyclic_permutation_operator,
    depolarizing_moment_signal,
    kron_power,
    moment,
)

NOISE_MODELS = ("depolarizing", "amplitude_damping", "dephasing")

_I2 = np.eye(2, dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def amplitude_damping_kraus(g: float) -> list[np.ndarray]:
    """Amplitude-damping Kraus operators at rate ``g``."""
    return [
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - g)]], dtype=np.complex128),
        np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=np.complex128),
    ]


def dephasing_kraus(g: float) -> list[np.ndarray]:
    """Dephasing Kraus operators at rate ``g``."""
    return [np.sqrt(1.0 - g / 2.0) * _I2, np.sqrt(g / 2.0) * _Z]


def _embed(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a single-qubit ``op`` on ``qubit`` into an ``n_qubits`` register."""
    factors = [op if i == qubit else _I2 for i in range(n_qubits)]
    return reduce(np.kron, factors)


def apply_channel_per_qubit(
    rho: np.ndarray, kraus_ops: list[np.ndarray], n_qubits: int
) -> np.ndarray:
    """Apply the single-qubit channel ``kraus_ops`` to every qubit of ``rho``."""
    out = np.asarray(rho, dtype=np.complex128)
    for qubit in range(n_qubits):
        embedded = [_embed(k, qubit, n_qubits) for k in kraus_ops]
        out = sum(e @ out @ e.conj().T for e in embedded)
    return out


def channel_collective_signal(
    rho: np.ndarray, k: int, kraus_ops: list[np.ndarray], n_qubits: int
) -> float:
    """Noisy k-copy cyclic-test signal for a per-qubit channel (factorized).

    Equals ``Tr(sigma^k)`` with ``sigma = N^{(x)n}(rho)`` — exactly the explicit
    construction :func:`explicit_channel_collective_signal`, but without forming
    the k-copy operator (used for the sweep at all sizes).
    """
    sigma = apply_channel_per_qubit(rho, kraus_ops, n_qubits)
    return moment(sigma, k)


def explicit_channel_collective_signal(
    rho: np.ndarray, k: int, kraus_ops: list[np.ndarray], n_qubits: int
) -> float:
    """Explicit noisy signal: build ``rho^{(x)k}``, apply the channel to all
    ``k*n`` qubits, and measure the cyclic operator ``Tr(C_k @ noisy)``.

    Ground-truth validation for :func:`channel_collective_signal` — feasible only
    for small ``(n, k)`` (the k-copy operator is ``2^(kn)``-dimensional).
    """
    big = kron_power(np.asarray(rho, dtype=np.complex128), k)
    noisy = apply_channel_per_qubit(big, kraus_ops, k * n_qubits)
    c_k = cyclic_permutation_operator(2 ** n_qubits, k)
    return float(np.trace(c_k @ noisy).real)


def collective_moment_signal(
    rho: np.ndarray, k: int, noise_model: str, rate: float, n_qubits: int
) -> float:
    """The noisy k-copy cyclic-test signal for a named noise model.

    ``depolarizing`` uses the global closed form; ``amplitude_damping`` and
    ``dephasing`` apply their explicit per-qubit Kraus channels.
    """
    if noise_model == "depolarizing":
        return depolarizing_moment_signal(moment(rho, k), k, rate, n_qubits)
    if noise_model == "amplitude_damping":
        return channel_collective_signal(rho, k, amplitude_damping_kraus(rate), n_qubits)
    if noise_model == "dephasing":
        return channel_collective_signal(rho, k, dephasing_kraus(rate), n_qubits)
    raise ValueError(f"unknown noise_model {noise_model!r}; expected one of {NOISE_MODELS}")
