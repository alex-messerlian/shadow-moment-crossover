"""Part 1; the exact, parameter-free collective bias laws.

The k-copy cyclic-permutation ("collective") test on a noise-damaged register
returns a *biased* estimate of ``Tr(rho^k)``.  The bias has an exact closed form,
and the form depends on the channel geometry:

* **Global depolarizing** at rate ``g`` on the whole ``k n``-qubit register: the
  depolarized k-copy state is ``(1-g) rho^{ox k} + g I / D`` with ``D = 2^{n k}``.
  Since ``Tr(C_k) = 2^n`` (the cyclic permutation is a single ``n``-qudit cycle),
  the measured value is ``(1-g) Tr(rho^k) + g 2^{n(1-k)}`` and

      bias = g * |Tr(rho^k) - 2^{n(1-k)}|          (linear in g, no compounding).

* **Per-qubit channel** ``E`` (amplitude damping, dephasing) applied to every
  qubit of every copy: this equals ``E^{ox n}`` applied to each copy
  independently, and the cyclic test of ``sigma^{ox k}`` returns ``Tr(sigma^k)``
  with ``sigma = E^{ox n}(rho)``.  So

      bias = |Tr(sigma^k) - Tr(rho^k)|.

Both are validated against a brute-force construction (build ``C_k`` and the
explicitly damaged k-copy state, evaluate ``Tr(C_k . noisy)``) in the tests.
"""

from __future__ import annotations

import numpy as np

from anrl.benchmark.channels import amplitude_damping_kraus, dephasing_kraus
from anrl.benchmark.moments import cyclic_permutation_operator, kron_power, moment
from anrl.benchmark.scaling import _apply_channel_dense

_PERQUBIT_KRAUS = {"amplitude_damping": amplitude_damping_kraus, "dephasing": dephasing_kraus}


def depolarizing_collective_value(moment_k: float, k: int, g: float, n: int) -> float:
    """Exact global-depolarizing collective signal ``(1-g) Tr(rho^k) + g 2^{n(1-k)}``."""
    return (1.0 - g) * moment_k + g * 2.0 ** (n * (1 - k))


def depolarizing_bias(moment_k: float, k: int, g: float, n: int) -> float:
    """Exact global-depolarizing collective bias ``g |Tr(rho^k) - 2^{n(1-k)}|``."""
    return abs(g) * abs(moment_k - 2.0 ** (n * (1 - k)))


def perqubit_channel_value(rho: np.ndarray, k: int, kraus: list[np.ndarray], n: int) -> float:
    """Exact per-qubit-channel collective signal ``Tr(sigma^k)``, ``sigma = E^{ox n}(rho)``."""
    sigma = _apply_channel_dense(np.asarray(rho, dtype=np.complex128), kraus, n)
    return moment(sigma, k)


def perqubit_channel_bias(rho: np.ndarray, k: int, kraus: list[np.ndarray], n: int) -> float:
    """Exact per-qubit-channel collective bias ``|Tr(sigma^k) - Tr(rho^k)|``."""
    return abs(perqubit_channel_value(rho, k, kraus, n) - moment(rho, k))


def collective_bias(rho: np.ndarray, k: int, noise_model: str, g: float, n: int) -> float:
    """Exact collective bias for a named noise model (dispatch on channel geometry)."""
    if noise_model == "depolarizing":
        return depolarizing_bias(moment(rho, k), k, g, n)
    if noise_model in _PERQUBIT_KRAUS:
        return perqubit_channel_bias(rho, k, _PERQUBIT_KRAUS[noise_model](g), n)
    raise ValueError(f"unknown noise_model {noise_model!r}")


def collective_value(rho: np.ndarray, k: int, noise_model: str, g: float, n: int) -> float:
    """Exact collective signal ``Tr(C_k . noisy)`` for a named noise model."""
    if noise_model == "depolarizing":
        return depolarizing_collective_value(moment(rho, k), k, g, n)
    if noise_model in _PERQUBIT_KRAUS:
        return perqubit_channel_value(rho, k, _PERQUBIT_KRAUS[noise_model](g), n)
    raise ValueError(f"unknown noise_model {noise_model!r}")


def _cyclic_trace(c_k: np.ndarray, noisy: np.ndarray) -> float:
    """``Tr(C_k . noisy)`` in ``O(D^2)`` (avoid the ``O(D^3)`` matmul)."""
    return float(np.einsum("ab,ba->", c_k, noisy).real)


def brute_force_collective_value(rho: np.ndarray, k: int, noise_model: str, g: float, n: int) -> float:
    """Ground-truth ``Tr(C_k . noisy_kcopy)`` from an explicit ``2^{nk}``-dim construction.

    Builds the full ``k``-copy state, damages it, and traces against the explicit
    cyclic-permutation operator ``C_k``, WITHOUT assuming either bias law.
    Depolarizing: mix ``rho^{ox k}`` with ``I/D`` at rate ``g``.  Per-qubit: apply
    the Kraus channel to all ``k n`` physical qubits of ``rho^{ox k}``.  Feasible
    only for small ``(n, k)`` (the register is ``2^{nk}``-dimensional).
    """
    rho = np.asarray(rho, dtype=np.complex128)
    d_big = (2 ** n) ** k
    state = kron_power(rho, k)
    if noise_model == "depolarizing":
        noisy = (1.0 - g) * state + g * np.eye(d_big, dtype=np.complex128) / d_big
    elif noise_model in _PERQUBIT_KRAUS:
        noisy = _apply_channel_dense(state, _PERQUBIT_KRAUS[noise_model](g), k * n)
    else:
        raise ValueError(f"unknown noise_model {noise_model!r}")
    return _cyclic_trace(cyclic_permutation_operator(2 ** n, k), noisy)
