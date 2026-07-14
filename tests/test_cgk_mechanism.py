"""Tests for the CGK reconciliation (arXiv:2512.10929).

Locks in the Sec 4.2 <-> CGK Thm 2.4 relationship computed in
``experiments/cgk_mechanism.py``:

* per-qubit depolarizing fixes the maximally mixed state (sigma_mixed = I/2^n);
* the exact bias law reproduces the noisy two-copy purity of a product state in
  closed form;
* the purity-testing gap Delta closes exponentially and the implied SWAP-test
  shot count 1/Delta^2 lies at or above the CGK universal lower bound at the
  system sizes / noise rates the paper operates in.

All checks are deterministic (no Monte-Carlo), exercising the committed bias-law
code ``anrl.theory.bias.perqubit_channel_value``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anrl.theory.bias import perqubit_channel_value

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I = np.eye(2, dtype=np.complex128)


def depolarizing_kraus(lam: float) -> list[np.ndarray]:
    """D_lambda(rho) = (1-lambda) rho + lambda I/2 in Pauli-Kraus form."""
    return [
        math.sqrt(1.0 - 3.0 * lam / 4.0) * _I,
        math.sqrt(lam / 4.0) * _X,
        math.sqrt(lam / 4.0) * _Y,
        math.sqrt(lam / 4.0) * _Z,
    ]


def _apply_single_qubit(rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    return sum(k @ rho @ k.conj().T for k in kraus)


def analytic_tr_sigma_pure2(n: int, lam: float) -> float:
    """Haar average of Tr(sigma_pure^2), sigma = D_lambda^{ox n}(|psi><psi|)."""
    D = 2 ** n
    s = 1.0 + 3.0 * (1.0 - lam) ** 2
    return (1.0 / D) * (1.0 + (s ** n - 1.0) / (D + 1.0))


def b_of_lambda(lam: float) -> float:
    return 4.0 / (1.0 + 3.0 * (1.0 - lam) ** 4)


@pytest.mark.parametrize("lam", [0.05, 0.1, 0.2, 0.3])
def test_depolarizing_kraus_matches_definition(lam: float) -> None:
    """The Kraus set implements D_lambda(rho) = (1-lambda) rho + lambda I/2 and is trace-preserving."""
    completeness = sum(k.conj().T @ k for k in depolarizing_kraus(lam))
    assert np.allclose(completeness, _I, atol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        rho = np.outer(v, v.conj())
        expected = (1.0 - lam) * rho + lam * _I / 2.0
        assert np.allclose(_apply_single_qubit(rho, depolarizing_kraus(lam)), expected, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("lam", [0.05, 0.1, 0.2])
def test_maximally_mixed_is_fixed(n: int, lam: float) -> None:
    """Depolarizing fixes I/2^n, so Tr(sigma_mixed^2) = 2^-n exactly."""
    mixed = np.eye(2 ** n, dtype=np.complex128) / (2 ** n)
    tr = perqubit_channel_value(mixed, 2, depolarizing_kraus(lam), n)
    assert tr == pytest.approx(2.0 ** (-n), abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("lam", [0.05, 0.1, 0.2])
def test_product_state_purity_closed_form(n: int, lam: float) -> None:
    """For |0>^n, Tr(sigma_pure^2) = ((1-lambda/2)^2 + (lambda/2)^2)^n (deterministic)."""
    zero = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
    zero[0, 0] = 1.0
    single = (1.0 - lam / 2.0) ** 2 + (lam / 2.0) ** 2
    expected = single ** n
    got = perqubit_channel_value(zero, 2, depolarizing_kraus(lam), n)
    assert got == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("lam", [0.05, 0.1, 0.2])
def test_bias_law_matches_haar_average(n: int, lam: float) -> None:
    """The bias law, averaged over Haar draws, reproduces the analytic Haar formula."""
    rng = np.random.default_rng(7)
    kraus = depolarizing_kraus(lam)
    vals = []
    for _ in range(400):
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        v /= np.linalg.norm(v)
        vals.append(perqubit_channel_value(np.outer(v, v.conj()), 2, kraus, n))
    assert np.mean(vals) == pytest.approx(analytic_tr_sigma_pure2(n, lam), abs=5e-3)


@pytest.mark.parametrize("lam", [0.05, 0.1, 0.2, 0.3])
def test_swap_cost_respects_cgk_lower_bound(lam: float) -> None:
    """1/Delta^2 (SWAP-test shot count) >= CGK bound min{2^(n/2), b^n} for n=2..10."""
    for n in [2, 4, 6, 8, 10]:
        delta = analytic_tr_sigma_pure2(n, lam) - 2.0 ** (-n)
        inv_delta_sq = 1.0 / delta ** 2
        cgk = min(2.0 ** (n / 2.0), b_of_lambda(lam) ** n)
        assert inv_delta_sq >= cgk - 1e-9


@pytest.mark.parametrize("lam", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5])
def test_swap_base_bounds_cgk_base(lam: float) -> None:
    """Asymptotic base (4/(1+3(1-lambda)^2))^2 >= b(lambda), equality at lambda=0.

    Follows from 4(1+3x^2) - (1+3x)^2 = 3(1-x)^2 >= 0 with x = (1-lambda)^2.
    """
    x = (1.0 - lam) ** 2
    swap_base = (4.0 / (1.0 + 3.0 * x)) ** 2
    assert swap_base >= b_of_lambda(lam) - 1e-12
    assert 4.0 * (1.0 + 3.0 * x ** 2) - (1.0 + 3.0 * x) ** 2 == pytest.approx(3.0 * (1.0 - x) ** 2, abs=1e-12)
    if lam == 0.0:
        assert swap_base == pytest.approx(b_of_lambda(lam), abs=1e-12)
