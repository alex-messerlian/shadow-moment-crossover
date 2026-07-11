"""Evaluation harness for purity estimators at a fixed copy budget.

``evaluate_estimator`` runs an estimator over a shared set of states at a fixed
total *copy* budget and reports the error against the exact purity.  Copy cost is
accounted for per estimator: single-copy shadows spend one copy per snapshot
(``copies_per_use = 1``), while the 2-copy SWAP test spends two copies per
measurement (``copies_per_use = 2``), so a shared ``budget`` becomes ``budget``
snapshots vs ``budget // 2`` SWAP tests.

The same ``states`` list is passed to every estimator, so comparisons are paired
by state.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from anrl.physics import purity

from .channels import collective_moment_signal
from .collective import collective_purity_estimate
from .moments import collective_moment_estimate, shadow_moment_estimate
from .shadows import shadow_purity_estimate

Estimator = Callable[[np.ndarray, int, np.random.Generator], float]


def make_shadow_estimator(n_pairs: int | None = None) -> Estimator:
    """Single-copy shadow estimator callable ``(rho, n_uses, rng) -> estimate``."""

    def estimator(rho: np.ndarray, n_uses: int, rng: np.random.Generator) -> float:
        return shadow_purity_estimate(rho, n_uses, rng, n_pairs=n_pairs)

    estimator.copies_per_use = 1  # type: ignore[attr-defined]
    estimator.name = "single-copy shadow"  # type: ignore[attr-defined]
    return estimator


def make_collective_estimator(p_gate: float, gate_count_fn: Callable[[int], int]) -> Estimator:
    """2-copy SWAP-test estimator callable ``(rho, n_uses, rng) -> estimate``."""

    def estimator(rho: np.ndarray, n_uses: int, rng: np.random.Generator) -> float:
        n_qubits = int(round(np.log2(rho.shape[0])))
        return collective_purity_estimate(
            rho, n_uses, p_gate, n_qubits, gate_count_fn, rng
        )

    estimator.copies_per_use = 2  # type: ignore[attr-defined]
    estimator.name = f"collective SWAP (p_gate={p_gate}, {gate_count_fn.__name__})"  # type: ignore[attr-defined]
    return estimator


def make_shadow_moment_estimator(k: int, n_tuples: int | None = None) -> Estimator:
    """Single-copy shadow estimator of ``Tr(rho^k)`` — ``copies_per_use = 1``."""

    def estimator(rho: np.ndarray, n_uses: int, rng: np.random.Generator) -> float:
        return shadow_moment_estimate(rho, k, n_uses, rng, n_tuples=n_tuples)

    estimator.copies_per_use = 1  # type: ignore[attr-defined]
    estimator.name = f"single-copy shadow (k={k})"  # type: ignore[attr-defined]
    return estimator


def make_collective_moment_estimator(k: int, noise_model: str, rate: float) -> Estimator:
    """k-copy cyclic-test estimator of ``Tr(rho^k)`` — ``copies_per_use = k``.

    The noisy signal is computed per state from ``noise_model`` and ``rate``
    (closed form for depolarizing; explicit Kraus channel otherwise).
    """

    def estimator(rho: np.ndarray, n_uses: int, rng: np.random.Generator) -> float:
        n = int(round(np.log2(rho.shape[0])))
        signal = collective_moment_signal(rho, k, noise_model, rate, n)
        return collective_moment_estimate(k, n_uses, signal, rng)

    estimator.copies_per_use = k  # type: ignore[attr-defined]
    estimator.name = f"collective (k={k}, {noise_model}, rate={rate})"  # type: ignore[attr-defined]
    return estimator


def evaluate_estimator(
    estimator: Estimator,
    states: Sequence[np.ndarray],
    budget: int,
    rng: np.random.Generator,
    copies_per_use: int | None = None,
    true_fn: Callable[[np.ndarray], float] = purity,
) -> dict:
    """Evaluate ``estimator`` over ``states`` at a fixed copy ``budget``.

    Returns a dict with:
        ``mean``   — mean absolute error ``|estimate - true purity|``.
        ``sem``    — standard error of that mean.
        ``errors`` — ``(len(states),)`` per-state absolute errors.
        ``rmse``   — root-mean-square error (derived from ``errors``).
        ``n_uses`` — measurements performed (``budget // copies_per_use``).
    """
    if copies_per_use is None:
        copies_per_use = getattr(estimator, "copies_per_use", 1)
    n_uses = budget // copies_per_use
    if n_uses < 1:
        raise ValueError(f"budget {budget} too small for copies_per_use {copies_per_use}")

    errors = np.array(
        [abs(estimator(state, n_uses, rng) - true_fn(state)) for state in states],
        dtype=np.float64,
    )
    mean = float(errors.mean())
    sem = float(errors.std(ddof=1) / np.sqrt(len(errors))) if len(errors) > 1 else 0.0
    return {
        "mean": mean,
        "sem": sem,
        "errors": errors,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "n_uses": n_uses,
    }
