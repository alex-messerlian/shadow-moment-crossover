"""Scaling study: single-copy vs collective purity estimation vs system size.

Measures the copy-fair single-copy shadow purity RMSE and the collective (2-copy
SWAP) RMSE as a function of ``n`` for two ensembles (:func:`noisy_pure`,
:func:`random_mixed`) under three noise channels, and locates the crossover in
``n``; the size at which collective starts beating single-copy.

Noise is modelled on the *collective* (2-copy) route only; the single-copy
shadow estimator is the ideal/noiseless statistical baseline, reflecting that a
joint 2-copy measurement is harder to implement noiselessly than local
single-qubit tomography.  So ``single_rmse`` is a function of ``(ensemble, n)``
only; the ``(noise_model, rate)`` axis degrades ``collective_rmse`` alone.

Copy budget: single-copy consumes exactly ``budget`` snapshots (forming all
pairs is free classical post-processing); collective consumes ``2 * (budget //
2)`` copies (two per SWAP-test measurement), equal for even ``budget``, off by
one copy for odd ``budget``.

Efficiency: shadow snapshots are drawn from the *factored* state (``U G`` only,
``O(R n 2^n)`` per snapshot, no ``2^n x 2^n`` rotation), and the exact
single-copy U-statistic uses the ``O(M^2 n)`` per-qubit factorization
(:func:`~anrl.benchmark.shadows.full_purity_ustatistic`).  The collective signal
for the per-qubit channels applies the channel to the dense ``rho`` with a
reshape-based ``O(n 2^{2n})`` contraction (feasible to ``n ~ 12``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .channels import amplitude_damping_kraus, dephasing_kraus
from .ensembles import NoisyState, noisy_pure, random_mixed
from .moments import collective_moment_estimate
from .shadows import _I2, _KET_BRA, full_purity_ustatistic, haar_unitary


def _apply_local_unitaries(columns: np.ndarray, unitaries: list[np.ndarray], n: int) -> np.ndarray:
    """Apply ``U_q`` (2x2) to qubit ``q`` of every column of ``columns`` (2^n x R)."""
    r = columns.shape[1]
    tensor = columns.reshape([2] * n + [r])
    for q in range(n):
        tensor = np.tensordot(unitaries[q], tensor, axes=([1], [q]))  # U axis0 -> front
        tensor = np.moveaxis(tensor, 0, q)
    return tensor.reshape(2 ** n, r)


def snapshots_factored(
    state: NoisyState, n_snapshots: int, rng: np.random.Generator
) -> np.ndarray:
    """``(M, n, 2, 2)`` local-shadow snapshots from a factored :class:`NoisyState`.

    Outcome probabilities ``p(b) = (1-q) * |U G|^2_b + q / 2^n`` are computed from
    ``U G`` only (never a dense ``2^n x 2^n`` rotation), so this scales to large n.
    """
    n, dim, g, q = state.n, state.dim, state.components, state.q
    snaps = np.empty((n_snapshots, n, 2, 2), dtype=np.complex128)
    shifts = n - 1 - np.arange(n)
    for s in range(n_snapshots):
        unitaries = [haar_unitary(2, rng) for _ in range(n)]
        ug = _apply_local_unitaries(g, unitaries, n)  # (dim, R)
        p_pure = (np.abs(ug) ** 2).sum(axis=1)  # sums to ||G||_F^2 = 1
        probs = np.clip((1.0 - q) * p_pure + q / dim, 0.0, None)
        probs /= probs.sum()
        outcome = int(rng.choice(dim, p=probs))
        bits = (outcome >> shifts) & 1
        for qb in range(n):
            u_q = unitaries[qb]
            rho_meas = u_q.conj().T @ _KET_BRA[bits[qb]] @ u_q
            snaps[s, qb] = 3.0 * rho_meas - _I2
    return snaps


def _apply_channel_dense(rho: np.ndarray, kraus: list[np.ndarray], n: int) -> np.ndarray:
    """Apply a single-qubit channel to every qubit of dense ``rho`` (O(n 2^{2n}))."""
    out = rho
    for q in range(n):
        tensor = out.reshape([2] * n + [2] * n)  # row axes 0..n-1, col axes n..2n-1
        new = np.zeros_like(tensor)
        for k in kraus:
            tk = np.tensordot(k, tensor, axes=([1], [q]))
            tk = np.moveaxis(tk, 0, q)
            tk = np.tensordot(k.conj(), tk, axes=([1], [n + q]))
            tk = np.moveaxis(tk, 0, n + q)
            new += tk
        out = new.reshape(2 ** n, 2 ** n)
    return out


def collective_purity_signal(state: NoisyState, noise_model: str, rate: float) -> float:
    """Noisy 2-copy SWAP-test signal for ``Tr(rho^2)`` under a named noise model.

    Depolarizing uses the closed form ``(1-p) purity + p / 2^n``; amplitude
    damping and dephasing apply their per-qubit Kraus channel to the dense state
    and return ``Tr(sigma^2)``.
    """
    n, d = state.n, state.dim
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"noise rate must be in [0, 1], got {rate}")
    if noise_model == "depolarizing":
        return (1.0 - rate) * state.purity() + rate / d
    if n > _MAX_DENSE_CHANNEL_N:
        raise ValueError(
            f"{noise_model} collective signal needs a dense 2^{n} x 2^{n} rho "
            f"(O(2^{{2n}}) memory); n={n} exceeds _MAX_DENSE_CHANNEL_N={_MAX_DENSE_CHANNEL_N}"
        )
    if noise_model == "amplitude_damping":
        kraus = amplitude_damping_kraus(rate)
    elif noise_model == "dephasing":
        kraus = dephasing_kraus(rate)
    else:
        raise ValueError(f"unknown noise_model {noise_model!r}")
    sigma = _apply_channel_dense(state.density_matrix(), kraus, n)
    return float((np.abs(sigma) ** 2).sum().real)  # Tr(sigma^2) = ||sigma||_F^2


def _rmse(errors: list[float]) -> float:
    arr = np.asarray(errors, dtype=np.float64)
    return float(np.sqrt(np.mean(arr ** 2)))


ENSEMBLES = {"noisy_pure": noisy_pure, "random_mixed": random_mixed}
# Deterministic integer id per ensemble for reproducible seeding (str hash() is
# salted across runs).
_ENSEMBLE_ID = {"noisy_pure": 0, "random_mixed": 1}
# The amplitude-damping / dephasing collective signal builds a dense 2^n x 2^n
# rho and applies an O(n 2^{2n}) channel; guard against silent multi-GB blowups.
# (depolarizing uses the closed form and is exempt.)  n=13 -> 2^13 x 2^13 ~ 1 GB.
_MAX_DENSE_CHANNEL_N = 13


def run_scaling(
    ensembles: tuple[str, ...] = ("noisy_pure", "random_mixed"),
    sizes: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    noise_models: tuple[str, ...] = ("depolarizing", "amplitude_damping", "dephasing"),
    rates: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1),
    ensemble_q: float = 0.1,
    budget: int = 2000,
    n_states: int = 12,
    seed: int = 0,
    max_random_mixed_n: int = 8,
) -> list[dict]:
    """Run the full scaling grid; one row per (ensemble, n, noise_model, rate)."""
    rows: list[dict] = []
    for ensemble in ensembles:
        make = ENSEMBLES[ensemble]
        max_n = max(sizes) if ensemble == "noisy_pure" else max_random_mixed_n
        for n in sizes:
            if n > max_n:
                continue
            state_rng = np.random.default_rng([seed, _ENSEMBLE_ID[ensemble], n, 0])
            states = [make(n, ensemble_q, state_rng) for _ in range(n_states)]
            true_purities = [s.purity() for s in states]

            # Single-copy fair RMSE.  Noise is modelled on the *collective* route
            # only (harder to implement jointly); the single-copy shadow baseline
            # is the ideal/noiseless estimator, so this is independent of
            # (noise_model, rate) for a fixed (ensemble, n).
            snap_rng = np.random.default_rng([seed, _ENSEMBLE_ID[ensemble], n, 1])
            single_errors = [
                abs(full_purity_ustatistic(snapshots_factored(s, budget, snap_rng)) - p)
                for s, p in zip(states, true_purities)
            ]
            single_rmse = _rmse(single_errors)

            for noise_idx, noise_model in enumerate(noise_models):
                for rate_idx, rate in enumerate(rates):
                    # Seed from loop *indices* (not value-derived) so distinct
                    # cells never share an RNG stream regardless of rate precision.
                    shot_rng = np.random.default_rng(
                        [seed, _ENSEMBLE_ID[ensemble], n, noise_idx, rate_idx]
                    )
                    coll_errors = [
                        abs(
                            collective_moment_estimate(
                                2, budget // 2, collective_purity_signal(s, noise_model, rate), shot_rng
                            )
                            - p
                        )
                        for s, p in zip(states, true_purities)
                    ]
                    collective_rmse = _rmse(coll_errors)
                    rows.append(
                        {
                            "ensemble": ensemble,
                            "n": int(n),
                            "noise_model": noise_model,
                            "rate": float(rate),
                            "budget": int(budget),
                            "mean_true_purity": float(np.mean(true_purities)),
                            "single_rmse": float(single_rmse),
                            "collective_rmse": float(collective_rmse),
                            "winner": "collective" if collective_rmse < single_rmse else "single-copy",
                        }
                    )
    return rows


def save_scaling(rows: list[dict], path: str | Path, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"metadata": metadata or {}, "rows": rows}, handle, indent=2)
    return path
