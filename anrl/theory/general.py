"""State-agnostic sampler and Hoeffding-component estimator (any density matrix).

The noisy-pure closed forms in :mod:`anrl.theory.variance` assume ``rho`` has the
rank-1-plus-identity spectrum ``lam1 |psi><psi| + lam0 (I - |psi><psi|)``.  For the
stress-test ensembles (Haar-pure is a special case; GHZ shares the structure;
low-rank does NOT) we estimate the exact Hoeffding components from DENSE ``rho``
powers, so the theory is tested rather than its noisy-pure shortcut.

* :func:`sample_batched_general` — vectorized local shadows from any
  ``rho = (1-q) G G^dag + q I/2^n`` (rank ``>= 1``).
* :func:`estimate_hoeffding_components_general` — the exact Hoeffding components
  ``[zeta_1, ..., zeta_k]`` for ``k in {2, 3}`` using ``zeta_c = Var(E[h|X_1..X_c])
  = Var(Re Tr(G_1..G_c rho^{k-c}))`` with dense ``rho^{k-c}``.
"""

from __future__ import annotations

import numpy as np

from anrl.benchmark.ensembles import NoisyState
from anrl.benchmark.moment_ustats import _apply_left_2x2
from anrl.benchmark.shadows import _I2
from .bias import collective_bias, collective_value
from .variance import _kernel_values, exact_single_copy_rmse


def sample_batched_general(
    state: NoisyState, n_snapshots: int, rng: np.random.Generator, chunk: int = 8192
) -> np.ndarray:
    """Vectorized ``(M, n, 2, 2)`` local shadows from any rank-``R`` noisy state.

    Outcome probability ``p(b) = (1-q) sum_c |U g_c|^2_b + q/2^n`` where ``g_c`` are
    the columns of ``G`` (``rho = (1-q) G G^dag + q I/2^n``).  Reduces to
    :func:`~anrl.benchmark.budget.sample_batched` for rank 1.
    """
    n, d, q = state.n, state.dim, state.q
    cols = state.components.reshape([2] * n + [state.components.shape[1]])  # (2,)*n x R
    snaps = np.empty((n_snapshots, n, 2, 2), dtype=np.complex128)
    shifts = n - 1 - np.arange(n)
    for s0 in range(0, n_snapshots, chunk):
        c = min(chunk, n_snapshots - s0)
        z = (rng.standard_normal((c, n, 2, 2)) + 1j * rng.standard_normal((c, n, 2, 2))) / np.sqrt(2.0)
        qr_q, qr_r = np.linalg.qr(z)
        u = qr_q * (np.diagonal(qr_r, axis1=-2, axis2=-1) / np.abs(np.diagonal(qr_r, axis1=-2, axis2=-1)))[:, :, None, :]
        # amp[chunk, (2,)*n, R] = apply ox_q U_q to each column of G
        amp = np.broadcast_to(cols, (c,) + cols.shape).astype(np.complex128).copy()
        for qb in range(n):
            amp = np.moveaxis(amp, qb + 1, 1)
            amp = np.einsum("cxr,cr...->cx...", u[:, qb], amp)
            amp = np.moveaxis(amp, 1, qb + 1)
        amp = amp.reshape(c, d, state.components.shape[1])
        p_pure = (np.abs(amp) ** 2).sum(axis=2)  # sum over columns -> (c, d)
        probs = (1.0 - q) * p_pure + q / d
        probs /= probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        outcome = np.minimum((cdf < rng.random(c)[:, None]).sum(axis=1), d - 1)
        bits = (outcome[:, None] >> shifts) & 1
        for qb in range(n):
            v = np.conj(u[np.arange(c), qb, bits[:, qb], :])
            snaps[s0:s0 + c, qb] = 3.0 * (v[:, :, None] * np.conj(v[:, None, :])) - _I2
    return snaps


_TR_CHUNK = 2048  # bound the (chunk, d, d) transient (~130 MB at d=64) — chunked over tuples


def _tr_G_word_rho(word: np.ndarray, rho_power: np.ndarray, n: int) -> np.ndarray:
    """``Tr((ox_q word^q) rho^p)`` per tuple; ``word`` is the per-qubit G-product ``(T,n,2,2)``.

    Chunked over tuples: the full ``(T, d, d)`` broadcast of ``rho^p`` would be
    multi-GB at ``T ~ 10^5`` (the crash culprit), so process ``_TR_CHUNK`` at a time.
    """
    t, d = word.shape[0], rho_power.shape[0]
    out = np.empty(t, dtype=np.complex128)
    for s in range(0, t, _TR_CHUNK):
        w = word[s:s + _TR_CHUNK]
        c = w.shape[0]
        y = np.broadcast_to(rho_power, (c, d, d)).copy().reshape([c] + [2] * n + [d])
        for q in range(n):
            y = _apply_left_2x2(w[:, q], y, q)  # left-multiply row-axis q of rho^p by word^q
        out[s:s + c] = np.einsum("tii->t", y.reshape(c, d, d))
    return out


def estimate_hoeffding_components_general(
    state: NoisyState, k: int, n_samples: int, rng: np.random.Generator
) -> list[float]:
    """Exact Hoeffding components ``[zeta_1, ..., zeta_k]`` for any state (``k in {2,3}``).

    ``zeta_c = Var(Re Tr(G_1..G_c rho^{k-c}))`` (dense ``rho^{k-c}``); ``zeta_k`` is
    the full symmetric-kernel variance.
    """
    if k not in (2, 3):
        raise ValueError(f"general estimator supports k in {{2, 3}}, got {k}")
    n = state.n
    rho = state.density_matrix()
    rho_pows = {p: np.linalg.matrix_power(rho, p) for p in range(1, k)}

    g1 = sample_batched_general(state, n_samples, rng)
    z1 = float(np.var(_tr_G_word_rho(g1, rho_pows[k - 1], n).real, ddof=1))  # zeta_1

    comps = [z1] + [0.0] * (k - 1)
    if k == 3:
        g2 = sample_batched_general(state, n_samples, rng)
        word = g1 @ g2  # per-qubit product g_1^q g_2^q
        comps[1] = float(np.var(_tr_G_word_rho(word, rho_pows[1], n).real, ddof=1))  # zeta_2 = Re Tr(G1 G2 rho)
    # zeta_k = full symmetric-kernel variance
    batches = [sample_batched_general(state, n_samples, rng) for _ in range(k)]
    comps[k - 1] = float(np.var(_kernel_values(batches), ddof=1))
    return comps


def predicted_collective_rmse_general(
    rhos: list[np.ndarray], k: int, noise_model: str, g: float, budget: int, n: int
) -> float:
    """Collective RMSE = bias floor (Part 1) + binomial shot noise, averaged over states."""
    biases = [collective_bias(r, k, noise_model, g, n) for r in rhos]
    signals = [collective_value(r, k, noise_model, g, n) for r in rhos]
    mean_bias_sq = float(np.mean(np.square(biases)))
    mean_signal = float(np.mean(signals))
    var_shot = max(0.0, 1.0 - mean_signal * mean_signal) / max(1, budget // k)
    return float((mean_bias_sq + var_shot) ** 0.5)


def predict_crossover_general(
    k: int, budget: int, sizes: list[int], components_by_n: dict, rhos_by_n: dict,
    noise_model: str, g: float,
) -> int | None:
    """Sustained crossover: smallest n from which single-copy RMSE exceeds collective."""
    ns = [n for n in sorted(sizes) if n in components_by_n]
    wins = {
        n: exact_single_copy_rmse(components_by_n[n], k, budget)
        > predicted_collective_rmse_general(rhos_by_n[n], k, noise_model, g, budget, n)
        for n in ns
    }
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None
