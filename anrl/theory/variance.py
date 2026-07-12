"""Part 2 — the single-copy variance law (Hoeffding decomposition).

The single-copy estimator of ``Tr(rho^k)`` is a ``k``-th order U-statistic.  Its
variance splits into a linear term and a higher-order term,

    Var(U_M) ~= k^2 zeta1 / M + zeta2 / M^2 ,

with ``zeta1`` the variance of the first-order Hoeffding projection and ``zeta2``
the variance of the kernel over independent tuples.  The two terms cross at the
budget threshold

    M*(n, k) = zeta2 / (k^2 zeta1).

When ``M >> M*`` the linear term wins -> ``RMSE ~ M^{-1/2}`` (alpha=0.5); when
``M << M*`` the higher-order term wins -> ``RMSE ~ M^{-1}`` (alpha=1.0).  The
effective exponent interpolates as ``alpha_eff = 0.5 w + 1.0 (1-w)`` with
``w = M / (M + M*)``.

Estimation (Monte Carlo from simulated shadow snapshots, noisy-pure ensemble):

* ``zeta2`` — the sample variance of the (real) kernel
  ``Re Tr(G_1 ... G_k) = Re prod_q Tr(g_1^q ... g_k^q)`` over independent k-tuples.
* ``zeta1`` — the sample variance of the first-order projection
  ``h1(x) = Tr(G_x rho^{k-1})``.  By cyclicity and ``E[G] = rho`` this is
  independent of where ``x`` sits, and for ``rho = lambda1 |psi><psi| + lambda0
  (I - |psi><psi|)`` (noisy pure) it reduces to
  ``h1(x) = lambda0^{k-1} + (lambda1^{k-1} - lambda0^{k-1}) <psi|G_x|psi>`` (since
  ``Tr(G_x) = 1``), so ``zeta1 = (lambda1^{k-1} - lambda0^{k-1})^2 Var_x <psi|G_x|psi>``.
"""

from __future__ import annotations

import numpy as np

from anrl.benchmark.budget import sample_batched
from anrl.benchmark.ensembles import NoisyState


def _psi_G_psi(state: NoisyState, snaps: np.ndarray) -> np.ndarray:
    """``<psi|G_x|psi>`` per snapshot (``G_x = ox_q g_x^q`` applied to the pure part)."""
    n, d = state.n, state.dim
    psi = state.components[:, 0].reshape([2] * n)
    amp = np.broadcast_to(psi, (snaps.shape[0],) + psi.shape).astype(np.complex128).copy()
    for q in range(n):
        amp = np.moveaxis(amp, q + 1, 1)
        amp = np.einsum("cxr,cr...->cx...", snaps[:, q], amp)
        amp = np.moveaxis(amp, 1, q + 1)
    amp = amp.reshape(snaps.shape[0], d)
    psi_flat = psi.reshape(d)
    return (np.conj(psi_flat)[None, :] * amp).sum(axis=1).real


def _eigs(state: NoisyState) -> tuple[float, float]:
    """The two eigenvalues of a noisy-pure ``rho``: ``lambda1`` (on psi), ``lambda0``."""
    q, d = state.q, state.dim
    return (1.0 - q) + q / d, q / d


def estimate_zeta1(state: NoisyState, k: int, n_samples: int, rng: np.random.Generator) -> float:
    """``zeta1 = Var_x Tr(G_x rho^{k-1})`` via the noisy-pure closed form."""
    lam1, lam0 = _eigs(state)
    snaps = sample_batched(state, n_samples, rng)
    p = _psi_G_psi(state, snaps)  # <psi|G_x|psi>
    coeff = lam1 ** (k - 1) - lam0 ** (k - 1)
    return float(coeff ** 2 * np.var(p, ddof=1))


# The SYMMETRIC U-statistic kernel is the average of ``Re Tr(G_{pi(1)} ... G_{pi(k)})``
# over the cyclic/reversal-inequivalent orderings (reversal = complex conjugate,
# already handled by ``Re``).  k=2 and k=3 each collapse to a single ordering; k=4
# has three (``1234``, ``1243``, ``1324``).
_KERNEL_ORDERINGS = {
    2: [(0, 1)],
    3: [(0, 1, 2)],
    4: [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)],
}


def _kernel_values(batches: list[np.ndarray]) -> np.ndarray:
    """Symmetric kernel ``mean_orderings Re Tr(G_{o(1)} ... G_{o(k)})`` per tuple."""
    t, n, k = batches[0].shape[0], batches[0].shape[1], len(batches)
    out = np.zeros(t)
    orderings = _KERNEL_ORDERINGS[k]
    for order in orderings:
        kernel = np.ones(t, dtype=np.complex128)
        for q in range(n):
            prod = batches[order[0]][:, q]
            for j in order[1:]:
                prod = prod @ batches[j][:, q]
            kernel *= np.einsum("tii->t", prod)  # Tr(g_{o(1)}^q ... g_{o(k)}^q)
        out += kernel.real
    return out / len(orderings)


def estimate_zeta2(state: NoisyState, k: int, n_samples: int, rng: np.random.Generator) -> float:
    """``zeta2 = Var`` of the symmetric kernel over ``n_samples`` independent k-tuples."""
    batches = [sample_batched(state, n_samples, rng) for _ in range(k)]
    return float(np.var(_kernel_values(batches), ddof=1))


def estimate_zetas(
    n: int, k: int, ensemble_q: float, n_samples: int, seed: int, n_states: int = 4
) -> dict:
    """Estimate the Hoeffding components (and the two-term ``zeta1, zeta2, M*``).

    One estimation pass gives the full component vector ``[zeta_1, ..., zeta_k]``
    (for the EXACT model); the two-term model's ``zeta1 = zeta_1`` and
    ``zeta2 = zeta_k`` (full kernel variance) are derived from it.  Averaged over a
    few states, with the across-state relative spread as a stability check.
    """
    from anrl.benchmark.ensembles import noisy_pure

    comp_list = []
    for s in range(n_states):
        state = noisy_pure(n, ensemble_q, np.random.default_rng([seed, n, s, 0]))
        comp_list.append(estimate_hoeffding_components(state, k, n_samples, np.random.default_rng([seed, n, s, k, 3])))
    comps = [float(np.mean([c[i] for c in comp_list])) for i in range(k)]
    z1, z2 = comps[0], comps[k - 1]  # zeta_1 (first projection), zeta_k (full kernel var)
    m_star = z2 / (k * k * z1) if z1 > 0 else float("inf")
    z1s = [c[0] for c in comp_list]
    z2s = [c[k - 1] for c in comp_list]
    return {
        "n": int(n), "k": int(k), "zeta1": z1, "zeta2": z2, "M_star": float(m_star),
        "components": comps,
        "zeta1_rel_spread": float(np.std(z1s) / z1) if z1 > 0 else float("nan"),
        "zeta2_rel_spread": float(np.std(z2s) / z2) if z2 > 0 else float("nan"),
    }


def _perqubit_trace_prod(batches_in_order: list[np.ndarray]) -> np.ndarray:
    """``prod_q Tr(g_{o(1)}^q ... g_{o(m)}^q)`` per tuple (Tr of a G-word, no rho)."""
    t, n = batches_in_order[0].shape[0], batches_in_order[0].shape[1]
    out = np.ones(t, dtype=np.complex128)
    for q in range(n):
        prod = batches_in_order[0][:, q]
        for b in batches_in_order[1:]:
            prod = prod @ b[:, q]
        out *= np.einsum("tii->t", prod)
    return out.real


def _psi_word_psi(state: NoisyState, batches_in_order: list[np.ndarray]) -> np.ndarray:
    """``<psi| (ox_q g_{o(1)}^q ... g_{o(m)}^q) |psi>`` per tuple (G-word around |psi>)."""
    n, d = state.n, state.dim
    psi = state.components[:, 0].reshape([2] * n)
    word = None  # per-qubit product matrices (T, n, 2, 2)
    for b in batches_in_order:
        word = b if word is None else word @ b
    amp = np.broadcast_to(psi, (word.shape[0],) + psi.shape).astype(np.complex128).copy()
    for q in range(n):
        amp = np.moveaxis(amp, q + 1, 1)
        amp = np.einsum("cxr,cr...->cx...", word[:, q], amp)
        amp = np.moveaxis(amp, 1, q + 1)
    amp = amp.reshape(word.shape[0], d)
    return (np.conj(psi.reshape(d))[None, :] * amp).sum(axis=1).real


def _projection_variance(state: NoisyState, k: int, c: int, n_samples: int, rng: np.random.Generator) -> float:
    """``zeta_c = Var(psi_c)`` with ``psi_c(x_1..x_c) = E[h | X_1..X_c]`` (closed form).

    ``psi_c`` inserts ``rho`` in the free slots of the symmetric kernel; with
    ``rho = lam0 I + (lam1-lam0) |psi><psi|`` every trace reduces to per-qubit
    trace products and ``<psi|word|psi>`` terms.  Implemented for the projections
    the sweep needs: ``(k=3, c=2)`` and ``(k=4, c in {2,3})``.
    """
    lam1, lam0 = _eigs(state)
    delta = lam1 - lam0
    g = [sample_batched(state, n_samples, rng) for _ in range(c)]

    if k == 3 and c == 2:
        psi = lam0 * _perqubit_trace_prod([g[0], g[1]]) + delta * _psi_word_psi(state, [g[0], g[1]])
    elif k == 4 and c == 2:
        t12 = _perqubit_trace_prod([g[0], g[1]])
        w12 = _psi_word_psi(state, [g[0], g[1]])
        w21 = _psi_word_psi(state, [g[1], g[0]])
        w1 = _psi_word_psi(state, [g[0]])
        w2 = _psi_word_psi(state, [g[1]])
        tr_g1g2_rho2 = lam0 ** 2 * t12 + (lam1 ** 2 - lam0 ** 2) * w12
        tr_g1rho_g2rho = lam0 ** 2 * t12 + lam0 * delta * (w12 + w21) + delta ** 2 * w1 * w2
        psi = (2.0 * tr_g1g2_rho2 + tr_g1rho_g2rho) / 3.0
    elif k == 4 and c == 3:
        t012 = _perqubit_trace_prod([g[0], g[1], g[2]])
        t021 = _perqubit_trace_prod([g[0], g[2], g[1]])
        w012 = _psi_word_psi(state, [g[0], g[1], g[2]])
        w201 = _psi_word_psi(state, [g[2], g[0], g[1]])
        w021 = _psi_word_psi(state, [g[0], g[2], g[1]])
        psi = (lam0 * (2.0 * t012 + t021) + delta * (w012 + w201 + w021)) / 3.0
    else:
        raise ValueError(f"projection (k={k}, c={c}) not implemented")
    return float(np.var(psi, ddof=1))


def estimate_hoeffding_components(
    state: NoisyState, k: int, n_samples: int, rng: np.random.Generator
) -> list[float]:
    """The exact Hoeffding variance components ``[zeta_1, ..., zeta_k]``.

    ``zeta_c = Var(E[h | X_1..X_c])`` (the c-th projection variance).  ``zeta_1``
    (first projection) and ``zeta_k`` (full kernel variance) have direct
    estimators; ``1 < c < k`` use the closed-form projection
    (:func:`_projection_variance`).  These feed the EXACT U-statistic variance
    ``Var(U_M) = C(M,k)^{-1} sum_c C(k,c) C(M-k,k-c) zeta_c`` — the correct model
    the two-term approximation (:func:`single_copy_variance`) collapses.
    """
    comps = [0.0] * k
    comps[0] = estimate_zeta1(state, k, n_samples, rng)  # zeta_1
    comps[k - 1] = estimate_zeta2(state, k, n_samples, rng)  # zeta_k = full kernel variance
    for c in range(2, k):
        comps[c - 1] = _projection_variance(state, k, c, n_samples, rng)
    return comps


def exact_ustatistic_variance(components: list[float], k: int, m: float) -> float:
    """EXACT k-th order U-statistic variance from the Hoeffding components.

    ``Var(U_M) = C(M,k)^{-1} sum_{c=1}^k C(k,c) C(M-k,k-c) zeta_c`` (Lee, U-statistics).
    """
    from math import comb

    m_int = int(round(m))
    if m_int < k:
        return float("inf")
    ck = comb(m_int, k)
    return float(sum(comb(k, c) * comb(m_int - k, k - c) / ck * components[c - 1] for c in range(1, k + 1)))


def exact_single_copy_rmse(components: list[float], k: int, m: float) -> float:
    """Exact single-copy RMSE ``sqrt(Var(U_M))`` from the Hoeffding components."""
    return float(np.sqrt(max(0.0, exact_ustatistic_variance(components, k, m))))


def exact_fitted_alpha(budgets: list[int], components: list[float], k: int) -> float:
    """The alpha a log-log fit of the EXACT-variance RMSE curve reports over ``budgets``."""
    x = np.log(np.asarray(budgets, dtype=np.float64))
    if x.size < 2:
        raise ValueError(f"exact_fitted_alpha needs >= 2 budgets, got {x.size}")
    y = np.log([exact_single_copy_rmse(components, k, m) for m in budgets])
    xm = x.mean()
    return -float(((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum())


def single_copy_variance(k: int, zeta1: float, zeta2: float, m: float) -> float:
    """Theory single-copy variance ``k^2 zeta1 / M + zeta2 / M^2`` (two-term model)."""
    return k * k * zeta1 / m + zeta2 / (m * m)


def single_copy_rmse(k: int, zeta1: float, zeta2: float, m: float) -> float:
    """Theory single-copy RMSE ``sqrt(k^2 zeta1 / M + zeta2 / M^2)``."""
    return float(np.sqrt(single_copy_variance(k, zeta1, zeta2, m)))


def alpha_eff(m: float, m_star: float) -> float:
    """Effective budget exponent ``0.5 w + 1.0 (1 - w)`` with ``w = M / (M + M*)``."""
    w = m / (m + m_star)
    return 0.5 * w + 1.0 * (1.0 - w)


def fitted_alpha(budgets: list[int], k: int, zeta1: float, zeta2: float) -> float:
    """The alpha a log-log fit of the THEORY RMSE curve would report over ``budgets``.

    This is the fair analogue of the measured fitted alpha (fit over the same
    budget grid), unlike the local :func:`alpha_eff` at a single ``M``.
    """
    x = np.log(np.asarray(budgets, dtype=np.float64))
    if x.size < 2:
        raise ValueError(f"fitted_alpha needs >= 2 budgets to fit a slope, got {x.size}")
    y = np.log([single_copy_rmse(k, zeta1, zeta2, m) for m in budgets])
    xm = x.mean()
    return -float(((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum())
