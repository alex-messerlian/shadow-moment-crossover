"""First-principles single-copy purity variance law (derived + independently verified).

The single-copy purity estimator is the second-order U-statistic
``U_M = C(M,2)^{-1} sum_{i<j} Tr(G_i G_j)`` over ``M`` classical-shadow snapshots
(``E[G_i] = rho``, kernel ``h = Tr(G_i G_j)``).  Its EXACT variance is the k=2
Hoeffding/Lee formula

    Var(U_M) = [ 4 (M-2) zeta1 + 2 zeta2 ] / [ M (M-1) ]                        (*)

with the two Hoeffding components

    zeta1 = Var_x[ Tr(G_x rho) ]      (first-order projection variance),
    zeta2 = Var[ Tr(G_i G_j) ]        (kernel variance over independent pairs).

VERIFIED here against brute-force Monte Carlo of the full estimator (ratio 0.98-1.01
across pure/noisy/GHZ states and M = 4..16); it is exactly ``exact_ustatistic_variance``
at k=2.  Two consequences that CORRECT the earlier two-term shortcut
``k^2 zeta1/M + zeta2/M^2``:

* the large-M expansion of (*) is ``4 zeta1/M + (2 zeta2 - 4 zeta1)/M^2 + ...``; the
  second-order coefficient is ``2 zeta2`` (asymptotically), not ``zeta2``;
* hence the budget crossover (where the linear and higher-order terms balance) is

    M* = zeta2 / (2 zeta1)                                                      (**)

  NOT ``zeta2 / (k^2 zeta1) = zeta2 / (4 zeta1)`` from the two-term model.

Scaling (noisy-pure Haar ensemble, depolarizing q = 0.1), from converged MC in this
derivation (500k-1M snapshots, cross-checked for convergence):

    zeta1 ~ 0.63 * 1.35^n   (n=2..7);  zeta2 ~ 1.10 * 6.93^n;  M* ~ 0.87 * 5.15^n

so the base of these FINITE-SIZE fits is ``zeta2_base / zeta1_base ~ 5.1-5.3``,
independent of the 2-vs-4 prefactor above.  That is a fit over the sizes reached, not the
asymptotic rate: the ensemble-averaged closed forms below give M* an exact asymptotic base
of ``28/5 = 5.6``, which the fits approach from below.

``zeta1`` and ``zeta2`` have NO simple weight-only closed form for n >= 2.  The reason is
NOT entanglement: because the snapshots are drawn from rho, the second moment is cubic in
rho while a weight-only sum of <P>^2 is quadratic in the Pauli expectations, and the
missing dependence is on the relative orientation of the marginal Bloch vectors through
the correlation matrix, an invariant already nonzero for separable, classically
correlated states (see Section 3.4 of the paper).  A weight-only ansatz therefore fails,
but the quantities themselves are NOT beyond closed form: at k = 2 the ensemble-averaged
``zeta1`` and ``zeta2`` are exact rational functions of ``(n, q)``, implemented here as
:func:`closed_form_zetas` by Haar-averaging the Huang--Kueng--Preskill second-moment
identity.  For k >= 3, and for a single arbitrary state, they are estimated numerically
(see :func:`~anrl.theory.variance.estimate_zetas`).
"""

from __future__ import annotations

import numpy as np

# Converged reference scalings from this derivation (noisy-pure, q=0.1). MC values,
# NOT closed forms, recorded so the law is reproducible, not to be taken on faith.
REFERENCE_SCALINGS_Q0_1 = {
    "n_range_2_7": {"zeta1": (0.63, 1.346), "zeta2": (1.10, 6.928), "M_star": (0.87, 5.147)},
    "n_range_2_9": {"zeta1": (0.72, 1.299), "zeta2": (1.09, 6.941), "M_star": (0.76, 5.345)},
    "empirical_M_star_base_earlier_phase": 5.343,
}


def hoeffding_variance(m: int, zeta1: float, zeta2: float) -> float:
    """EXACT k=2 U-statistic variance ``[4(M-2)zeta1 + 2 zeta2] / [M(M-1)]``, eq. (*).

    Valid for ``M >= 2``.  Equal to :func:`~anrl.theory.variance.exact_ustatistic_variance`
    with ``k = 2`` and ``components = [zeta1, zeta2]``.
    """
    if m < 2:
        raise ValueError(f"purity U-statistic needs M >= 2, got {m}")
    return (4.0 * (m - 2) * zeta1 + 2.0 * zeta2) / (m * (m - 1))


def hoeffding_rmse(m: int, zeta1: float, zeta2: float) -> float:
    """RMSE ``sqrt(Var(U_M))`` of the single-copy purity estimator (exact k=2 law)."""
    return float(np.sqrt(max(0.0, hoeffding_variance(m, zeta1, zeta2))))


def crossover_budget(zeta1: float, zeta2: float) -> float:
    """Budget threshold ``M* = zeta2 / (2 zeta1)``, eq. (**), the EXACT-formula crossover.

    Below ``M*`` the ``zeta2`` (kernel-variance) term dominates and RMSE ~ 1/M
    (alpha -> 1); above it the ``zeta1`` term dominates and RMSE ~ 1/sqrt(M)
    (alpha -> 1/2).  This corrects the two-term model's ``zeta2/(4 zeta1)``.
    """
    if zeta1 <= 0:
        return float("inf")
    return zeta2 / (2.0 * zeta1)


def predicted_alpha(budgets: list[int], zeta1: float, zeta2: float) -> float:
    """Effective exponent ``alpha = -d log(RMSE)/d log(M)`` from a log-log fit of the
    EXACT-formula RMSE over ``budgets`` (>= 2 budgets).  This is the out-of-sample
    predictor of the measured budget-scaling exponent."""
    x = np.log(np.asarray(budgets, dtype=np.float64))
    if x.size < 2:
        raise ValueError(f"predicted_alpha needs >= 2 budgets, got {x.size}")
    y = np.log([hoeffding_rmse(m, zeta1, zeta2) for m in budgets])
    xm = x.mean()
    return -float(((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum())


def single_qubit_second_moment(t: float) -> float:
    """EXACT single-qubit shadow identity ``E[ Tr(G r)^2 ] = 1/4 + (5/4) t^2 = (5/2) p - 1``
    for a state with Bloch length ``t`` (purity ``p = (1 + t^2)/2``).  Verified to ~1e-4.
    """
    return 0.25 + 1.25 * t * t


def single_qubit_zeta1(t: float) -> float:
    """Closed-form single-qubit ``zeta1 = Var[Tr(G r)] = E[Tr(G r)^2] - p^2 = (3/4) t^2 - (1/4) t^4``.

    (``p = (1+t^2)/2``, so ``p^2 = (1 + 2 t^2 + t^4)/4`` and the identity collapses.)
    This is the ONLY n for which ``zeta1`` has a simple closed form; for n >= 2 the
    weight-only Pauli ansatz ``zeta1 = sum_P c_{|P|} <P>^2`` fails; not because of
    entanglement, but because the ansatz is quadratic in the Pauli expectations while the
    second moment is cubic in rho, and it misses the correlation-matrix invariant that is
    already nonzero for separable states (Section 3.4).  ``zeta1`` must be computed
    numerically.
    """
    return 0.75 * t * t - 0.25 * t ** 4


def closed_form_zetas(n: int, q: float) -> tuple[float, float]:
    """ENSEMBLE-AVERAGED k=2 projection variances ``(zeta1, zeta2)`` for the noisy-pure ensemble.

    Closed form for ``E_psi[zeta_c]`` over ``rho = (1-q)|psi><psi| + q I/2^n`` with
    ``|psi>`` Haar-random, obtained by Haar-averaging the per-state second-moment identity
    of Huang--Kueng--Preskill (arXiv:2002.08953, Lemma 4, eq. S52): the exact
    ``E[hat x_P hat x_Q] = 3^{|supp(P) cap supp(Q)|} Tr(rho, P ominus Q)`` for two
    reconstructed Pauli coefficients under local shadows.  Averaging uses the Haar moments
    ``E[<P>^2] = 1/(d+1)`` and ``E[<P_u><P_s><P_s'>] = 2/((d+1)(d+2))`` (when
    ``P_s P_s' = P_u``) and four weighted counts over compatible Pauli pairs
    (``sum 3^|overlap| = 16^n`` with diagonal ``10^n``; ``sum 9^|overlap| = 34^n`` with
    diagonal ``28^n``).  With ``d = 2^n``, ``u = 1-q``, ``p2 = u^2 + q(2-q)/d``,

        zeta1 = 4^-n [ 1 + u^2(10^n-1)/(d+1) + 2u^2(4^n-1)/(d+1)
                       + 2u^3(16^n-10^n-2*4^n+2)/((d+1)(d+2)) ] - p2^2,
        zeta2 = 4^-n [ 28^n + u^2(34^n-28^n)/(d+1) ] - p2^2.

    Consequences (verified): ``zeta2/7^n -> 1`` (so ``M*`` has base ``28/5 = 5.6`` and
    prefactor ``1/(2(1-q)^2)`` exactly).  These are **k=2 only**; the ``k >= 3`` projection
    variances have no closed form here (their kernels are dominated by state-dependent Haar
    terms and require compatible-tuple counts beyond these four).

    Exact rational arithmetic (``fractions.Fraction`` over Python big integers) is used so
    there is no float overflow or precision loss at any ``n`` (the ``float`` cast is applied
    only to the final results).
    """
    from fractions import Fraction

    d = 2 ** n
    uf = Fraction(q).limit_denominator(10 ** 9)
    u = Fraction(1) - uf  # u = 1 - q, exact for the study's q values (e.g. 0.1 -> 9/10)
    p2 = u * u + uf * (2 - uf) / Fraction(d)
    inv4n = Fraction(1, 4 ** n)
    z1 = inv4n * (
        1
        + u ** 2 * Fraction(10 ** n - 1, d + 1)
        + 2 * u ** 2 * Fraction(4 ** n - 1, d + 1)
        + 2 * u ** 3 * Fraction(16 ** n - 10 ** n - 2 * 4 ** n + 2, (d + 1) * (d + 2))
    ) - p2 ** 2
    z2 = inv4n * (28 ** n + u ** 2 * Fraction(34 ** n - 28 ** n, d + 1)) - p2 ** 2
    return float(z1), float(z2)
