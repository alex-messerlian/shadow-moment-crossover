"""PASS 47.3(b): can ``M*(rho)`` be estimated from a PILOT shadow budget?

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/pass47_pilot_estimator.py

``M*(rho) = zeta_2(rho) / (2 zeta_1(rho))`` is exact but needs the full state
(PASS 47.2(b)).  An experimenter has snapshots, not ``rho``.  This measures whether the
threshold can be estimated from the snapshots themselves, against the exact value from
:mod:`anrl.theory.statewise_zetas` for the same state.

The estimators are unbiased and use nothing but the pilot snapshots.  Split the pilot into
four disjoint blocks ``A, B, C, D`` and write ``K_ij = Tr(G_i G_j) = prod_q Tr(G_i^q G_j^q)``:

* ``zeta_2 = Var[K_ij]``: the sample variance of ``K`` over DISJOINT snapshot pairs.  Each
  pair is independent, so this is unbiased directly.
* ``E[Tr(G rho)^2]``: ``mean_{i in A} K_iB-bar * K_iC-bar`` with ``K_iB-bar = mean_{j in B} K_ij``.
  Since ``B`` and ``C`` are disjoint from each other and from ``A``, both inner means are
  independent unbiased estimates of ``Tr(G_i rho)``, so the product is unbiased for
  ``Tr(G_i rho)^2``.
* ``Tr(rho^2)^2``: ``K_AB-bar * K_CD-bar`` over four disjoint blocks, unbiased since the two
  factors are independent and each is unbiased for ``Tr(rho^2)``.
* ``zeta_1 = E[Tr(G rho)^2] - Tr(rho^2)^2`` is the difference of the last two.

``zeta_1`` is therefore a DIFFERENCE OF TWO ESTIMATES whose common scale is ``Tr(rho^2)^2``,
each carrying noise of order ``zeta_2 / M`` -- the kernel variance, which grows as ``7^n``.
That is the mechanism this script quantifies.

Writes ``results/pass47_pilot_estimator.json``.  No paper edit.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import low_rank, noisy_pure
from anrl.theory.general import sample_batched_general
from anrl.theory.statewise_zetas import exact_zeta1, exact_zeta2, pauli_expectations, pauli_weights

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass47_pilot_estimator.json"

SEED = 47
SIZES = (2, 3, 4, 5, 6)
PILOTS = (500, 2000, 8000, 32000, 128000)
N_STATES = 4
N_REPS = 40             # independent pilot draws per (state, budget)
MAX_WORKERS = 4
ENSEMBLES = ("noisy_pure_q0.1", "variable_rank")
_ENS_ID = {"noisy_pure_q0.1": 0, "variable_rank": 5}


def make_state(ens: str, n: int, s: int):
    rng = np.random.default_rng([SEED, _ENS_ID[ens], n, s])
    if ens == "noisy_pure_q0.1":
        return noisy_pure(n, 0.1, rng)
    return low_rank(n, int(rng.integers(1, min(8, 2 ** n) + 1)), rng)


_SIGMA_VEC = np.array(                       # rows: I, X, Y, Z as flattened 2x2 matrices
    [[1, 0, 0, 1], [0, 1, 1, 0], [0, -1j, 1j, 0], [1, 0, 0, -1]], dtype=complex
) / 2.0


def snapshot_features(snaps: np.ndarray) -> np.ndarray:
    """``(M, 4^n)`` real features ``x_i`` with ``Tr(G_i G_j) = 2^n <x_i, x_j>``.

    Each snapshot factor is a Hermitian ``2 x 2``, so writing it in the Pauli basis
    ``G^q = sum_alpha a^q_alpha sigma_alpha`` with real ``a`` gives
    ``Tr(G_i^q G_j^q) = 2 <a_i^q, a_j^q>``; the product over qubits is ``2^n`` times the
    inner product of the Kronecker products.  Every block mean the estimators below need is
    then a mean of these ``4^n``-vectors, so the whole computation is ``O(M 4^n)`` instead of
    ``O(M^2 n)`` -- no pairwise matrix is ever formed.
    """
    coeff = np.einsum("ap,inp->ina", _SIGMA_VEC.conj(), snaps.reshape(*snaps.shape[:2], 4)).real
    x = coeff[:, 0, :]
    for q in range(1, snaps.shape[1]):
        x = (x[:, :, None] * coeff[:, q, None, :]).reshape(x.shape[0], -1)
    return x


_FEATURE_CHUNK = 8192   # bound the (chunk, 4^n) transient: 268 MB at n = 8


def _block_mean(snaps: np.ndarray, block: slice, dim: int) -> np.ndarray:
    """Mean feature vector over ``block``, accumulated in chunks (never materialized whole)."""
    total = np.zeros(dim)
    count = 0
    start, stop = block.start, block.stop
    for s in range(start, stop, _FEATURE_CHUNK):
        e = min(s + _FEATURE_CHUNK, stop)
        total += snapshot_features(snaps[s:e]).sum(axis=0)
        count += e - s
    return total / count


def pilot_zetas(snaps: np.ndarray) -> tuple[float, float]:
    """Unbiased ``(zeta_1_hat, zeta_2_hat)`` from ``snaps`` alone (four disjoint blocks).

    Chunked over snapshots so peak memory is ``O(_FEATURE_CHUNK * 4^n)``, independent of the
    pilot budget.
    """
    n = snaps.shape[1]
    scale = 2.0 ** n
    dim = 4 ** n
    q = snaps.shape[0] // 4
    a, b, c, d = slice(0, q), slice(q, 2 * q), slice(2 * q, 3 * q), slice(3 * q, 4 * q)

    mb, mc, md = (_block_mean(snaps, blk, dim) for blk in (b, c, d))
    ma = _block_mean(snaps, a, dim)

    # zeta_2 from the q disjoint pairs (A_i, B_i): each pair is independent.
    # E[Tr(G rho)^2]: block means of B and C are independent unbiased copies of rho.
    pair_k = np.empty(q)
    prod = np.empty(q)
    for s in range(0, q, _FEATURE_CHUNK):
        e = min(s + _FEATURE_CHUNK, q)
        xa = snapshot_features(snaps[s:e])
        xb = snapshot_features(snaps[q + s:q + e])
        pair_k[s:e] = scale * np.einsum("ij,ij->i", xa, xb)
        prod[s:e] = (scale * (xa @ mb)) * (scale * (xa @ mc))
    z2 = float(np.var(pair_k, ddof=1))
    e_sq = float(prod.mean())

    # Tr(rho^2)^2 from (A,B) against (C,D): four disjoint blocks, so the factors are independent.
    p2_ab = scale * float(ma @ mb)
    p2_cd = scale * float(mc @ md)
    return e_sq - p2_ab * p2_cd, z2


def _worker(task):
    ens, n, s = task
    state = make_state(ens, n, s)
    weights = pauli_weights(n)
    m = pauli_expectations(state.density_matrix(), n)
    z1_ex, z2_ex = exact_zeta1(m, n), exact_zeta2(m, n, weights)
    ms_ex = z2_ex / (2 * z1_ex)
    rng = np.random.default_rng([SEED, 909, _ENS_ID[ens], n, s])
    out = {}
    for M in PILOTS:
        z1s, z2s, mss = [], [], []
        for _ in range(N_REPS):
            snaps = sample_batched_general(state, M, rng)
            z1h, z2h = pilot_zetas(snaps)
            z1s.append(z1h)
            z2s.append(z2h)
            mss.append(z2h / (2 * z1h) if z1h > 0 else np.nan)
        z1s, z2s, mss = np.array(z1s), np.array(z2s), np.array(mss)
        finite = np.isfinite(mss)
        out[str(M)] = {
            "zeta1_exact": z1_ex, "zeta2_exact": z2_ex, "m_star_exact": ms_ex,
            "zeta1_hat_mean": float(z1s.mean()), "zeta1_hat_std": float(z1s.std(ddof=1)),
            "zeta1_rel_bias": float((z1s.mean() - z1_ex) / z1_ex),
            "zeta1_rel_rmse": float(np.sqrt(np.mean((z1s - z1_ex) ** 2)) / z1_ex),
            "zeta2_hat_mean": float(z2s.mean()), "zeta2_hat_std": float(z2s.std(ddof=1)),
            "zeta2_rel_bias": float((z2s.mean() - z2_ex) / z2_ex),
            "zeta2_rel_rmse": float(np.sqrt(np.mean((z2s - z2_ex) ** 2)) / z2_ex),
            "n_negative_zeta1": int((z1s <= 0).sum()),
            "m_star_median": float(np.median(mss[finite])) if finite.any() else None,
            "m_star_rel_rmse": (float(np.sqrt(np.mean((mss[finite] - ms_ex) ** 2)) / ms_ex)
                                if finite.any() else None),
            "m_star_rel_mad": (float(np.median(np.abs(mss[finite] - ms_ex)) / ms_ex)
                               if finite.any() else None),
            "m_star_p10_over_exact": (float(np.percentile(mss[finite], 10) / ms_ex)
                                      if finite.any() else None),
            "m_star_p90_over_exact": (float(np.percentile(mss[finite], 90) / ms_ex)
                                      if finite.any() else None),
            "n_reps": N_REPS,
        }
    return task, out


def main() -> None:
    t0 = time.time()
    grid = [(e, n, s) for e in ENSEMBLES for n in SIZES for s in range(N_STATES)]
    print(f"pilot-estimator grid: {len(grid)} units x {len(PILOTS)} budgets x {N_REPS} reps")
    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for task, res in pool.map(_worker, grid):
            results["|".join(map(str, task))] = res
            print(f"  done {task}", flush=True)

    print("\n47.3(b) relative error of the pilot-estimated M* against the exact value")
    print(f"  {'ens':16s} {'n':>2s} " + " ".join(f"{M:>10d}" for M in PILOTS))
    summary = {}
    for e in ENSEMBLES:
        for n in SIZES:
            keys = [f"{e}|{n}|{s}" for s in range(N_STATES)]
            row_mad, row_z1, row_z2 = [], [], []
            for M in PILOTS:
                mads = [results[k][str(M)]["m_star_rel_mad"] for k in keys
                        if results[k][str(M)]["m_star_rel_mad"] is not None]
                row_mad.append(float(np.median(mads)) if mads else None)
                row_z1.append(float(np.median([results[k][str(M)]["zeta1_rel_rmse"] for k in keys])))
                row_z2.append(float(np.median([results[k][str(M)]["zeta2_rel_rmse"] for k in keys])))
            # rel error ~ C M^{-p}: fit p and the budget needed for 10% relative accuracy
            good = [(M, v) for M, v in zip(PILOTS, row_mad) if v is not None and v < 0.6]
            slope = intercept = budget10 = None
            if len(good) >= 2:
                lx = np.log([g[0] for g in good]); ly = np.log([g[1] for g in good])
                slope = float(np.polyfit(lx, ly, 1)[0])
                intercept = float(np.polyfit(lx, ly, 1)[1])
                budget10 = float(np.exp((np.log(0.10) - intercept) / slope))
            m_star_ex = float(np.median([results[k][str(PILOTS[0])]["m_star_exact"] for k in keys]))
            summary[f"{e}|n{n}"] = {
                "m_star_rel_mad": row_mad, "zeta1_rel_rmse": row_z1, "zeta2_rel_rmse": row_z2,
                "fitted_convergence_exponent": slope,
                "pilot_budget_for_10pct": budget10,
                "m_star_exact_median": m_star_ex,
                "pilot_over_m_star": (budget10 / m_star_ex) if budget10 else None,
            }
            cells = " ".join(f"{v*100:9.1f}%" if v is not None else "        --" for v in row_mad)
            extra = (f"  | M^{slope:+.2f}   10%-budget {budget10:>9.0f}   M*={m_star_ex:>9.0f}   "
                     f"ratio {budget10/m_star_ex:6.2f}" if budget10 else "")
            print(f"  {e:16s} {n:>2d} {cells}{extra}")
    print("\n  zeta1 relative RMSE (the hard factor):")
    for key, v in summary.items():
        print(f"  {key:24s} " + " ".join(f"{x*100:9.1f}%" for x in v["zeta1_rel_rmse"]))
    print("\n  zeta2 relative RMSE (the easy factor):")
    for key, v in summary.items():
        print(f"  {key:24s} " + " ".join(f"{x*100:9.1f}%" for x in v["zeta2_rel_rmse"]))

    payload = {
        "description": "PASS 47.3(b): pilot-budget estimation of M*(rho) vs the exact value",
        "config": {"seed": SEED, "sizes": list(SIZES), "pilots": list(PILOTS),
                   "n_states": N_STATES, "n_reps": N_REPS, "ensembles": list(ENSEMBLES)},
        "estimator": {
            "zeta2": "sample variance of Tr(G_i G_j) over M/4 disjoint pairs (unbiased)",
            "E_Tr_G_rho_sq": "mean_{i in A} Kbar_iB * Kbar_iC over disjoint blocks (unbiased)",
            "purity_squared": "Kbar_AB * Kbar_CD over four disjoint blocks (unbiased)",
            "zeta1": "difference of the previous two",
        },
        "summary": summary,
        "per_unit": results,
        "wall_seconds": time.time() - t0,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
