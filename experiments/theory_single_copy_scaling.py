"""Claim 2: recompute the Hoeffding components zeta1, zeta2 for the single-copy purity
U-statistic (noisy-pure, q=0.1) at n=2..9 with convergence checks, and fit the scalings.

Independent chunked/streaming Monte Carlo (does not reuse the theory estimators); writes
results/theory_zetas_recomputed.json. Convergence: `conv` mode shows zeta1/zeta2 are
stable to ~1% by 1M snapshots (n=7 zeta2 moves 0.3% from 100k->3M).

Run:  PYTHONPATH=. python -m experiments.theory_single_copy_scaling conv   # convergence
      PYTHONPATH=. python -m experiments.theory_single_copy_scaling full   # scan + fit
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from anrl.benchmark.budget import sample_batched
from anrl.benchmark.ensembles import noisy_pure

Q = 0.1
OUT = Path(__file__).resolve().parent.parent / "results" / "theory_zetas_recomputed.json"


def _welford_merge(agg, x):
    cnt, mean, M2 = agg
    n2 = x.size
    if n2 == 0:
        return agg
    m2, v2 = x.mean(), x.var()
    d = m2 - mean
    tot = cnt + n2
    return (tot, mean + d * n2 / tot, M2 + v2 * n2 + d * d * cnt * n2 / tot)


def _var(agg):
    cnt, _, M2 = agg
    return M2 / (cnt - 1)


def _psi_G_psi(state, snaps):
    n, d = state.n, state.dim
    psi = state.components[:, 0].reshape([2] * n)
    amp = np.broadcast_to(psi, (snaps.shape[0],) + psi.shape).astype(np.complex128).copy()
    for q in range(n):
        amp = np.moveaxis(amp, q + 1, 1)
        amp = np.einsum("cxr,cr...->cx...", snaps[:, q], amp)
        amp = np.moveaxis(amp, 1, q + 1)
    amp = amp.reshape(snaps.shape[0], d)
    return (np.conj(psi.reshape(d))[None, :] * amp).sum(axis=1).real


def zeta1_zeta2_state(state, N, rng, chunk=200_000):
    """zeta1=Var[Tr(G rho)]=(1-q)^2 Var<psi|G|psi>+const; zeta2=Var[Tr(Gi Gj)] over indep pairs."""
    n, d, q = state.n, state.dim, state.q
    a1 = a2 = (0, 0.0, 0.0)
    done = 0
    while done < N:
        c = min(chunk, N - done)
        a1 = _welford_merge(a1, (1.0 - q) * _psi_G_psi(state, sample_batched(state, c, rng)) + q / d)
        sa, sb = sample_batched(state, c, rng), sample_batched(state, c, rng)
        kern = np.ones(c)
        for qb in range(n):
            kern *= np.einsum("kij,kji->k", sa[:, qb], sb[:, qb]).real
        a2 = _welford_merge(a2, kern)
        done += c
    return _var(a1), _var(a2)


def run(n_list, N_by_n, states_by_n, seed=0):
    out = {}
    for n in n_list:
        N, ns = N_by_n(n), states_by_n(n)
        z1s, z2s = zip(*[zeta1_zeta2_state(noisy_pure(n, Q, np.random.default_rng([seed, n, s])),
                                           N, np.random.default_rng([seed, n, s, 7])) for s in range(ns)])
        z1, z2 = float(np.mean(z1s)), float(np.mean(z2s))
        out[n] = {"N": N, "n_states": ns, "zeta1": z1, "zeta2": z2,
                  "zeta1_spread": float(np.std(z1s) / z1), "zeta2_spread": float(np.std(z2s) / z2),
                  "M_star_2z1": z2 / (2 * z1), "M_star_4z1": z2 / (4 * z1)}
        print(f"n={n} N={N} states={ns}: zeta1={z1:.4f} ({out[n]['zeta1_spread']:.1%}) "
              f"zeta2={z2:.4g} ({out[n]['zeta2_spread']:.1%}) M*(2z1)={out[n]['M_star_2z1']:.1f}", flush=True)
    return out


def fit_scaling(out, key):
    ns = np.array(sorted(out))
    y = np.log([out[n][key] for n in ns])
    A = np.vstack([ns, np.ones_like(ns)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    s2 = ((y - A @ coef) ** 2).sum() / max(1, len(ns) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    return {"base": float(np.exp(coef[0])), "base_se": float(np.exp(coef[0]) * np.sqrt(cov[0, 0])),
            "prefactor": float(np.exp(coef[1])), "prefactor_se": float(np.exp(coef[1]) * np.sqrt(cov[1, 1]))}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "conv":
        for n in (3, 5, 7):
            print(f"--- convergence n={n} ---")
            for N in (100_000, 300_000, 1_000_000, 3_000_000):
                z1, z2 = zeta1_zeta2_state(noisy_pure(n, Q, np.random.default_rng([0, n, 0])),
                                           N, np.random.default_rng([0, n, 0, 7]))
                print(f"   N={N:>9}: zeta1={z1:.4f} zeta2={z2:.5g} M*(2z1)={z2/(2*z1):.1f}")
        return
    out = run(range(2, 10), lambda n: 500_000 if n <= 6 else 1_000_000,
              lambda n: 4 if n <= 7 else (3 if n == 8 else 2))
    fits27 = {k: fit_scaling({n: out[n] for n in range(2, 8)}, k) for k in ("zeta1", "zeta2", "M_star_2z1")}
    fits29 = {k: fit_scaling(out, k) for k in ("zeta1", "zeta2", "M_star_2z1")}
    for lbl, f in (("n2-7", fits27), ("n2-9", fits29)):
        print(f"--- fits {lbl} ---")
        for k, v in f.items():
            print(f"  {k}: {v['prefactor']:.3f}±{v['prefactor_se']:.3f} * ({v['base']:.3f}±{v['base_se']:.3f})^n")
    OUT.write_text(json.dumps({"q": Q, "per_n": {str(k): v for k, v in out.items()},
                               "fits_n2_7": fits27, "fits_n2_9": fits29}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
