"""The weight-only zeta1 ansatz fitted on the NARROW family only.

Section 3.4 of the paper makes two claims about the weight-only ansatz
``zeta1 = sum_P c_{|P|} <P>^2`` at n = 2:

* fitted across a *diverse* 19-state family it leaves a residual of up to
  12.3% -- computed by ``theory_single_copy_verify.py:partE`` and stored in
  ``results/theory_derivation.json`` as ``partE.weight_only_n2.max_resid``;
* fitted only within the *narrow* single-parameter family of Haar-pure states
  at a fixed depolarizing rate ``q = 0.1``, it appears to hold, because that
  family spans a low-dimensional invariant subspace.

The second number had no generating script.  This module supplies it, reusing
``partE``'s estimator and design matrix unchanged so the two residuals are
computed the same way and are directly comparable.

Because the residual on the narrow family is small, the Monte-Carlo noise floor
of the ``zeta1`` estimator is reported alongside it: a residual at or below that
floor means the ansatz is exact on this family to the precision we can measure,
not merely close.

Writes ``results/weight_only_narrow_family.json``.
Run:  PYTHONPATH=. python -m experiments.weight_only_narrow_family
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.benchmark.ensembles import noisy_pure
from experiments.theory_single_copy_verify import _weight_sums, _zeta1_general

R = Path(__file__).resolve().parent.parent / "results"

N_STATES = 12       # matches the noisy-pure count inside partE's diverse family
N_SAMPLES = 2_000_000
Q = 0.1


def _fit(states, n_samples, seed0):
    """partE's fit, verbatim in form: lstsq of zeta1 on the weight sums."""
    A, y = [], []
    for i, st in enumerate(states):
        y.append(_zeta1_general(st, n_samples, seed0 + i))
        A.append(_weight_sums(st.density_matrix(), 2))
    A, y = np.array(A), np.array(y)
    pred = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    resid = np.abs(pred - y) / y
    return float(resid.max()), float(np.sqrt((resid ** 2).mean())), y


def narrow_family():
    """Haar-pure states at the single fixed depolarizing rate q = 0.1."""
    return [noisy_pure(2, Q, np.random.default_rng([1, int(Q * 100), s]))
            for s in range(N_STATES)]


def mc_noise_floor(states, n_samples):
    """Relative spread of the zeta1 estimator itself, from independent replicas."""
    rel = []
    for i, st in enumerate(states[:4]):
        reps = [_zeta1_general(st, n_samples, 9000 + 100 * i + r) for r in range(3)]
        rel.append(float(np.std(reps) / np.mean(reps)))
    return float(np.mean(rel))


def main():
    states = narrow_family()
    max_resid, rms_resid, zetas = _fit(states, N_SAMPLES, 700)
    floor = mc_noise_floor(states, N_SAMPLES)
    out = {
        "description": "weight-only zeta1 ansatz fitted on the narrow q=0.1 "
                       "Haar-pure family at n=2 (Section 3.4)",
        "family": {"kind": "noisy_pure", "n": 2, "q": Q, "n_states": N_STATES,
                   "seeds": f"default_rng([1, {int(Q * 100)}, s]) for s in range({N_STATES})"},
        "n_samples_per_zeta1": N_SAMPLES,
        "max_relative_residual": round(max_resid, 5),
        "rms_relative_residual": round(rms_resid, 5),
        "mc_noise_floor_relative": round(floor, 5),
        "residual_at_or_below_mc_noise": bool(max_resid <= 2 * floor),
        "zeta1_values": [round(float(z), 6) for z in zetas],
        "contrast_diverse_family": "partE.weight_only_n2.max_resid in "
                                   "results/theory_derivation.json (0.123)",
    }
    (R / "weight_only_narrow_family.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"narrow family (q={Q}, {N_STATES} states): "
          f"max relative residual {max_resid:.4%}, rms {rms_resid:.4%}")
    print(f"zeta1 MC noise floor: {floor:.4%}  "
          f"(residual at or below 2x floor: {out['residual_at_or_below_mc_noise']})")
    print("contrast: 12.3% max residual on the diverse 19-state family")


if __name__ == "__main__":
    main()
