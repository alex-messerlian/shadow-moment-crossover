"""PASS 37.4: the criterion re-derived for a CLIPPED estimator.

Scoring a clipped measurement against a prediction built for the raw estimator is a
mismatch: it asks whether the paper's existing criterion still describes a different
experiment.  The question a revision actually needs is whether the criterion is
still accurate when applied consistently -- both sides clipped.

The predicted single-copy RMSE is the exact Hoeffding value, which for an unbiased
estimator is its standard deviation, so the clipped prediction is
``clipped_rmse(mu, sigma, 2^{n(1-k)}, 1)`` from :mod:`anrl.theory.clipping`.  The
collective side is unchanged.  The crossover rule is the sustained one of
``predict_crossover``, used identically for both.

Writes ``results/pass37_clipped_prediction.json``.
Run:  PYTHONPATH=. python -m experiments.run_clipped_prediction
"""

from __future__ import annotations

import json
from pathlib import Path

from anrl.theory.clipping import clipped_rmse
from anrl.theory.crossover import (
    noisy_pure_moment, predicted_collective_rmse, predicted_single_rmse,
)

R = Path(__file__).resolve().parent.parent / "results"
Q = 0.1


def _zetas() -> dict:
    raw = json.loads((R / "theory_zetas.json").read_text())["zetas"]
    return {(e["n"], e["k"]): e for e in raw}


def crossover(k, nm, g, budget, sizes, zetas, clipped: bool):
    ns = [n for n in sorted(sizes) if (n, k) in zetas]
    wins = {}
    for n in ns:
        s = predicted_single_rmse(n, k, budget, zetas, Q, "exact")
        if clipped:
            mu = noisy_pure_moment(n, k, Q)
            s = clipped_rmse(mu, s, 2.0 ** (n * (1 - k)), 1.0)
        wins[n] = s > predicted_collective_rmse(n, k, nm, g, budget, Q)
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None


def main() -> None:
    zet = _zetas()
    rule = json.loads((R / "pass37_rule_audit.json").read_text())["all_cells"]
    rows = []
    for c in rule:
        sizes = c["sizes"]
        rows.append({
            "source": c["source"], "k": c["k"], "noise_model": c["noise_model"],
            "rate": c["rate"], "budget": c["budget"],
            "predicted_raw": crossover(c["k"], c["noise_model"], c["rate"],
                                       c["budget"], sizes, zet, False),
            "predicted_clipped": crossover(c["k"], c["noise_model"], c["rate"],
                                           c["budget"], sizes, zet, True),
        })
    shifts = [r["predicted_clipped"] - r["predicted_raw"] for r in rows
              if r["predicted_raw"] is not None and r["predicted_clipped"] is not None]
    lost = sum(1 for r in rows if r["predicted_raw"] is not None
               and r["predicted_clipped"] is None)
    import collections
    print(f"PREDICTED crossover, raw vs clipped criterion, {len(rows)} noisy-pure cells")
    print(f"  resolve raw {sum(1 for r in rows if r['predicted_raw'] is not None)}, "
          f"clipped {sum(1 for r in rows if r['predicted_clipped'] is not None)}")
    print(f"  shift distribution: {dict(sorted(collections.Counter(shifts).items()))}")
    print(f"  prediction LOST under clipping: {lost}")
    (R / "pass37_clipped_prediction.json").write_text(json.dumps({
        "description": "PASS 37.4: predicted crossover with the criterion re-derived for "
                       "a clipped estimator (both sides clipped), noisy-pure cells",
        "rows": rows,
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass37_clipped_prediction.json'}")


if __name__ == "__main__":
    main()
