"""PASS 36: the crossover grid under RAW, CLIPPED and SHRINKAGE single-copy estimators.

The single-copy U-statistic is unbiased, so its standard deviation equals its RMSE,
and the committed sweeps record both that RMSE and the mean true moment for every
cell.  For an estimator that is Gaussian to good approximation -- verified in
``results/clipping_correction.json`` (kurtosis ~3.1 at M >= 2000) and re-verified
directly against Monte Carlo by ``run_clipping_audit.py`` -- the RMSE after a
pointwise projection is a deterministic function of ``(mu, sigma)``.  The whole grid
can therefore be re-scored under each estimator from the committed measurements,
without re-running the sweep.

The projected RMSEs are obtained by Gauss-Legendre quadrature over ``N(mu, sigma^2)``.
The collective side is untouched.

Crossover rule, identical for all three estimators: the SUSTAINED crossover of
:func:`anrl.theory.crossover.predict_crossover` -- the smallest ``n`` from which
single-copy RMSE exceeds collective RMSE and stays above for the rest of the range.

VALIDATION GATE: re-scoring the committed sweeps under RAW must reproduce the
committed ``crossover_n``.  If it does not, the re-scoring is wrong and the clipped
numbers cannot be trusted; the script reports the agreement.

Writes ``results/pass36_clipping_grid.json``.
Run:  PYTHONPATH=. python -m experiments.run_clipping_crossover_grid
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from anrl.benchmark.constrained import clip_moment, shrink_moment

R = Path(__file__).resolve().parent.parent / "results"

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(4001)
_SPAN = 12.0  # integrate mu +- 12 sigma


def _projected_rmse(mu: float, sigma: float, n: int, k: int, kind: str) -> float:
    """RMSE about ``mu`` of a projected N(mu, sigma^2) estimate, by quadrature."""
    if sigma <= 0:
        return 0.0
    z = _NODES * _SPAN
    x = mu + sigma * z
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    if kind == "raw":
        y = x
    elif kind == "clipped":
        y = clip_moment(x, n, k)
    elif kind == "shrunk":
        y = shrink_moment(x, n, k, sigma)
    else:
        raise ValueError(kind)
    mse = float(np.sum(_WEIGHTS * _SPAN * pdf * (y - mu) ** 2))
    return float(np.sqrt(max(0.0, mse)))


def _sustained_crossover(sizes, single, collective):
    ns = sorted(sizes)
    wins = {n: single[n] > collective[n] for n in ns}
    for i, n in enumerate(ns):
        if wins[n] and all(wins[m] for m in ns[i:]):
            return n
    return None


def _cells_from(path: Path, default_budget: int, ensemble: str) -> dict:
    payload = json.loads(path.read_text())
    cells: dict[tuple, dict] = {}
    for r in payload["rows"]:
        key = (ensemble, r["k"], r["noise_model"], float(r["rate"]),
               int(r.get("budget", default_budget)))
        c = cells.setdefault(key, {"sizes": [], "single": {}, "collective": {}, "mu": {}})
        n = int(r["n"])
        c["sizes"].append(n)
        c["single"][n] = float(r["single_rmse"])
        c["collective"][n] = float(r["collective_rmse"])
        c["mu"][n] = float(r.get("mean_true_moment", r.get("mean_true_purity")))
    committed = {}
    for e in payload.get("crossover_table", []):
        key = (ensemble, int(e["k"]), e["noise_model"], float(e["rate"]),
               int(e.get("budget", default_budget)))
        committed[key] = e["crossover_n"]
    return {"cells": cells, "committed": committed}


def main() -> None:
    sources = [
        (R / "budget_scaling.json", 2000, "noisy_pure"),
        (R / "moment_sweep_corrected.json", 2000, "noisy_pure"),
    ]
    rows = []
    for path, bud, ens in sources:
        got = _cells_from(path, bud, ens)
        tag = path.stem
        for key, c in sorted(got["cells"].items(), key=lambda kv: str(kv[0])):
            _, k, nm, rate, budget = key
            sizes = sorted(set(c["sizes"]))
            out = {"source": tag, "ensemble": ens, "k": k, "noise_model": nm,
                   "rate": rate, "budget": budget, "sizes": sizes}
            for kind in ("raw", "clipped", "shrunk"):
                single = {n: (c["single"][n] if kind == "raw"
                              else _projected_rmse(c["mu"][n], c["single"][n], n, k, kind))
                          for n in sizes}
                out[f"single_{kind}"] = {str(n): single[n] for n in sizes}
                out[f"crossover_{kind}"] = _sustained_crossover(sizes, single, c["collective"])
            out["collective"] = {str(n): c["collective"][n] for n in sizes}
            out["committed_crossover"] = got["committed"].get(key)
            rows.append(out)

    checked = [r for r in rows if r["committed_crossover"] is not None
               or r["crossover_raw"] is not None]
    agree = sum(1 for r in checked if r["crossover_raw"] == r["committed_crossover"])
    print(f"VALIDATION: RAW re-scoring reproduces the committed crossover in "
          f"{agree}/{len(checked)} cells")
    for r in checked:
        if r["crossover_raw"] != r["committed_crossover"]:
            print(f"   MISMATCH k={r['k']} {r['noise_model']}@{r['rate']} M={r['budget']} "
                  f"[{r['source']}]: committed {r['committed_crossover']} "
                  f"vs re-scored {r['crossover_raw']}")

    print(f"\nCROSSOVER COUNTS over {len(rows)} cells")
    for kind in ("raw", "clipped", "shrunk"):
        res = sum(1 for r in rows if r[f"crossover_{kind}"] is not None)
        print(f"  {kind:9s}: {res}/{len(rows)} resolve a crossover")

    for kind in ("clipped", "shrunk"):
        sh = [r[f"crossover_{kind}"] - r["crossover_raw"] for r in rows
              if r["crossover_raw"] is not None and r[f"crossover_{kind}"] is not None]
        lost = sum(1 for r in rows
                   if r["crossover_raw"] is not None and r[f"crossover_{kind}"] is None)
        gained = sum(1 for r in rows
                     if r["crossover_raw"] is None and r[f"crossover_{kind}"] is not None)
        a = np.array(sh) if sh else np.array([0])
        print(f"\nSHIFT raw -> {kind}: {len(sh)} cells resolve under both; "
              f"mean {a.mean():+.3f}, median {np.median(a):+.1f}, "
              f"min {a.min():+d}, max {a.max():+d}")
        print(f"  distribution: {dict(sorted(collections.Counter(a.tolist()).items()))}")
        print(f"  crossover LOST: {lost}   GAINED: {gained}")

    (R / "pass36_clipping_grid.json").write_text(json.dumps({
        "description": "PASS 36: crossover grid re-scored under RAW / CLIPPED / SHRINKAGE "
                       "single-copy estimators; collective side unchanged",
        "method": "projected RMSE by Gauss-Legendre quadrature over N(mu, sigma^2), "
                  "sigma = committed measured single_rmse (estimator unbiased), "
                  "mu = mean true moment; sustained-crossover rule",
        "validation_raw_reproduces_committed": f"{agree}/{len(checked)}",
        "rows": rows,
    }, indent=1) + "\n")
    print(f"\nwrote {R / 'pass36_clipping_grid.json'}")


if __name__ == "__main__":
    main()
