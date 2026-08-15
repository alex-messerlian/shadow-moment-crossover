"""PASS 59.1: the weight-resolved readout series, extracted from raw counts and fitted.

    PYTHONPATH=. .venv/bin/python experiments/pass59_saturation.py

The committed analysis reports P(1|0) at a few weights.  This rebuilds the whole series from
``results/hardware/ro_w*_counts.json`` so the saturation claim can be tested rather than quoted,
and fits it against two models:

  linear      p(w) = p0 + a*w          -- what additive/independent crosstalk predicts
  saturating  p(w) = p0 + A*(1 - e^{-w/w0})

A shared feedline carries one multiplexed readout tone per resonator.  Excited neighbours shift
and broaden the spectator's resonance, but the demodulation window is finite: once the neighbour
population is enough to push the spectator's response outside the discrimination threshold, adding
further excitations cannot move it further.  That bounds the effect, so the mechanism predicts
saturation with an amplitude set by the discrimination margin, not unbounded growth.

Writes ``results/pass59_saturation.json``.
"""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
HW = REPO / "results" / "hardware"
OUT = REPO / "results" / "pass59_saturation.json"

BITS = [0, 1, 2, 3, 9, 10, 11, 12]          # c[0]..c[7] -> physical qubit
CELLS = {"w0": [], "w2": [0, 1], "w4a": [0, 1, 2, 3], "w4b": [9, 10, 11, 12],
         "w6a": [0, 1, 2, 3, 9, 10], "w6b": [2, 3, 9, 10, 11, 12],
         "w8": [0, 1, 2, 3, 9, 10, 11, 12]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def bit_order(counts: dict, excited: list[int]) -> bool:
    """True if c[0] is the LEFTMOST character.  Decided from the cell's own dominant outcome."""
    top = max(counts, key=counts.get)
    want = "".join("1" if q in excited else "0" for q in BITS)
    return top == want


def main() -> None:
    cells = {}
    for name, exc in CELLS.items():
        f = HW / f"ro_{name}_counts.json"
        if not f.exists():
            continue
        cells[name] = (json.loads(f.read_text()), exc)

    # settle the bit order on the highest-weight cell, where the signal is unambiguous
    ref, refexc = cells["w4a"]
    left0 = bit_order(ref, refexc)
    print(f"  bit order: c[0] is {'LEFTMOST' if left0 else 'RIGHTMOST'} "
          f"(decided on w4a, dominant outcome {max(ref, key=ref.get)})")

    def idx(q: int) -> int:
        i = BITS.index(q)
        return i if left0 else len(BITS) - 1 - i

    rows = []
    print(f"\n  {'cell':5s} {'w':>2s} {'spectator':>10s} {'shots':>6s} {'P(1|0)':>8s}  95% Wilson CI")
    for name, (counts, exc) in cells.items():
        n = sum(counts.values())
        spectators = [q for q in BITS if q not in exc]
        for q in spectators:
            j = idx(q)
            k = sum(v for b, v in counts.items() if b[j] == "1")
            p, lo, hi = wilson(k, n)
            rows.append({"cell": name, "weight": len(exc), "spectator": q,
                         "shots": n, "p10": p, "ci": [lo, hi]})
            print(f"  {name:5s} {len(exc):>2d} {('$'+str(q)):>10s} {n:>6d} {p:>8.4f}  [{lo:.4f}, {hi:.4f}]")

    # pooled per weight
    print(f"\n  {'weight':>6s} {'cells':>5s} {'spectators':>10s} {'pooled P(1|0)':>14s}  95% CI")
    pooled = []
    for w in sorted({r["weight"] for r in rows}):
        sel = [r for r in rows if r["weight"] == w]
        k = sum(r["p10"] * r["shots"] for r in sel)
        n = sum(r["shots"] for r in sel)
        p, lo, hi = wilson(int(round(k)), n)
        pooled.append({"weight": w, "p10": p, "ci": [lo, hi],
                       "n_spectators": len(sel), "shots": n})
        print(f"  {w:>6d} {len({r['cell'] for r in sel}):>5d} {len(sel):>10d} {p:>14.4f}  [{lo:.4f}, {hi:.4f}]")

    ws = np.array([q["weight"] for q in pooled], float)
    ps = np.array([q["p10"] for q in pooled], float)
    se = np.array([(q["ci"][1] - q["ci"][0]) / 3.92 for q in pooled], float)

    fits = {}
    # linear
    A = np.vstack([np.ones_like(ws), ws]).T
    W = np.diag(1 / se**2)
    beta = np.linalg.solve(A.T @ W @ A, A.T @ W @ ps)
    lin = A @ beta
    fits["linear"] = {"p0": beta[0], "slope": beta[1],
                      "resid": (ps - lin).tolist(),
                      "chi2": float(((ps - lin) / se) ** 2 @ np.ones_like(ws)),
                      "dof": len(ws) - 2}
    # saturating, grid search on w0 then linear solve for p0 and A
    best = None
    for w0 in np.linspace(0.4, 12.0, 400):
        B = np.vstack([np.ones_like(ws), 1 - np.exp(-ws / w0)]).T
        try:
            b = np.linalg.solve(B.T @ W @ B, B.T @ W @ ps)
        except np.linalg.LinAlgError:
            continue
        pred = B @ b
        chi2 = float(((ps - pred) / se) ** 2 @ np.ones_like(ws))
        if best is None or chi2 < best[0]:
            best = (chi2, w0, b, pred)
    chi2, w0, b, pred = best
    fits["saturating"] = {"p0": b[0], "amplitude": b[1], "w_scale": w0,
                          "resid": (ps - pred).tolist(), "chi2": chi2,
                          "dof": len(ws) - 3}

    print(f"\n  LINEAR      p(w) = {fits['linear']['p0']:.4f} + {fits['linear']['slope']:.5f} w"
          f"   chi2 = {fits['linear']['chi2']:.1f} on {fits['linear']['dof']} dof")
    print(f"    residuals {[f'{r:+.4f}' for r in fits['linear']['resid']]}")
    print(f"  SATURATING  p(w) = {fits['saturating']['p0']:.4f} + "
          f"{fits['saturating']['amplitude']:.4f}(1 - e^-w/{w0:.2f})"
          f"   chi2 = {chi2:.1f} on {fits['saturating']['dof']} dof")
    print(f"    residuals {[f'{r:+.4f}' for r in fits['saturating']['resid']]}")

    OUT.write_text(json.dumps({"per_spectator": rows, "pooled": pooled, "fits": fits,
                               "bit_order_c0_leftmost": bool(left0)}, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
