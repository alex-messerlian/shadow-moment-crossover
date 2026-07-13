"""Analyze the GHZ-ladder hardware experiment (post-processing; no credits).

Collective route (n=2,3,4): measured purity vs the locked v2 band, the readout-dominated
scaling test, and the readout-vs-gate decomposition.  Single-copy anchor (n=2): the
15-basis U-statistic vs its locked prediction (pipeline validation).  Single-copy n=3,4:
model prediction only (labeled).  Route comparison, copy-fair.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware import swap_sign
from anrl.hardware.shadows import snapshots_from_outcomes
from anrl.benchmark.shadows import full_purity_ustatistic

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"


def _load(name: str) -> dict:
    return {k: int(v) for k, v in json.loads((HW / f"hg_{name}_counts.json").read_text()).items()}


def swap_purity(counts: dict, n: int) -> float:
    shots = sum(counts.values())
    return float(sum(swap_sign(b.replace(" ", ""), n) * c for b, c in counts.items()) / shots)


def collective_analysis(locked):
    out = []
    for n in (2, 3, 4):
        counts = _load(f"coll_n{n}")
        mu = swap_purity(counts, n)
        mu_rev = swap_purity({k[::-1]: v for k, v in counts.items()}, n)  # endianness check
        signs = np.array([swap_sign(b, n) for b, c in counts.items() for _ in range(c)], dtype=float)
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(signs, size=len(signs), replace=True).mean() for _ in range(4000)])
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        band = locked["collective"][str(n)]["v2_swap_band"]
        inside = band["hi"] <= mu <= band["lo"]  # band is lo>=mid>=hi (lo CZ -> higher purity)
        dist = 0.0 if inside else min(abs(mu - band["lo"]), abs(mu - band["hi"]))
        out.append({"n": n, "shots": sum(counts.values()), "measured": round(mu, 4),
                    "measured_rev": round(mu_rev, 4), "ci95": [round(x, 4) for x in ci],
                    "band": band, "inside_band": inside, "dist_outside": round(dist, 4),
                    "gate_penalty": locked["collective"][str(n)]["v2_gate_penalty"],
                    "readout_penalty": locked["collective"][str(n)]["v2_readout_penalty"]})
    return out


def anchor_analysis(locked):
    a = locked["single_anchor_n2"]
    bases = np.array(a["bases"])
    allb, allo = [], []
    for i in range(a["n_bases"]):
        counts = _load(f"single_n2_b{i:02d}")
        for s, c in counts.items():
            s = s.replace(" ", "")
            bit = [int(s[1 - q]) for q in range(2)]  # clbit q = s[1-q]
            for _ in range(c):
                allb.append(bases[i]); allo.append(bit)
    snaps = snapshots_from_outcomes(np.array(allb), np.array(allo), 2)
    measured = full_purity_ustatistic(snaps)
    # bootstrap over snapshots (resample within the fixed 15 bases)
    m = snaps.shape[0]
    rng = np.random.default_rng(1)
    boot = []
    for _ in range(300):
        idx = rng.integers(0, m, size=m)
        boot.append(full_purity_ustatistic(snaps[idx]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    pred = a["locked_prediction"]["mid"]
    validated = pred["mean"] - 3 * pred["se"] <= measured <= pred["mean"] + 3 * pred["se"] or \
        ci[0] <= pred["mean"] <= ci[1]
    return {"measured_ustatistic": round(measured, 4), "ci95": [round(x, 4) for x in ci],
            "locked_prediction": pred, "n_snapshots": m,
            "validated": bool(validated),
            "deviation": round(measured - pred["mean"], 4)}


def main() -> None:
    locked = json.loads((HW / "hg_locked.json").read_text())
    coll = collective_analysis(locked)
    anchor = anchor_analysis(locked)

    print("=== COLLECTIVE route: measured vs locked v2 band ===")
    for c in coll:
        verdict = "INSIDE" if c["inside_band"] else f"OUTSIDE by {c['dist_outside']:.4f}"
        print(f"  n={c['n']}: measured {c['measured']:.4f} (CI {c['ci95']}, rev {c['measured_rev']:.4f}) "
              f"vs band {c['band']['hi']:.3f}-{c['band']['lo']:.3f}  -> {verdict}")
    print("\n=== SINGLE-COPY anchor n=2 (pipeline validation) ===")
    print(f"  measured U-statistic {anchor['measured_ustatistic']:.4f} (CI {anchor['ci95']}) "
          f"vs locked {anchor['locked_prediction']['mean']:.3f}+/-{anchor['locked_prediction']['se']:.3f} "
          f"-> {'VALIDATED' if anchor['validated'] else 'MISMATCH'} (dev {anchor['deviation']:+.4f})")

    result = {"collective": coll, "single_anchor_n2": anchor}
    (HW / "hardware_grid_analysis.json").write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
