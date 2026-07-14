"""Register-fault prediction: SWAP purity for three explicit-layout cells from a
same-session readout calibration. Reuses the validated same-session readout parsing
and confusion model; adds explicit-layout prediction (the gate-noise distribution is
layout-independent for zero-routing GHZ SWAP, so only the readout confusion differs
per cell)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware import avg_gate_error_to_depol_param, swap_sign
from anrl.hardware.grid_predict import swap_gate_noisy_probs
from anrl.hardware.state_prep import ghz_state
from experiments.same_session_lib import (
    CZ_AVGS, P1, PHYS_8, STATES, build_confusion, build_tables, wilson,
)

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"

# clbit c -> physical qubit, in clbit order (copy A first, then copy B)
CELLS = {
    "n3_std": {"n": 3, "phys": [0, 1, 2, 9, 10, 11]},
    "n4":     {"n": 4, "phys": [0, 1, 2, 3, 9, 10, 11, 12]},
    "n3_alt": {"n": 3, "phys": [1, 2, 3, 10, 11, 12]},  # localization: parity pairs (1,10)(2,11)(3,12)
}


def parse_readout_rf(block: str) -> dict:
    """Per-qubit readout rates from register-fault block counts rf_<block>_w*_counts.json."""
    counts = {name: {k: int(v) for k, v in json.loads((HW / f"rf_{block}_{name}_counts.json").read_text()).items()}
              for name in STATES}
    out = {}
    for c in range(8):
        n0 = f10 = n1 = f01 = 0
        byw = {}
        for name, excited in STATES.items():
            W = len(excited); prep = 1 if c in excited else 0; w = W - prep
            for s, cnt in counts[name].items():
                m = int(s.replace(" ", "")[7 - c])
                if prep == 0:
                    n0 += cnt; f10 += cnt if m == 1 else 0
                    dd = byw.setdefault(w, [0, 0]); dd[0] += cnt; dd[1] += cnt if m == 1 else 0
                else:
                    n1 += cnt; f01 += cnt if m == 0 else 0
        out[PHYS_8[c]] = {"p10": f10 / n0, "p10_ci": wilson(f10, n0), "n0": n0,
                          "p01": f01 / n1, "p01_ci": wilson(f01, n1), "n1": n1,
                          "p10_by_w": {w: {"p10": d[1] / d[0], "ci": wilson(d[1], d[0]), "n": d[0]}
                                       for w, d in sorted(byw.items())}}
    return out


def _gate_dist(n: int, avg: float) -> np.ndarray:
    """Ideal-readout SWAP outcome distribution under CZ depolarizing noise (layout-independent)."""
    q, _ = swap_gate_noisy_probs(ghz_state(n), avg_gate_error_to_depol_param(avg, 2), P1)
    return q


def predict_cell(n: int, phys: list[int], tables: dict) -> dict:
    """Predicted collective SWAP purity band for an explicit physical layout (clbit order)."""
    signs = np.array([swap_sign(format(b, f"0{2 * n}b"), n) for b in range(2 ** (2 * n))], dtype=float)
    band = {}
    for tag, avg in CZ_AVGS.items():
        q = _gate_dist(n, avg)
        band[tag] = round(float(signs @ (build_confusion(phys, tables) @ q)), 4)
    qm = _gate_dist(n, 0.009)
    gate_only = float(signs @ qm)
    meas = float(signs @ (build_confusion(phys, tables) @ qm))
    return {"n": n, "phys": phys, "band": band,
            "gate_penalty": round(1 - gate_only, 4), "readout_penalty": round(gate_only - meas, 4)}


def predict_all(tables: dict) -> dict:
    return {name: predict_cell(c["n"], c["phys"], tables) for name, c in CELLS.items()}
