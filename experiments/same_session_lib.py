"""Same-session readout parsing + SWAP prediction (shared by locking and analysis).

Block A/C readout characterization = the validated 7 basis states on the 8 GHZ-ladder
qubits {0,1,2,3,9,10,11,12}.  Parses their counts into per-qubit P(1|0)(w) / P(0|1)
tables (saturating, from the measured points — no cross-session blending, unlike the
readout-extension phase), builds the joint correlated confusion, and predicts the
collective SWAP purity band for n=2,3,4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware import avg_gate_error_to_depol_param, swap_sign
from anrl.hardware.grid_predict import swap_gate_noisy_probs
from anrl.hardware.state_prep import ghz_state

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
PHYS_8 = [0, 1, 2, 3, 9, 10, 11, 12]                     # clbit c -> physical PHYS_8[c]
STATES = {"w0": [], "w2": [0, 1], "w4a": [0, 1, 2, 3], "w4b": [4, 5, 6, 7],
          "w6a": [0, 1, 2, 3, 4, 5], "w6b": [2, 3, 4, 5, 6, 7], "w8": [0, 1, 2, 3, 4, 5, 6, 7]}
P1 = 0.001
CZ_AVGS = {"lo": 0.005, "mid": 0.009, "hi": 0.015}


def _mbit(s: str, c: int) -> int:
    return int(s[7 - c])


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def parse_readout(block: str) -> dict:
    """Per-qubit readout rates from block's 7 readout-state counts (ss_<block>_w*_counts.json)."""
    counts = {name: {k: int(v) for k, v in json.loads((HW / f"ss_{block}_{name}_counts.json").read_text()).items()}
              for name in STATES}
    out = {}
    for c in range(8):
        n0 = f10 = n1 = f01 = 0
        byw = {}
        for name, excited in STATES.items():
            W = len(excited); prep = 1 if c in excited else 0; w = W - prep
            for s, cnt in counts[name].items():
                m = _mbit(s.replace(" ", ""), c)
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


def build_tables(rates: dict) -> dict:
    """Per-qubit {'tbl': {w: p10}, 'p01': p01} from the block's own measurements (fresh only)."""
    return {q: {"tbl": {w: d["p10"] for w, d in rates[q]["p10_by_w"].items()}, "p01": rates[q]["p01"]}
            for q in rates}


def _interp(tbl: dict, w: int) -> float:
    ws = sorted(tbl)
    if w <= ws[0]:
        return tbl[ws[0]]
    if w >= ws[-1]:
        return tbl[ws[-1]]
    for i in range(len(ws) - 1):
        if ws[i] <= w <= ws[i + 1]:
            f = (w - ws[i]) / (ws[i + 1] - ws[i])
            return tbl[ws[i]] * (1 - f) + tbl[ws[i + 1]] * f
    return tbl[ws[-1]]


def build_confusion(phys_qubits: list[int], tables: dict) -> np.ndarray:
    m = len(phys_qubits); dim = 2 ** m
    fit = [tables[q] for q in phys_qubits]
    R = np.zeros((dim, dim))
    for t in range(dim):
        pop = bin(t).count("1")
        for mo in range(dim):
            prob = 1.0
            for c in range(m):
                t_c = (t >> c) & 1; m_c = (mo >> c) & 1
                if t_c == 0:
                    p1_0 = min(max(_interp(fit[c]["tbl"], pop - t_c), 0.0), 1.0)
                    prob *= (1 - p1_0) if m_c == 0 else p1_0
                else:
                    prob *= fit[c]["p01"] if m_c == 0 else (1 - fit[c]["p01"])
            R[mo, t] = prob
    return R


def predict_swap(n: int, tables: dict) -> dict:
    prep = ghz_state(n)
    signs = np.array([swap_sign(format(b, f"0{2 * n}b"), n) for b in range(2 ** (2 * n))], dtype=float)
    band = {}
    for tag, avg in CZ_AVGS.items():
        q, phys = swap_gate_noisy_probs(prep, avg_gate_error_to_depol_param(avg, 2), P1)
        band[tag] = round(float(signs @ (build_confusion(phys, tables) @ q)), 4)
    q_mid, phys = swap_gate_noisy_probs(prep, avg_gate_error_to_depol_param(0.009, 2), P1)
    gate_only = float(signs @ q_mid)
    meas = float(signs @ (build_confusion(phys, tables) @ q_mid))
    return {"n": n, "phys": phys, "band": band, "gate_penalty": round(1 - gate_only, 4),
            "readout_penalty": round(gate_only - meas, 4)}


def swap_purity_from_counts(counts: dict, n: int) -> float:
    shots = sum(counts.values())
    return float(sum(swap_sign(b.replace(" ", ""), n) * c for b, c in counts.items()) / shots)
