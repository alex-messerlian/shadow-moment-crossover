"""Analyze the n=3/n=4 readout characterization and re-lock the grid cells (no credits).

Reads results/hardware/ro_*_counts.json (7 basis states, 8 physical qubits
{0,1,2,3,9,10,11,12} in clbit order).  Produces per-qubit P(1|0)/P(0|1) with Wilson
CIs, the measured excitation-weight correlation (P(1|0) vs number of excited others),
compares it to the linear w-extrapolation used in the locked grid, re-locks the n=3/n=4
GHZ+Haar cells with the measured readout, and checks the n=2 Bell reproduction.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from anrl.hardware import avg_gate_error_to_depol_param, bell_state, swap_sign
from anrl.hardware.calibration import gate_noisy_probs
from anrl.hardware.grid_predict import swap_gate_noisy_probs, swap_signs
from anrl.hardware.readout_model import MEASURED_READOUT, correlated_confusion
from anrl.hardware.state_prep import ghz_state, haar_pure

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
PHYS = [0, 1, 2, 3, 9, 10, 11, 12]          # clbit c -> physical PHYS[c]
STATES = {"w0": [], "w2": [0, 1], "w4a": [0, 1, 2, 3], "w4b": [4, 5, 6, 7],
          "w6a": [0, 1, 2, 3, 4, 5], "w6b": [2, 3, 4, 5, 6, 7], "w8": [0, 1, 2, 3, 4, 5, 6, 7]}
P1 = 0.001
CZ_MID = avg_gate_error_to_depol_param(0.009, 2)
BELL_MEASURED = 0.7184


def _mbit(s: str, c: int) -> int:
    return int(s[7 - c])  # s[0]=clbit7 ... s[7]=clbit0


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_counts():
    return {name: {k: int(v) for k, v in json.loads((HW / f"ro_{name}_counts.json").read_text()).items()}
            for name in STATES}


def per_qubit_rates(counts):
    """Aggregate P(1|0), P(0|1) and per-w P(1|0) for each physical qubit."""
    out = {}
    for c in range(8):
        n0 = f10 = n1 = f01 = 0
        byw = {}  # w -> [n0, f10]
        for name, excited in STATES.items():
            cts = counts[name]; W = len(excited); prep = 1 if c in excited else 0
            w = W - prep  # number of OTHER excited qubits
            for s, cnt in cts.items():
                m = _mbit(s, c)
                if prep == 0:
                    n0 += cnt; f10 += cnt if m == 1 else 0
                    d = byw.setdefault(w, [0, 0]); d[0] += cnt; d[1] += cnt if m == 1 else 0
                else:
                    n1 += cnt; f01 += cnt if m == 0 else 0
        out[PHYS[c]] = {
            "p10": f10 / n0, "p10_ci": _wilson(f10, n0), "n0": n0,
            "p01": f01 / n1, "p01_ci": _wilson(f01, n1), "n1": n1,
            "p10_by_w": {w: {"p10": d[1] / d[0], "ci": _wilson(d[1], d[0]), "n": d[0]}
                         for w, d in sorted(byw.items())},
        }
    return out


def p10_table(q: int, byw: dict) -> dict:
    """Measured P(1|0) vs w lookup for a qubit: this-run points + device-char w0/w2.

    The correlation is nonlinear ($0 rises steeply 0->2 then saturates), so we keep the
    measured points and interpolate, rather than fitting a single slope.  Device-char
    supplies the w=2 point for {0,1,9,10} (this run's states did not always sample it),
    which is what preserves the n=2 Bell reproduction.
    """
    tbl = {}
    if q in MEASURED_READOUT:
        idle, exc, _ = MEASURED_READOUT[q]
        tbl[0] = idle; tbl[2] = exc            # device-char w0, w2 (validated by the Bell run) — kept
        for w, d in byw.items():
            if w not in (0, 2):                # add only the NEW high-w points (saturation)
                tbl[w] = d["p10"]
    else:
        for w, d in byw.items():               # new qubits: use this run's measurement fully
            tbl[w] = d["p10"]
    return tbl


def _p10_at(tbl: dict, w: int) -> float:
    """Piecewise-linear interpolation of the measured table, clamped past the endpoints."""
    ws = sorted(tbl)
    if w <= ws[0]:
        return tbl[ws[0]]
    if w >= ws[-1]:
        return tbl[ws[-1]]                        # saturate at the highest measured w
    for i in range(len(ws) - 1):
        if ws[i] <= w <= ws[i + 1]:
            f = (w - ws[i]) / (ws[i + 1] - ws[i])
            return tbl[ws[i]] * (1 - f) + tbl[ws[i + 1]] * f
    return tbl[ws[-1]]


def build_confusion_v2(phys_qubits, model):
    """Joint confusion using the MEASURED saturating P(1|0)(w) table per qubit.

    ``model[q] = {'tbl': {w: p10}, 'p01': p01}``.  Qubits not measured fall back to mean
    rates with no correlation (grid assumption) — flagged where it applies (Haar n=4).
    """
    from anrl.hardware.readout_model import _MEAN_P01, _MEAN_P10

    m = len(phys_qubits)
    dim = 2 ** m
    fit = [model.get(q, {"tbl": {0: _MEAN_P10}, "p01": _MEAN_P01}) for q in phys_qubits]
    R = np.zeros((dim, dim))
    for t in range(dim):
        pop = bin(t).count("1")
        for mo in range(dim):
            prob = 1.0
            for c in range(m):
                t_c = (t >> c) & 1; m_c = (mo >> c) & 1
                if t_c == 0:
                    p1_0 = min(max(_p10_at(fit[c]["tbl"], pop - t_c), 0.0), 1.0)
                    prob *= (1 - p1_0) if m_c == 0 else p1_0
                else:
                    prob *= fit[c]["p01"] if m_c == 0 else (1 - fit[c]["p01"])
            R[mo, t] = prob
    return R


def main() -> None:
    counts = load_counts()
    rates = per_qubit_rates(counts)

    # --- assemble v2 model: measured saturating P(1|0)(w) table + measured p01 ---
    v2 = {q: {"tbl": p10_table(q, rates[q]["p10_by_w"]), "p01": rates[q]["p01"]} for q in PHYS}

    # --- how far off was the grid's LINEAR w-extrapolation? (measured vs v1 at high w) ---
    from anrl.hardware.readout_model import _MEAN_P10

    def grid_p10(q, w):
        """v1 grid model P(1|0)(w): linear for {0,1,9,10}, mean-constant for others."""
        if q in MEASURED_READOUT:
            idle, exc, _ = MEASURED_READOUT[q]
            return idle + 0.5 * (exc - idle) * w
        return _MEAN_P10
    extrap_error = {}
    for q in PHYS:
        rows = []
        for w, d in rates[q]["p10_by_w"].items():
            rows.append({"w": w, "measured": round(d["p10"], 4), "grid_linear": round(grid_p10(q, w), 4),
                         "error": round(grid_p10(q, w) - d["p10"], 4)})
        extrap_error[q] = rows

    # --- re-lock n=3,4 GHZ + Haar with v2 readout ---
    def relock(prep):
        n = prep.n
        q, phys_order = swap_gate_noisy_probs(prep, CZ_MID, P1)
        signs = swap_signs(n)
        R2 = build_confusion_v2(phys_order, v2)
        return float(signs @ (R2 @ q)), phys_order

    v1 = json.loads((HW / "locked_grid_predictions.json").read_text())
    v1_swap = {(c["n"], c["state"]): c["swap"]["purity_mid"] for c in v1["grid"]}
    relocked = []
    for n in (3, 4):
        for name, prep in (("ghz", ghz_state(n)), ("haar", haar_pure(n, 0))):
            new, phys = relock(prep)
            old = v1_swap[(n, name)]
            relocked.append({"n": n, "state": name, "phys": phys,
                             "old_purity": old, "new_purity": round(new, 4),
                             "delta": round(new - old, 4)})

    # --- n=2 regression check (Bell) under the updated {0,1,9,10} rates ---
    signs2 = np.array([swap_sign(format(b, "04b"), 2) for b in range(16)])
    qb = gate_noisy_probs(bell_state(), CZ_MID, P1)
    bell_v1 = float(signs2 @ (correlated_confusion([0, 1, 9, 10], True) @ qb))
    bell_v2 = float(signs2 @ (build_confusion_v2([0, 1, 9, 10], v2) @ qb))

    # --- build the v2 grid: copy v1, re-lock the n=3,4 SWAP cells with measured readout ---
    cz_band = {"lo": avg_gate_error_to_depol_param(0.005, 2), "mid": CZ_MID,
               "hi": avg_gate_error_to_depol_param(0.015, 2)}
    v2_grid = json.loads(json.dumps(v1))  # deep copy
    v2_grid["timestamp"] = datetime.now().isoformat(timespec="seconds")
    v2_grid["derived_from"] = "locked_grid_predictions.json (v1)"
    v2_grid["readout_update"] = "measured 8-qubit correlated readout (readout-extension phase); " \
        "n=3,4 SWAP cells re-locked; n=2 kept (device-char, Bell-validated)"
    for cell in v2_grid["grid"]:
        if cell["n"] in (3, 4):
            prep = ghz_state(cell["n"]) if cell["state"] == "ghz" else haar_pure(cell["n"], 0)
            band = {}
            for k, p2 in cz_band.items():
                q, phys = swap_gate_noisy_probs(prep, p2, P1)
                R2 = build_confusion_v2(phys, v2)
                band[k] = round(float(swap_signs(cell["n"]) @ (R2 @ q)), 4)
            q_mid, phys = swap_gate_noisy_probs(prep, CZ_MID, P1)
            R2 = build_confusion_v2(phys, v2)
            gate_only = float(swap_signs(cell["n"]) @ q_mid)
            meas = float(swap_signs(cell["n"]) @ (R2 @ q_mid))
            cell["swap"]["purity_band_v1"] = cell["swap"]["purity_band"]
            cell["swap"]["purity_mid_v1"] = cell["swap"]["purity_mid"]
            cell["swap"]["purity_band"] = band
            cell["swap"]["purity_mid"] = round(meas, 4)
            cell["swap"]["readout_penalty"] = round(gate_only - meas, 4)
            cell["swap"]["se_10k"] = round(float(np.sqrt(max(0, 1 - meas * meas) / 10000)), 4)
            cell["swap"]["delta_vs_v1"] = round(meas - cell["swap"]["purity_mid_v1"], 4)
    (HW / "locked_grid_predictions_v2.json").write_text(json.dumps(v2_grid, indent=2))

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "physical_qubits": {"n3": [0, 1, 2, 9, 10, 11], "n4": PHYS, "nests": True},
        "new_qubits_first_characterized": [2, 3, 11, 12],
        "per_qubit": {str(q): {"p10": round(rates[q]["p10"], 4), "p10_ci": [round(x, 4) for x in rates[q]["p10_ci"]],
                               "p01": round(rates[q]["p01"], 4), "p01_ci": [round(x, 4) for x in rates[q]["p01_ci"]],
                               "p10_by_w": {str(w): round(d["p10"], 4) for w, d in rates[q]["p10_by_w"].items()},
                               "grid_linear_vs_measured": extrap_error[q]}
                      for q in PHYS},
        "relocked_cells": relocked,
        "n2_bell_check": {"measured": BELL_MEASURED, "v1_model": round(bell_v1, 4),
                          "v2_model": round(bell_v2, 4), "moved": round(bell_v2 - bell_v1, 4)},
    }
    (HW / "readout_extension_analysis.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    main()
