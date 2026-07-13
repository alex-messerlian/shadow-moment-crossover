"""Locked, out-of-sample predictions for the Cepheus purity experiment (ZERO credits).

Runs entirely on local simulation.  Produces, and freezes to disk BEFORE any
hardware job:

* the device noise model and its reference rates (with the readout assumption
  stated and its sensitivity mapped);
* the calibration surface — measured Bell purity vs ``(p2, p_ro)`` — and its
  inversion;
* the locked measured-purity predictions for four states x two routes, from the
  full Qiskit noise simulation AND the analytic global-depolarizing bias law,
  side by side, with the discrepancy characterized;
* the recommended shot budget and its credit cost.

Writes ``results/cepheus_predictions.json`` (regenerable) and the committed,
version-controlled ``experiments/cepheus_locked_predictions.json`` +
``experiments/cepheus_prediction_report.md`` (the locked deliverable).

Run:  ``python -m experiments.hardware_prediction``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware import (
    REF_P1,
    REF_P2,
    REF_P_RO,
    bell_state,
    bell_calibration_surface,
    consistent_error_rates,
    depol_param_to_avg_gate_error,
    effective_g_from_purity,
    haar_pure,
    measured_swap_purity,
    measured_swap_purity_ensemble,
    mixed_ensemble,
    purity_from_g,
)
from anrl.hardware.shadow_noise import noisy_shadow_dists, predict_shadow_purity
from anrl.hardware.shot_budget import (
    CREDIT_BUDGET,
    MAX_SHOTS_PER_CIRCUIT,
    shots_to_credits,
    swap_shot_se,
    swap_shots_for_se,
)

REPO = Path(__file__).resolve().parent.parent
P2, P1, P_RO = REF_P2, REF_P1, REF_P_RO
SHADOW_M_REF = 2000        # reference budget for shadow SE (U-statistic is O(M^2) memory)
SHADOW_N_EXP = 80          # independent experiments -> SE estimate
SIGMA_TARGET = 5.0         # resolve the collective bias at this many sigma
RECOMMENDED_SHOTS = 20_000 # per configuration (see budget section)


def _swap_measured(state, p2, p1, p_ro):
    if hasattr(state, "components"):
        return measured_swap_purity_ensemble(state, p2, p1, p_ro)
    return measured_swap_purity(state, p2, p1, p_ro)


def build_states():
    """The four locked states: two pure (Bell, Haar) and two mixed (P~0.7, 0.5)."""
    return [
        ("bell", bell_state(), 1.0),
        ("haar", haar_pure(2, 0), 1.0),
        ("mixed_P0.7", mixed_ensemble(2, 0.70, seed=1), None),
        ("mixed_P0.5", mixed_ensemble(2, 0.50, seed=1), None),
    ]


def predict_all() -> dict:
    states = build_states()
    bell = bell_state()

    # Calibrate one effective global-depolarizing g from the Bell state.
    bell_measured = measured_swap_purity(bell, P2, P1, P_RO)
    g_bell = effective_g_from_purity(bell_measured, 1.0)

    per_state = []
    for name, st, tp in states:
        true_purity = tp if tp is not None else st.purity()
        swap_meas = _swap_measured(st, P2, P1, P_RO)
        swap_meas_gate_only = _swap_measured(st, P2, P1, 0.0)
        biaslaw = purity_from_g(g_bell, true_purity)
        own_g_full = effective_g_from_purity(swap_meas, true_purity)
        own_g_gate = effective_g_from_purity(swap_meas_gate_only, true_purity)

        dists = noisy_shadow_dists(st, P2, P1, P_RO)
        shadow = predict_shadow_purity(st, dists, SHADOW_M_REF, SHADOW_N_EXP, base_seed=100)
        se_at_budget = shadow["std"] * np.sqrt(SHADOW_M_REF / RECOMMENDED_SHOTS)

        per_state.append({
            "state": name,
            "true_purity": round(true_purity, 6),
            "swap_measured_qiskit": round(swap_meas, 6),
            "swap_biaslaw_g_bell": round(biaslaw, 6),
            "swap_discrepancy": round(swap_meas - biaslaw, 6),
            "swap_own_g_full": round(own_g_full, 6),
            "swap_own_g_gate_only": round(own_g_gate, 6),
            "shadow_measured_mean": round(shadow["mean"], 6),
            "shadow_se_at_Mref": round(shadow["std"], 6),
            "shadow_se_at_budget": round(float(se_at_budget), 6),
            "collective_bias": round(abs(true_purity - swap_meas), 6),
        })

    return {"g_bell": round(g_bell, 6), "bell_measured": round(bell_measured, 6),
            "per_state": per_state}


def readout_sensitivity() -> list[dict]:
    """Measured Bell purity across the plausible readout band (readout dominates)."""
    bell = bell_state()
    out = []
    for p_ro in (0.005, 0.01, 0.02, 0.03, 0.05):
        out.append({"p_ro": p_ro, "bell_measured": round(measured_swap_purity(bell, P2, P1, p_ro), 6)})
    return out


def calibration() -> dict:
    p2_grid = np.round(np.linspace(0.0, 0.05, 26), 6)
    p_ro_grid = np.round(np.linspace(0.005, 0.05, 10), 6)
    surface = bell_calibration_surface(p2_grid, p_ro_grid, P1)
    # inversion demo: invert the reference measured Bell purity into the (p_ro, p2) level set
    bell_measured = measured_swap_purity(bell_state(), P2, P1, P_RO)
    level_set = consistent_error_rates(bell_measured, P1, np.array([0.005, 0.01, 0.02, 0.03, 0.05]))
    return {
        "p2_grid": p2_grid.tolist(),
        "p_ro_grid": p_ro_grid.tolist(),
        "surface": np.round(surface, 6).tolist(),
        "inversion_demo": {
            "measured": round(bell_measured, 6),
            "consistent_p_ro_p2": [[round(a, 6), round(b, 6)] for a, b in level_set],
        },
    }


def shot_budget(per_state: list[dict]) -> dict:
    # Collective route: shots to resolve each state's bias at SIGMA_TARGET.
    rows = []
    for s in per_state:
        bias = s["collective_bias"]
        n_needed = swap_shots_for_se(s["swap_measured_qiskit"], bias / SIGMA_TARGET) if bias > 0 else None
        rows.append({
            "state": s["state"], "collective_bias": bias,
            "shots_for_%dsigma" % int(SIGMA_TARGET): n_needed,
            "swap_se_at_recommended": round(swap_shot_se(s["swap_measured_qiskit"], RECOMMENDED_SHOTS), 6),
            "swap_bias_sigma_at_recommended": round(bias / swap_shot_se(s["swap_measured_qiskit"], RECOMMENDED_SHOTS), 2) if bias > 0 else None,
        })
    binding = max((r["shots_for_5sigma"] for r in rows if r["shots_for_5sigma"]), default=0)
    n_configs = len(per_state) * 2  # states x {collective, single-copy}
    total_shots = RECOMMENDED_SHOTS * n_configs
    return {
        "recommended_shots_per_config": RECOMMENDED_SHOTS,
        "device_max_shots_per_circuit": MAX_SHOTS_PER_CIRCUIT,
        "binding_shots_to_resolve_all_biases_5sigma": binding,
        "n_configs": n_configs,
        "total_shots": total_shots,
        "total_credits": round(shots_to_credits(total_shots), 2),
        "credit_budget": CREDIT_BUDGET,
        "within_budget": shots_to_credits(total_shots) <= CREDIT_BUDGET,
        "per_state": rows,
    }


def render_report(pred, sens, budget, calib) -> str:
    L = []
    A = L.append
    A("# Cepheus purity experiment — locked predictions (ZERO credits)\n")
    A(f"Noise model (Qiskit Aer depolarizing parameters): p2(CZ)={P2}, p1(rx)={P1}, "
      f"p_ro(readout)={P_RO}. rz is virtual (noiseless).")
    A(f"CZ avg gate error at p2={P2} is {depol_param_to_avg_gate_error(P2,2):.5f} "
      f"(fidelity {1-depol_param_to_avg_gate_error(P2,2):.4%}); datasheet 99.1% -> p2=0.012.\n")
    A(f"Bell-calibrated effective global-depolarizing g = {pred['g_bell']:.4f} "
      f"(from measured Bell purity {pred['bell_measured']:.4f}).\n")

    A("## Locked measured-purity predictions (p_ro = %.3f)\n" % P_RO)
    A("| state | true | SWAP (Qiskit) | SWAP (bias law) | discrep | own g (full) | own g (gate-only) | shadow mean | shadow SE@20k |")
    A("|---|---|---|---|---|---|---|---|---|")
    for s in pred["per_state"]:
        A(f"| {s['state']} | {s['true_purity']:.3f} | {s['swap_measured_qiskit']:.4f} | "
          f"{s['swap_biaslaw_g_bell']:.4f} | {s['swap_discrepancy']:+.4f} | {s['swap_own_g_full']:.4f} | "
          f"{s['swap_own_g_gate_only']:.4f} | {s['shadow_measured_mean']:.4f} | {s['shadow_se_at_budget']:.4f} |")
    A("")
    A("Key finding: gate-only effective-g is nearly state-independent (~0.04), but readout")
    A("error breaks the universality (full g spans %.3f–%.3f). The global-depolarizing bias"
      % (min(s['swap_own_g_full'] for s in pred['per_state']),
         max(s['swap_own_g_full'] for s in pred['per_state'])))
    A("law captures the GATE noise but not the state-dependent readout suppression.\n")

    A("## Readout sensitivity (readout dominates the collective bias)\n")
    A("| p_ro | Bell measured |")
    A("|---|---|")
    for r in sens:
        A(f"| {r['p_ro']:.3f} | {r['bell_measured']:.4f} |")
    A("")

    A("## Shot budget\n")
    A("| state | collective bias | shots for 5σ | SWAP SE @20k | bias significance @20k |")
    A("|---|---|---|---|---|")
    for r in budget["per_state"]:
        A(f"| {r['state']} | {r['collective_bias']:.4f} | {r['shots_for_5sigma']} | "
          f"{r['swap_se_at_recommended']:.4f} | {r['swap_bias_sigma_at_recommended']}σ |")
    A("")
    A(f"Recommended: {budget['recommended_shots_per_config']:,} shots/config x "
      f"{budget['n_configs']} configs = {budget['total_shots']:,} shots = "
      f"{budget['total_credits']} credits (budget {budget['credit_budget']}; "
      f"within budget: {budget['within_budget']}).")
    A(f"Device max {budget['device_max_shots_per_circuit']:,} shots/circuit. Inversion level-set "
      f"(measured {calib['inversion_demo']['measured']:.4f} -> consistent (p_ro,p2)): "
      f"{calib['inversion_demo']['consistent_p_ro_p2']}\n")
    A("Caveats (stated for honesty):")
    A("* The collective route resolves every bias at >=4.6σ at 20k shots; the single-copy")
    A("  (shadow) route has ~2-4x larger error bars at the SAME shot budget — that gap is")
    A("  itself the n=2 collective advantage. Its shadow SE is a 1/sqrt(M) extrapolation")
    A("  from M=2000, which is a CONSERVATIVE upper bound (the U-statistic SE falls slightly")
    A("  faster than 1/sqrt(M) at small M), so the shadow route is if anything a touch better.")
    A("* Copy accounting: one SWAP-test shot consumes 2 copies (2n=4 qubits); one shadow")
    A("  shot consumes 1 copy (n=2 qubits). Open Quantum bills per shot, so equal-shots is")
    A("  the fair credit comparison; at equal COPIES the shadow route would get sqrt(2) more")
    A("  shots, still short of the collective route at n=2.\n")
    A("No hardware job submitted — ZERO quantum credits spent.")
    return "\n".join(L)


def main() -> None:
    pred = predict_all()
    sens = readout_sensitivity()
    calib = calibration()
    budget = shot_budget(pred["per_state"])

    full = {
        "config": {"p2": P2, "p1": P1, "p_ro": P_RO, "sigma_target": SIGMA_TARGET,
                   "recommended_shots_per_config": RECOMMENDED_SHOTS},
        "predictions": pred, "readout_sensitivity": sens,
        "calibration": calib, "shot_budget": budget,
    }
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "cepheus_predictions.json").write_text(json.dumps(full, indent=2))

    # committed, version-controlled locked deliverable (no big surface array)
    locked = {k: v for k, v in full.items() if k != "calibration"}
    locked["calibration_inversion_demo"] = calib["inversion_demo"]
    (REPO / "experiments" / "cepheus_locked_predictions.json").write_text(json.dumps(locked, indent=2))

    report = render_report(pred, sens, budget, calib)
    (REPO / "experiments" / "cepheus_prediction_report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
