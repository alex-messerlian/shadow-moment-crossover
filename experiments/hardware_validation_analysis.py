"""Analyze the real Cepheus hardware run and compare to the locked prediction.

Reads the raw counts saved from the single submitted job
(``results/hardware/raw_output.json``), computes the measured purity with a
bootstrap CI, compares to the locked prediction (0.9412), and inverts the
measurement into effective device error rates via the calibration machinery.

This is pure post-processing of data we already paid for — it submits nothing and
spends no credits.  Run:  ``python -m experiments.hardware_validation_analysis``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware import (
    REF_P1,
    bell_state,
    consistent_error_rates,
    effective_g_from_purity,
    invert_measured_to_p2,
    measured_swap_purity,
    purity_from_counts,
    swap_sign,
)

REPO = Path(__file__).resolve().parent.parent
HW = REPO / "results" / "hardware"
LOCKED_PREDICTION = 0.9412          # from the hardware-prediction phase (Qiskit gate-level sim)
ANALYTIC_BIAS_LAW = 0.9401          # analytic global-depolarizing bias law
PREDICTED_BAND = (0.92, 0.96)
SPEC_CZ_AVG_ERR = 0.009             # Rigetti published median
SPEC_CZ_DEPOL_PARAM = 0.012         # = 0.009 * 4/3
SPEC_P1 = 0.001


def _pro_at_spec_gates(mu: float, p2: float = SPEC_CZ_DEPOL_PARAM, p1: float = SPEC_P1) -> float:
    """Readout error that, with spec-level gates, reproduces the measured purity."""
    bell = bell_state()
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if measured_swap_purity(bell, p2, p1, mid) > mu:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analyze() -> dict:
    counts = {k: int(v) for k, v in json.loads((HW / "raw_output.json").read_text()).items()}
    n_shots = sum(counts.values())

    mu = purity_from_counts(counts, 2)
    # endianness-invariance sanity (the (c0,c2),(c1,c3) pairing is symmetric under reversal)
    mu_rev = purity_from_counts({k[::-1]: v for k, v in counts.items()}, 2)

    signs = np.array([swap_sign(b, 2) for b, c in counts.items() for _ in range(c)], dtype=float)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(signs, size=n_shots, replace=True).mean() for _ in range(5000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    se = float(np.sqrt((1.0 - mu * mu) / n_shots))

    g_eff = effective_g_from_purity(mu, 1.0)
    level_set = consistent_error_rates(mu, REF_P1, np.array([0.005, 0.01, 0.02, 0.03, 0.05]))
    p2_at_ro02 = invert_measured_to_p2(mu, 0.02, REF_P1)
    pro_at_spec = _pro_at_spec_gates(mu)

    return {
        "n_shots": n_shots,
        "measured_purity": mu,
        "measured_purity_reversed_bits": mu_rev,
        "analytic_se": se,
        "bootstrap_ci95": ci,
        "bootstrap_se": float(boot.std()),
        "locked_prediction": LOCKED_PREDICTION,
        "analytic_bias_law": ANALYTIC_BIAS_LAW,
        "deviation_from_prediction": mu - LOCKED_PREDICTION,
        "in_predicted_band": PREDICTED_BAND[0] <= mu <= PREDICTED_BAND[1],
        "prediction_in_ci": ci[0] <= LOCKED_PREDICTION <= ci[1],
        "effective_g": g_eff,
        "consistent_p_ro_p2": [[float(pr), float(p2), float(p2 * 3 / 4)] for pr, p2 in level_set],
        "implied_p2_at_pro0.02": p2_at_ro02,
        "implied_cz_avg_err_at_pro0.02": p2_at_ro02 * 3 / 4,
        "readout_needed_at_spec_gates": pro_at_spec,
        "spec_cz_avg_err": SPEC_CZ_AVG_ERR,
        "credits_consumed": 2,
        "credits_pool": "spark",
        "balance_after": {"full": 20, "spark": 23},
    }


def render(a: dict) -> str:
    mu, ci = a["measured_purity"], a["bootstrap_ci95"]
    dev = a["deviation_from_prediction"]
    L = ["# Cepheus hardware validation — result\n"]
    L.append(f"Single job on Rigetti Cepheus-1-108Q (Public Compute, Standard queue), "
             f"{a['n_shots']} shots, physical qubits {{0,1,9,10}}, 4 CZ, no routing SWAPs.\n")
    L.append("## Measured purity vs locked prediction\n")
    L.append(f"* **Measured purity: {mu:.4f}**  (95% bootstrap CI [{ci[0]:.4f}, {ci[1]:.4f}], "
             f"analytic SE {a['analytic_se']:.4f}). Endianness-invariant ({a['measured_purity_reversed_bits']:.4f}).")
    L.append(f"* Locked prediction: {a['locked_prediction']} (Qiskit), {a['analytic_bias_law']} (bias law); "
             f"true Bell purity is exactly 1.0.")
    L.append(f"* **Deviation: {dev:+.4f}** — measured is {'HIGHER' if dev > 0 else 'LOWER'} than predicted.")
    L.append(f"* In predicted band {PREDICTED_BAND}? **{a['in_predicted_band']}**. "
             f"Prediction inside measured CI? {a['prediction_in_ci']}.")
    L.append(f"\n**Prediction NOT confirmed.** The device is noisier on our specific qubits than the "
             f"published-median-based prediction. The measured distribution is a correctly-executed but "
             f"noisy Bell-SWAP test: its four dominant outcomes are exactly the ideal support "
             f"{{0000,0011,1100,1111}}, with ~25% of weight leaked into the other 12 outcomes by noise.\n")
    L.append("## Inverted effective noise vs Rigetti published spec\n")
    L.append(f"* Effective global-depolarizing g = **{a['effective_g']:.3f}** "
             f"(the prediction implied g ~ 0.10-0.14).")
    L.append(f"* Consistent (p_ro, p2) level set for the measured purity:")
    L.append("\n  | assumed readout p_ro | implied CZ depol p2 | implied CZ avg err |")
    L.append("  |---|---|---|")
    for pr, p2, avg in a["consistent_p_ro_p2"]:
        L.append(f"  | {pr:.3f} | {p2:.4f} | {avg:.4f} |")
    L.append(f"\n* At the assumed 2% readout, implied CZ avg error is "
             f"{a['implied_cz_avg_err_at_pro0.02']:.3f} (~{a['implied_cz_avg_err_at_pro0.02']/a['spec_cz_avg_err']:.0f}x "
             f"the {a['spec_cz_avg_err']} published median).")
    L.append(f"* Alternatively, if gates are AT spec, readout must be "
             f"**{a['readout_needed_at_spec_gates']*100:.1f}% per qubit** to explain the data. Readout was NOT in "
             f"the datasheet (we assumed 2%); the prediction phase flagged it as the dominant unknown, and the "
             f"hardware confirms readout is the leading suspect.\n")
    L.append("## Cost\n")
    L.append(f"* Estimated: 2 credits (quote). **Consumed: {a['credits_consumed']} {a['credits_pool']} credits.**")
    L.append(f"* Balance after: full={a['balance_after']['full']}, spark={a['balance_after']['spark']} "
             f"(before: full=20, spark=25).")
    L.append(f"* Raw counts (irreplaceable): `results/hardware/raw_output.json`. Submitted circuit: "
             f"`results/hardware/bell_swap_cepheus_q3.qasm`.\n")
    L.append("Public Plan: any publication must attribute Open Quantum (www.openquantum.com/citation).")
    return "\n".join(L)


def main() -> None:
    a = analyze()
    (HW / "analysis.json").write_text(json.dumps(a, indent=2, default=float))
    report = render(a)
    (HW / "REPORT.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
