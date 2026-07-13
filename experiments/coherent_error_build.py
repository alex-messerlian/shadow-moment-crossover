"""Build + verify the Pauli-twirled (randomized-compiling) SWAP circuits (NO credits).

For n=3 (priority) and n=4: an untwirled baseline, a positive-control circuit, and
K independent Pauli-twirl realizations of the destructive SWAP test on the GHZ-ladder
registers.  Every twirled circuit is verified to be logically identical to the
untwirled one in noiseless simulation (SWAP-test purity 1.0 on GHZ), and every
twirl's explicit Pauli gates are confirmed present and distinct across realizations.

Writes QASM to results/hardware/ce_*.qasm and a manifest ce_manifest.json.
Run:  PYTHONPATH=. .venv/bin/python -m experiments.coherent_error_build
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anrl.hardware import twirl as T

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
BASE_SEED = 20260713
N_TWIRLS = {3: 12, 4: 4}
SHOTS_BASE = 2500      # untwirled + positive control
SHOTS_TWIRL = 2000     # each twirl realization
# depolarizing-model (independent readout + depolarizing gate) predictions, prior phases
DEPOL_PRED = {3: 0.606, 4: 0.579}
# untwirled measured purity, same-session-grid phase (drift-controlled, historical)
UNTW_HIST = {3: 0.3784, 4: 0.4204}
POSCTRL_QUBIT = 0          # copy-A qubit 0
POSCTRL_CZ = (0, 9)        # first cross-copy (SWAP-body) CZ at n=3


def _gate_counts(instrs: list) -> dict:
    out: dict = {}
    for name, *_ in instrs:
        out[name] = out.get(name, 0) + 1
    return out


def main() -> None:
    T.verify_pauli_algebra()  # natives + all 16 CZ twirl corrections preserve CZ
    manifest: dict = {"base_seed": BASE_SEED, "shots_base": SHOTS_BASE,
                      "shots_twirl": SHOTS_TWIRL, "depol_prediction": DEPOL_PRED,
                      "untwirled_measured_historical": UNTW_HIST, "circuits": {}}

    for n in (3, 4):
        untw = T.untwirled_instrs(n)
        untw_probs, untw_pur = T.ideal_distribution(untw, n)
        assert abs(untw_pur - 1.0) < 1e-9, untw_pur
        (HW / f"ce_n{n}_untw.qasm").write_text(T.emit_qasm(untw, 2 * n))
        n_cz = _gate_counts(untw).get("cz", 0)
        untw_qasm = T.emit_qasm(untw, 2 * n)
        manifest["circuits"][f"n{n}_untw"] = {
            "n": n, "shots": SHOTS_BASE, "kind": "untwirled", "cz": n_cz,
            "ideal_purity": untw_pur}

        # ----- twirl realizations -----
        seen_qasm = {untw_qasm}
        for k in range(N_TWIRLS[n]):
            rng = np.random.default_rng(BASE_SEED + n * 100 + k)
            tw, rec = T.twirl_instrs(untw, rng)
            v = T.verify_twirl(tw, untw, n)
            assert v["ok"], (n, k, v)
            qasm = T.emit_qasm(tw, 2 * n)
            # survival prerequisite: explicit Pauli gates present, and this twirl is
            # a DISTINCT circuit from the untwirled one and from every other twirl.
            gc_tw, gc_un = _gate_counts(tw), _gate_counts(untw)
            extra_1q = (gc_tw.get("rx", 0) + gc_tw.get("rz", 0)) - (gc_un.get("rx", 0) + gc_un.get("rz", 0))
            assert extra_1q > 0, (n, k, "no Pauli gates inserted")
            assert qasm not in seen_qasm, (n, k, "duplicate twirl QASM")
            seen_qasm.add(qasm)
            (HW / f"ce_n{n}_tw{k:02d}.qasm").write_text(qasm)
            manifest["circuits"][f"n{n}_tw{k:02d}"] = {
                "n": n, "shots": SHOTS_TWIRL, "kind": "twirl", "seed": BASE_SEED + n * 100 + k,
                "cz": gc_tw.get("cz", 0), "inserted_1q_pauli_gates": extra_1q,
                "ideal_purity": v["purity_twirled"], "max_prob_dev_vs_untwirled": v["max_prob_deviation"],
                "twirls": rec}

        # ----- positive control (n=3 only): X before CZ(0,9), uncorrected -----
        if n == 3:
            pc = T.insert_x_before_cz(untw, POSCTRL_QUBIT, POSCTRL_CZ)
            pc_probs, pc_pur = T.ideal_distribution(pc, n)
            (HW / f"ce_n{n}_posctrl.qasm").write_text(T.emit_qasm(pc, 2 * n))
            # dominant ideal support (bitstrings, clbit order) for both, for the survival check
            def top_support(probs, k=8):
                idx = np.argsort(probs)[::-1][:k]
                return {format(int(i), f"0{2*n}b"): round(float(probs[i]), 4) for i in idx if probs[i] > 1e-6}
            # total-variation distance between posctrl-ideal and untwirled-ideal
            tvd = 0.5 * float(np.abs(pc_probs - untw_probs).sum())
            manifest["circuits"][f"n{n}_posctrl"] = {
                "n": n, "shots": SHOTS_BASE, "kind": "positive_control",
                "insert": f"rx(pi) on ${POSCTRL_QUBIT} before cz{POSCTRL_CZ}",
                "ideal_purity": pc_pur, "tvd_vs_untwirled_ideal": tvd,
                "posctrl_ideal_support": top_support(pc_probs),
                "untwirled_ideal_support": top_support(untw_probs)}

    (HW / "ce_manifest.json").write_text(json.dumps(manifest, indent=2, default=float))

    # ---------- summary ----------
    print("=== COHERENT-ERROR (randomized-compiling) CIRCUITS BUILT ===\n")
    for n in (3, 4):
        print(f"n={n}: untwirled ideal purity 1.0, {N_TWIRLS[n]} twirls, "
              f"{_gate_counts(T.untwirled_instrs(n)).get('cz')} CZ each")
        print(f"   depol-model prediction {DEPOL_PRED[n]}, untwirled measured (hist) {UNTW_HIST[n]}")
    pc = manifest["circuits"]["n3_posctrl"]
    print(f"\npositive control (n=3): {pc['insert']}")
    print(f"   ideal purity {pc['ideal_purity']:.4f} (!= 1.0), TVD vs untwirled ideal {pc['tvd_vs_untwirled_ideal']:.4f}")
    print(f"   posctrl support:   {pc['posctrl_ideal_support']}")
    print(f"   untwirled support: {pc['untwirled_ideal_support']}")
    print("\nAll twirls verified logically identical to untwirled (purity 1.0, max dev ~1e-16),")
    print("Pauli gates present + every twirl QASM distinct. Manifest: ce_manifest.json")


if __name__ == "__main__":
    main()
