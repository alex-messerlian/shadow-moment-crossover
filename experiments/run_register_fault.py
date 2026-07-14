"""Bracketed register-fault localization experiment (REAL credits, ceiling 20).

Back-to-back, one session:
  Block A  opening readout calibration on {0,1,2,3,9,10,11,12} (7 states x 2500, ~7 cr)
  -> LOCK predictions for 3 SWAP cells from Block A (frozen before any SWAP result)
  Block B  SWAP: n3_std {0,1,2,9,10,11}, n4 {0,1,2,3,9,10,11,12}, n3_alt {1,2,3,10,11,12}
           (5000 shots each, ~2 cr each = 6 cr; byte-identical circuits verified)
  Block C  closing readout calibration, identical to A (~7 cr) -> within-session drift

Order puts the lock + SWAP within 13 cr; the closing bracket is last (lowest risk).
Granular flow per job: prepare+quote (FREE) -> hard gate (<=20) -> create_job -> poll
(6h timeout, resumable, never resubmit) -> save counts verbatim.
Run:  PYTHONPATH=. python -m experiments.run_register_fault
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from anrl.hardware.submit import (
    balance, clients, download_counts, org_id, poll_job, prepare_and_quote, submit_prepared,
)
from experiments.register_fault_lib import build_tables, parse_readout_rf, predict_all, CELLS

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 20
PROGRESS = HW / "rf_progress.json"
LOCKED = HW / "rf_locked.json"
RO_STATES = ["w0", "w2", "w4a", "w4b", "w6a", "w6b", "w8"]
BLOCK_A = [(f"A_{w}", f"ro_{w}.qasm", 2500) for w in RO_STATES]
BLOCK_B = [("B_n3_std", "hg_coll_n3.qasm", 5000), ("B_n4", "hg_coll_n4.qasm", 5000),
           ("B_n3_alt", "rf_n3alt.qasm", 5000)]
BLOCK_C = [(f"C_{w}", f"ro_{w}.qasm", 2500) for w in RO_STATES]


def _load():
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}


def _byte_identity_gate():
    """Hard gate: the n3_std / n4 SWAP QASM we submit must equal the committed circuits."""
    for n in (3, 4):
        f = f"results/hardware/hg_coll_n{n}.qasm"
        committed = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
        if committed != (HW / f"hg_coll_n{n}.qasm").read_text() or not committed:
            raise SystemExit(f"BYTE-IDENTITY FAILED for hg_coll_n{n}.qasm — aborting")
    print("[gate] byte-identity vs committed hg_coll_n3/n4: OK")


def run_block(jobs, progress, sched, mgmt, org, running):
    for name, fname, shots in jobs:
        rec = progress.get(name, {})
        if rec.get("counts_file"):
            continue
        if rec.get("job_id"):
            print(f"[{name}] resuming poll of {rec['job_id'][:8]} (no resubmit)")
            job = poll_job(sched, rec["job_id"], timeout=21600, interval=30)
        else:
            bal_before = balance(mgmt, org)
            quo = prepare_and_quote(sched, org, (HW / fname).read_bytes(), shots, f"rf {name}")
            print(f"\n[{name}] {shots} shots; estimate {quo['cost']} cr; running {running}/{CEILING}; "
                  f"balance full={bal_before['full']} spark={bal_before['spark']}")
            if running + quo["cost"] > CEILING:
                print(f"[{name}] STOP: {running}+{quo['cost']} > {CEILING}. Not submitting.")
                return running, False
            job = submit_prepared(sched, org, quo)
            running += quo["cost"]
            rec = {"job_id": job.id, "cost": quo["cost"], "shots": shots, "status": job.status,
                   "submitted_at": job.submitted_at, "bal_before": bal_before}
            progress[name] = rec
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] submitted {job.id} — polling...")
            job = poll_job(sched, job.id, timeout=21600, interval=30)
        if job.status != "Completed":
            rec["status"] = job.status
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] did NOT complete ({job.status}). Stopping.")
            return running, False
        counts = download_counts(sched, job)
        (HW / f"rf_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        ba = balance(mgmt, org); bb = rec.get("bal_before")
        actual = ((bb["full"] - ba["full"]) + (bb["spark"] - ba["spark"])) if bb else rec["cost"]
        rec.update(status="Completed", counts_file=f"rf_{name}_counts.json", actual_cost=actual, bal_after=ba)
        progress[name] = rec
        PROGRESS.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETED. actual {actual} cr; balance full={ba['full']} spark={ba['spark']}; "
              f"shots {sum(counts.values())}")
    return running, True


def lock_from_block_a():
    """Freeze the 3-cell predictions from Block A's fresh calibration (before any SWAP)."""
    tables = build_tables(parse_readout_rf("A"))
    preds = predict_all(tables)
    n3 = preds["n3_std"]["band"]["mid"]
    sanity = "OK" if 0.45 <= n3 <= 0.75 else f"WARNING: n3_std mid {n3:.3f} far from expected ~0.6"
    locked = {"timestamp": datetime.now().isoformat(timespec="seconds"),
              "source": "Block A opening readout (fresh, same session)",
              "cz_band_avg_error": [0.005, 0.015], "p1": 0.001, "cells": CELLS,
              "readout_rates_A": {str(q): {"p10": round(parse_readout_rf("A")[q]["p10"], 4),
                                           "p01": round(parse_readout_rf("A")[q]["p01"], 4)}
                                  for q in [0, 1, 2, 3, 9, 10, 11, 12]},
              "predictions": preds, "n3_std_sanity": sanity}
    LOCKED.write_text(json.dumps(locked, indent=2))
    print("\n=== LOCKED PREDICTIONS from Block A (frozen before Block B) ===")
    for name, p in preds.items():
        b = p["band"]
        print(f"  {name:7s} phys={p['phys']}: band {b['hi']:.3f}/{b['mid']:.3f}/{b['lo']:.3f}  "
              f"(readout pen {p['readout_penalty']:.3f}, gate pen {p['gate_penalty']:.3f})")
    print(f"  n3_std sanity: {sanity}\n")


def main():
    progress = _load()
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running}/{CEILING}")

    running, ok = run_block(BLOCK_A, progress, sched, mgmt, org, running)
    if not ok:
        return
    if not LOCKED.exists():
        lock_from_block_a()
    else:
        print("[lock] rf_locked.json exists — not recomputing")
    _byte_identity_gate()  # hard gate before SWAP
    running, ok = run_block(BLOCK_B, progress, sched, mgmt, org, running)
    if not ok:
        return
    running, ok = run_block(BLOCK_C, progress, sched, mgmt, org, running)
    print(f"\nrunning credit total: {running}/{CEILING}  (all blocks done: {ok})")


if __name__ == "__main__":
    main()
