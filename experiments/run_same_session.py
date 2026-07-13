"""Bracketed same-session GHZ-ladder experiment (REAL credits, ceiling 32).

Submission order IS the experiment, back to back:
  Block A  — opening readout characterization (7 states x 2500 shots, ~7 cr)
  -> LOCK predictions from Block A's fresh parameters (before any Block B result exists)
  Block B  — collective SWAP on GHZ n=2,3,4 (10k shots, ~9 cr)
  Block C  — closing readout characterization (identical to A, ~7 cr)

Reuses the validated readout circuits (ro_*.qasm) and SWAP circuits (hg_coll_n*.qasm).
Granular flow per job: prepare+quote (FREE) -> hard gate on running total (<=32) ->
create_job -> poll (6h timeout, resumable, never resubmits) -> save counts verbatim.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from anrl.hardware.submit import (
    balance, clients, download_counts, org_id, poll_job, prepare_and_quote, submit_prepared,
)
from experiments.same_session_lib import build_tables, parse_readout, predict_swap

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 32
PROGRESS = HW / "ss_progress.json"
LOCKED = HW / "locked_same_session.json"
RO_STATES = ["w0", "w2", "w4a", "w4b", "w6a", "w6b", "w8"]
BLOCK_A = [(f"A_{w}", f"ro_{w}.qasm", 2500) for w in RO_STATES]
BLOCK_B = [(f"B_n{n}", f"hg_coll_n{n}.qasm", 10000) for n in (2, 3, 4)]
BLOCK_C = [(f"C_{w}", f"ro_{w}.qasm", 2500) for w in RO_STATES]


def _load():
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}


def run_block(jobs, progress, sched, mgmt, org, running):
    """Submit+poll each job in the block (resumable, quote-gated). Returns (running, ok)."""
    for name, fname, shots in jobs:
        rec = progress.get(name, {})
        if rec.get("counts_file"):
            continue
        if rec.get("job_id"):
            print(f"[{name}] resuming poll of {rec['job_id'][:8]} (no resubmit)")
            job = poll_job(sched, rec["job_id"], timeout=21600, interval=30)
        else:
            bal_before = balance(mgmt, org)
            quo = prepare_and_quote(sched, org, (HW / fname).read_bytes(), shots, f"ss {name}")
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
        (HW / f"ss_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        ba = balance(mgmt, org); bb = rec.get("bal_before")
        actual = ((bb["full"] - ba["full"]) + (bb["spark"] - ba["spark"])) if bb else rec["cost"]
        rec.update(status="Completed", counts_file=f"ss_{name}_counts.json", actual_cost=actual, bal_after=ba)
        progress[name] = rec
        PROGRESS.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETED. actual {actual} cr; balance full={ba['full']} spark={ba['spark']}")
    return running, True


def lock_from_block_a():
    """Compute + freeze the predictions from Block A's fresh parameters. Called before Block B."""
    rates = parse_readout("A")
    tables = build_tables(rates)
    preds = {n: predict_swap(n, tables) for n in (2, 3, 4)}
    n2_mid = preds[2]["band"]["mid"]
    sanity = "OK (near Bell region ~0.70-0.74)" if 0.66 <= n2_mid <= 0.78 else \
        f"WARNING: n=2 pred {n2_mid:.3f} far from Bell region -> device may have moved a lot"
    locked = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": "Block A opening readout characterization (fresh, same session)",
        "cz_band_avg_error": [0.005, 0.015], "p1": 0.001,
        "readout_rates_A": {str(q): {"p10": round(rates[q]["p10"], 4),
                                     "p01": round(rates[q]["p01"], 4),
                                     "p10_by_w": {str(w): round(d["p10"], 4)
                                                  for w, d in rates[q]["p10_by_w"].items()}} for q in rates},
        "predictions": preds, "n2_sanity_gate": sanity,
    }
    LOCKED.write_text(json.dumps(locked, indent=2))
    print("\n=== LOCKED PREDICTIONS from Block A (frozen before Block B) ===")
    for n in (2, 3, 4):
        b = preds[n]["band"]
        print(f"  n={n}: band {b['lo']:.3f}/{b['mid']:.3f}/{b['hi']:.3f}  "
              f"(readout pen {preds[n]['readout_penalty']:.3f}, gate pen {preds[n]['gate_penalty']:.3f})")
    print(f"  n=2 sanity gate: {sanity}\n")


def main():
    progress = _load()
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running}/{CEILING}")

    # BLOCK A
    running, ok = run_block(BLOCK_A, progress, sched, mgmt, org, running)
    if not ok:
        return
    # LOCK before B (idempotent)
    if not LOCKED.exists():
        lock_from_block_a()
    else:
        print("[lock] locked_same_session.json already exists — not recomputing")
    # BLOCK B
    running, ok = run_block(BLOCK_B, progress, sched, mgmt, org, running)
    if not ok:
        return
    # BLOCK C
    running, ok = run_block(BLOCK_C, progress, sched, mgmt, org, running)
    print(f"\nrunning credit total: {running}/{CEILING}  (all blocks done: {ok})")


if __name__ == "__main__":
    main()
