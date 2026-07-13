"""Submit joint (multi-qubit) readout characterization on the SWAP registers (ceiling 12).

n=3 register {0,1,2,9,10,11} (7 states, priority) + n=4 register {0,1,2,3,9,10,11,12}
(3 states).  X-gate basis states chosen to expose cross-copy pair (i, i+n) correlation.
Granular flow per job: prepare+quote (FREE) -> hard gate (<=12) -> create_job -> poll
(6h timeout, resumable, never resubmit) -> save counts verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path

from anrl.hardware.submit import (
    balance, clients, download_counts, org_id, poll_job, prepare_and_quote, submit_prepared,
)

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 12
PROGRESS = HW / "jr_progress.json"
ORDER = ["n3_s0", "n3_h0", "n3_h1", "n3_h2", "n3_copyA", "n3_copyB", "n3_all",
         "n4_s0", "n4_copyA", "n4_all"]
SHOTS = 2500


def main():
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running}/{CEILING}")

    for name in ORDER:
        rec = progress.get(name, {})
        if rec.get("counts_file"):
            continue
        if rec.get("job_id"):
            print(f"[{name}] resuming poll of {rec['job_id'][:8]} (no resubmit)")
            job = poll_job(sched, rec["job_id"], timeout=21600, interval=30)
        else:
            bal_before = balance(mgmt, org)
            quo = prepare_and_quote(sched, org, (HW / f"jr_{name}.qasm").read_bytes(), SHOTS, f"jr {name}")
            print(f"\n[{name}] estimate {quo['cost']} cr; running {running}/{CEILING}; "
                  f"balance full={bal_before['full']} spark={bal_before['spark']}")
            if running + quo["cost"] > CEILING:
                print(f"[{name}] STOP: {running}+{quo['cost']} > {CEILING}. Not submitting.")
                break
            job = submit_prepared(sched, org, quo)
            running += quo["cost"]
            rec = {"job_id": job.id, "cost": quo["cost"], "shots": SHOTS, "status": job.status,
                   "submitted_at": job.submitted_at, "bal_before": bal_before}
            progress[name] = rec
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] submitted {job.id} — polling...")
            job = poll_job(sched, job.id, timeout=21600, interval=30)
        if job.status != "Completed":
            rec["status"] = job.status
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] did NOT complete ({job.status}). Stopping.")
            break
        counts = download_counts(sched, job)
        (HW / f"jr_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        ba = balance(mgmt, org); bb = rec.get("bal_before")
        actual = ((bb["full"] - ba["full"]) + (bb["spark"] - ba["spark"])) if bb else rec["cost"]
        rec.update(status="Completed", counts_file=f"jr_{name}_counts.json", actual_cost=actual, bal_after=ba)
        progress[name] = rec
        PROGRESS.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETED. actual {actual} cr; balance full={ba['full']} spark={ba['spark']}")

    print(f"\nrunning credit total: {running}/{CEILING}")


if __name__ == "__main__":
    main()
