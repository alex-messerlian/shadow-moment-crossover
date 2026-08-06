"""Submit readout characterization for the n=3/n=4 GHZ SWAP qubits (REAL credits, ceiling 8).

Eight physical qubits {0,1,2,3,9,10,11,12} (n=3 set {0,1,2,9,10,11} nests inside).
Seven X-only basis states spanning excitation weights 0,2,4,4,6,6,8 so every qubit is
sampled prepared-0 and prepared-1 at several excited-neighbor counts w, measuring the
weight-correlation directly instead of extrapolating the $0 result.  3000 shots each.

Granular flow per circuit: prepare+quote (FREE) -> hard gate on running total (<=8) ->
create_job (the ONLY charge) -> poll -> save raw counts.  Resumable via ro_progress.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from anrl.hardware.submit import (
    balance, clients, download_counts, org_id, poll_job, prepare_and_quote, submit_prepared,
)

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 8
ORDER = ["w0", "w2", "w4a", "w4b", "w6a", "w6b", "w8"]
SHOTS = 3000
PROGRESS = HW / "ro_progress.json"


def main() -> None:
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running}/{CEILING} credits")

    for name in ORDER:
        rec = progress.get(name, {})
        if rec.get("counts_file"):
            print(f"[{name}] already complete, skipping")
            continue
        if rec.get("job_id"):
            print(f"[{name}] resuming poll of {rec['job_id'][:8]} (no resubmit)")
            job = poll_job(sched, rec["job_id"], timeout=21600, interval=30)
        else:
            qasm = (HW / f"ro_{name}.qasm").read_bytes()
            bal_before = balance(mgmt, org)
            quo = prepare_and_quote(sched, org, qasm, SHOTS, f"readout-ext {name}")
            print(f"\n[{name}] estimate {quo['cost']} cr; balance before "
                  f"full={bal_before['full']} spark={bal_before['spark']}")
            if running + quo["cost"] > CEILING:
                print(f"[{name}] STOP: running {running} + {quo['cost']} > ceiling {CEILING}. Not submitting.")
                break
            job = submit_prepared(sched, org, quo)
            running += quo["cost"]
            rec = {"job_id": job.id, "cost": quo["cost"], "shots": SHOTS,
                   "status": job.status, "submitted_at": job.submitted_at, "bal_before": bal_before}
            progress[name] = rec
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] submitted {job.id}, polling...")
            job = poll_job(sched, job.id, timeout=21600, interval=30)

        if job.status != "Completed":
            rec["status"] = job.status
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] did NOT complete ({job.status}). Reporting and stopping.")
            break
        counts = download_counts(sched, job)
        (HW / f"ro_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        bal_after = balance(mgmt, org)
        bb = rec.get("bal_before")
        actual = ((bb["full"] - bal_after["full"]) + (bb["spark"] - bal_after["spark"])) if bb else rec["cost"]
        rec.update(status="Completed", counts_file=f"ro_{name}_counts.json",
                   actual_cost=actual, bal_after=bal_after)
        progress[name] = rec
        PROGRESS.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETED. actual {actual} cr; balance after "
              f"full={bal_after['full']} spark={bal_after['spark']}; shots {sum(counts.values())}")

    print(f"\nrunning credit total: {running}/{CEILING}")


if __name__ == "__main__":
    main()
