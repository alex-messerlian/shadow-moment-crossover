"""Submit the GHZ-ladder hardware experiment (REAL credits, ceiling 35).

3 collective SWAP jobs (GHZ n=2,3,4, 10k shots) + 15 single-copy anchor jobs (n=2,
600 shots each). Granular flow per circuit: prepare+quote (FREE) -> hard gate on running
total (<=35) -> create_job (the ONLY charge) -> poll -> save raw counts. Resumable via
hg_progress.json; 6h poll timeout to ride out calibration windows; never resubmits.
"""

from __future__ import annotations

import json
from pathlib import Path

from anrl.hardware.submit import (
    balance, clients, download_counts, org_id, poll_job, prepare_and_quote, submit_prepared,
)

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 35
PROGRESS = HW / "hg_progress.json"
JOBS = ([(f"coll_n{n}", f"hg_coll_n{n}.qasm", 10000) for n in (2, 3, 4)]
        + [(f"single_n2_b{i:02d}", f"hg_single_n2_b{i:02d}.qasm", 600) for i in range(15)])


def main() -> None:
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running}/{CEILING} credits")

    for name, fname, shots in JOBS:
        rec = progress.get(name, {})
        if rec.get("counts_file"):
            print(f"[{name}] already complete, skipping")
            continue
        if rec.get("job_id"):
            print(f"[{name}] resuming poll of {rec['job_id'][:8]} (no resubmit)")
            job = poll_job(sched, rec["job_id"], timeout=21600, interval=30)
        else:
            bal_before = balance(mgmt, org)
            quo = prepare_and_quote(sched, org, (HW / fname).read_bytes(), shots, f"hg {name}")
            print(f"\n[{name}] {shots} shots; estimate {quo['cost']} cr; running {running}/{CEILING}; "
                  f"balance before full={bal_before['full']} spark={bal_before['spark']}")
            if running + quo["cost"] > CEILING:
                print(f"[{name}] STOP: running {running} + {quo['cost']} > ceiling {CEILING}. Not submitting.")
                break
            job = submit_prepared(sched, org, quo)
            running += quo["cost"]
            rec = {"job_id": job.id, "cost": quo["cost"], "shots": shots,
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
        (HW / f"hg_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        bal_after = balance(mgmt, org)
        bb = rec.get("bal_before")
        actual = ((bb["full"] - bal_after["full"]) + (bb["spark"] - bal_after["spark"])) if bb else rec["cost"]
        rec.update(status="Completed", counts_file=f"hg_{name}_counts.json",
                   actual_cost=actual, bal_after=bal_after)
        progress[name] = rec
        PROGRESS.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETED. actual {actual} cr; balance after full={bal_after['full']} "
              f"spark={bal_after['spark']}; shots {sum(counts.values())}")

    print(f"\nrunning credit total: {running}/{CEILING}")


if __name__ == "__main__":
    main()
