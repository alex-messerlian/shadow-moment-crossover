"""Submit the Pauli-twirled (randomized-compiling) SWAP jobs to Cepheus (REAL credits).

Tests whether coherent CZ error explains the collective-purity gap: twirling converts
coherent error to stochastic, so if coherent error is the mechanism the twirled purity
moves toward the depolarizing-model prediction.  n=3 is the priority (anomalous) cell;
n=4 runs if budget allows.

Order runs the n=3 baseline + positive control + first twirls FIRST (a survival pilot),
then the rest.  ``MAX_JOBS`` limits how many NEW jobs a single invocation submits, so the
pilot can be assessed before the bulk spend.  Ceiling 20 credits, quote-gated per job,
resumable (a job with a job_id is polled, never resubmitted), 6h poll timeout for the
device's calibration windows.  Raw counts are saved verbatim.

Run:  PYTHONPATH=. MAX_JOBS=6 .venv/bin/python -m experiments.run_coherent_error   (pilot)
      PYTHONPATH=. .venv/bin/python -m experiments.run_coherent_error              (rest)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from anrl.hardware.submit import (
    balance, clients, download_counts, org_id, poll_job, prepare_and_quote, submit_prepared,
)

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 20
PROGRESS = HW / "ce_progress.json"
MANIFEST = HW / "ce_manifest.json"

ORDER = (
    ["n3_untw", "n3_posctrl", "n3_tw00", "n3_tw01", "n3_tw02", "n3_tw03"]   # survival pilot
    + [f"n3_tw{k:02d}" for k in range(4, 12)]                               # rest of n=3
    + ["n4_untw"] + [f"n4_tw{k:02d}" for k in range(4)]                     # n=4 (secondary)
)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())["circuits"]
    progress = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    max_new = int(os.environ.get("MAX_JOBS", "999"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running}/{CEILING} credits; MAX_JOBS this run = {max_new}")

    submitted_this_run = 0
    for name in ORDER:
        rec = progress.get(name, {})
        if rec.get("counts_file"):
            continue
        if rec.get("job_id"):
            print(f"[{name}] resuming poll of {rec['job_id'][:8]} (no resubmit)")
            job = poll_job(sched, rec["job_id"], timeout=21600, interval=30)
        else:
            if submitted_this_run >= max_new:
                print(f"[{name}] MAX_JOBS={max_new} reached this run — stopping (resumable).")
                break
            shots = manifest[name]["shots"]
            bal_before = balance(mgmt, org)
            quo = prepare_and_quote(sched, org, (HW / f"ce_{name}.qasm").read_bytes(), shots, f"ce {name}")
            print(f"\n[{name}] {shots} shots ({manifest[name]['kind']}); estimate {quo['cost']} cr; "
                  f"running {running}/{CEILING}; balance full={bal_before['full']} spark={bal_before['spark']}")
            if running + quo["cost"] > CEILING:
                print(f"[{name}] STOP: {running}+{quo['cost']} > {CEILING}. Not submitting.")
                break
            job = submit_prepared(sched, org, quo)
            running += quo["cost"]
            submitted_this_run += 1
            rec = {"job_id": job.id, "cost": quo["cost"], "shots": shots, "kind": manifest[name]["kind"],
                   "status": job.status, "submitted_at": job.submitted_at, "bal_before": bal_before}
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
        (HW / f"ce_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        ba = balance(mgmt, org); bb = rec.get("bal_before")
        actual = ((bb["full"] - ba["full"]) + (bb["spark"] - ba["spark"])) if bb else rec["cost"]
        rec.update(status="Completed", counts_file=f"ce_{name}_counts.json", actual_cost=actual, bal_after=ba)
        progress[name] = rec
        PROGRESS.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETED. actual {actual} cr; balance full={ba['full']} spark={ba['spark']}; "
              f"shots {sum(counts.values())}")

    done = sum(1 for v in progress.values() if v.get("counts_file"))
    print(f"\nrunning credit total: {running}/{CEILING}; completed {done}/{len(ORDER)} jobs")


if __name__ == "__main__":
    main()
