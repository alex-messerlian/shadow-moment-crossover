"""Submit the CZ-depth-sweep jobs to Cepheus (REAL credits, ceiling 6).

Non-cancellable CZ echo (X-twirl) on physical {0,1,9,10}: |+>^4 -> [CZ layer,
X, CZ layer, X]^reps -> H -> measure, inside a Braket verbatim box (X = two native
rx(pi/2)).  Logically identity on |+>^4 at every depth, so survival vs CZ count
isolates the per-CZ error.  Depths: 0, 8, 16, 24, 32, 48 CZ; 3000 shots each.

Granular flow per circuit: prepare+quote (FREE) -> hard gate on running total (<=6)
-> create_job (the ONLY charge) -> poll -> save raw counts verbatim.  Resumable via
cz_progress.json (a circuit already recorded with a job_id is never resubmitted).
"""

from __future__ import annotations

import json
from pathlib import Path

from anrl.hardware.submit import (
    balance, clients, download_counts, org_id, poll_job, prepare_and_quote, submit_prepared,
)

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 6
ORDER = ["d0", "d8", "d16", "d24", "d32", "d48"]
SHOTS = 3000
PROGRESS = HW / "cz_progress.json"


def _load() -> dict:
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}


def main() -> None:
    progress = _load()
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running}/{CEILING} credits")

    for name in ORDER:
        rec = progress.get(name, {})
        if rec.get("counts_file"):
            print(f"[{name}] already complete ({rec['job_id'][:8]}), skipping")
            continue

        if rec.get("job_id"):
            # Submitted on a prior run but not yet downloaded (e.g. network drop while
            # polling). Resume polling — NEVER resubmit. Cost already counted in `running`.
            print(f"[{name}] resuming poll of already-submitted {rec['job_id'][:8]} (no resubmit)")
            job = poll_job(sched, rec["job_id"], timeout=600, interval=15)
        else:
            qasm = (HW / f"cz_{name}.qasm").read_bytes()
            bal_before = balance(mgmt, org)
            quo = prepare_and_quote(sched, org, qasm, SHOTS, f"czdepth {name}")
            ncz = qasm.decode().count("cz $")
            print(f"\n[{name}] {ncz} CZ; estimate {quo['cost']} cr; "
                  f"balance before full={bal_before['full']} spark={bal_before['spark']}")
            if running + quo["cost"] > CEILING:
                print(f"[{name}] STOP: running {running} + {quo['cost']} > ceiling {CEILING}. Not submitting.")
                break
            job = submit_prepared(sched, org, quo)  # THE charge
            running += quo["cost"]
            rec = {"job_id": job.id, "cost": quo["cost"], "shots": SHOTS, "ncz": ncz,
                   "status": job.status, "submitted_at": job.submitted_at, "bal_before": bal_before}
            progress[name] = rec
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] submitted {job.id} — polling...")
            job = poll_job(sched, job.id, timeout=600, interval=15)

        if job.status != "Completed":
            rec["status"] = job.status
            PROGRESS.write_text(json.dumps(progress, indent=2))
            print(f"[{name}] did NOT complete ({job.status}). Reporting and stopping.")
            break
        counts = download_counts(sched, job)
        (HW / f"cz_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        bal_after = balance(mgmt, org)
        bb = rec.get("bal_before")
        actual = ((bb["full"] - bal_after["full"]) + (bb["spark"] - bal_after["spark"])) if bb else rec["cost"]
        rec.update(status="Completed", counts_file=f"cz_{name}_counts.json",
                   actual_cost=actual, bal_after=bal_after)
        progress[name] = rec
        PROGRESS.write_text(json.dumps(progress, indent=2))
        print(f"[{name}] COMPLETED. actual {actual} cr; balance after full={bal_after['full']} "
              f"spark={bal_after['spark']}; shots {sum(counts.values())}")

    print(f"\nrunning credit total: {running}/{CEILING}")


if __name__ == "__main__":
    main()
