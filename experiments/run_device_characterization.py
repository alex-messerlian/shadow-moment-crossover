"""Submit the device-characterization jobs to Cepheus (REAL credits, ceiling 6).

Job A (readout): |0000>, |0011>, |1100>, |1111> on physical {0,1,9,10}, X gates
only (|0000> uses identity gates so the platform registers physical-qubit usage),
2000 shots each.  Job B (gates): |+>^4 -> 8 CZ echo (4 Bell-SWAP pairs, twice, with
barriers so none cancel) -> H -> measure, 4000 shots.

Granular flow per circuit: prepare+quote (FREE) -> hard gate on the running credit
total (<= 6) -> create_job (the ONLY charge) -> poll -> save raw counts verbatim.
Resumable: a circuit already recorded with a job_id in char_progress.json is never
resubmitted.  Prints balance before/after each submission.
"""

from __future__ import annotations

import json
from pathlib import Path

from anrl.hardware.submit import (
    balance,
    clients,
    download_counts,
    org_id,
    poll_job,
    prepare_and_quote,
    submit_prepared,
)

HW = Path(__file__).resolve().parent.parent / "results" / "hardware"
CEILING = 6
ORDER = ["A_0000", "A_0011", "A_1100", "A_1111", "B_cz_echo"]
SHOTS = {"A_0000": 2000, "A_0011": 2000, "A_1100": 2000, "A_1111": 2000, "B_cz_echo": 4000}
PROGRESS = HW / "char_progress.json"


def _load() -> dict:
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {}


def _save(p: dict) -> None:
    PROGRESS.write_text(json.dumps(p, indent=2))


def main() -> None:
    progress = _load()
    running = sum(v.get("cost", 0) for v in progress.values() if v.get("job_id"))
    sched, mgmt = clients()
    org = org_id(mgmt)
    print(f"running total so far: {running} credits (ceiling {CEILING})")

    for name in ORDER:
        if progress.get(name, {}).get("job_id"):
            print(f"[{name}] already submitted (job {progress[name]['job_id'][:8]}), skipping")
            continue

        qasm = (HW / f"char_{name}.qasm").read_bytes()
        shots = SHOTS[name]
        bal_before = balance(mgmt, org)
        quo = prepare_and_quote(sched, org, qasm, shots, f"char {name}")
        print(f"\n[{name}] estimate {quo['cost']} cr ({shots} shots); "
              f"balance before full={bal_before['full']} spark={bal_before['spark']}")

        if running + quo["cost"] > CEILING:
            print(f"[{name}] STOP: running {running} + quote {quo['cost']} > ceiling {CEILING}. Not submitting.")
            break

        job = submit_prepared(sched, org, quo)  # THE charge
        progress.setdefault(name, {}).update(
            job_id=job.id, cost=quo["cost"], shots=shots, status=job.status,
            submitted_at=job.submitted_at)
        _save(progress)
        print(f"[{name}] submitted job {job.id} — polling...")

        job = poll_job(sched, job.id, timeout=540, interval=15)
        if job.status != "Completed":
            progress[name]["status"] = job.status
            _save(progress)
            print(f"[{name}] did NOT complete (status {job.status}). Reporting and stopping.")
            break

        counts = download_counts(sched, job)
        (HW / f"char_{name}_counts.json").write_text(json.dumps(counts, indent=2))
        bal_after = balance(mgmt, org)
        actual = (bal_before["full"] - bal_after["full"]) + (bal_before["spark"] - bal_after["spark"])
        running += quo["cost"]
        progress[name].update(status="Completed", counts_file=f"char_{name}_counts.json",
                              actual_cost=actual, bal_before=bal_before, bal_after=bal_after)
        _save(progress)
        print(f"[{name}] COMPLETED. actual cost {actual} cr; "
              f"balance after full={bal_after['full']} spark={bal_after['spark']}; "
              f"total shots {sum(counts.values())}")

    print(f"\nrunning credit total: {running}/{CEILING}")
    print("submitted:", {k: v.get("job_id", "")[:8] for k, v in progress.items() if v.get("job_id")})


if __name__ == "__main__":
    main()
