"""Granular Open Quantum submission flow with a hard spend gate.

The paid path is split so the cost is always seen and gated BEFORE any charge:

    prepare_and_quote(...)   # FREE: upload + prepare, returns the real quote
    # caller checks the running credit total against its ceiling here
    submit_prepared(...)     # THE ONLY charging call (create_job)
    poll_job(...)            # FREE: read-only status polling (never resubmits)
    download_counts(...)     # FREE: fetch the raw measurement counts

Physical-qubit circuits are submitted as OpenQASM 3 with ``$N`` addressing (a
virtual register with non-contiguous qubits is rejected by the platform).
"""

from __future__ import annotations

import time

from openquantum_sdk.enums import ExecutionPlanType, QueuePriorityType
from openquantum_sdk.models import JobCreate, JobPreparationCreate

from .backend import CEPHEUS_SHORT_CODE, _clients

SUBCAT_PHYSICS_OTHER = "dcd158af-d14f-46b2-b6b7-75f14267c02a"
PUBLIC = ExecutionPlanType.PUBLIC.value
STANDARD = QueuePriorityType.STANDARD.value


def clients():
    return _clients()


def balance(mgmt, org_id: str) -> dict:
    b = mgmt.get_credit_balance(org_id)
    return {"full": b.full_credits, "spark": b.spark_credits}


def org_id(mgmt) -> str:
    return mgmt.list_user_organizations().organizations[0].id


def prepare_and_quote(sched, org: str, qasm: bytes, shots: int, name: str) -> dict:
    """FREE: upload + prepare + fetch the real Public/Standard quote. No charge."""
    upload_id = sched.upload_job_input(file_content=qasm)
    prep = JobPreparationCreate(
        organization_id=org, backend_class_id=CEPHEUS_SHORT_CODE, name=name,
        upload_endpoint_id=upload_id, job_subcategory_id=SUBCAT_PHYSICS_OTHER,
        shots=shots, configuration_data={},
    )
    prep_resp = sched.prepare_job(prep)
    result = sched._wait_for_preparation(prep_resp.id, timeout=300, interval=5, verbose=False)
    if result.status != "Completed":
        raise RuntimeError(f"preparation not completed: {result.status} — {result.message}")
    cost = next((p.price + q.price_increase for p in result.quote if p.execution_plan_id == PUBLIC
                 for q in p.queue_priorities if q.queue_priority_id == STANDARD), None)
    if cost is None:
        raise RuntimeError("Public+Standard quote not found")
    return {"name": name, "upload_id": upload_id, "preparation_id": prep_resp.id,
            "exec_plan_id": PUBLIC, "queue_priority_id": STANDARD, "shots": shots, "cost": cost}


def submit_prepared(sched, org: str, prepared: dict):
    """THE single charging call (create_job) for one prepared job."""
    job = sched.create_job(JobCreate(
        organization_id=org, job_preparation_id=prepared["preparation_id"],
        execution_plan_id=prepared["exec_plan_id"], queue_priority_id=prepared["queue_priority_id"],
    ))
    return job


def poll_job(sched, job_id: str, timeout: int = 600, interval: int = 15, log=print):
    """FREE read-only polling until terminal status (no resubmission).

    Tolerates transient network errors: a dropped connection on ``get_job`` (a
    read-only call) is retried, never a resubmission.
    """
    import requests

    last, deadline = None, time.time() + timeout
    while time.time() < deadline:
        try:
            job = sched.get_job(job_id)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            log(f"    [poll] transient network error, retrying: {type(e).__name__}")
            time.sleep(interval)
            continue
        if job.status != last:
            log(f"    [{time.strftime('%H:%M:%S')}] {job_id[:8]} -> {job.status}"
                + (f" ({job.message})" if job.message else ""))
            last = job.status
        if job.status in ("Completed", "Failed", "Canceled"):
            return job
        time.sleep(interval)
    return sched.get_job(job_id)  # last known (still running)


def download_counts(sched, job) -> dict:
    out = sched.download_job_output(job)
    return {k: int(v) for k, v in out.items()}
