"""Open Quantum backend access for Rigetti Cepheus — auth, metadata, coupling map.

Credentials are read from the gitignored ``.env`` (never committed).  Listing
backends and fetching capabilities is FREE (no credits); this module never submits
a job.  Device metadata is cached to ``results/cepheus_metadata.json`` so
transpilation and tests work OFFLINE (no network, no credentials).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# Committed package data (public device info: no secrets) so transpilation and
# tests work offline without network or credentials.
METADATA_PATH = Path(__file__).resolve().parent / "cepheus_metadata.json"
CEPHEUS_SHORT_CODE = "rigetti:cepheus-1-108q"
CEPHEUS_BASIS_GATES = ["cz", "rx", "rz"]  # native ops reported by the device


def load_credentials() -> tuple[str, str]:
    """Read ``OPENQUANTUM_CLIENT_ID/SECRET`` from the environment or ``.env``."""
    cid, sec = os.environ.get("OPENQUANTUM_CLIENT_ID"), os.environ.get("OPENQUANTUM_CLIENT_SECRET")
    env = REPO / ".env"
    if (not cid or not sec) and env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k == "OPENQUANTUM_CLIENT_ID" and not cid:
                    cid = v
                elif k == "OPENQUANTUM_CLIENT_SECRET" and not sec:
                    sec = v
    if not cid or not sec:
        raise RuntimeError("Open Quantum credentials not found (env vars or .env)")
    return cid, sec


def _clients():
    """Authenticate and return ``(scheduler, management)`` clients (network + creds)."""
    import openquantum_sdk as oq
    from openquantum_sdk import auth

    cid, sec = load_credentials()
    a = auth.ClientCredentialsAuth(auth.ClientCredentials(client_id=cid, client_secret=sec))
    return oq.SchedulerClient(auth=a), oq.ManagementClient(auth=a)


def list_backend_classes() -> list[dict]:
    """Live catalog of backend classes (FREE).  One dict per device."""
    _, mgmt = _clients()
    page = mgmt.list_backend_classes(limit=100)
    return [{"short_code": b.short_code, "name": b.name, "type": b.type, "status": b.status,
             "accepting_jobs": b.accepting_jobs} for b in page.backend_classes]


def fetch_cepheus_metadata(save: bool = True) -> dict:
    """Fetch Cepheus device metadata (FREE) and optionally cache it to disk.

    Records qubit count, native gates, coupling map, and limits.  The device does
    NOT expose error rates / readout / T1 / T2 (``constraint_data.noise`` is empty),
    so ``error_rates`` is ``None`` with an explanatory note — we do not fabricate them.
    """
    from openquantum_sdk import backends

    sched, _ = _clients()
    bc = sched.get_backend_class(CEPHEUS_SHORT_CODE)
    cap = backends.fetch_capabilities(CEPHEUS_SHORT_CODE, sched)
    noise = bc.get("constraint_data", {}).get("noise", {})
    meta = {
        "short_code": cap.short_code,
        "name": bc.get("name"),
        "provider": "rigetti",
        "n_qubits": cap.n_qubits,
        "native_ops": [g.name for g in cap.native_ops],
        "coupling_map": [list(e) for e in cap.topology.coupling_map],
        "limits": cap.limits,
        "features": cap.features,
        "error_rates": (noise or None),
        "error_rates_note": (
            "constraint_data.noise is empty; the platform's free metadata API does "
            "not expose two-qubit gate error, readout error, or T1/T2 for this device."
        ) if not noise else None,
    }
    if save:
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def load_cepheus_metadata() -> dict:
    """Load the cached Cepheus metadata (OFFLINE; no network/credentials)."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} missing; run fetch_cepheus_metadata() once")
    return json.loads(METADATA_PATH.read_text())


def cepheus_coupling_map():
    """Cepheus coupling map as a Qiskit ``CouplingMap`` (from the cached metadata)."""
    from qiskit.transpiler import CouplingMap

    return CouplingMap(couplinglist=[tuple(e) for e in load_cepheus_metadata()["coupling_map"]])
