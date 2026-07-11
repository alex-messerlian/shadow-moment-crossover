"""Environment verification for adaptive-negativity-rl.

Imports the core scientific/RL stack, prints versions, and reports hardware
acceleration availability (CUDA and Apple MPS). Run inside the project venv:

    .venv/bin/python check_env.py

This is a scaffold-only sanity check — it exercises no simulation, environment,
or agent logic.
"""

from __future__ import annotations

import platform
import sys


def _version(module_name: str) -> str:
    """Import a module and return its ``__version__`` (or an error marker)."""
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001 - report, don't crash the check
        return f"IMPORT FAILED: {exc}"


def main() -> int:
    print("=" * 60)
    print("adaptive-negativity-rl — environment check")
    print("=" * 60)

    print(f"Python           : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform         : {platform.platform()}")
    print(f"Machine          : {platform.machine()}")
    print("-" * 60)

    # Core stack the project depends on.
    for name in (
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "tensordict",
        "torchrl",
        "gymnasium",
        "tqdm",
        "pytest",
    ):
        print(f"{name:<16} : {_version(name)}")

    print("-" * 60)

    # Hardware acceleration report. On this machine (Apple Silicon) CUDA is
    # unavailable; MPS is the relevant accelerator. On Linux/NVIDIA hosts CUDA
    # would report available.
    import torch  # noqa: PLC0415 - imported here so version check runs first

    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    print(f"CUDA available   : {cuda_ok}")
    if cuda_ok:
        print(f"CUDA device      : {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"MPS available    : {mps_ok}")

    if cuda_ok:
        device = "cuda"
    elif mps_ok:
        device = "mps"
    else:
        device = "cpu"
    print(f"Default device   : {device}")

    # Prove torchrl actually initializes (imports its compiled extensions).
    import torchrl  # noqa: PLC0415

    print("-" * 60)
    print(f"torchrl import OK (v{torchrl.__version__}); GPU accelerator: "
          f"{'yes' if (cuda_ok or mps_ok) else 'no (CPU only)'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
