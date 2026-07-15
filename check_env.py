"""Environment verification for shadow-moment-crossover.

Imports the project's dependency stack and prints each package's version, so a
reader can confirm a working environment. Run inside the project venv:

    .venv/bin/python check_env.py
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
    print("shadow-moment-crossover — environment check")
    print("=" * 60)

    print(f"Python           : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform         : {platform.platform()}")
    print(f"Machine          : {platform.machine()}")
    print("-" * 60)

    # The stack the project depends on.
    for name in (
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pytest",
    ):
        print(f"{name:<16} : {_version(name)}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
