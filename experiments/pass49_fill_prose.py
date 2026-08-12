"""PASS 49.4/49.5: fill the restructure placeholders with the hand-written prose.

    PYTHONPATH=. .venv/bin/python experiments/pass49_fill_prose.py

``pass49_restructure.py`` left ten ``%%PASS49_*%%`` markers where new or rewritten prose goes.
This substitutes each one exactly once from ``paper/sections/``, rewrites the Data-and-code
section into a Methods section that the AI-use statement can point at, and asserts that no
marker survives.  Idempotent by construction: it refuses to run if any marker is already gone.

Writes ``paper/paper.tex`` in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper" / "paper.tex"
SEC = REPO / "paper" / "sections"

MARKERS = {
    "%%PASS49_TITLE%%": "title.tex",
    "%%PASS49_ABSTRACT%%": "abstract.tex",
    "%%PASS49_INTRO%%": "intro.tex",
    "%%PASS49_COMPILATION%%": "sec2_6_compilation.tex",
    "%%PASS49_S3_PROMOTED%%": "sec3_5_promoted.tex",
    "%%PASS49_S3_ALGORITHM%%": "sec3_6_evaluation.tex",
    "%%PASS49_STATEWISE%%": "sec4_statewise.tex",
    "%%PASS49_PILOT%%": "sec5_pilot.tex",
    "%%PASS49_CONCLUSION%%": "conclusion.tex",
    "%%PASS49_ACK%%": "acknowledgements.tex",
}

OLD_DATACODE = """\\hypertarget{data-and-code-availability}{%
\\section*{Data and code
availability}\\label{data-and-code-availability}}

All code, raw measurement counts, and analysis are available at
\\url{https://github.com/alex-messerlian/shadow-moment-crossover}. Raw hardware counts were committed to version
control before any analysis was performed, and every locked prediction
was committed before the corresponding measurement was submitted."""

NEW_METHODS = """\\hypertarget{methods-data-and-code}{%
\\section*{Methods, data, and code}\\label{sec:methods}}

Every number in this paper is produced by a script in the public repository at
\\url{https://github.com/alex-messerlian/shadow-moment-crossover}, which also holds the
generated artifacts each figure and table is built from.

The computational work has three layers. The exact evaluators are sampling-free: the statewise
projection variances of Section~\\ref{sec:zeta-closed-form} are contractions of the
second-moment identity computed in exact or double-precision arithmetic, with the
ensemble-averaged closed forms carried in exact rational arithmetic so no size overflows. The
estimators of Section~\\ref{sec:pilot} and the crossover sweeps are Monte Carlo over simulated
local-shadow snapshots, with all randomness drawn from seeded, value-based generators, so every
figure regenerates bit-for-bit from the committed seeds. The verification layer is a test suite
that locks each derivation as an executable check --- the identities against independent
brute-force computations, the closed forms against their Haar averages, the estimators against
their exact targets, and the chunked implementations against their unchunked values.

Simulations used Python 3.12 with NumPy 2.5, SciPy 1.18 and Matplotlib 3.11; the transpiled gate
counts of Section~\\ref{sec:compilation} used Qiskit 2.5 against a coupling map carried as
committed package data, so they reproduce offline with no credentials. Sizes reported as feasible
were timed on a single core."""

REF_FIXES = [
    ("Section~\\ref{sec:where-useless}",
     "Section~\\ref{where-single-copy-estimation-stops-being-useful}"),
    ("Section~\\ref{sec:crossover-validation}",
     "Section~\\ref{validation-on-83-cells-and-four-ensembles}"),
]


def die(m: str) -> None:
    print(f"ABORT: {m}")
    sys.exit(1)


def main() -> None:
    text = PAPER.read_text()
    missing = [m for m in MARKERS if m not in text]
    if missing:
        die(f"markers already filled or absent: {missing}. This script runs once.")

    for marker, fname in MARKERS.items():
        p = SEC / fname
        if not p.exists():
            die(f"missing section file {p}")
        body = p.read_text().rstrip("\n")
        if text.count(marker) != 1:
            die(f"{marker} appears {text.count(marker)} times")
        text = text.replace(marker, body)
        print(f"  filled {marker:28s} <- paper/sections/{fname} "
              f"({len(body.split(chr(10)))} lines)")

    if text.count(OLD_DATACODE) != 1:
        die("the Data-and-code block did not match verbatim")
    text = text.replace(OLD_DATACODE, NEW_METHODS)
    print("  rewrote Data-and-code -> Methods, data, and code (labelled sec:methods)")

    for old, new in REF_FIXES:
        if old in text:
            text = text.replace(old, new)
            print(f"  repointed {old} -> {new}")

    left = [m for m in MARKERS if m in text]
    if left:
        die(f"markers survived: {left}")
    PAPER.write_text(text)
    print(f"\nwrote {PAPER.relative_to(REPO)} ({len(text.split(chr(10)))} lines); no markers remain")


if __name__ == "__main__":
    main()
