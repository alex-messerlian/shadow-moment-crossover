"""PASS 51: extract the rendered manuscript section by section, for reading.

    PYTHONPATH=. .venv/bin/python experiments/pass51_read_rendered.py [section]

Sections 6 through 10 and the appendices moved as verbatim blocks during the restructure and
have never been read as rendered output.  That is exactly what was believed about Section 3.5
before the compile revealed a seam in it, so this pass reads them.

With no argument it prints an index and the mechanical checks: every cross-reference in each
section with the section it resolves to, every citation, and the label/number map.  With a
section name it prints that section's rendered prose in full.

Writes ``results/pass51_read_index.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
PDF = REPO / "paper" / "paper.pdf"
AUX = REPO / "paper" / "paper.aux"
OUT = REPO / "results" / "pass51_read_index.json"

# (key, rendered heading as it appears in the extracted text)
MARKS = [
    ("1", "1.\nINTRODUCTION"),
    ("2", "2.\nSETTING AND ESTIMATORS"),
    ("3", "3.\nTHE PROJECTION VARIANCES"),
    ("4", "4.\nSTATEWISE VALIDATION"),
    ("5", "5.\nESTIMATING THE THRESHOLD FROM A PILOT BUDGET"),
    ("6", "6.\nTHE COLLECTIVE ROUTE: TWO EXACT BIAS LAWS"),
    ("7", "7.\nTHE CROSSOVER"),
    ("8", "8.\nRELATED WORK"),
    ("9", "9.\nLIMITATIONS"),
    ("10", "10.\nCONCLUSION"),
    ("ack", "ACKNOWLEDGEMENTS"),
    ("methods", "Methods, data, and code"),
    ("appA", "Appendix A: The exact fourth-moment"),
    ("appB", "Appendix B: The destructive SWAP"),
    ("appC", "Appendix C: Reproducibility and statistics"),
]


def load_text() -> tuple[list[str], str]:
    with fitz.open(PDF) as d:
        pages = [p.get_text() for p in d]
    return pages, "\n".join(pages)


def label_map() -> dict:
    """label -> (rendered number, page) from the .aux."""
    aux = AUX.read_text(errors="replace")
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux):
        out[m.group(1)] = {"number": m.group(2), "page": int(m.group(3))}
    return out


def split_sections(whole: str) -> dict:
    idx = {}
    for key, mark in MARKS:
        i = whole.find(mark)
        idx[key] = i
    order = [k for k, _ in MARKS if idx[k] >= 0]
    out = {}
    for a, b in zip(order, order[1:] + [None]):
        out[a] = whole[idx[a]: idx[b] if b else len(whole)]
    return out


def page_of(pages: list[str], needle: str) -> int | None:
    n = re.sub(r"\s+", " ", needle)
    for i, t in enumerate(pages, 1):
        if n in re.sub(r"\s+", " ", t):
            return i
    return None


def main() -> None:
    pages, whole = load_text()
    secs = split_sections(whole)
    labels = label_map()

    if len(sys.argv) > 1:
        key = sys.argv[1]
        body = secs.get(key)
        if body is None:
            print(f"unknown section {key!r}; known: {list(secs)}")
            sys.exit(1)
        print(re.sub(r"\s+", " ", body))
        return

    report = {"sections": {}, "labels": labels}
    print(f"{'sec':8s} {'words':>6s} {'pages':>10s}  cross-references it makes")
    for key, body in secs.items():
        flat = re.sub(r"\s+", " ", body)
        words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", flat))
        refs = sorted(set(re.findall(r"Sections?\s+(\d+(?:\.\d+)?)", flat)))
        figs = sorted(set(re.findall(r"Figs?\.\s*(\d+)", flat)), key=int)
        tabs = sorted(set(re.findall(r"Table\s+([IVX]+)", flat)))
        apps = sorted(set(re.findall(r"Appendix~?\s*([ABC])", flat)))
        cites = sorted(set(re.findall(r"\[(\d+(?:,\s*\d+)*)\]", flat)))
        first = page_of(pages, body[:70])
        report["sections"][key] = {"words": words, "first_page": first,
                                   "section_refs": refs, "figure_refs": figs,
                                   "table_refs": tabs, "appendix_refs": apps,
                                   "citation_groups": cites[:24]}
        print(f"  {key:6s} {words:>6d} {str(first):>10s}  secs={refs} figs={figs} "
              f"tabs={tabs} apps={apps}")

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
