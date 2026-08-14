"""PASS 50.2: compile both documents and report everything the engine says.

    PYTHONPATH=. .venv/bin/python experiments/pass50_compile_report.py

The manuscript went through a full restructure without ever being compiled, so this is the
first render of the current source.  Reports page counts, every error, every warning, undefined
references and citations, overfull/underfull boxes above a threshold, and where each float
actually landed relative to the page that calls it.

Float placement is measured, not guessed: the call site is located by searching the rendered
text for the ``Fig.~\\ref``/``Table~\\ref`` mention, and the float itself by its caption's
opening words, both via PyMuPDF.

Writes ``results/pass50_compile_report.json``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO / "paper"
OUT = REPO / "results" / "pass50_compile_report.json"

BOX_THRESHOLD_PT = 5.0


def compile_doc(stem: str, passes: int) -> dict:
    """Compile with tectonic; it reruns internally, so `passes` is a belt-and-braces loop."""
    logs = []
    for i in range(passes):
        r = subprocess.run(
            ["tectonic", "-X", "compile", f"{stem}.tex", "--keep-logs", "--keep-intermediates"],
            cwd=PAPER_DIR, capture_output=True, text=True)
        logs.append({"pass": i + 1, "returncode": r.returncode,
                     "stderr_tail": r.stderr.strip().split("\n")[-40:]})
        if r.returncode != 0:
            break
    return {"stem": stem, "passes_run": len(logs), "runs": logs,
            "final_returncode": logs[-1]["returncode"]}


def parse_log(stem: str) -> dict:
    log = (PAPER_DIR / f"{stem}.log").read_text(errors="replace")
    errors = re.findall(r"(?m)^! (.+)$", log)
    undef_ref = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
    undef_cite = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", log)))
    multiply = sorted(set(re.findall(r"Label `([^']+)' multiply defined", log)))

    boxes = []
    for m in re.finditer(
            r"(Overfull|Underfull) \\([hv])box \(((?:\d+\.?\d*)pt too (?:wide|high)|badness \d+)\)"
            r"[^\n]*?(?:at lines (\d+)--(\d+)|has occurred while \\output is active)", log):
        kind, hv, detail, l1, l2 = m.groups()
        pt = None
        mm = re.match(r"([\d.]+)pt", detail)
        if mm:
            pt = float(mm.group(1))
        boxes.append({"kind": kind, "box": hv, "detail": detail, "pt": pt,
                      "lines": [int(l1), int(l2)] if l1 else None})
    big = [b for b in boxes if b["pt"] is not None and b["pt"] > BOX_THRESHOLD_PT]

    pages = None
    mp = re.search(r"Output written on \S+ \((\d+) pages", log)
    if mp:
        pages = int(mp.group(1))
    return {"n_errors": len(errors), "errors": errors[:20],
            "undefined_references": undef_ref, "undefined_citations": undef_cite,
            "multiply_defined_labels": multiply,
            "n_boxes_total": len(boxes),
            "boxes_over_threshold": big,
            "n_underfull": sum(1 for b in boxes if b["kind"] == "Underfull"),
            "pages_from_log": pages}


def doc_text(pdf: Path) -> list[str]:
    with fitz.open(pdf) as d:
        return [p.get_text() for p in d]


def float_placement(pages: list[str], items: list[tuple[str, str, str]]) -> list[dict]:
    """For each (name, call_pattern, caption_opening) find the calling page and the float page."""
    out = []
    for name, call_pat, cap in items:
        call_pages = [i + 1 for i, t in enumerate(pages)
                      if re.search(call_pat, re.sub(r"\s+", " ", t))]
        cap_norm = re.sub(r"\s+", " ", cap)
        float_pages = [i + 1 for i, t in enumerate(pages)
                       if cap_norm in re.sub(r"\s+", " ", t)]
        first_call = min(call_pages) if call_pages else None
        landed = min(float_pages) if float_pages else None
        out.append({"item": name, "first_call_page": first_call, "float_page": landed,
                    "drift_pages": (landed - first_call) if (landed and first_call) else None,
                    "all_call_pages": call_pages[:6], "all_float_pages": float_pages[:4]})
    return out


FLOATS = [
    ("Fig 1 crossover map", r"Fig\. 1|Figure 1", "The crossover for purity"),
    ("Fig 2 boundary", r"Fig\. 2|Figure 2", "Crossover validation across"),
    ("Fig 3 alpha", r"Fig\. 3|Figure 3", "The effective budget-scaling exponent"),
    ("Fig 4 out-of-ensemble", r"Fig\. 4|Figure 4", "Out-of-ensemble validation"),
    ("Fig 5 wall", r"Fig\. 5|Figure 5", "Single-copy error growth"),
    ("Fig 6 weight truncation", r"Fig\. 6|Figure 6", "The two projection variances read"),
    ("Fig 7 statewise", r"Fig\. 7|Figure 7", "Statewise validation. Predicted versus"),
    ("Fig 8 pilot", r"Fig\. 8|Figure 8", "Estimating the threshold from data"),
    ("Table I pilot", r"Table I\b|TABLE I\b", "Pilot cost against the threshold"),
]


def main() -> None:
    report = {}
    for stem, npass in (("paper", 3), ("supplementary", 2)):
        print(f"=== compiling {stem}.tex ({npass} passes) ===")
        c = compile_doc(stem, npass)
        p = parse_log(stem)
        pdf = PAPER_DIR / f"{stem}.pdf"
        with fitz.open(pdf) as d:
            n_pages = d.page_count
        report[stem] = {**c, **p, "pdf_pages": n_pages,
                        "pdf_bytes": pdf.stat().st_size}
        print(f"  returncode {c['final_returncode']}, {n_pages} pages, "
              f"{p['n_errors']} errors, {len(p['undefined_references'])} undefined refs, "
              f"{len(p['undefined_citations'])} undefined citations, "
              f"{len(p['boxes_over_threshold'])} boxes over {BOX_THRESHOLD_PT}pt")

    pages = doc_text(PAPER_DIR / "paper.pdf")
    report["paper"]["float_placement"] = float_placement(pages, FLOATS)

    # rendered front matter and section numbering
    head = re.sub(r"\s+", " ", pages[0])
    report["front_matter"] = {
        "page1_first_600_chars": head[:600],
        "authors_on_paper": sorted(set(re.findall(r"(Alexander Messerlian|Ziwei Gu)", head))),
    }
    supp_pages = doc_text(PAPER_DIR / "supplementary.pdf")
    supp_head = re.sub(r"\s+", " ", supp_pages[0])
    report["front_matter"]["authors_on_supplementary"] = sorted(
        set(re.findall(r"(Alexander Messerlian|Ziwei Gu)", supp_head)))
    report["front_matter"]["supp_page1_first_400"] = supp_head[:400]

    whole = "\n".join(pages)
    secs = re.findall(r"(?m)^\s*(\d{1,2})\.\s+([A-Z][^\n]{3,70})$", whole)
    report["section_headings_rendered"] = [f"{a}. {b.strip()}" for a, b in secs][:24]

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    if report["paper"]["final_returncode"] != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
