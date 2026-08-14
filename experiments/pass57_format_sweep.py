r"""PASS 57.2: audit what a document-class change can break without touching a number.

    PYTHONPATH=. .venv/bin/python experiments/pass57_format_sweep.py

PASS 55 converted REVTeX -> quantumarticle and verified page counts, figures, references,
every numeric token and every claim.  All of those passed while two pointers in Section 3.5
silently broke, because REVTeX letters ``\paragraph`` run-in headings ("a.", "b.") and
quantumarticle does not.  This checks the formatting a numeric gate cannot see:

  (a) run-in headings, and whether any prose names one by a marker the class does not print;
  (b) positional references -- "the table above", "the display below", "item (e)";
  (c) enumerated and lettered lists, and whether the text assumes markers the class renders;
  (d) float drift, so a caption or body reference by position is still true;
  (e) footnotes, placed differently by the two classes;
  (f) bibliography fields, where the prose names a source inline.

Writes ``results/pass57_format_sweep.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
TEX = REPO / "paper" / "paper.tex"
PDF = REPO / "paper" / "paper.pdf"
OUT = REPO / "results" / "pass57_format_sweep.json"


def main() -> None:
    src = TEX.read_text()
    with fitz.open(PDF) as d:
        pages = [re.sub(r"\s+", " ", p.get_text()) for p in d]
        toc = d.get_toc()
    flat = " ".join(pages)
    rep: dict = {}

    # (a) run-in headings: does each render, and with what marker in front of it?
    heads = re.findall(r"\\(?:sub)?paragraph\{([^}]*)\}", src)
    hr = []
    for h in heads:
        txt = re.sub(r"\\[a-zA-Z]+|[{}\\$]", "", h).strip().rstrip(".")
        key = re.sub(r"[^A-Za-z ]", "", txt)[:38]
        hay = re.sub(r"[^A-Za-z ]", "", flat)
        i = hay.find(key)
        i = -1 if i < 0 else i
        raw = flat.find(txt[:18].replace("'", "\u2019"))
        if raw < 0: raw = flat.find(txt[:18])
        before = flat[max(0, raw - 16):raw].strip() if raw >= 0 else ""
        hr.append({"heading": txt[:56], "rendered": i >= 0,
                   "preceding_chars": before[-12:],
                   "lettered": bool(re.search(r"(?:^|\s)[a-z]\.\s*$", before))})
    rep["run_in_headings"] = hr
    print(f"(a) run-in headings: {len(hr)} declared, {sum(h['rendered'] for h in hr)} rendered, "
          f"{sum(h['lettered'] for h in hr)} carry a letter marker")
    missing = [h["heading"] for h in hr if not h["rendered"]]
    if missing:
        print(f"    NOT RENDERED: {missing}")

    # (b) positional and marker-based references in the prose
    PAT = {
        "item (x)": r"item\s*\([a-z]\)",
        "lettered blocks": r"lettered\s+(?:block|item|paragraph)",
        "the table above/below": r"the table (?:above|below)",
        "the display above/below": r"the display (?:above|below)",
        "rows above/below": r"rows? (?:above|below)",
        "the paragraph above/below": r"paragraph (?:above|below)",
        "the figure above/below": r"figure (?:above|below)",
        "as shown above": r"(?:as )?(?:shown|noted|stated) above",
        "the equation above/below": r"equation (?:above|below)",
    }
    pos = {}
    for lbl, p in PAT.items():
        hits = [m.group(0) for m in re.finditer(p, flat, re.I)]
        pos[lbl] = hits
    rep["positional_references"] = pos
    print("\n(b) positional references in the rendered text")
    for lbl, hits in pos.items():
        if hits:
            print(f"    {lbl:26s} {len(hits)}x  {hits[:4]}")
    if not any(pos.values()):
        print("    none")

    # (c) lists
    envs = re.findall(r"\\begin\{(itemize|enumerate|description)\}", src)
    rep["list_envs"] = envs
    print(f"\n(c) list environments: {envs or 'none'}")
    if "enumerate" in envs:
        print("    NOTE: enumerate markers differ between classes; check any text naming an item")

    # (d) float drift, against the .aux label map
    aux = (REPO / "paper" / "paper.aux")
    drift = []
    if aux.exists():
        lab = {m.group(1): (m.group(2), int(m.group(3))) for m in
               re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux.read_text(errors="replace"))}
        for k, (num, pg) in sorted(lab.items()):
            if not k.startswith(("fig:", "tab:")):
                continue
            kind = "Figure" if k.startswith("fig:") else "Table"
            pat = (rf"(?:Figure|Figs?\.)\s*{num}\b|Figs?\.\s*\d+\s*and\s*{num}\b"
               if kind == "Figure" else rf"Table\s*{num}\b")
            calls = [i + 1 for i, t in enumerate(pages) if re.search(pat, t) and i + 1 != pg]
            first = min(calls) if calls else None
            drift.append({"label": k, "number": num, "lands_page": pg,
                          "first_call_page": first,
                          "drift": (pg - first) if first else 0})
    rep["float_drift"] = drift
    print("\n(d) float placement")
    for f in drift:
        flag = "  <- backward" if f["drift"] < -1 else ""
        print(f"    {f['label']:16s} {f['number']:>3s}  call p{f['first_call_page']}  "
              f"lands p{f['lands_page']}  drift {f['drift']}{flag}")

    # (e) footnotes
    fn = re.findall(r"\\footnote\{", src)
    rep["footnotes"] = len(fn)
    print(f"\n(e) footnotes in source: {len(fn)}")

    # (f) bibliography: does every rendered entry carry the fields the prose assumes?
    bib = re.findall(r"\\bibitem\{([^}]+)\}", src)
    nums = re.findall(r"\[(\d+)\]\s+[A-Z]", flat)
    rep["bibitems"] = len(bib)
    rep["rendered_entry_markers"] = len(set(nums))
    inline = re.findall(r"([A-Z][a-z]+(?:\s+et\s+al\.|\s+and\s+[A-Z][a-z]+)?)\s*\[(\d+)\]", flat)
    rep["inline_named_citations"] = len(inline)
    print(f"\n(f) bibliography: {len(bib)} bibitems, {len(set(nums))} distinct rendered markers, "
          f"{len(inline)} inline named citations")

    OUT.write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
