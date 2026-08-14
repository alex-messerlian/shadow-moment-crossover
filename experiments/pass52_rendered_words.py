"""PASS 52: rendered word counts per section and subsection, anchored on the PDF bookmarks.

    PYTHONPATH=. .venv/bin/python experiments/pass52_rendered_words.py [label]

The pass targets are quoted in rendered words -- what a reader meets, including captions, table
bodies and math read aloud -- while edits are located by source-prose words.  The two differ by
roughly 15%, so both are needed and neither substitutes for the other.

Splitting on hyperref's own bookmark titles is what makes this reliable: matching rendered
headings by their LaTeX titles fails on any heading carrying math or \\texorpdfstring.

Writes ``results/pass52_rendered_words_<label>.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
PDF = REPO / "paper" / "paper.pdf"


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "now"
    with fitz.open(PDF) as d:
        flat = re.sub(r"\s+", " ", "\n".join(p.get_text() for p in d))
        toc = d.get_toc()
        n_pages = d.page_count

    # locate each bookmark title in the rendered flow, scanning forward so repeated
    # words in a title cannot match an earlier occurrence
    marks, cur = [], 0
    for level, title, page in toc:
        if level == 1:
            continue
        t = re.sub(r"\s+", " ", title).strip()
        i = flat.upper().find(t.upper(), cur)
        if i < 0:                                   # title broken by a line-hyphen
            head = " ".join(t.split()[:3]).upper()
            i = flat.upper().find(head, cur)
        if i >= 0:
            cur = i + len(t)
            marks.append({"level": level, "title": t, "page": page, "at": i})

    out, total = [], 0
    for m, nxt in zip(marks, marks[1:] + [{"at": len(flat)}]):
        w = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", flat[m["at"]: nxt["at"]]))
        out.append({**m, "words": w})
        total += w

    print(f"{'unit':52s} {'p':>4s} {'words':>7s}")
    sec_total, sec_name = 0, None
    body = 0
    for r in out:
        pad = "" if r["level"] == 2 else "    "
        print(f"  {pad}{r['title'][:48 - len(pad)]:{48 - len(pad)}s} {r['page']:>4d} {r['words']:>7d}")
    # roll up
    roll, cur_sec = [], None
    for r in out:
        if r["level"] == 2:
            cur_sec = {"title": r["title"], "words": r["words"], "subs": []}
            roll.append(cur_sec)
        elif cur_sec:
            cur_sec["words"] += r["words"]
            cur_sec["subs"].append({"title": r["title"], "words": r["words"]})
    print(f"\n{'rolled up':52s} {'words':>12s}")
    for s in roll:
        print(f"  {s['title'][:50]:52s} {s['words']:>12d}")
    # Everything from Acknowledgements on is back matter.  Identifying it by title prefix
    # broke the moment an appendix was renamed, so cut on position instead.
    titles = [s["title"] for s in roll]
    stop = titles.index("Acknowledgements") if "Acknowledgements" in titles else len(titles)
    NUMBERED = roll[:stop]
    print(f"\n  pages {n_pages}   body (abstract + numbered sections) "
          f"{sum(s['words'] for s in NUMBERED)}   whole document {total}")

    p = REPO / "results" / f"pass52_rendered_words_{label}.json"
    p.write_text(json.dumps({"pages": n_pages, "units": out, "rolled": roll,
                             "body_words": sum(s["words"] for s in NUMBERED),
                             "document_words": total}, indent=1))
    print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
