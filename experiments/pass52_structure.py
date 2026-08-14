"""PASS 52.1: the manuscript's structure with measured word counts, from the source.

    PYTHONPATH=. .venv/bin/python experiments/pass52_structure.py [--dump SECTION]

Counts prose words only: display math, tabular bodies, figure/table environments and
LaTeX control sequences are stripped before counting, so the numbers are what a reader
reads rather than what the file holds.  Line spans are printed so an edit can be located.

Writes ``results/pass52_structure.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEX = REPO / "paper" / "paper.tex"
OUT = REPO / "results" / "pass52_structure.json"

STRIP_ENVS = ("figure", "table", "tabular", "equation", "align", "center", "widetext")


def prose(chunk: str) -> str:
    s = chunk
    for env in STRIP_ENVS:
        s = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", " ", s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)          # display math
    s = re.sub(r"\\\(.*?\\\)", " X ", s, flags=re.S)        # inline math -> one token
    s = re.sub(r"\$[^$]*\$", " X ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)                  # control sequences
    s = re.sub(r"[{}\\~&%^_]", " ", s)
    return s


def words(chunk: str) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", prose(chunk)))


def main() -> None:
    src = TEX.read_text()
    lines = src.split("\n")

    heads = []
    for i, ln in enumerate(lines):
        m = re.match(r"\\(section|subsection)\*?\{(.*)", ln)
        if m:
            # titles wrap; join until braces balance
            title, j = m.group(2), i
            while title.count("{") + 1 > title.count("}") and j + 1 < len(lines):
                j += 1
                title += " " + lines[j]
            title = re.sub(r"\\texorpdfstring\{", "", title)
            title = re.sub(r"\\label\{[^}]*\}\}?\s*$", "", title).strip()
            title = re.sub(r"[{}]", "", title).strip().rstrip("}")
            heads.append({"kind": m.group(1), "line": i + 1, "title": " ".join(title.split())})

    for h, nxt in zip(heads, heads[1:] + [{"line": len(lines) + 1}]):
        h["end_line"] = nxt["line"] - 1
        h["words"] = words("\n".join(lines[h["line"]: nxt["line"] - 1]))

    # roll subsection words into their parent section
    secs = []
    for h in heads:
        if h["kind"] == "section":
            secs.append({**h, "subs": []})
        elif secs:
            secs[-1]["subs"].append(h)
    for s in secs:
        s["total_words"] = s["words"] + sum(x["words"] for x in s["subs"])
        s["end_line"] = s["subs"][-1]["end_line"] if s["subs"] else s["end_line"]

    if "--dump" in sys.argv:
        want = sys.argv[sys.argv.index("--dump") + 1].lower()
        for s in secs:
            if want in s["title"].lower():
                print("\n".join(lines[s["line"] - 1: s["end_line"]]))
        return

    print(f"{'section':52s} {'lines':>13s} {'lead':>6s} {'total':>7s}")
    for i, s in enumerate(secs, 1):
        print(f"{i:2d}. {s['title'][:48]:48s} {s['line']:>6d}-{s['end_line']:<6d} "
              f"{s['words']:>6d} {s['total_words']:>7d}")
        for j, x in enumerate(s["subs"], 1):
            print(f"      {i}.{j} {x['title'][:42]:42s} {x['line']:>6d}-{x['end_line']:<6d} "
                  f"{x['words']:>6d}")
    print(f"\n  body total (numbered sections): "
          f"{sum(s['total_words'] for s in secs if not s['title'].startswith(('Ack','Meth')))}")

    OUT.write_text(json.dumps({"sections": secs}, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
