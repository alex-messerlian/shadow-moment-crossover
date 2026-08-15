"""PASS 60.3 -- locate every em-dash and en-dash used as punctuation.

A dash counts as punctuation when it is not a numeric range, not inside math, and
not inside the bibliography.  The detector reports the enclosing sentence so each
instance can be judged on what the sentence actually wants in the dash's place.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"

# Regions whose dashes are never prose punctuation.
MATH_PATTERNS = [
    (r"\\\(", r"\\\)"),
    (r"\\\[", r"\\\]"),
    (r"\\begin\{equation\*?\}", r"\\end\{equation\*?\}"),
    (r"\\begin\{align\*?\}", r"\\end\{align\*?\}"),
    (r"\\begin\{gather\*?\}", r"\\end\{gather\*?\}"),
]

DASH = re.compile(r"---|--|—|–")
RANGE = re.compile(r"\d\s*(?:---|--|—|–)\s*\d")


def masked_spans(text: str) -> list[tuple[int, int]]:
    """Character spans covering math, verbatim $...$, and the bibliography."""
    spans: list[tuple[int, int]] = []

    for op, cl in MATH_PATTERNS:
        for m in re.finditer(op, text):
            end = re.search(cl, text[m.start():])
            if end:
                spans.append((m.start(), m.start() + end.end()))

    # inline $...$ (non-greedy, single line)
    for m in re.finditer(r"(?<!\\)\$[^$\n]*\$", text):
        spans.append((m.start(), m.end()))

    bib = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", text, re.S)
    if bib:
        spans.append((bib.start(), bib.end()))

    return spans


def in_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def sentence_around(text: str, pos: int) -> str:
    """The sentence containing pos, bounded by sentence-final punctuation or blank lines."""
    start = 0
    for m in re.finditer(r"(?<![A-Z])[.!?]\s+|\n\n|\\par\b", text[:pos]):
        start = m.end()
    tail = re.search(r"(?<![A-Z])[.!?](?:\s|$)|\n\n|\\par\b", text[pos:])
    end = pos + (tail.end() if tail else 120)
    return " ".join(text[start:end].split())


# A range may be written bare (293--325) or with the numbers in math delimiters
# (\(2000\)--\(128{,}000\) or $2000$--$128000$).  None of these is punctuation.
RANGE_LEFT = re.compile(r"(?:\d|\\\)|\$)\s*$")
RANGE_RIGHT = re.compile(r"^\s*(?:\d|\\\(|\$)")
# Compound proper names take an en-dash by correct typography, not as punctuation.
PROPER_NAME = re.compile(r"[A-Z][a-z]+$")


def sweep(path: Path) -> list[dict]:
    text = path.read_text()
    spans = masked_spans(text)
    found = []
    for m in DASH.finditer(text):
        pos = m.start()
        if in_span(pos, spans):
            continue
        left = text[max(0, pos - 12):pos]
        right = text[m.end():m.end() + 12]
        if RANGE_LEFT.search(left) and RANGE_RIGHT.match(right):
            continue
        if PROPER_NAME.search(left) and re.match(r"^[A-Z][a-z]", right):
            continue
        line = text[:pos].count("\n") + 1
        found.append({
            "file": path.name,
            "line": line,
            "dash": m.group(0),
            "pos": pos,
            "sentence": sentence_around(text, pos),
        })
    return found


def main() -> int:
    all_found: list[dict] = []
    for name in ("paper.tex", "supplementary.tex"):
        all_found.extend(sweep(PAPER / name))

    by_kind: dict[str, int] = {}
    for f in all_found:
        by_kind[f["dash"]] = by_kind.get(f["dash"], 0) + 1

    # Group into edit sites: dashes sharing a sentence are one parenthetical.
    sites: dict[tuple[str, str], list[dict]] = {}
    for f in all_found:
        sites.setdefault((f["file"], f["sentence"]), []).append(f)

    print(f"punctuation dashes found: {len(all_found)}")
    print(f"by kind: {by_kind}")
    print(f"edit sites (sentences): {len(sites)}")
    paired = sum(1 for v in sites.values() if len(v) == 2)
    print(f"  paired (parenthetical): {paired}   single: {len(sites) - paired}")
    for i, ((fn, sent), group) in enumerate(sites.items(), 1):
        lines = ",".join(str(g["line"]) for g in group)
        print(f"\n[{i:2d}] {fn}:{lines}  n={len(group)}")
        print(f"     {sent[:340]}")

    out = PAPER.parent / "results" / "pass60_dash_sweep.json"
    out.write_text(json.dumps({"count": len(all_found), "by_kind": by_kind,
                               "instances": all_found}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
