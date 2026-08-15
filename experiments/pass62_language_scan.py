"""PASS 62.5 -- narrow language scan.

Four categories only, each with a deletion-or-one-word fix as the bar for editing:
signposting, vague attribution, stacked hedging, and ``-ing`` padding where a finite verb
is shorter.  Sentences carrying a numeric claim are excluded by rule, not by judgement.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"

MATH = [(r"\\\(", r"\\\)"), (r"\\\[", r"\\\]"),
        (r"\\begin\{equation\*?\}", r"\\end\{equation\*?\}"),
        (r"\\begin\{align\*?\}", r"\\end\{align\*?\}"),
        (r"\\begin\{tabular\}", r"\\end\{tabular\}")]

PATTERNS = {
    "signposting": [
        r"\bIn this (?:section|subsection|paper|paragraph),? we (?:will |shall )?(?:show|describe|discuss|present|explain|consider|turn)\b",
        r"\bWe (?:will |shall )?now (?:turn|describe|discuss|present|show|consider)\b",
        r"\bThe (?:following|next) (?:section|subsection|paragraph) (?:will )?(?:describes?|discusses?|presents?|shows?)\b",
        r"\bIt is (?:worth|important) (?:noting|to note) that\b",
        r"\bIn what follows,? we\b",
    ],
    "vague_attribution": [
        r"\bit is (?:well[- ])?known that\b",
        r"\bit is (?:widely |generally |commonly )?(?:believed|thought|accepted|understood) that\b",
        r"\bit has been (?:suggested|argued|noted|observed|shown) that\b",
        r"\bsome (?:have )?(?:argue|argued|claim|claimed|suggest|suggested)\b",
        r"\b(?:many|most) (?:authors|researchers|workers) \w+\b",
        r"\bin the literature,? it\b",
    ],
    "stacked_hedging": [
        r"\b(?:may|might|could|can) (?:possibly|perhaps|potentially|conceivably)\b",
        r"\bappears? to (?:suggest|indicate|imply)\b",
        r"\bseems? to (?:suggest|indicate|imply)\b",
        r"\b(?:somewhat|rather|fairly|quite) (?:likely|possibly|probably)\b",
        r"\bit (?:may|might|could) be (?:the case )?that\b",
        r"\bwe (?:believe|think) (?:it (?:is )?)?(?:may|might|could)\b",
    ],
    "ing_padding": [
        r"\bis (?:currently )?(?:serving|helping|allowing|enabling|providing) to\b",
        r"\bin order to be able to\b",
        r"\bwith the aim of \w+ing\b",
        r"\bfor the purpose of \w+ing\b",
        r"\bhas the effect of \w+ing\b",
        r"\bplays? an? (?:important|key|crucial|significant) role in\b",
    ],
}

NUM = re.compile(r"(?<![\w.\\])(\d+(?:[.,]\d+)*)(?![\w])")


def masked(text: str) -> list[tuple[int, int]]:
    spans = []
    for op, cl in MATH:
        for m in re.finditer(op, text):
            e = re.search(cl, text[m.start():])
            if e:
                spans.append((m.start(), m.start() + e.end()))
    for m in re.finditer(r"(?<!\\)\$[^$\n]*\$", text):
        spans.append((m.start(), m.end()))
    bib = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", text, re.S)
    if bib:
        spans.append((bib.start(), bib.end()))
    return spans


def sentence(text: str, pos: int) -> tuple[str, int, int]:
    start = 0
    for m in re.finditer(r"(?<![A-Z])[.!?]\s+|\n\n|\\par\b", text[:pos]):
        start = m.end()
    t = re.search(r"(?<![A-Z])[.!?](?:\s|$)|\n\n|\\par\b", text[pos:])
    end = pos + (t.end() if t else 160)
    return text[start:end], start, end


def scan(path: Path) -> list[dict]:
    text = path.read_text()
    spans = masked(text)
    found = []
    for cat, pats in PATTERNS.items():
        for p in pats:
            for m in re.finditer(p, text, re.I):
                if any(a <= m.start() < b for a, b in spans):
                    continue
                sent, s, e = sentence(text, m.start())
                flat = " ".join(sent.split())
                found.append({
                    "file": path.name,
                    "line": text[:m.start()].count("\n") + 1,
                    "category": cat,
                    "match": m.group(0),
                    "sentence": flat,
                    # 62.5d: a sentence carrying a numeric claim is out of bounds
                    "has_number": bool(NUM.search(re.sub(r"\\(section|subsection|ref|cite)\{[^}]*\}", "", sent))),
                })
    return found


def main() -> int:
    all_found = []
    for name in ("paper.tex", "supplementary.tex"):
        all_found.extend(scan(PAPER / name))

    print(f"candidate instances: {len(all_found)}")
    by_cat: dict[str, int] = {}
    for f in all_found:
        by_cat[f["category"]] = by_cat.get(f["category"], 0) + 1
    print(f"by category: {by_cat or '(none)'}")
    editable = [f for f in all_found if not f["has_number"]]
    print(f"eligible to edit (no numeric claim in sentence): {len(editable)}")
    print(f"excluded by the numeric-claim rule: {len(all_found) - len(editable)}")

    for i, f in enumerate(all_found, 1):
        flag = "EDITABLE" if not f["has_number"] else "EXCLUDED (numeric claim)"
        print(f"\n[{i:2d}] {f['file']}:{f['line']}  {f['category']}  '{f['match']}'  -- {flag}")
        print(f"     {f['sentence'][:300]}")

    out = PAPER.parent / "results" / "pass62_language_scan.json"
    out.write_text(json.dumps({"count": len(all_found), "by_category": by_cat,
                               "instances": all_found}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
