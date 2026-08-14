"""PASS 51.5: one consistency read over the whole rendered document.

    PYTHONPATH=. .venv/bin/python experiments/pass51_consistency.py

Four checks, all against ``paper/paper.pdf`` as rendered, not the source:

(a) claims that appear in more than one place, compared for number, scope and hedging;
(b) every number in the abstract, located in the body and compared;
(c) the AI-use disclosure in Methods against the one in Acknowledgements;
(d) word count by section and the page count.

Writes ``results/pass51_consistency.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
PDF = REPO / "paper" / "paper.pdf"
OUT = REPO / "results" / "pass51_consistency.json"

# Claims the restructure could plausibly have left stated two ways.  Each entry is a label and
# the list of anchor phrases whose surrounding sentence should agree.
REPEATED = [
    ("28/5 asymptotic base", ["28", "5.6"]),
    ("variable_rank rank correlation", ["+0.87"]),
    ("82 of 84 within 2 SE", ["82 of 84"]),
    ("pilot crossing size", ["7.95"]),
    ("n=8 tail exponent", ["1.14"]),
    ("statewise sequences exact", ["82 of the"]),
    ("all-cells within one", ["42"]),
]


def sentences(flat: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?]) +", flat) if s.strip()]


def main() -> None:
    with fitz.open(PDF) as d:
        pages = [p.get_text() for p in d]
        n_pages = d.page_count
    flat = re.sub(r"[ \t]+", " ", "\n".join(pages))
    one = re.sub(r"\s+", " ", flat)
    sents = sentences(one)

    report: dict = {"pages": n_pages}

    # ---- (d) section word counts
    MARKS = [("abstract", "Classical shadows"), ("1 Introduction", "1. INTRODUCTION"),
             ("2 Setting", "2. SETTING AND ESTIMATORS"),
             ("3 Projection variances", "3. THE PROJECTION VARIANCES"),
             ("4 Statewise", "4. STATEWISE VALIDATION"),
             ("5 Pilot", "5. ESTIMATING THE THRESHOLD"),
             ("6 Collective", "6. THE COLLECTIVE ROUTE"),
             ("7 Crossover", "7. THE CROSSOVER"),
             ("8 Related work", "8. RELATED WORK"),
             ("9 Limitations", "9. LIMITATIONS"),
             ("10 Conclusion", "10. CONCLUSION"),
             ("Acknowledgements", "ACKNOWLEDGEMENTS"),
             ("Methods", "METHODS, DATA, AND CODE"),
             ("References", "[1] Hsin-Yuan Huang"),
             ("Appendix A", "Appendix A:"),
             ("Appendix B", "Appendix B:"),
             ("Appendix C", "Appendix C:")]
    pos = [(lbl, one.find(re.sub(r"\s+", " ", m))) for lbl, m in MARKS]
    pos = [(l, i) for l, i in pos if i >= 0]
    counts = {}
    for (lbl, i), nxt in zip(pos, [p[1] for p in pos[1:]] + [len(one)]):
        counts[lbl] = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", one[i:nxt]))
    report["word_counts"] = counts
    report["body_words"] = sum(v for k, v in counts.items()
                               if k[0].isdigit() or k == "abstract")

    # ---- (b) every number in the abstract, found in the body
    a0 = one.find("Classical shadows")
    a1 = one.find("1. INTRODUCTION")
    abstract = one[a0:a1]
    body = one[a1:]
    nums = sorted(set(re.findall(r"(?<![\w.])(\d+(?:[.,]\d+)*(?:%)?)(?![\w])", abstract)))
    abs_check = []
    for v in nums:
        bare = v.rstrip("%")
        hits = len(re.findall(re.escape(v), body)) or len(re.findall(re.escape(bare), body))
        ctx = ""
        m = re.search(r"[^.]*" + re.escape(v) + r"[^.]*\.", abstract)
        if m:
            ctx = m.group(0).strip()[:120]
        abs_check.append({"value": v, "occurrences_in_body": hits,
                          "abstract_context": ctx, "ok": hits > 0})
    report["abstract_numbers"] = abs_check

    # ---- (a) repeated claims
    rep = []
    for label, anchors in REPEATED:
        hits = []
        for a in anchors:
            for s in sents:
                if a in s:
                    hits.append(s[:190])
        seen, uniq = set(), []
        for h in hits:
            k = re.sub(r"\W", "", h)[:60]
            if k not in seen:
                seen.add(k)
                uniq.append(h)
        rep.append({"claim": label, "n_statements": len(uniq), "statements": uniq[:6]})
    report["repeated_claims"] = rep

    # ---- (c) AI disclosure, Methods vs Acknowledgements
    ai = {}
    for lbl, start in (("acknowledgements", one.find("ACKNOWLEDGEMENTS")),
                       ("methods", one.find("METHODS, DATA, AND CODE"))):
        end = one.find("[1] Hsin-Yuan Huang", start) if lbl == "methods" else one.find(
            "METHODS, DATA, AND CODE", start)
        seg = one[start: end if end > start else start + 4000]
        ai[lbl] = {
            "mentions_AI": bool(re.search(r"AI (coding )?(assistant|tools?)", seg)),
            "mentions_Claude": "Claude" in seg,
            "authors_responsible": bool(re.search(r"authors?\b[^.]*\bresponsible for", seg, re.I)),
            "verbs": sorted(set(re.findall(
                r"\b(implement\w*|refactor\w*|draft\w*|edit\w*|derivation\w*|verif\w*|"
                r"analys\w*|generat\w*|test suite|figures?)\b", seg))),
            "excerpt": seg[:600],
        }
    report["ai_disclosure"] = ai

    # ---- print
    print(f"pages {n_pages}   body words {report['body_words']}")
    print("\n(d) words by section")
    for k, v in counts.items():
        print(f"   {k:26s} {v:>6d}")
    bad = [c for c in abs_check if not c["ok"]]
    print(f"\n(b) abstract numbers: {len(abs_check) - len(bad)}/{len(abs_check)} appear in the body")
    for c in bad:
        print(f"   MISSING {c['value']!r}  from: {c['abstract_context']}")
    print("\n(a) repeated claims")
    for r in rep:
        print(f"   {r['claim']:34s} stated {r['n_statements']}x")
        for s in r["statements"]:
            print(f"        - {s}")
    print("\n(c) AI disclosure")
    for k, v in ai.items():
        print(f"   {k}: AI={v['mentions_AI']} Claude={v['mentions_Claude']} "
              f"responsibility={v['authors_responsible']} verbs={v['verbs']}")

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
