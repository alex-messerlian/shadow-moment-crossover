"""PASS 52: the conservation gate -- no claim and no number may be lost to compression.

    PYTHONPATH=. .venv/bin/python experiments/pass52_conservation.py BEFORE.tex [AFTER.tex]

Compares two revisions of ``paper.tex`` and answers, for every numeric token whose count falls:
does it still appear somewhere in the document, and where?  A token reaching zero document-wide
is a STOP unless it is on the explicitly permitted list.

Two gates run on top of that:

FRACTION-SENTENCE INTEGRITY -- every ``N of M`` pair in the before must still appear, as a pair,
in the after.  Compression that keeps ``82`` and drops ``of 83`` turns a rate into a bare count,
which is the specific failure this gate exists to catch.

PER-SECTION NUMERIC MULTISET -- the same accounting per section, so a token that merely MOVES
between sections is reported as a move rather than a loss.

Writes ``results/pass52_conservation.json``; exit status is non-zero if any gate fails.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "pass52_conservation.json"

# Tokens whose disappearance is authorised by the pass brief, with the reason.
PERMITTED_LOSS: dict[str, str] = {
    # arXiv:2604.22627 (zeng2026exact) was withdrawn by its author on 11 August 2026 -- the abs
    # page reads "withdrawn by Shuai Zeng" and "No PDF available" -- so the citation and its
    # supporting clause were removed in PASS 55.  Nothing else in the paper rests on it.
    "2604.22627": "cited preprint withdrawn by its author, 11 August 2026",
}

NUM = re.compile(r"(?<![\w.\\])(\d+(?:[.,]\d+)*)(?![\w])")
# "of" / "out of" only.  Accepting a bare "/" made this match exponents -- M^{-1/2} scored as
# the pair "1 of 2" -- so a rewording that dropped one exponent raised a spurious STOP.
FRACTION = re.compile(r"\\?\(?(\d[\d,{}\\]*)\\?\)?\s*(?:of|out of)\s+(?:the\s+)?\\?\(?(\d[\d,{}\\]*)\\?\)?")


def clean(s: str) -> str:
    """Drop LaTeX bookkeeping that carries no reader-facing number."""
    s = re.sub(r"\\label\{[^}]*\}", " ", s)
    s = re.sub(r"\\hypertarget\{[^}]*\}", " ", s)
    s = re.sub(r"\\includegraphics(\[[^]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\(?:auto)?ref\{[^}]*\}", " ", s)
    s = re.sub(r"\\cite\{[^}]*\}", " ", s)
    s = re.sub(r"0\.\d+\\(?:line|column)width", " ", s)     # float sizing
    s = re.sub(r"\{[\d.]+pt\}", " ", s)
    s = re.sub(r"\\allowbreak", "", s)
    s = s.replace("{,}", ",").replace("\\,", "")
    return s


def sections(src: str) -> dict[str, str]:
    lines = src.split("\n")
    marks = [(i, re.match(r"\\section\*?\{(.*)", ln)) for i, ln in enumerate(lines)]
    marks = [(i, m.group(1)) for i, m in marks if m]
    out, names = {}, []
    for (i, t), (j, _) in zip(marks, marks[1:] + [(len(lines), "")]):
        name = re.sub(r"[{}\\]|label.*", "", t).strip()[:44] or f"sec@{i}"
        while name in names:
            name += "'"
        names.append(name)
        out[name] = clean("\n".join(lines[i:j]))
    return out


def tokens(text: str) -> Counter:
    return Counter(NUM.findall(text))


def fractions(text: str) -> Counter:
    t = re.sub(r"[{}\\]", "", text)
    return Counter(f"{a} of {b}" for a, b in FRACTION.findall(t))


def main() -> None:
    before_p, after_p = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2
                                                else REPO / "paper" / "paper.tex")
    before, after = clean(before_p.read_text()), clean(after_p.read_text())
    bt, at = tokens(before), tokens(after)
    bs, asec = sections(before_p.read_text()), sections(after_p.read_text())

    # ---- document-wide conservation
    lost, thinned = [], []
    for tok, n in sorted(bt.items(), key=lambda x: -x[1]):
        m = at.get(tok, 0)
        if m == 0 and n > 0:
            lost.append({"token": tok, "was": n,
                         "permitted": tok in PERMITTED_LOSS,
                         "reason": PERMITTED_LOSS.get(tok, "")})
        elif m < n:
            # find a surviving site
            sites = [s for s, txt in asec.items() if tok in tokens(clean(txt))]
            thinned.append({"token": tok, "was": n, "now": m, "surviving_sections": sites})
    gained = [{"token": t, "n": c} for t, c in at.items() if t not in bt]

    # ---- fraction-sentence integrity
    bf, af = fractions(before), fractions(after)
    frac_lost = [{"fraction": f, "was": c, "now": af.get(f, 0)}
                 for f, c in bf.items() if af.get(f, 0) < c]

    # ---- per-section multiset
    per = []
    for name in bs:
        b, a = tokens(bs[name]), tokens(asec.get(name, ""))
        moved_out = {t: b[t] - a.get(t, 0) for t in b if a.get(t, 0) < b[t]}
        landed = {}
        for t in moved_out:
            landed[t] = [s for s, txt in asec.items()
                         if s != name and tokens(txt).get(t, 0) > tokens(bs.get(s, "")).get(t, 0)]
        per.append({"section": name, "before": sum(b.values()), "after": sum(a.values()),
                    "tokens_reduced": moved_out,
                    "relocated_to": {k: v for k, v in landed.items() if v}})

    hard_lost = [l for l in lost if not l["permitted"]]
    print(f"document-wide: {len(hard_lost)} tokens reached ZERO (unpermitted), "
          f"{len(thinned)} thinned, {len(gained)} new")
    for l in hard_lost:
        print(f"   STOP  token {l['token']!r} was used {l['was']}x and is now absent")
    print(f"\nfraction-sentence integrity: {len(frac_lost)} pairs lost")
    for f in frac_lost:
        print(f"   STOP  {f['fraction']!r}  {f['was']} -> {f['now']}")
    print("\nper-section numeric multiset")
    for p in per:
        d = p["after"] - p["before"]
        if d or p["tokens_reduced"]:
            print(f"   {p['section'][:40]:42s} {p['before']:>4d} -> {p['after']:<4d} ({d:+d})")
            for t, n in sorted(p["tokens_reduced"].items(), key=lambda x: -x[1])[:12]:
                dest = p["relocated_to"].get(t)
                print(f"       -{n}x {t:<10s} {'-> ' + ', '.join(dest) if dest else '(absorbed)'}")
    print("\nthinned tokens and their surviving sites")
    for t in thinned:
        print(f"   {t['token']:<10s} {t['was']}->{t['now']}  in: {', '.join(t['surviving_sections'])}")

    OUT.write_text(json.dumps({"lost": lost, "thinned": thinned, "gained": gained,
                               "fraction_pairs_lost": frac_lost, "per_section": per}, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    if hard_lost or frac_lost:
        sys.exit(1)


if __name__ == "__main__":
    main()
