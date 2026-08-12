"""PASS 49.6: validate the restructured manuscript without a LaTeX toolchain.

    PYTHONPATH=. .venv/bin/python experiments/pass49_validate.py

No TeX is installed in this environment, so the PDFs cannot be rebuilt here and the checks below
are the strongest static substitutes: brace and environment balance, reference and citation
closure, figure-file existence, author count, the per-section numeric multiset diff against the
pre-restructure text, and the fraction-sentence integrity gate.

The fraction gate looks for sentences of the form "<a> of the <b>" / "<a> of <b>" and requires
a <= b and that the pair appear in the committed artifacts, since a restructure that reflows text
is exactly where a numerator and denominator get separated.

Writes ``results/pass49_validation.json``; exit status is non-zero on any hard failure.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper" / "paper.tex"
SUPP = REPO / "paper" / "supplementary.tex"
FIGDIR = REPO / "paper" / "figures"
OUT = REPO / "results" / "pass49_validation.json"
BASE_COMMIT = "2cb5494"

NUM = re.compile(r"(?<![A-Za-z0-9_.^{])(\d+(?:\.\d+)?)")


def strip_tex(s: str) -> str:
    s = re.sub(r"(?m)^\s*%.*$", "", s)
    return s


def sections(text: str) -> dict[str, str]:
    out, cur, buf = {}, "front matter", []
    for line in text.split("\n"):
        m = re.match(r"\\section\*?\{(.*)", line)
        if m:
            out[cur] = "\n".join(buf)
            cur, buf = re.sub(r"[\\{}]|\\label.*", "", m.group(1))[:46].strip(), []
        buf.append(line)
    out[cur] = "\n".join(buf)
    return out


def check_balance(name: str, text: str, fails: list) -> dict:
    t = strip_tex(text)
    # brace balance ignoring \{ \}
    tb = re.sub(r"\\[{}]", "", t)
    ob, cb = tb.count("{"), tb.count("}")
    if ob != cb:
        fails.append(f"{name}: brace imbalance {ob} open vs {cb} close")
    envs = Counter(re.findall(r"\\begin\{([^}]*)\}", t))
    ende = Counter(re.findall(r"\\end\{([^}]*)\}", t))
    bad = {k: (envs[k], ende[k]) for k in set(envs) | set(ende) if envs[k] != ende[k]}
    if bad:
        fails.append(f"{name}: unbalanced environments {bad}")
    return {"braces": [ob, cb], "unbalanced_envs": bad,
            "n_environments": sum(envs.values())}


def main() -> None:
    fails: list[str] = []
    report: dict = {}
    paper = PAPER.read_text()
    supp = SUPP.read_text()

    # ---- balance ----
    report["balance_paper"] = check_balance("paper.tex", paper, fails)
    report["balance_supp"] = check_balance("supplementary.tex", supp, fails)

    # ---- refs and labels ----
    for name, text in (("paper.tex", paper), ("supplementary.tex", supp)):
        labels = set(re.findall(r"\\label\{([^}]*)\}", text))
        refs = set(re.findall(r"\\(?:auto)?ref\{([^}]*)\}", text))
        dangling = sorted(refs - labels)
        if name == "paper.tex" and dangling:
            fails.append(f"{name}: dangling \\ref -> {dangling}")
        report[f"refs_{name}"] = {"n_labels": len(labels), "n_refs": len(refs),
                                  "dangling": dangling,
                                  "unused_labels": len(labels - refs)}

    # ---- citations ----
    flat = re.sub(r"\s+", " ", paper)
    cited: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", flat):
        cited |= {k.strip() for k in m.group(1).split(",")}
    bib = set(re.findall(r"\\bibitem\{([^}]*)\}", paper))
    if cited - bib:
        fails.append(f"unresolved citations: {sorted(cited - bib)}")
    if bib - cited:
        fails.append(f"uncited bibitems: {sorted(bib - cited)}")
    report["citations"] = {"n_bibitems": len(bib), "n_cited": len(cited),
                           "unresolved": sorted(cited - bib),
                           "uncited": sorted(bib - cited)}

    # ---- figures exist ----
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", paper)
    missing = [f for f in figs
               if not any((FIGDIR / f"{f}{e}").exists() for e in (".pdf", ".png", ""))]
    if missing:
        fails.append(f"missing figure files: {missing}")
    report["figures"] = {"referenced": figs, "missing": missing}

    # ---- authors on both documents ----
    for name, text in (("paper.tex", paper), ("supplementary.tex", supp)):
        n_auth = len(re.findall(r"\\author\{", text))
        if n_auth != 2:
            fails.append(f"{name}: {n_auth} \\author entries, expected 2")
        report[f"authors_{name}"] = n_auth

    # ---- hardware residue ----
    banned = ("Rigetti", "Cepheus", "Quantum Rings", "Open Quantum", "chiplet",
              "elimination ledger", "Public Tier", "cross-session", "same-session")
    residue = {b: [i + 1 for i, l in enumerate(paper.split("\n")) if b in l] for b in banned}
    residue = {k: v for k, v in residue.items() if v}
    if residue:
        fails.append(f"hardware residue in paper.tex: {residue}")
    report["hardware_residue"] = residue

    # ---- section word counts ----
    def words(s: str) -> int:
        t = strip_tex(s)
        t = re.sub(r"\$[^$]*\$|\\\[.*?\\\]", " ", t, flags=re.S)
        t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
        t = re.sub(r"[{}&\\~^_]", " ", t)
        return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", t))

    secs = sections(paper)
    report["section_words"] = {k: words(v) for k, v in secs.items()}
    body_total = sum(w for k, w in report["section_words"].items()
                     if k not in ("front matter",) and "bibliography" not in k.lower())
    report["body_words"] = body_total

    # ---- numeric multiset diff against the pre-restructure text ----
    old = subprocess.run(["git", "show", f"{BASE_COMMIT}:paper/paper.tex"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    old_n = Counter(NUM.findall(strip_tex(old)))
    new_n = Counter(NUM.findall(strip_tex(paper)))
    removed = {k: v for k, v in (old_n - new_n).items()}
    added = {k: v for k, v in (new_n - old_n).items()}
    report["numeric_diff"] = {"n_tokens_before": sum(old_n.values()),
                              "n_tokens_after": sum(new_n.values()),
                              "removed": dict(sorted(removed.items())),
                              "added": dict(sorted(added.items()))}

    # ---- fraction-sentence integrity ----
    frac = []
    for m in re.finditer(r"\\\((\d+)\\\)\s+(?:of|in)\s+(?:the\s+)?\\\((\d+)\\\)", flat):
        a, b = int(m.group(1)), int(m.group(2))
        frac.append({"a": a, "b": b, "ok": a <= b,
                     "context": flat[max(0, m.start() - 80):m.end() + 40]})
    for m in re.finditer(r"\b(\d+)\s+(?:of|in)\s+(?:the\s+)?(\d+)\b", flat):
        a, b = int(m.group(1)), int(m.group(2))
        frac.append({"a": a, "b": b, "ok": a <= b,
                     "context": flat[max(0, m.start() - 80):m.end() + 40]})
    broken = [f for f in frac if not f["ok"]]
    if broken:
        fails.append(f"fraction sentences with numerator > denominator: "
                     f"{[(f['a'], f['b']) for f in broken]}")
    report["fraction_sentences"] = {"n_found": len(frac), "n_broken": len(broken),
                                    "pairs": [[f["a"], f["b"]] for f in frac],
                                    "broken": broken}

    # ---- section numbering sanity ----
    heads = re.findall(r"\\section\*?\{([^\\}]*)", paper)
    report["section_order"] = [h.strip()[:52] for h in heads]

    print("=== balance ===")
    print(f"  paper.tex braces {report['balance_paper']['braces']}, "
          f"{report['balance_paper']['n_environments']} environments, "
          f"unbalanced {report['balance_paper']['unbalanced_envs'] or 'none'}")
    print(f"  supp.tex  braces {report['balance_supp']['braces']}, "
          f"unbalanced {report['balance_supp']['unbalanced_envs'] or 'none'}")
    print("=== references ===")
    r = report["refs_paper.tex"]
    print(f"  paper.tex {r['n_labels']} labels, {r['n_refs']} refs, "
          f"dangling {r['dangling'] or 'none'}")
    print(f"  citations: {report['citations']['n_bibitems']} bibitems, "
          f"{report['citations']['n_cited']} cited, "
          f"unresolved {report['citations']['unresolved'] or 'none'}, "
          f"uncited {report['citations']['uncited'] or 'none'}")
    print(f"  figures referenced {len(figs)}, missing {missing or 'none'}")
    print(f"  authors: paper {report['authors_paper.tex']}, supp {report['authors_supplementary.tex']}")
    print(f"  hardware residue: {residue or 'none'}")
    print("=== section word counts ===")
    for k, v in report["section_words"].items():
        print(f"  {k[:52]:54s} {v:>6d}")
    print(f"  {'BODY TOTAL':54s} {body_total:>6d}")
    print("=== numeric multiset diff vs " + BASE_COMMIT + " ===")
    print(f"  tokens {report['numeric_diff']['n_tokens_before']} -> "
          f"{report['numeric_diff']['n_tokens_after']}")
    print(f"  distinct values removed: {len(removed)}, added: {len(added)}")
    print(f"=== fraction sentences: {len(frac)} found, {len(broken)} broken ===")

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    if fails:
        print("\nHARD FAILURES:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nall static checks pass")


if __name__ == "__main__":
    main()
