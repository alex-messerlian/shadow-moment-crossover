"""PASS 49.2 gate: nothing is deleted from paper one unless paper two already has it.

    PYTHONPATH=. .venv/bin/python experiments/pass49_deletion_gate.py

Reads the CURRENT ``paper/paper.tex`` and ``paper/supplementary.tex``, takes each planned
deletion range, and requires that the exact text appear inside the corresponding read-only
``paper2/source/`` fragment.  Comparison is on normalized whitespace so a reflow cannot hide a
missing sentence, and the check is per-line so a partial fragment fails loudly.

Exit status is non-zero if any range is not fully covered, so this can gate the edit.

Writes ``results/pass49_deletion_gate.json``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper" / "paper.tex"
SUPP = REPO / "paper" / "supplementary.tex"
SRC = REPO / "paper2" / "source"
OUT = REPO / "results" / "pass49_deletion_gate.json"

# (label, source file, first line, last line, covering paper2 fragment)
DELETIONS = (
    ("abstract hardware paragraph", "paper.tex", 64, 71, "01_abstract_hardware_paragraph.tex"),
    ("intro hardware sentences", "paper.tex", 158, 164, "02_intro_hardware_sentences.tex"),
    ("sec2.3 hardware pointer", "paper.tex", 245, 246, "03_sec2_3_hardware_pointers.tex"),
    ("sec5.2 readout asymmetry (partial: rewritten, not removed)",
     "paper.tex", 906, 915, "05_sec5_2_readout_asymmetry.tex"),
    ("sec5.4 hardware pointer", "paper.tex", 1082, 1083, "06_sec5_4_hardware_pointer.tex"),
    ("SECTION 6 ENTIRE", "paper.tex", 1094, 1350, "07_section6_full.tex"),
    ("sec7 hardware comparison", "paper.tex", 1417, 1434, "08_sec7_hardware_comparison.tex"),
    ("sec7 bias floor on hardware", "paper.tex", 1435, 1450, "09_sec7_bias_floor_on_hardware.tex"),
    ("sec7 NISQ stability", "paper.tex", 1526, 1536, "10_sec7_nisq_stability.tex"),
    ("sec8 hardware limitations (4 items)", "paper.tex", 1554, 1580,
     "11_sec8_hardware_limitations.tex"),
    ("sec8 readout simulation item", "paper.tex", 1607, 1612, "12_sec8_readout_simulation.tex"),
    ("sec9 hardware conclusion", "paper.tex", 1664, 1676, "13_sec9_hardware_conclusion.tex"),
    ("acknowledgements platform paragraph", "paper.tex", 1690, 1695,
     "14_acknowledgements_platform.tex"),
    ("supplementary S1-S6", "supplementary.tex", 52, 230, "15_supplementary_S1_S6.tex"),
)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    lines = {"paper.tex": PAPER.read_text().split("\n"),
             "supplementary.tex": SUPP.read_text().split("\n")}
    results, failures = [], []
    for label, src, first, last, frag in DELETIONS:
        body = lines[src][first - 1:last]
        cover = (SRC / frag).read_text()
        cover_norm = norm(cover)
        missing = [ln for ln in body if norm(ln) and norm(ln) not in cover_norm]
        ok = not missing
        results.append({"label": label, "source": src, "first_line": first, "last_line": last,
                        "fragment": frag, "n_lines": len(body),
                        "n_nonblank": sum(1 for ln in body if norm(ln)),
                        "covered": ok, "missing_lines": missing[:5]})
        if not ok:
            failures.append(label)
        flag = "COVERED" if ok else f"*** {len(missing)} LINE(S) NOT IN FRAGMENT ***"
        print(f"  {label:52s} {src:18s} L{first:>5d}-{last:<5d} -> {frag:36s} {flag}")
        for m in missing[:3]:
            print(f"        missing: {m[:100]}")

    print(f"\n{len(results) - len(failures)}/{len(results)} deletion ranges are fully covered "
          f"by paper2/source/")
    frag_files = sorted(p.name for p in SRC.glob("*.tex"))
    unused = [f for f in frag_files if f not in {d[4] for d in DELETIONS}]
    print(f"paper2/source/ holds {len(frag_files)} fragments; {len(unused)} not referenced by a "
          f"deletion range: {unused}")
    print("  (04_sec2_3_kge3_gate_ratios.tex is retained in paper one as compilation facts,")
    print("   so it is copied to paper two but NOT deleted here -- see PASS 48.)")

    # paper2/source must be untouched: compare against the committed blobs.
    dirty = subprocess.run(["git", "status", "--porcelain", "paper2"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    print(f"\npaper2/ working-tree status: {'CLEAN (read-only respected)' if not dirty else dirty}")

    OUT.write_text(json.dumps({
        "description": "PASS 49.2 gate: every deletion range is covered by a paper2/source fragment",
        "ranges": results,
        "all_covered": not failures,
        "failures": failures,
        "unreferenced_fragments": unused,
        "paper2_dirty": dirty,
    }, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    if failures or dirty:
        sys.exit(1)


if __name__ == "__main__":
    main()
