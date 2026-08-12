"""PASS 48.1: extract paper two's source material from the frozen manuscript.

    PYTHONPATH=. .venv/bin/python experiments/pass48_extract_paper2.py

Paper one (theory, target Quantum) drops the hardware case study entirely.  This script COPIES
every hardware-bearing fragment out of ``paper/paper.tex`` and ``paper/supplementary.tex`` into
``paper2/source/`` so nothing is lost when paper one's rewrite deletes it.  Nothing is moved or
removed: both manuscripts stay byte-frozen this pass, and the extraction is read-only.

Extraction is by explicit line range against the blob at the frozen commit, so every fragment
is reproducible and auditable rather than hand-copied.  Each output file carries a provenance
header naming the source file, blob hash, line range, and what the fragment is.

Writes ``paper2/source/*.tex``, ``paper2/MATERIAL_INVENTORY.md`` and
``results/pass48_extraction_manifest.json``.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper" / "paper.tex"
SUPP = REPO / "paper" / "supplementary.tex"
OUT_DIR = REPO / "paper2" / "source"
INVENTORY = REPO / "paper2" / "MATERIAL_INVENTORY.md"
MANIFEST = REPO / "results" / "pass48_extraction_manifest.json"

FROZEN_PAPER_BLOB = "13590dbfe8119022ae95ff7a374910e30146ab8c"
FROZEN_SUPP_BLOB = "2076cda654e58c43e1cc96dfc46ee239900dffec"

# (output stem, source, first line, last line, section reference, description)
FRAGMENTS: tuple[tuple[str, str, int, int, str, str], ...] = (
    ("01_abstract_hardware_paragraph", "paper.tex", 64, 71, "Abstract, final paragraph",
     "The hardware result as the abstract states it: the 108-qubit backend, both locked "
     "predictions missed (0.8932 -> 0.7184 and 1.500 -> 1.344), and the disclaimer that the "
     "crossover was not demonstrated."),
    ("02_intro_hardware_sentences", "paper.tex", 158, 164, "Section 1, final paragraph",
     "The introduction's framing of the hardware test: the Peng et al. precedent, the "
     "locked-in-advance claim, and the shape-of-the-obstacle summary."),
    ("03_sec2_3_hardware_pointers", "paper.tex", 243, 246, "Section 2.3",
     "The sentence tying the destructive SWAP gate count to hardware affordability. Paper one "
     "keeps the gate count but must reword this pointer; paper two inherits the claim."),
    ("04_sec2_3_kge3_gate_ratios", "paper.tex", 255, 262, "Section 2.3",
     "The k>=3 ancilla-based Hadamard-test gate ratios (~10x at k=3, ~16x at k=4) transpiled to "
     "the tested coupling map, and the Section 6.2 scoping note. NOTE: these counts are "
     "reproducible offline from the committed anrl/hardware/cepheus_metadata.json, so paper one "
     "KEEPS them as compilation facts; only the Section 6.2 pointer needs rewording."),
    ("05_sec5_2_readout_asymmetry", "paper.tex", 902, 920, "Section 5.2",
     "The noise-model asymmetry paragraph, including the measured per-qubit readout rates fed "
     "back through both routes and the resulting conservative-upper-bound statement. Paper one "
     "loses the measured rates and must either cite a rate or drop the quantitative claim."),
    ("06_sec5_4_hardware_pointer", "paper.tex", 1075, 1083, "Section 5.4",
     "The 'entangling overhead is real but not the binding constraint' paragraph and its "
     "forward reference to the hardware section. The claim itself survives in paper one on the "
     "transpilation counts; the forward reference does not."),
    ("07_section6_full", "paper.tex", 1094, 1350, "Section 6, all six subsections",
     "The hardware case study in full: platform and execution-path disclosure (6.1), the "
     "entangling-overhead measurement (6.2), the readout and SPAM characterization with the "
     "four-qubit table (6.3), cross-session variation with Figure 6 (6.4), the seven-mechanism "
     "elimination ledger as a longtable (6.5), and what remains (6.6). This is paper two's core."),
    ("08_sec7_hardware_comparison", "paper.tex", 1417, 1434, "Section 7",
     "Related Work: 'Crossovers and hardware comparison' -- the Pauli/Clifford ensemble "
     "crossover and the Peng et al. hardware comparison of the two routes."),
    ("09_sec7_bias_floor_on_hardware", "paper.tex", 1435, 1450, "Section 7",
     "Related Work: 'Statistical-to-bias-floor transitions on hardware' -- the March 2026 "
     "report of O(M^-1/2) decay saturating at a hardware floor, and how it differs from ours."),
    ("10_sec7_nisq_stability", "paper.tex", 1526, 1536, "Section 7",
     "Related Work: 'NISQ device stability' -- Dasgupta and Humble over 22 months, and the "
     "calibration-drift analyses. This is paper two's direct literature."),
    ("11_sec8_hardware_limitations", "paper.tex", 1554, 1580, "Section 8",
     "Four Limitations items: the channel abstraction (which points at Section 6), the "
     "crossover not demonstrated on hardware, the computed-not-measured single-copy baseline, "
     "and the unverifiable Public Tier execution path."),
    ("12_sec8_readout_simulation", "paper.tex", 1607, 1612, "Section 8",
     "The fifth hardware Limitations item: the readout sensitivity check is a simulation using "
     "the Section 6.3 rates."),
    ("13_sec9_hardware_conclusion", "paper.tex", 1654, 1676, "Section 9",
     "The conclusion's two hardware paragraphs: the entangling-overhead accounting against the "
     "measured deficit, and 'what blocks the demonstration is readout fidelity and "
     "session-to-session variation' -- which is paper two's thesis sentence."),
    ("14_acknowledgements_platform", "paper.tex", 1687, 1696, "Acknowledgements",
     "The platform paragraph: Open Quantum / Quantum Rings, Public Tier, the licence "
     "conditions requiring citation and anonymized data contribution, and the backend "
     "attribution. Paper two must carry this verbatim -- it is a licence obligation."),
    ("15_supplementary_S1_S6", "supplementary.tex", 52, 230,
     "Supplementary S1-S6",
     "The hardware supplement in full: S1 hardware protocol, S2 weight-resolved readout "
     "crosstalk, S3 the bracketed same-session experiment, S4 the three-session comparison, "
     "S5 the single-copy anchor, S6 cloud-QPU economics. S7 (the full crossover table) is "
     "theory and stays with paper one."),
)

# results/ files that are hardware-specific.  Listed, NOT moved: paper one's repo stays intact
# and both manuscripts can cite the same public repository.
HARDWARE_RESULT_GLOBS = ("hardware/**/*",)
HARDWARE_RESULT_CANDIDATES = ("cepheus_predictions.json",)

# Scripts are classified by an EXPLICIT list, not a name regex: a regex on "grid" catches the
# theory clipping grids, and a regex on "readout" catches the two readout-simulation modules
# that paper one's Section 5.2 depends on.  Each entry below was checked against its docstring.
HARDWARE_SCRIPTS = (
    "coherent_error_analysis.py", "coherent_error_build.py", "run_coherent_error.py",
    "cz_characterization_analysis.py", "run_cz_characterization.py",
    "device_characterization_analysis.py", "run_device_characterization.py",
    "grid_predictions.py",                  # locked Cepheus grid predictions, device params only
    "hardware_grid_analysis.py", "hardware_grid_build.py", "run_hardware_grid.py",
    "hardware_prediction.py", "hardware_validation.py", "hardware_validation_analysis.py",
    "joint_readout_analysis.py", "run_joint_readout.py",
    "readout_extension_analysis.py", "run_readout_extension.py",
    "register_fault_analysis.py", "register_fault_build.py", "register_fault_lib.py",
    "run_register_fault.py",
    "run_same_session.py", "same_session_analysis.py", "same_session_lib.py",
)
# Theory scripts a "hardware" name regex would wrongly claim; they stay with paper one.
THEORY_SCRIPTS_NEAR_MISS = (
    "run_clipping_crossover_grid.py",       # PASS 36 clipping grid -- theory
    "run_clipping_paired_grid.py",          # PASS 36 paired z-test grid -- theory
)
# Simulation modules that USE the measured readout rates but belong to paper one's Section 5.2.
SHARED_READOUT_MODULES = (
    "anrl/benchmark/readout_shadows.py",
    "anrl/benchmark/readout_correction.py",
)


def blob_hash(path: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()


def words(text: str) -> int:
    """Prose word count: strip LaTeX macros, math, and comments."""
    t = re.sub(r"(?m)%.*$", "", text)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}&\\~^_]", " ", t)
    return len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", t))


def main() -> None:
    t0 = time.time()
    # Freeze gate: refuse to extract from anything but the frozen text.
    actual = {"paper.tex": blob_hash(PAPER), "supplementary.tex": blob_hash(SUPP)}
    expected = {"paper.tex": FROZEN_PAPER_BLOB, "supplementary.tex": FROZEN_SUPP_BLOB}
    if actual != expected:
        raise SystemExit(f"FREEZE GATE FAILED: {actual} != {expected}")
    print(f"freeze gate: paper.tex {actual['paper.tex'][:8]}  "
          f"supplementary.tex {actual['supplementary.tex'][:8]}  OK")

    lines = {"paper.tex": PAPER.read_text().split("\n"),
             "supplementary.tex": SUPP.read_text().split("\n")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_words = 0
    total_lines = 0
    for stem, src, first, last, ref, desc in FRAGMENTS:
        body = "\n".join(lines[src][first - 1:last])
        w = words(body)
        total_words += w
        total_lines += last - first + 1
        header = (
            f"% EXTRACTED FOR PAPER TWO (the hardware case study) -- PASS 48.1\n"
            f"% source     : paper/{src}\n"
            f"% blob       : {actual[src]}\n"
            f"% lines      : {first}-{last} (1-indexed, inclusive)\n"
            f"% section    : {ref}\n"
            f"% prose words: {w}\n"
            f"%\n"
            f"% {desc}\n"
            f"%\n"
            f"% This is a VERBATIM COPY. The original is unmodified and still frozen.\n"
            f"% Regenerate with: experiments/pass48_extract_paper2.py\n\n"
        )
        (OUT_DIR / f"{stem}.tex").write_text(header + body + "\n")
        manifest.append({"file": f"paper2/source/{stem}.tex", "source": f"paper/{src}",
                         "blob": actual[src], "first_line": first, "last_line": last,
                         "n_lines": last - first + 1, "section": ref, "prose_words": w,
                         "description": desc})
        print(f"  {stem:34s} {src:18s} L{first:>5d}-{last:<5d} {w:>5d} words")

    # ---- hardware artifacts (listed, not moved) ----
    hw_files = sorted(str(p.relative_to(REPO)) for g in HARDWARE_RESULT_GLOBS
                      for p in (REPO / "results").glob(g) if p.is_file())
    for c in HARDWARE_RESULT_CANDIDATES:
        p = REPO / "results" / c
        if p.is_file() and str(p.relative_to(REPO)) not in hw_files:
            hw_files.append(str(p.relative_to(REPO)))
    hw_bytes = sum((REPO / f).stat().st_size for f in hw_files)
    missing = [s for s in HARDWARE_SCRIPTS if not (REPO / "experiments" / s).is_file()]
    if missing:
        raise SystemExit(f"HARDWARE_SCRIPTS names files that do not exist: {missing}")
    hw_scripts = [f"experiments/{s}" for s in sorted(HARDWARE_SCRIPTS)]
    hw_package = sorted(str(p.relative_to(REPO)) for p in (REPO / "anrl" / "hardware").glob("*.py"))

    print(f"\nhardware artifacts that would accompany paper two (NOT moved):")
    print(f"  results/hardware/ and siblings : {len(hw_files)} files, {hw_bytes/1e6:.2f} MB")
    print(f"  experiments/ hardware scripts  : {len(hw_scripts)} files (explicit list, audited)")
    print(f"  anrl/hardware/ package         : {len(hw_package)} files")
    print(f"  stays with paper one           : {', '.join(THEORY_SCRIPTS_NEAR_MISS)} "
          f"and {len(SHARED_READOUT_MODULES)} readout-simulation modules")

    # ---- inventory ----
    # Full descriptions: splitting on the first period mangles them, because several contain
    # decimals ("0.8932"), subsection numbers ("6.1") and abbreviations ("Peng et al.").
    rows = "\n".join(
        f"| [`{m['file'].split('/')[-1]}`](source/{m['file'].split('/')[-1]}) | {m['section']} | "
        f"`{m['source']}` L{m['first_line']}-{m['last_line']} | {m['prose_words']} | "
        f"{m['description']} |"
        for m in manifest)
    INVENTORY.write_text(f"""# Paper two: material inventory

The hardware case study, extracted from paper one's frozen manuscript so that paper one's
rewrite can delete it without losing anything.

**Status.** Source material only. Paper two is **not written**. Nothing here has been edited:
every file in `source/` is a verbatim copy carrying a provenance header with its source blob
and line range.

**Provenance.** Extracted at commit `f07f38b` from
`paper/paper.tex` blob `{FROZEN_PAPER_BLOB}` and
`paper/supplementary.tex` blob `{FROZEN_SUPP_BLOB}`, both byte-frozen.
Regenerate with `experiments/pass48_extract_paper2.py` (it re-checks the blobs and refuses to
run against modified text).

**Totals.** {len(manifest)} fragments, {total_lines} source lines, **{total_words} prose words**.

## Fragments

| File | Section | Source | Words | What it is |
|---|---|---|---|---|
{rows}

## What paper one keeps

Two fragments are listed above but are **not** purely paper two's:

- `04_sec2_3_kge3_gate_ratios.tex` — the k>=3 gate ratios (~10x, ~16x) and the n=2 four-CZ /
  zero-routing counts are **transpilation facts**, reproducible offline from the committed
  `anrl/hardware/cepheus_metadata.json` with no credentials and no device access. Paper one
  keeps them in Section 2 as compilation costs; only the "Section 6.2" pointer is rewritten.
- `06_sec5_4_hardware_pointer.tex` — the claim that the entangling overhead is real but not the
  binding constraint rests on those same transpilation counts, so it survives in paper one. Only
  the closing forward reference to the hardware section goes.

Everything else in the list leaves paper one entirely.

## Backing artifacts (in this repository, not moved)

Both manuscripts cite the same public repository, so the hardware data stays where it is.

- `results/hardware/` — {len(hw_files)} files, {hw_bytes/1e6:.2f} MB: raw OpenQASM 3 circuits as
  submitted, per-job measurement counts, the twirled coherent-error series, the calibration
  brackets, the three-session repeats, and the analysis JSONs.
- `experiments/` — {len(hw_scripts)} hardware scripts (explicit audited list, not a name match):
  {", ".join('`' + Path(s).name + '`' for s in hw_scripts)}.
- `anrl/hardware/` — {len(hw_package)} modules: backend and credentials, calibration, the noise
  and readout models, SWAP-test and shadow circuit construction, Pauli twirling, shot budgeting,
  and the committed `cepheus_metadata.json` device description.

Three near-misses stay with **paper one**, and a name-based split would get them wrong:
{chr(10).join('- `experiments/' + s + '`' for s in THEORY_SCRIPTS_NEAR_MISS)}
  — the PASS 36 clipping and paired-z-test grids, which are theory.
{chr(10).join('- `' + s + '`' for s in SHARED_READOUT_MODULES)}
  — readout *simulation* modules. They consume the measured per-qubit rates, but they implement
  paper one's Section 5.2 sensitivity check, not a hardware measurement.

## Figure

Figure 6 (`paper/figures/fig6_hardware.*`) is paper two's; its caption is inside
`07_section6_full.tex`. Figures 1-5 are paper one's and are untouched.

## Not extracted, deliberately

- Supplementary S7, the full crossover table: theory, stays with paper one.
- Section 7's "Collective measurement on noisy hardware" paragraph: despite the title it is the
  theoretical reconciliation with the noisy-purity-testing lower bound, so it stays with paper
  one. Paper one should rename the paragraph, since "on noisy hardware" will no longer be apt.
""")
    MANIFEST.write_text(json.dumps({
        "description": "PASS 48.1: paper-two extraction manifest (copies, nothing moved)",
        "frozen_blobs": expected, "verified_blobs": actual,
        "fragments": manifest,
        "totals": {"n_fragments": len(manifest), "source_lines": total_lines,
                   "prose_words": total_words},
        "hardware_artifacts": {"results_files": hw_files, "results_bytes": hw_bytes,
                               "experiment_scripts": hw_scripts, "package_modules": hw_package},
        "wall_seconds": time.time() - t0,
    }, indent=1))

    print(f"\n48.1(d) TOTAL EXTRACTED: {total_words} prose words over {total_lines} source lines "
          f"in {len(manifest)} fragments")
    print(f"wrote paper2/source/ ({len(manifest)} files), "
          f"{INVENTORY.relative_to(REPO)}, {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
