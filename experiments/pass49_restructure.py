"""PASS 49.3: the mechanical half of paper one's restructure.

    PYTHONPATH=. .venv/bin/python experiments/pass49_restructure.py

Does only the deterministic, auditable part: split ``paper/paper.tex`` into blocks, delete the
hardware fragments by EXACT string anchor (never by line range, which overran a boundary in
PASS 48), reassemble in the target order, remap every textual section cross-reference, and drop
placeholder markers where PASS 49.4/49.5 prose goes.

Every step asserts.  An anchor that does not appear exactly once aborts the run, so a silent
partial deletion is impossible.  The prose (abstract, introduction, the two new sections, the
conclusion) is written by hand into the placeholders afterwards.

Old -> new section numbering:
    1,2,3 unchanged;  3.6 -> 3.7 (a new 3.6 holds the evaluation cost);
    4 -> 6 (bias laws);  5 -> 7 (crossover);  6 DELETED;  7 -> 8;  8 -> 9;  9 -> 10.

Writes ``paper/paper.tex`` in place and ``results/pass49_restructure_log.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper" / "paper.tex"
LOG = REPO / "results" / "pass49_restructure_log.json"

EXPECT_BLOB = "13590dbfe8119022ae95ff7a374910e30146ab8c"

# Block boundaries, 1-indexed inclusive, with the text each must start with (asserted).
BLOCKS = (
    ("preamble", 1, 24, "\\documentclass"),
    ("doc_open", 25, 26, "\\begin{document}"),
    ("title", 27, 28, "\\title{"),
    ("authors", 29, 33, "\\author{Alexander"),
    ("date", 34, 35, "\\date{"),
    ("abstract", 36, 72, "\\begin{abstract}"),
    ("maketitle", 73, 75, ""),
    ("S1_intro", 76, 165, "\\hypertarget{introduction}"),
    ("rule_a", 166, 167, "\\begin{center}\\rule"),
    ("S2_setting", 168, 325, "\\hypertarget{setting-and-estimators}"),
    ("rule_b", 326, 327, "\\begin{center}\\rule"),
    ("S3_variance", 328, 717, "\\hypertarget{the-single-copy-variance-law}"),
    ("rule_c", 718, 719, "\\begin{center}\\rule"),
    ("S4_bias", 720, 828, "\\hypertarget{the-collective-route-two-exact-bias-laws}"),
    ("rule_d", 829, 830, "\\begin{center}\\rule"),
    ("S5_crossover", 831, 1084, "\\hypertarget{the-crossover}"),
    ("rule_e", 1085, 1086, "\\begin{center}\\rule"),
    ("fig5", 1087, 1093, "\\begin{figure}"),
    ("S6_hardware", 1094, 1350, "\\hypertarget{hardware-a-pre-registered-test"),
    ("rule_f", 1351, 1352, "\\begin{center}\\rule"),
    ("S7_related", 1353, 1533, "\\hypertarget{related-work}"),
    ("rule_g", 1534, 1535, "\\begin{center}\\rule"),
    ("S8_limitations", 1536, 1613, "\\hypertarget{limitations}"),
    ("rule_h", 1614, 1615, "\\begin{center}\\rule"),
    ("S9_conclusion", 1616, 1684, "\\hypertarget{conclusion}"),
    ("rule_i", 1685, 1686, "\\begin{center}\\rule"),
    ("ack", 1687, 1696, "\\hypertarget{acknowledgements}"),
    ("datacode", 1697, 1708, "\\hypertarget{data-and-code-availability}"),
    ("bib", 1709, 1892, "\\begin{thebibliography}"),
    ("appendix", 1893, 2192, "\\appendix"),
    ("doc_close", 2193, 2193, "\\end{document}"),
)

# Deletions by exact anchor.  Each must match exactly once in its block.
# (block, anchor text, replacement, note)
DELETIONS = [
    # 49.2(b) intro hardware sentences -- the sentence starts mid-line, so anchor the sentence.
    ("S1_intro",
     " A\nhardware comparison of the two routes has been reported\n"
     "\\cite{peng2025beyond}; what we add is the predictive law and a test of it\n"
     "locked in advance. That test failed, and what it established is the shape\n"
     "of the obstacle: the dominant error was readout rather than the entangling\n"
     "overhead the crossover argument prices, and device parameters\n"
     "characterized in earlier sessions did not describe the device we ran on.",
     "", "intro hardware sentences -> paper two"),
    # 49.2(b)+(e) sec2.3: drop the hardware-affordability pointer, keep the gate count.
    ("S2_setting",
     " That gate count is why the\nentangling overhead is affordable on real hardware (Section 6).",
     " That gate count is what makes the\nmeasurement overhead of the collective route two two-qubit gates, "
     "a compilation\nfact we return to in Section 2.6.",
     "sec2.3 hardware pointer -> compilation-fact pointer"),
    # 49.2(e) sec2.3 k>=3 ratios: reframe as a compilation fact, drop the Section 6.2 pointer.
    ("S2_setting",
     "transpiled to the tested coupling map, the controlled cyclic shift on",
     "transpiled to a published square-lattice coupling map, the controlled cyclic shift on",
     "k>=3 ratios reframed as compilation facts"),
    ("S2_setting",
     "and needs an ancilla and routing it does not. Our \\(k \\geq 3\\) results are\n"
     "simulated, and the gate counts in Section 6.2 are for \\(k = 2\\) only.",
     "and needs an ancilla and routing it does not. Our \\(k \\geq 3\\) results are\n"
     "simulated, and the gate counts of Section 2.6 are for \\(k = 2\\) only.",
     "sec2.3 Section 6.2 pointer -> Section 2.6"),
    # 49.2(b) sec5.4 hardware pointer (sentence starts mid-line).
    ("S5_crossover",
     " Whether the collective route survives on a real device is the\nquestion we take up next.",
     "", "sec5.4 forward reference to the hardware section -> paper two"),
    # 49.2(g) rename the Related Work paragraph that is a theory reconciliation.
    ("S7_related",
     "\\textbf{Collective measurement on noisy hardware.} The exponential",
     "\\textbf{Noise and the collective advantage: testing versus estimation.} The exponential",
     "sec7 paragraph renamed (it is a theory reconciliation)"),
    # 49.2(d) keep the DIRECTION of the readout-sensitivity result, drop the quantitative
    # "one to two qubits", which depended on device-specific measured rates.
    ("S5_crossover",
     "Real shadow readout is not noiseless, and Section 6.3 finds readout to be\n"
     "the dominant error on the device we tested, so this understates the\n"
     "single-copy cost, in the direction that costs us, and we have measured\n"
     "by how much. Simulating readout on both routes at the per-qubit rates\n"
     "measured in Section 6.3, and correcting the collective parity for readout\n"
     "exactly through its outcome-dependent per-pair contraction rather than a\n"
     "blanket factor, moves the crossover to smaller \\(n\\) in every resolved\n"
     "cell: by one to two qubits when both routes calibrate readout out, and\n"
     "further when both leave it uncorrected. The sizes reported here are\n"
     "therefore a conservative upper bound under this noise model.",
     "Real shadow readout is not noiseless, so this understates the single-copy\n"
     "cost, in the direction that costs us. Simulating readout on both routes,\n"
     "and correcting the collective parity for readout exactly through its\n"
     "outcome-dependent per-pair contraction rather than a blanket factor, moves\n"
     "the crossover to smaller \\(n\\) in every resolved cell. The magnitude of\n"
     "that shift depends on device-specific per-qubit rates and is quantified in a\n"
     "companion study of the hardware case; the direction is what matters here,\n"
     "and the sizes reported below are therefore a conservative upper bound under\n"
     "this noise model.",
     "sec5.2 -> 7.2: readout direction kept, the one-to-two-qubit figure deferred"),
    # 49.2(b) the channel-abstraction Limitations item pointed at the deleted section.
    ("S8_limitations",
     "exact given the channel, and they say nothing about whether the channel\n"
     "describes any particular device. Section 6 documents exactly where a\n"
     "real device departs from them.",
     "exact given the channel, and they say nothing about whether the channel\n"
     "describes any particular device; a companion study of a superconducting\n"
     "backend reports where a real device departs from them.",
     "sec9 channel-abstraction item: Section 6 pointer -> companion study"),
]

# Paragraph deletions, as half-open spans [start_anchor, stop_before_anchor).  Anchoring the END
# on the anchor of the paragraph that must SURVIVE is what makes this safe: an end-of-paragraph
# sentence anchor silently swallowed two surviving paragraphs on the first attempt here.
# stop_before = None means "to the end of the block".
PARA_DELETIONS = [
    ("S7_related", "\\textbf{Crossovers and hardware comparison.}",
     "\\textbf{Sample-complexity lower bounds for purity.}",
     "sec7: the hardware-comparison and bias-floor-on-hardware paragraphs -> paper two"),
    ("S7_related", "\\textbf{NISQ device stability.}", None,
     "sec7: NISQ device stability paragraph -> paper two"),
    ("S8_limitations", "\\textbf{The crossover was not demonstrated on hardware.}",
     "\\textbf{Statistical caveats.}",
     "sec9: three hardware Limitations items -> paper two"),
    ("S8_limitations", "\\textbf{The readout sensitivity check is a simulation.}", None,
     "sec9: the readout-simulation Limitations item -> paper two"),
]

# Textual section-reference remap.  Applied in ONE pass so nothing is remapped twice.
REF_MAP = {
    "3.6": "3.7",
    "4": "6", "4.1": "6.1", "4.2": "6.2", "4.3": "6.3",
    "5": "7", "5.1": "7.1", "5.2": "7.2", "5.3": "7.3", "5.4": "7.4",
    "7": "8", "8": "9", "9": "10",
}

PLACEHOLDERS = {
    "title": "%%PASS49_TITLE%%\n",
    "abstract": "%%PASS49_ABSTRACT%%\n",
    "S1_intro": "%%PASS49_INTRO%%\n",
    "S4_statewise": "%%PASS49_STATEWISE%%\n",
    "S5_pilot": "%%PASS49_PILOT%%\n",
    "S10_conclusion": "%%PASS49_CONCLUSION%%\n",
    "S2_compilation": "%%PASS49_COMPILATION%%\n",
    "S3_promoted": "%%PASS49_S3_PROMOTED%%\n",
    "S3_algorithm": "%%PASS49_S3_ALGORITHM%%\n",
    "ack_new": "%%PASS49_ACK%%\n",
}


def die(msg: str) -> None:
    print(f"ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    import subprocess
    blob = subprocess.run(["git", "hash-object", str(PAPER)], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    if blob != EXPECT_BLOB:
        die(f"paper.tex is {blob}, expected the PASS 48 frozen blob {EXPECT_BLOB}. "
            "This script transforms the frozen text exactly once.")
    lines = PAPER.read_text().split("\n")
    log: dict = {"source_blob": blob, "steps": []}

    # --- split into blocks, asserting each start ---
    blocks: dict[str, str] = {}
    for name, first, last, starts in BLOCKS:
        body = "\n".join(lines[first - 1:last])
        if starts and not body.lstrip().startswith(starts):
            die(f"block {name} (L{first}) does not start with {starts!r}; got "
                f"{body.lstrip()[:70]!r}")
        blocks[name] = body
    log["steps"].append({"step": "split", "n_blocks": len(blocks)})
    print(f"split into {len(blocks)} blocks, all start anchors verified")

    # --- confirm the deleted section is the hardware one, then drop it ---
    if "Rigetti" not in blocks["S6_hardware"] or "elimination" not in blocks["S6_hardware"]:
        die("S6_hardware block does not look like the hardware section")
    dropped_words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", blocks.pop("S6_hardware")))
    blocks.pop("rule_f")
    print(f"deleted S6_hardware ({dropped_words} raw word tokens) and its trailing separator")
    log["steps"].append({"step": "delete_section6", "raw_word_tokens": dropped_words})

    # --- targeted deletions / rewordings ---
    for block, anchor, repl, note in DELETIONS:
        n = blocks[block].count(anchor)
        if n != 1:
            die(f"anchor for {note!r} appears {n} times in {block} (need exactly 1):\n{anchor[:120]}")
        blocks[block] = blocks[block].replace(anchor, repl)
        log["steps"].append({"step": "anchor_edit", "block": block, "note": note,
                             "removed_chars": len(anchor) - len(repl)})
        print(f"  edited {block:16s} {note}")

    # --- paragraph deletions (half-open spans) ---
    for block, start, stop_before, note in PARA_DELETIONS:
        text = blocks[block]
        if text.count(start) != 1:
            die(f"start anchor for {note!r} appears {text.count(start)} times in {block}")
        i = text.find(start)
        if stop_before is None:
            j = len(text)
        else:
            if text.count(stop_before) != 1:
                die(f"stop anchor for {note!r} appears {text.count(stop_before)} times in {block}")
            j = text.find(stop_before)
            if j <= i:
                die(f"stop anchor precedes start anchor for {note!r}")
        removed = text[i:j]
        blocks[block] = text[:i] + text[j:]
        words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", removed))
        leads = removed.count("\\textbf{")
        log["steps"].append({"step": "para_delete", "block": block, "note": note,
                             "raw_word_tokens": words, "bold_leads_removed": leads})
        print(f"  removed {words:>5d} raw word tokens / {leads} bold lead(s) from {block:16s} {note}")

    # --- reassemble in the target order, with placeholders ---
    rule = blocks["rule_a"]
    order = [
        blocks["preamble"], blocks["doc_open"], PLACEHOLDERS["title"], blocks["authors"],
        blocks["date"], PLACEHOLDERS["abstract"], blocks["maketitle"],
        PLACEHOLDERS["S1_intro"], rule,
        blocks["S2_setting"] + "\n" + PLACEHOLDERS["S2_compilation"], rule,
        blocks["S3_variance"], rule,
        PLACEHOLDERS["S4_statewise"], rule,
        PLACEHOLDERS["S5_pilot"], rule,
        blocks["S4_bias"], rule,
        blocks["S5_crossover"] + "\n" + blocks["fig5"], rule,
        blocks["S7_related"], rule,
        blocks["S8_limitations"], rule,
        PLACEHOLDERS["S10_conclusion"], rule,
        PLACEHOLDERS["ack_new"], blocks["datacode"], blocks["bib"], blocks["appendix"],
        blocks["doc_close"],
    ]
    text = "\n".join(order)

    # --- insert the two §3 placeholders at their anchors ---
    a = "\\hypertarget{the-alpha-transition}{%"
    if text.count(a) != 1:
        die("alpha-transition anchor not unique")
    # The promoted 3.5 material and the new 3.6 both go immediately before the alpha
    # transition, which then becomes 3.7.
    text = text.replace(a, PLACEHOLDERS["S3_promoted"] + PLACEHOLDERS["S3_algorithm"] + a)

    # --- remap textual section references in ONE pass ---
    remapped: dict[str, int] = {}

    def _sub(m: re.Match) -> str:
        word, sep, num = m.group(1), m.group(2), m.group(3)
        if num in REF_MAP:
            remapped[f"{num}->{REF_MAP[num]}"] = remapped.get(f"{num}->{REF_MAP[num]}", 0) + 1
            return f"{word}{sep}{REF_MAP[num]}"
        return m.group(0)

    text = re.sub(r"\b(Sections?|Secs?\.)([~ ])(\d+(?:\.\d+)?)", _sub, text)
    # "Sections 3 and 4" -> the trailing number is not caught by the pattern above
    pair = "Sections 3 and 4; nothing is fitted"
    if pair in text:
        text = text.replace(pair, "Sections 3 and 6; nothing is fitted")
        remapped["pair 3 and 4 -> 3 and 6"] = 1
    print("\nreference remap:")
    for k, v in sorted(remapped.items()):
        print(f"  {k:28s} x{v}")
    log["steps"].append({"step": "ref_remap", "counts": remapped})

    for unused in ("title", "abstract", "S1_intro", "S9_conclusion", "ack"):
        blocks.pop(unused)          # replaced wholesale by hand-written prose
    PAPER.write_text(text)
    LOG.write_text(json.dumps(log, indent=1))

    # --- report what still points at the deleted section ---
    stale = [(i + 1, ln) for i, ln in enumerate(text.split("\n"))
             if re.search(r"\bSections?[~ ]6(\.\d+)?\b", ln)]
    print(f"\nreferences still pointing at a Section 6.x: {len(stale)}")
    for i, ln in stale:
        print(f"  L{i}: {ln.strip()[:110]}")
    print(f"\nwrote {PAPER.relative_to(REPO)} ({len(text.split(chr(10)))} lines) and "
          f"{LOG.relative_to(REPO)}")
    print("PLACEHOLDERS remaining (filled by hand next): "
          + ", ".join(sorted(p.strip().strip('%') for p in PLACEHOLDERS.values())))


if __name__ == "__main__":
    main()
