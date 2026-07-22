# Submission checklist (pre-arXiv)

Author reference only. The integrity program does **not** act on this list — every
open item below needs an author decision or an out-of-band action. Refreshed at
PASS 32.

Locations are given as sections and labels, not line numbers, so they stop going
stale as the paper is edited.

---

## OPEN — these three block submission

1. **Ziwei Gu's authorship decision.** The paper currently compiles
   **acknowledgement-only**: the byline is Alexander Messerlian / Independent
   Researcher, and the co-author lines sit commented immediately below it in
   `paper/paper.tex`, marked `CO-AUTHOR BYLINE, pending Ziwei Gu's authorship
   decision`. The Acknowledgements thank him by name.
   - *If he accepts co-authorship:* uncomment the two `%\author{Ziwei Gu}` /
     `%\affiliation{Harvard …}` lines **and** remove the "I thank Ziwei Gu…"
     sentence from the Acknowledgements. Two-line edit, no other change.
   - *If he declines or prefers acknowledgement only:* nothing to do — the
     current state is already correct.

2. **`\date{\today}` → a fixed submission date.** In the preamble of
   `paper/paper.tex`. It currently recompiles to the build date, so every
   rebuild stamps a different day. Must be pinned before upload.

3. **Make the GitHub repository public.** The Data and Code Availability
   statement asserts public availability at
   `https://github.com/alex-messerlian/shadow-moment-crossover`. Per the project
   record the repo is private pending arXiv; it must be public at that exact URL
   before or at submission, or the statement is false. (Not verified over the
   network — the passes restrict network access to fetch/push.)

---

## DONE

- **Mentor acknowledgement — filled (PASS 30, commit `03a7cf5`).** The
  `[Mentor acknowledgement.]` placeholder is gone; the Acknowledgements read
  "I thank Ziwei Gu for mentoring this work and for his guidance through my
  first research paper." Verified in the rendered PDF: the placeholder text does
  not appear anywhere, and "Harvard" appears nowhere in the document.
- **Byline resolved to a safe default (PASS 30).** Acknowledgement-only, with
  the co-author lines preserved in comments for a one-line revert. The two
  mutually exclusive states the paper used to carry — co-author byline *and* an
  unfilled mentor placeholder — no longer coexist.
- **Number audit (PASS 27, `results/pass27_number_audit.json`).** 131 targets in
  Sections 3 and 6 recomputed from committed code and data. Section 6 is clean.
  The two Section 3 findings were fixed in PASS 28: the low-rank `M*` base
  (5.8 → 5.60) and the previously unverifiable narrow-family fit, which now has
  a generating script.
- **Every result in the Section 3.5 table is reproducible.**
  `experiments/beta_law_regenerate.py` regenerates `results/beta_law_test.json`
  from the committed ensemble code and reproduces the low-rank `M*` base to
  1.8e-15.
- All `\cite` resolve; **22 bibitems**, none uncited, none undefined.
- No dangling `\ref`; no `??` in the rendered PDF.
- `paper/paper.pdf` at HEAD is the exact build of `paper/paper.tex` at HEAD.
- Figures 1, 3 and 5 are the closed-form regenerations, and their plotted-data
  CSVs match the committed PDFs.

---

## Pre-upload actions (not blocking, but do them before the public push)

- [ ] **Untracked working notes.** Nine scratch files sit untracked under
      `paper/` in the primary worktree (`ANCHOR_DISCLOSURE_DRAFT.md`,
      `APPENDIX_A_REVISION.md`, `CITATIONS_TO_CONFIRM.md`,
      `CORRECTIONS_MASTER.md`, `FINAL_SWEEP.md`, `REFEREE_RESPONSE.md`,
      `REFEREE2_RESPONSE.md`, `REFRAME_OPTIONS.md`,
      `SECTION_3.4_REVISION.md`). They are intentionally untracked. Decide
      whether they stay local or are removed from the working tree.
- [ ] **`paper/DRAFT_asymptotic_section.tex`** is a superseded draft (its header
      says so; the content is now Section 3.5). Keep as a provenance record or
      delete — if deleted, also remove the reference to it in
      `paper/PRIOR_ART_MAP.md`.
- [ ] **Tracked provenance `.md` files** (`paper/PRIOR_ART_MAP.md` and this
      file). Decide whether they belong in the public repo.
- [ ] **arXiv upload set:** `paper/paper.tex`, `paper/supplementary.tex`,
      `paper/figures/*.pdf` (six figures), and `paper/refs.bib` if you switch
      away from the inline `thebibliography` (22 entries, already present).
      Primary category `quant-ph`; choose a license.
- [ ] **Confirm the source compiles on arXiv's TeX Live.** Built here with
      `tectonic`; REVTeX 4-2, no custom `\usepackage` beyond the standard set.
- [ ] **Final rebuild after the date and byline edits** — confirm the page count
      is stable, no LaTeX errors, no unresolved `\ref`.

---

## Current build

43 pages, 0 LaTeX errors, 0 undefined references, 0 undefined citations.
Supplementary material is a separate 9-page document (`paper/supplementary.tex`)
carrying the hardware protocol, the weight-resolved readout data, the session
tables, the single-copy anchor construction, the credit accounting, and the full
83-cell crossover listing.
