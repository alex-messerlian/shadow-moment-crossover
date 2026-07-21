# Submission checklist (pre-arXiv)

Author reference only. This list is **not acted on** by the integrity program — every
item below requires an author decision or an out-of-band action. Prepared at commit
`c1d8ea9`.

## 1. Repository visibility (blocks the Data and Code Availability statement)

The Data and Code Availability statement (`paper/paper.tex`, "Data and code
availability") reads:

> All code, raw measurement counts, and analysis are available at
> https://github.com/alex-messerlian/shadow-moment-crossover. Raw hardware counts were
> committed to version control before any analysis was performed, and every locked
> prediction was committed before the corresponding measurement was submitted.

- [ ] **Make the GitHub repository public.** The statement asserts public availability;
      per the project record the repo is currently **private pending arXiv**. It must be
      public (at the exact URL above) before or at submission, or the statement is false.
      (Not verified over the network here — the pass restricts network to fetch/push.)
- [ ] Confirm the public tree contains what the statement promises: code (`anrl/`,
      `experiments/`, `tests/`), raw hardware counts (`results/hardware/`), and analysis
      (`results/*.json`). All are tracked at HEAD.

## 2. Authorship decision (four blocked items in the source)

These were left untouched by every pass, pending an authorship outcome:

| Item | Location | 
|---|---|
| Author byline | `paper/paper.tex` L25 `\author{Alexander Messerlian}`, L26 `\affiliation{Independent Researcher}`, L27 `\author{Ziwei Gu}`, L28 `\affiliation{Harvard John A. Paulson School of Engineering and Applied Sciences}` |
| Date | L30 `\date{\today}` |
| Mentor acknowledgement | L1798 `[Mentor acknowledgement.]` (inside the Acknowledgements section) |

Actions by outcome:

- **Ziwei as co-author:** keep the byline as-is; resolve or fill `[Mentor
  acknowledgement.]` (or remove if redundant with co-authorship).
- **Ziwei in acknowledgements only:** remove `\author{Ziwei Gu}` and its
  `\affiliation` (L27-28); replace `[Mentor acknowledgement.]` with a proper
  acknowledgement of Ziwei.
- **Ziwei in neither:** remove L27-28; remove the `[Mentor acknowledgement.]`
  placeholder.
- [ ] **In every outcome:** replace `\date{\today}` with a fixed submission date
      (it currently recompiles to the build date).

## 3. Repository hygiene before the public push

- [ ] Nine untracked working notes exist under `paper/` (`ANCHOR_DISCLOSURE_DRAFT.md`,
      `APPENDIX_A_REVISION.md`, `CITATIONS_TO_CONFIRM.md`, `CORRECTIONS_MASTER.md`,
      `FINAL_SWEEP.md`, `REFEREE_RESPONSE.md`, `REFEREE2_RESPONSE.md`,
      `REFRAME_OPTIONS.md`, `SECTION_3.4_REVISION.md`). They are intentionally
      untracked scratch. Decide whether they stay local or are removed from the
      working tree before making the repo public.
- [ ] `paper/DRAFT_asymptotic_section.tex` is a **superseded** v3 draft (header marks
      it so; content is in `paper.tex` Section 3.5 as of commit `1a973d3`). Decide
      whether to keep it as a provenance record or delete it; if deleted, also remove
      the reference in `paper/PRIOR_ART_MAP.md` (PASS 7 header).
- [ ] `paper/PRIOR_ART_MAP.md`, `paper/SUBMISSION_CHECKLIST.md` (this file), and other
      provenance `.md` files are tracked. Decide whether they belong in the public repo
      or should be pruned before release.

## 4. Final build and arXiv upload

- [ ] After the byline/date/acknowledgement edits, rebuild `paper/paper.pdf` and
      confirm: page count stable, no LaTeX errors, no unresolved `\ref` (`??`).
- [ ] arXiv upload set: `paper/paper.tex`, `paper/refs.bib` (or the inline
      `thebibliography`, already present with 21 entries), and `paper/figures/*.pdf`.
      Set the primary category to `quant-ph` and choose a license.
- [ ] Confirm the submission compiles on arXiv's TeX Live (currently built with
      `tectonic`; REVTeX 4-2, no custom `\usepackage`).

## 5. Consistency already verified (no action needed)

- All `\cite` resolve; 21 bibitems; no uncited or undefined citations (PASS 17).
- Figures 1, 3, 5 are the closed-form regenerations; their plotted-data CSVs are now
  consistent with the committed PDFs (PASS 18).
- No numeric contradictions across sections; cascade section references clean; no
  duplicate or orphan sentences (PASS 17).
- `paper/paper.pdf` at HEAD is the exact build of `paper/paper.tex` at HEAD (PASS 18).
