# Submission checklist (pre-arXiv)

Author reference only. The integrity program does **not** act on this list — every
open item below needs an author decision or an out-of-band action. Refreshed at
PASS 41.

Locations are given as sections and labels, not line numbers, so they stop going
stale as the paper is edited.

---

## OPEN — these four are the only remaining blockers

1. **Ziwei Gu's approval of *this* version.** He accepted co-authorship and the
   byline was applied in PASS 39, but the paper has changed substantially since
   he last would have seen it: new title, a new Section 5.3 on range-constrained
   estimators, a new opening claim, and dual accuracy figures throughout. He
   needs to read and approve the current draft, not the one he agreed to.

2. **Both authors' cold read.** The manuscript has been revised across many
   passes without an end-to-end human read of the final text. Do that before
   upload.

3. **`\date{\today}` → a fixed submission date.** In the preamble of
   `paper/paper.tex` *and* `paper/supplementary.tex`. Both currently recompile to
   the build date, so every rebuild stamps a different day. Must be pinned before
   upload, in both documents.

4. **Make the GitHub repository public.** The Data and Code Availability
   statement asserts public availability at
   `https://github.com/alex-messerlian/shadow-moment-crossover`, which matches the
   configured remote exactly. Per the project record the repo is private pending
   arXiv; it must be public at that exact URL before or at submission, or the
   statement is false. (Not verified over the network — the passes restrict
   network access to fetch/push.)

---

## DONE

- **Byline resolved (PASS 39).** Ziwei Gu is a co-author with his Harvard
  affiliation; the mentor sentence was removed from the Acknowledgements, since a
  co-author is not thanked for mentoring in the same paper. `paper.tex` and
  `supplementary.tex` now carry character-identical author blocks — they had
  disagreed since PASS 30, and an external reviewer caught it. Both rendered
  title blocks show two authors.
- **The clipping objection is answered (PASSES 36–40).** External review pointed
  out that $\mathrm{Tr}(\rho^k)$ lies in $[2^{n(1-k)}, 1]$, so an out-of-range
  estimate can be projected back in, which weakly reduces squared error
  pointwise — making the paper's reported RMSE of 11.98 at $n=10$ unreachable for
  any deployed estimator. Measured, and the criterion survives and improves:
  - at $n=10$, 97.9% of raw estimates fall outside the physical range, and
    projection cuts RMSE from 11.98 to 0.58, a factor of 20.6, with zero of 4320
    samples worsened;
  - re-derived for the projected estimator, the criterion places **all 72** cells
    that resolve a crossover within one qubit (100.0%), **63 of 72** exactly
    (87.5%), and **119 of 123** swept cells correctly (96.7%) — against 82 of 83,
    73 of 83, and 118 of 123 for the unbiased estimator;
  - the cost, disclosed in the paper: projection removes the crossover from 11
    cells, so 72 rather than 83 resolve, and no surviving crossover moves earlier.
  Both sets of figures are reported throughout; neither supersedes the other.
- **Title changed (PASS 39).** "The exponential wall" described the raw variance
  only and overstated what the paper shows once estimates are projected. Now
  *A finite-size crossover criterion for shadow-based moment estimation, with a
  hardware case study*, in both documents.
- **Headline claim replaced (PASS 39).** "Fifteen times the quantity being
  estimated" is retired at every site. The current claim is that the projected
  single-copy estimator is beaten from $n=7$ by a constant fixed at the **midpoint
  of the physical range** — stated explicitly as the midpoint everywhere it
  appears, because on three of four ensembles the estimand is deterministic and a
  constant that knew $q$ would win trivially.
- **Three documentation errors fixed (PASS 39).** Appendix C now names all three
  crossover rules the code uses and which produced which figures, corrects the
  trial count for the second noisy-pure sweep, and discloses that the estimand is
  degenerate on three of the four ensembles.
- **Test suite: 259 tests, all passing.**
- **Number audit (PASS 27).** 131 targets in Sections 3 and 6 recomputed from
  committed code and data. Section 6 is clean. The two Section 3 findings were
  fixed in PASS 28: the low-rank `M*` base (5.8 → 5.60) and the previously
  unverifiable narrow-family fit, which now has a generating script. The audit
  record itself is no longer in the repo — see the local archive below.
- **Every result in the Section 3.5 table is reproducible.**
  `experiments/beta_law_regenerate.py` regenerates `results/beta_law_test.json`
  from the committed ensemble code and reproduces the low-rank `M*` base to
  1.8e-15.
- All `\cite` resolve; **22 bibitems**, none uncited, none undefined.
- No dangling `\ref`; no `??` in the rendered PDF.
- `paper/paper.pdf` at HEAD is the exact build of `paper/paper.tex` at HEAD.
- Figures 1, 3 and 5 are the closed-form regenerations, and their plotted-data
  CSVs match the committed PDFs.
- **Repository cleaned of prose (PASSES 33-35).** Every write-up, provenance
  note, derivation narrative and verification record was removed, so the repo
  now carries only the paper, its build dependencies, and the code and data
  needed to run the verification. Local copies of all of it live outside the
  repository at `~/shadow-moment-crossover-notes/`, with an `INDEX.md`
  explaining each file and how to recover it from git history.
- **Repository cleaned of process artifacts (PASS 33).** `DRAFT_asymptotic_section.tex`
  (superseded draft), `PRIOR_ART_MAP.md` (provenance note) and `refs.bib` (dead,
  and stale at 15 entries against the paper's 22) were removed after a dependency
  check confirmed neither document referenced them. Both PDFs rebuild from
  `paper.tex`, `supplementary.tex` and `figures/` alone. The removals are in git
  history and recoverable.
- **Repo-facing documentation re-audited (PASS 41).** `README.md` had been stale
  since 2026-07-16 and still carried the pre-PASS-39 title, an obsolete claim
  about the `M*` base, a module table that put the state ensembles in the wrong
  package, and a pointer to the per-campaign `*_REPORT.md` narratives deleted in
  PASS 34. Rewritten. `pyproject.toml`'s description and author list, the
  `single_copy_law` module docstring (which denied the existence of a closed form
  the same module exports), and the figure-5 caption builder were corrected.

---

## Pre-upload actions (not blocking, but do them before the public push)

- [ ] **Untracked working notes.** Nine scratch files sit untracked under
      `paper/` in the primary worktree (`ANCHOR_DISCLOSURE_DRAFT.md`,
      `APPENDIX_A_REVISION.md`, `CITATIONS_TO_CONFIRM.md`,
      `CORRECTIONS_MASTER.md`, `FINAL_SWEEP.md`, `REFEREE_RESPONSE.md`,
      `REFEREE2_RESPONSE.md`, `REFRAME_OPTIONS.md`,
      `SECTION_3.4_REVISION.md`). They are intentionally untracked. Decide
      whether they stay local or are removed from the working tree.
- [ ] **This file.** `paper/SUBMISSION_CHECKLIST.md` is the last tracked
      process artifact under `paper/`. Delete it after submission.
- [ ] **GitHub-side metadata** (settings, not files — see PASS 41's report):
      the repository description, topics, and About section still describe the
      old title and a single author. Nothing in the repo can change these.
- [ ] **Confirm the source compiles on arXiv's TeX Live.** Built here with
      `tectonic`; REVTeX 4-2, no custom `\usepackage` beyond the standard set.
- [ ] **Final rebuild after the date edit** — confirm the page count is stable,
      no LaTeX errors, no unresolved `\ref`.

---

## The arXiv upload set

`paper/paper.tex`, `paper/supplementary.tex`, and `paper/figures/*.pdf` (six
figures) — that is the whole set. Both documents carry their bibliography inline
as `thebibliography` (22 entries in the paper, 1 in the supplement), so no `.bib`
file is needed or present. Primary category `quant-ph`; choose a license.

**arXiv posting is blocked pending endorsement** for `quant-ph`, which requires
an endorser or a prior qualifying submission. This does not block journal
submission: PRA accepts submissions that have not been posted to arXiv, so the
endorsement question can be resolved in parallel rather than gating the paper.

---

## Current build

46 pages, 0 LaTeX errors, 0 undefined references, 0 undefined citations.
Supplementary material is a separate 9-page document (`paper/supplementary.tex`)
carrying the hardware protocol, the weight-resolved readout data, the session
tables, the single-copy anchor construction, the credit accounting, and the full
83-cell crossover listing.
