# Building the paper

## Files

| file | what it is |
|---|---|
| `paper.tex` | **The single source of truth.** Self-contained LaTeX: the bibliography is embedded, so there is no separate `.bib` step. |
| `paper.pdf` | Compiled output, 36 pages. Builds with zero errors. |
| `refs.bib` | BibTeX version of the references (all author-verified against arXiv), if you later want to switch to `\bibliography{refs}`. |
| `VERIFICATION.md` | The verification record. Every paper number was checked against the saved results, and all 16 references were verified against arXiv (closure note at the top). |
| `DISCREPANCIES.md` | Audit log of the corrections made during verification. |
| `archive/PAPER_draft.md` | Historical Markdown draft, superseded by `paper.tex`. Do not edit. |
| `archive/OUTLINE_draft.md` | Historical planning outline. Do not edit. |

## Easiest path: Overleaf

1. Go to overleaf.com, New Project, Upload Project.
2. Upload `paper.tex`.
3. It compiles in the browser. Edit there.
4. Overleaf has a direct "Submit to arXiv" button when you are ready.

This is the right path if you are not set up with a local LaTeX install. REVTeX is preinstalled on Overleaf.

## Local build

```
pdflatex paper.tex
pdflatex paper.tex     # second pass resolves cross-references
```

Two passes. No bibtex step, because the bibliography is embedded in a `thebibliography` environment.

## Layout

The document class line is:

```latex
\documentclass[aps,prx,preprint,superscriptaddress,nofootinbib]{revtex4-2}
```

`preprint` gives a single-column, generously spaced layout, which is what you want for review and for a mentor to read. To switch to the two-column journal look, change `preprint` to `reprint`. Be warned: the wide tables (the elimination ledger in Section 6.5) will need to become `table*` environments to span both columns, or they will overflow.

## Two things fixed during conversion that would have broken the build

**Dollar-sign qubit labels.** The draft used Rigetti's physical-qubit syntax (`$0`, `$9`) in prose and tables. In LaTeX a bare `$` opens math mode, and this would have cascaded errors through the whole document in a way that is genuinely unpleasant to debug. Physical qubits are now written $q_0$, $q_9$, and so on, with a note in Section 6.3 tying that notation to the platform's `$N` addressing.

**Pandoc `longtable`.** Pandoc emits `longtable` with `calc`-computed column widths, which conflicts with REVTeX. All tables were converted to plain `tabular` inside `table` environments.

## Done during verification

1. **Verified every number against the `anrl` repo.** Each quantitative claim was checked against the saved results; corrections are logged in `DISCREPANCIES.md`.
2. **Figures.** All six figures are inserted (`\includegraphics`); they regenerate from `anrl/figures/` — see the repository `README.md`.
3. **References.** All 16 were verified against arXiv (one, the misapplied coherence-of-noise paper, was removed, leaving 15).

## Still to do before submission

1. **Affiliation and mentor authorship.** Placeholders remain in the preamble (`[affiliation]`, mentor name).
