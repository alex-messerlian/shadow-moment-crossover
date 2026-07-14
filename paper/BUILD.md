# Building the paper

## Files

| file | what it is |
|---|---|
| `paper.tex` | The LaTeX source. Self-contained: the bibliography is embedded, so there is no separate `.bib` step. |
| `paper.pdf` | Compiled output, 29 pages. Verified to build with zero errors. |
| `PAPER.md` | The markdown source. Edit here if you prefer, then regenerate. |
| `VERIFICATION.md` | The 40-row checklist. **Nothing is submittable until this is green.** |
| `refs.bib` | BibTeX version of the references, if you later want to switch to `\bibliography{refs}`. |

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

## Still to do before submission

1. **Run `VERIFICATION.md` against the `anrl` repo.** Every number in the paper was written from the conversation record, not from the saved results. One number in this project was previously misreported. Do not skip this.
2. **Figures.** The paper currently has no `\includegraphics`. Five figures exist in `results/figures/` and a sixth (hardware: measured vs pre-registered bands, plus cross-session drift) does not yet exist and must be generated.
3. **Affiliation and mentor authorship.** Placeholders in the preamble.
4. **Confirm all 16 references bibliographically.** Titles came from web search; author lists mostly did not.
