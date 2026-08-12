# The committed PDFs are STALE as of the restructure

`paper/paper.pdf` and `paper/supplementary.pdf` in this repository still render the
pre-restructure manuscript: the finite-size-crossover paper with the hardware case study as
Section 6. They do **not** correspond to the current `paper.tex` and `supplementary.tex`.

They were not rebuilt because no LaTeX toolchain was available in the environment where the
restructure was performed. Consequently the following are **unverified** for the current source:

- page count,
- undefined-reference and overfull-box warnings,
- float placement of Figures 1-5 and 7-10 and of the four tables,
- the rendered two-author byline on both documents.

Rebuild both before circulating or submitting:

    cd paper && pdflatex paper && pdflatex paper && pdflatex paper
    cd paper && pdflatex supplementary && pdflatex supplementary

Three passes are needed for the cross-references to settle. `fig6_hardware.pdf` is retained in
`figures/` for the companion hardware paper and is no longer referenced by `paper.tex`.
