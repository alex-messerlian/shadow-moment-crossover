# The one thing that could not be settled here

`paper.tex` and `supplementary.tex` open with an engine conditional:

```latex
\ifdefined\XeTeXversion\else\pdfoutput=1\fi
\ifdefined\XeTeXversion
  \documentclass[a4paper,superscriptaddress,nofootinbib,onecolumn,nopdfoutputerror]{quantumarticle}
\else
  \documentclass[a4paper,superscriptaddress,nofootinbib,onecolumn]{quantumarticle}
\fi
```

**Under pdflatex** (what Quantum and the arXiv run): `\XeTeXversion` is undefined, so
`\pdfoutput=1` is set on the first line, before any `\usepackage`, exactly as the class and the
arXiv require; the class is loaded *without* `nopdfoutputerror`, so its check runs and must pass.

**Under XeTeX** (tectonic, the only toolchain in this repository): the assignment is skipped, so
hyperref selects the xetex driver instead of `hpdftex.def`; `nopdfoutputerror` stops the class
complaining about the `\pdfoutput` it cannot see.

## What has been verified here, and what has not

Verified locally, by loading the class with `\pdfoutput=1` and *without* `nopdfoutputerror`:

* the class's `\pdfoutput` check **passes** — its error does not fire;
* the build then proceeds to hyperref and dies inside `hpdftex.def` on a pdfTeX primitive XeTeX
  does not have.

That is the expected and harmless failure: `hpdftex.def` is the *correct* driver under pdflatex,
where those primitives exist. So the assignment reaches the class, and the class accepts it.

Not verified: that the whole document compiles under pdflatex. No pdfTeX is installed on the
machine these passes ran on (`pdflatex`, `pdftex`, `lualatex`, `tlmgr`, `kpsewhich` all absent; no
TeX Live or MacTeX tree; no container runtime).

## How to settle it in five minutes

Upload `paper/` to Overleaf, set the compiler to **pdfLaTeX** (Menu -> Compiler), and compile.
Or, with any TeX Live:

```bash
cd paper && pdflatex paper.tex && pdflatex paper.tex
```

Two runs, because cross-references need the second.

### A correct result looks like

* **31 pages**, no `!` errors, no `Undefined control sequence`;
* **no** `Class quantumarticle Error: ... please add \pdfoutput=1` — if this appears, the
  conditional did not fire and the `\ifdefined` guard is wrong;
* **no** `Package hyperref Warning: ... driver` complaint, and no `hpdftex.def` error;
* the log line `Output written on paper.pdf (31 pages` ;
* `LaTeX Warning: There were undefined references` **absent** after the second run.

### If it fails

The likely cause is the guard, not the document. Replace the first six lines with the plain form

```latex
\pdfoutput=1
\documentclass[a4paper,superscriptaddress,nofootinbib,onecolumn]{quantumarticle}
```

which is what Quantum's own template uses. That form cannot be built with tectonic, but it is the
form the journal expects, and once the paper is being compiled with pdflatex the conditional has
no further purpose.
