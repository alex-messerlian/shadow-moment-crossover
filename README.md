# shadow-moment-crossover

Code, data, and analysis for the paper:

> **A finite-size crossover criterion for shadow-based moment estimation, with a hardware case study**
> Alexander Messerlian (Independent Researcher) and Ziwei Gu (Harvard John A. Paulson School of Engineering and Applied Sciences).
> Compiled manuscript: [`paper/paper.pdf`](paper/paper.pdf) · LaTeX source: [`paper/paper.tex`](paper/paper.tex) · Supplement: [`paper/supplementary.pdf`](paper/supplementary.pdf)

## What this repository is

Everything the paper's numbers depend on: the estimators and theory as an
installable package, one runnable script per result, the saved outputs those
scripts produce, and the raw hardware measurement counts. Nothing in the paper
is computed anywhere else.

The paper itself asks when a collective two-copy measurement beats single-copy
classical shadows at estimating $\mathrm{Tr}(\rho^k)$ at equal copy budget, derives
a criterion for the crossover size, validates it in simulation, and reports a
committed-in-advance hardware test that failed. Read the PDF for the argument;
this file only explains how to run things.

## Layout

| path | contents |
|---|---|
| `anrl/theory/` | the exact U-statistic variance law, the Hoeffding projection variances and their $k=2$ closed forms, the two collective bias laws, the threshold $M^*$, and the crossover predictor |
| `anrl/benchmark/` | Monte-Carlo estimators (single-copy shadows, collective SWAP test), the state ensembles, noise channels, moment operators, readout models, and the range-constrained estimators in `constrained.py` |
| `anrl/physics/` | states, Pauli machinery, measurement, entanglement witnesses |
| `anrl/figures/` | the publication figure builders (Okabe–Ito palette; PDF/PNG/CSV export) |
| `anrl/hardware/` | Open Quantum / Rigetti Cepheus backend, circuit builders, the destructive-SWAP protocol |
| `experiments/` | one runnable script per result |
| `results/` | the saved outputs, including the raw hardware counts |
| `tests/` | the test suite (259 tests) |

`anrl/benchmark/constrained.py` is worth pointing out: $\mathrm{Tr}(\rho^k)$ lies in
$[2^{n(1-k)}, 1]$, so an estimate outside that range can be projected back into it,
which weakly reduces squared error pointwise. The paper reports its accuracy figures
for both the unbiased estimator and this range-projected one, and that module is where
the projection lives.

## Setup

No GPU and no deep-learning dependencies. The pinned set installs everything:

```bash
git clone https://github.com/alex-messerlian/shadow-moment-crossover.git
cd shadow-moment-crossover
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The theory results and the figures need only the core scientific stack, so
`pip install numpy scipy matplotlib` is enough for those. Everything else needs
`qiskit` and `qiskit-aer`, which `requirements.txt` pins: `anrl.hardware` imports them at
module level, so they are required to re-analyse the committed hardware counts, and three
of the test modules import them, which means `pytest` aborts at collection without them.
Submitting new jobs additionally needs the `openquantum_sdk` and `requests`
(see [Hardware](#hardware-runs-cost-real-money) below); reproducing the paper does not.

## Reproduce the theory results

None of these touch hardware or cost anything. Each reads/writes `results/*.json`.

```bash
.venv/bin/python experiments/theory_single_copy_verify.py   # exact variance law vs brute force (k=2,3,4)
.venv/bin/python experiments/run_budget_sweep.py            # the alpha transition (RMSE vs budget)
.venv/bin/python experiments/run_scaling.py                 # zeta_1, zeta_2, M* vs system size
.venv/bin/python experiments/run_crossover_theory.py        # the crossover criterion across the swept cells
.venv/bin/python experiments/general_k_variance.py          # generalization to k=3, k=4
.venv/bin/python experiments/run_stress_test.py             # out-of-ensemble validation (Haar/low-rank/GHZ)
.venv/bin/python experiments/cgk_mechanism.py               # reconciliation with Cotler-Gong-Kannan (ref [6])
```

The range-constrained re-analysis, which the paper reports alongside the unbiased
figures, is a second group:

```bash
PYTHONPATH=. .venv/bin/python -m experiments.run_clipping_audit    # how far out of range the raw estimates fall
PYTHONPATH=. .venv/bin/python -m experiments.run_trivial_baseline  # both routes against a data-free constant
PYTHONPATH=. .venv/bin/python -m experiments.run_estimand_spread   # per-state spread of the estimand, by ensemble
PYTHONPATH=. .venv/bin/python -m experiments.build_pass38_final    # the reconciled accuracy table
```

## Reproduce the figures

```bash
.venv/bin/python experiments/make_figures.py
```

This reads the saved `results/*.json` (it never re-runs the science) and writes each of
the six figures as a vector PDF, a 300-dpi PNG, and a CSV of the exact plotted data into
`results/figures/`. The PDFs used in the paper live in `paper/figures/`.

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

259 tests, all passing at HEAD.

## Building the paper

The paper source is [`paper/paper.tex`](paper/paper.tex); the supplement is
[`paper/supplementary.tex`](paper/supplementary.tex). Build either with:

```bash
tectonic paper/paper.tex
```

or with two `pdflatex` passes (the second resolves cross-references). Each document
carries its bibliography inline, so there is no separate BibTeX step and no `.bib` file
in the repository.

## Hardware runs cost real money

The hardware experiments in `anrl/hardware/` and the `experiments/run_*` / `experiments/*_analysis.py`
scripts submit circuits to the [Open Quantum](https://www.openquantum.com) platform (Rigetti
Cepheus-1-108Q). **Each job is billed in platform credits — running them costs money.**

- Credentials are read from a **gitignored** `.env` file (never commit it):

  ```
  OPENQUANTUM_CLIENT_ID=...
  OPENQUANTUM_CLIENT_SECRET=...
  ```

- You do **not** need credentials to reproduce any number in the paper. Every raw
  measurement count is already committed under `results/hardware/`, and every
  `*_analysis.py` script recomputes the paper's hardware numbers from those committed counts
  offline and for free. They do need `qiskit` and `qiskit-aer` installed (they import
  `anrl.hardware`), which `requirements.txt` provides — but no credentials and no credits.

## Data availability

`.gitignore` ignores `results/*` by default (to keep large regenerable outputs out of the
tree); the load-bearing files were force-added, so what a reader sees is exactly what the
paper's numbers depend on:

- **Raw hardware counts** — every submitted circuit (`results/hardware/*.qasm`, 69 files)
  and its raw shot counts (`results/hardware/*_counts.json`, 96 files), plus the
  per-campaign analysis JSONs. These are **irreplaceable** (they cost real credits to
  obtain) and were committed verbatim *before* any analysis; every locked prediction was
  committed *before* the corresponding job was submitted.
- **Theory / analysis outputs** — the variance-law, budget-scaling, crossover, general-$k$,
  stress-test, range-constrained, and CGK-reconciliation JSONs (39 files at the top level
  of `results/`).
- **Figure data** — `results/figures/*.csv` (the exact plotted values). The rendered
  `*.pdf`/`*.png` regenerate from these with `make_figures.py`.

Not tracked (regenerable or unrelated to the paper): a few ephemeral hardware working
files (credit quotes, status).

## License

[MIT](LICENSE) for the code in `anrl/`, `experiments/` and `tests/`; the committed
measurement data in `results/` may be reused under the same terms. The manuscript
in `paper/` is not MIT-licensed — it is © 2026 Alexander Messerlian and Ziwei Gu,
all rights reserved pending publication.
