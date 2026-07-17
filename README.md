# Budget-scaling transition in shadow-based moment estimation

Code, data, and analysis for the paper:

> **Budget-scaling transition in shadow-based moment estimation and the crossover to collective measurement**
> Alexander Messerlian (Independent Researcher) and Ziwei Gu (Harvard John A. Paulson School of Engineering and Applied Sciences).
> Compiled manuscript: [`paper/paper.pdf`](paper/paper.pdf) · LaTeX source: [`paper/paper.tex`](paper/paper.tex)

## What the paper is

Nonlinear functionals of a quantum state — the purity $\mathrm{Tr}(\rho^2)$ and the
higher moments $\mathrm{Tr}(\rho^k)$ — can be estimated two ways: from **single-copy**
randomized (classical-shadow) measurements, at a cost that grows exponentially with
system size, or from **collective** measurements on $k$ copies, at $O(1)$ variance but
at the price of entangling gates and the noise they carry. The paper works out both
sides of that ledger exactly.

The main results:

- An **exact, state-dependent variance law** for the $k$-th-moment U-statistic estimator
  under local random-unitary classical shadows, derived via the Hoeffding decomposition and
  verified against brute-force Monte Carlo for $k = 2, 3, 4$.
- A **budget-scaling exponent transition**: the *effective* exponent $\alpha$ in
  $\mathrm{RMSE} \propto M^{-\alpha}$, over the budget $M$ in use, is not the constant $1/2$ —
  it migrates continuously toward $1$, and past it for higher moment orders, as $M$ falls below
  a threshold $M^* \approx 5.3^n$ that diverges exponentially in $n$. The asymptotic scaling at
  fixed $n$ remains the familiar square-root law; the migration is a finite-budget effect.
- Two **exact, parameter-free collective bias laws** (global-depolarizing and per-qubit-channel).
- A **parameter-free crossover law** for the system size at which collective measurement
  becomes cheaper, validated across 83 cells and four state ensembles.
- A **pre-registered hardware test** on a 108-qubit superconducting processor. The prediction
  fails, and the paper reports the failure and its diagnosis (readout error and cross-session
  drift dominate; the entangling overhead does not).

## What the code does

Everything is in the `anrl` package:

| module | contents |
|---|---|
| `anrl/theory/` | the exact variance law, the Hoeffding/Lee U-statistic decomposition, the two bias laws, the crossover, and the analytic threshold $M^*$ |
| `anrl/benchmark/` | Monte-Carlo estimators (single-copy shadows, collective SWAP test), noise channels, moment operators |
| `anrl/physics/` | state ensembles (Haar-pure, noisy-pure, low-rank, GHZ) and Pauli machinery |
| `anrl/figures/` | the publication figure builders (Okabe–Ito palette; PDF/PNG/CSV export) |
| `anrl/hardware/` | Open Quantum / Rigetti Cepheus backend, circuit builders, the destructive-SWAP protocol |

`experiments/` holds one runnable script per result; `results/` holds the saved outputs;
`tests/` is the test suite.

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
.venv/bin/python experiments/run_crossover_theory.py        # crossover law across 83 cells
.venv/bin/python experiments/general_k_variance.py          # generalization to k=3, k=4
.venv/bin/python experiments/run_stress_test.py             # out-of-ensemble validation (Haar/low-rank/GHZ)
.venv/bin/python experiments/cgk_mechanism.py               # reconciliation with Cotler-Gong-Kannan (ref [6])
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

## Building the paper

The paper source is [`paper/paper.tex`](paper/paper.tex). Build it with:

```bash
tectonic paper/paper.tex
```

or with two `pdflatex` passes (the second resolves cross-references). The bibliography is
embedded in the `.tex`, so there is no separate BibTeX step.

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

- **Raw hardware counts** — every submitted circuit (`results/hardware/*.qasm`) and its raw
  shot counts (`results/hardware/*_counts.json`), plus per-campaign analysis JSONs and
  reports. These are **irreplaceable** (they cost real credits to obtain) and were committed
  verbatim *before* any analysis; every locked prediction was committed *before* the
  corresponding job was submitted.
- **Theory / analysis outputs** — the variance-law, budget-scaling, crossover, general-$k$,
  stress-test, and CGK-reconciliation JSONs.
- **Figure data** — `results/figures/*.csv` (the exact plotted values). The rendered
  `*.pdf`/`*.png` regenerate from these with `make_figures.py`.

Not tracked (regenerable or unrelated to the paper): a few ephemeral hardware working
files (credit quotes, status).

## License

[MIT](LICENSE) for the code; the committed measurement data may be reused under the same terms.
