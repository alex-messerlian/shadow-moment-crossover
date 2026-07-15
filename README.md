# Sample-complexity transition in shadow-based estimation of quantum state moments

Code, data, and analysis for the paper:

> **Sample-complexity transition in shadow-based estimation of quantum state moments and the crossover to collective measurement**
> Alexander Messerlian.
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
  under local-Pauli classical shadows, derived via the Hoeffding decomposition and
  verified against brute-force enumeration for $k = 2, 3, 4$.
- A **sample-complexity exponent transition**: the exponent $\alpha$ in
  $\mathrm{RMSE} \propto M^{-\alpha}$ is not the constant $1/2$ — it migrates to $1$ as the
  budget $M$ falls below a threshold $M^* \approx 5.3^n$ that diverges exponentially in $n$.
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

The theory and figures need only the core scientific stack (no GPU, no deep-learning deps):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pytest
```

For the full pinned versions, use `pip install -r requirements.txt` instead. The hardware
module additionally needs `qiskit` and the `openquantum_sdk`
(see [Hardware](#hardware-runs-cost-real-money) below).

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
  offline and for free.

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
