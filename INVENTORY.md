# Repository inventory

**Generated for the public-release decision.** Read-only audit — nothing was deleted, moved, or
edited to produce this file.

> **Post-audit update (repo-final-trim pass):** Decision 1 (prune the torch/torchrl/tensordict/gymnasium
> stack) and Decision 3 (delete `VOICE_NOTES.md`) were executed, and the stale RL line in
> `witness.py`'s docstring was fixed. The tables below still describe the pre-trim state; treat the
> `torch`/`VOICE_NOTES` rows and Decision 1 as historical.

- **Tracked files:** 362 · **Tracked size:** ~3.8 MB (results/ 2.0 MB, paper/ 0.9 MB, anrl/ 0.4 MB, experiments/ 0.3 MB, tests/ 0.2 MB).
- **State of the repo.** After the RL/witness declutter, the repository is the code, data, and paper
  for one project: the exact single-copy variance law, the two collective bias laws, the crossover,
  and the pre-registered hardware test. The import package is `anrl`; the live science lives in
  `anrl/{theory,benchmark,hardware,figures}` plus the shared `anrl/physics` utilities. Every paper
  number traces to a tracked file, and the suite passes from a clean clone (229 tests). What remains
  to decide is mostly *tidiness*, not correctness: a now-unused deep-learning dependency stack that
  only the environment-check script touches, a handful of abandoned entanglement-witness functions
  that survive inside the otherwise-live `physics` package, and several internal process/audit
  documents that are honest provenance but not everyone ships them.

Legend — **Recommendation**: KEEP (load-bearing or clearly belongs public) · OPTIONAL (harmless, judgment call) · CONSIDER CUTTING (dead weight / redundant / internal-only).

---

## Root files

| Path | What it is | Paper-referenced | Recommendation | Reason |
|---|---|---|---|---|
| `README.md` | Project README: what the paper is, how to reproduce theory/figures, hardware/credentials note, data-availability. | — (describes it) | KEEP | Front door of a public repo. |
| `LICENSE` | MIT license (2026 Alexander Messerlian). | — | KEEP | A public repo needs one. |
| `pyproject.toml` | Package metadata; dist name `shadow-moment-crossover`, import name `anrl`. Lists numpy/scipy/matplotlib + **torch/tensordict/torchrl/gymnasium** (lines 21–24) + tqdm. | — | KEEP (edit) | Correct, but the torch stack is now unused — see Decision 1. |
| `requirements.txt` | Pinned, mutually-compatible dependency set. Pins torch 2.12.1 / tensordict 0.13.0 / torchrl 0.13.2 (lines 17–19) and gymnasium (line 22). | — | KEEP (edit) | Same torch question — see Decision 1. |
| `check_env.py` | Env-check utility: imports the stack, prints versions, reports CUDA/MPS. **Safe to publish — reads no `os.environ`, touches no credential.** | No | OPTIONAL | Only consumer of torch/torchrl (lines 42–45, 56–79); its usefulness is tied to Decision 1. |
| `.gitignore` | Ignores `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.env`, `results/*` (with force-added exceptions). | — | KEEP | Correct and complete. |

---

## `anrl/` — the package

### `anrl/theory/` — the analytic results (all load-bearing)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `variance.py` | Single-copy Hoeffding/U-statistic variance law (the boxed §3.1 formula). | §3.1 | KEEP |
| `single_copy_law.py` | First-principles single-copy purity variance law + `M*` base; derived and independently verified. | §3, §3.6 | KEEP |
| `general_k.py` | Exact Hoeffding components ζ₁…ζₖ for k=2,3,4. | §3.5, general-k | KEEP |
| `general.py` | State-agnostic sampler + Hoeffding-component estimator (any density matrix). | §3 (supports general_k) | KEEP |
| `bias.py` | The two exact collective bias laws (global-depolarizing + per-qubit channel). | §4 | KEEP |
| `crossover.py` | The crossover, solved from the variance + bias laws. | §5 | KEEP |
| `clipping.py` | Clipped-estimator RMSE for a near-Gaussian purity estimate (backs the "finite-trial noise, not systematic" §8 caveat). | §8 caveat | KEEP |

### `anrl/benchmark/` — Monte-Carlo estimators and sweeps (all load-bearing)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `shadows.py` | Single-copy local-Pauli classical-shadow snapshots and estimator. | §2.1 | KEEP |
| `moments.py` | Cyclic-permutation moment operators; exact k=4 U-statistic (Möbius/coincidence tensor). | §2.2, App. A | KEEP |
| `moment_ustats.py` | Exact moment U-statistics (k=2,3,4). | §2.2 | KEEP |
| `collective.py` | Collective SWAP-test estimator (imports `purity`). | §2.3 | KEEP |
| `channels.py` | Noise channels: depolarizing, amplitude damping, dephasing. | §4, §5 | KEEP |
| `ensembles.py` | State ensembles: noisy-pure, GHZ, low-rank, Haar. | §2.5, §5.2 | KEEP |
| `scaling.py` | Single-vs-collective RMSE vs n at fixed budget. | Fig 1, §5 | KEEP |
| `sweep.py` | Benchmark sweep incl. subsampled-vs-exact inflation. | §2.2 | KEEP |
| `sweep_hardened.py`, `hardened.py` | Hardened (large-n) sweep and estimator. | §5.4, Fig 5 | KEEP |
| `budget.py`, `budget_sweep.py` | Budget-scaling (the α transition). | §3.5, Fig 1 | KEEP |
| `functionals.py` | Target functional (purity), re-exported from physics. | §2 | KEEP |
| `evaluation.py` | Estimator-evaluation utilities. | §2–§5 | KEEP |

### `anrl/hardware/` — device backend, circuits, protocol (all load-bearing for §6)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `backend.py` | Open Quantum / Cepheus auth + backend listing (reads `.env`, never hardcodes). | §6.1 | KEEP |
| `submit.py` | Resumable OpenQASM-3 job submission. | §6 | KEEP |
| `swap_test.py` | Destructive-SWAP circuit + parity sign rule. | §2.3, App. B | KEEP |
| `state_prep.py` | GHZ-ladder state preparation (imports `random_density`, `states.ghz`). | §6.2 | KEEP |
| `calibration.py`, `readout_model.py`, `noise_model.py` | Readout/gate calibration and device noise model. | §6.2, §6.3 | KEEP |
| `grid_predict.py` | Locked grid predictions from measured device parameters. | §6.2 | KEEP |
| `twirl.py` | Pauli twirling / randomized compiling (coherent-error test). | §6.5 | KEEP |
| `shadow_noise.py`, `shadows.py` | Shadow-on-hardware cost/noise helpers. | §6.7 | OPTIONAL — support the "shadows are economically infeasible" finding; verify still needed. |
| `shot_budget.py` | Shot-budget accounting. | §6 | KEEP |
| `cepheus_metadata.json` | Device metadata (topology, native gates). | §6.1 | KEEP |

### `anrl/physics/` — shared quantum-state utilities (KEEP the package; see Decision 2)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `states.py` | State construction: `depolarize`, `random_density`, `ghz`, `maximally_mixed`. **Live** (benchmark + hardware import these). | §2.5, §5, §6 | KEEP |
| `pauli.py` | Pauli machinery: `kron_all`, Pauli matrices/strings. **Live** (benchmark imports `kron_all`). | §2.1, §3 | KEEP |
| `entanglement.py` | **Mixed:** defines `purity()` — **live**, imported by 12 benchmark/hardware files — *and* the abandoned witness-line functions `partial_transpose`/`negativity`/`pt_moment` (0 live callers). | `purity`: yes; rest: no | KEEP (file), but see Decision 2 |
| `witness.py` | Entanglement-witness estimators (`witness_weights`, `negativity_witness_estimator`). **0 live callers** — only `test_witness`. | No | CONSIDER CUTTING — abandoned witness line (Decision 2) |
| `measurement.py` | Local-Pauli measurement simulator (`estimate_pauli_expectations`, `sample_counts`, …). **0 live callers** — only `test_measurement`. | No | CONSIDER CUTTING — abandoned witness line (Decision 2) |

### `anrl/figures/` — publication figure builders (load-bearing)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `figures.py` | The six `make_figN()` builders. | Figs 1–6 | KEEP |
| `data.py` | Reads `results/*.json` into figure-ready arrays. | Figs 1–6 | KEEP |
| `style.py` | Okabe–Ito palette + `save_figure`/`write_csv`. | Figs 1–6 | KEEP |

---

## `experiments/` — one runnable script per result (reproducibility scripts)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `run_scaling.py`, `run_budget_sweep.py`, `run_crossover_theory.py`, `run_stress_test.py`, `run_moment_sweep.py`, `run_hardened_scaling.py`, `run_benchmark_sweep.py` | **Local-only** theory reproducers — no hardware, no credentials; each writes a `results/*.json`. | §2–§5, Figs 1–5 | KEEP |
| `general_k_variance.py`, `cgk_mechanism.py`, `clipping_investigation.py` | Local analyses: general-k variance; CGK ([6]) reconciliation; the clipping negative result. | §3.5, §7, §8 | KEEP |
| `theory_single_copy_verify.py`, `theory_single_copy_scaling.py` | Verify the variance law vs brute force; ζ/M* scaling. | §3.1, §3.5 | KEEP |
| `make_figures.py` | Renders all six figures (PDF/PNG/CSV) from `results/*.json`. | Figs 1–6 | KEEP |
| `run_*` + `*_analysis.py` hardware pairs: `run_hardware_grid`/`hardware_grid_build`/`hardware_grid_analysis`, `run_coherent_error`/`coherent_error_build`/`coherent_error_analysis`, `run_register_fault`/`register_fault_build`/`register_fault_lib`/`register_fault_analysis`, `run_same_session`/`same_session_lib`/`same_session_analysis`, `run_joint_readout`/`joint_readout_analysis`, `run_readout_extension`/`readout_extension_analysis`, `run_cz_characterization`/`cz_characterization_analysis`, `run_device_characterization`/`device_characterization_analysis`, `hardware_validation`/`hardware_validation_analysis`, `hardware_prediction`, `grid_predictions` | Per hardware campaign: a **build/submit** script (costs credits — not needed to reproduce) and an **analysis** script that recomputes the paper's hardware numbers offline from committed raw counts. The `*_lib.py` files are shared helpers. | §6 | KEEP — the analysis halves are the offline reproducers; the build halves document exactly what was submitted. |
| `cepheus_locked_predictions.json`, `cepheus_prediction_report.md` | Early hardware-prediction-phase locked predictions + human-readable report (predecessor of the `results/hardware` locked grids). | §6 (context) | OPTIONAL — provenance of the prediction phase; author should confirm it isn't superseded by `results/hardware/locked_grid_predictions*.json`. |
| `.gitkeep` | Keeps the dir in tree. | — | KEEP |

---

## `results/` — saved outputs (force-added; the paper's numbers live here)

### Theory / analysis JSONs (root of `results/`)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `theory_zetas.json` | ζ₁, ζ₂, M* arrays. **Canonical** — read by the figure pipeline and tests. | §3.5, Fig 1 | KEEP |
| `theory_zetas_recomputed.json` | An independent MC recompute of the ζ/M* base (the 5.343 reproducibility check). Read only by `theory_single_copy_{verify,scaling}`. | §3.5 (the 5.343 cross-check) | OPTIONAL — verification recompute; author should confirm which ζ file is canonical (Decision 4). |
| `theory_derivation.json` | The M* derivation record (5.147 / 5.345). | §3.6 | KEEP |
| `budget_scaling.json` | Measured α at n=2,4,9 (0.495/0.528/1.006). Figure source. | §3.5, Fig 1 | KEEP |
| `general_k_variance.json` | k=3,4 variance verification data. | §3.5 | KEEP |
| `crossover_theory.json` | 83-cell crossover predicted-vs-measured. Figure source. | §5.2, Figs 2–3 | KEEP |
| `scaling_hardened.json` | Hardened exponential-wall RMSE (0.043→11.98). **Figure source** for Fig 5. | §5.4, Fig 5 | KEEP |
| `scaling_crossover.json` | Output of `run_scaling.py`; backs the §2.4 "purity stays near 0.8" claim. **Not** read by figures. | §2.4 | OPTIONAL — distinct from `scaling_hardened.json` (different claim); author should confirm both are intended (Decision 4). |
| `moment_sweep_corrected.json` | Corrected moment-family sweep (the "corrected" supersedes an earlier, non-committed run). | §3.5 (general-k) | KEEP |
| `benchmark_sweep.json` | Subsampled-vs-exact inflation sweep (~5× at n=2 → >50× at n=4). | §2.2 | KEEP |
| `stress_test.json`, `stress_components.json`, `stress_measurements.json` | Out-of-ensemble stress test (median 6.7%). `stress_test.json` is the figure source; the other two are intermediate. | §8, Fig 4 | KEEP (`stress_test`); OPTIONAL (`_components`,`_measurements` — intermediates) |
| `clipping_correction.json` | Clipping-does-not-explain-the-gap data. | §8 | KEEP |
| `cgk_bound.json`, `cgk_mechanism.json` | CGK ([6]) bound table + mechanism reconciliation. | §7 | KEEP |
| `cepheus_predictions.json`, `cepheus_metadata.json` | Early Cepheus predictions/metadata. | §6 (context) | OPTIONAL — predecessor of the locked hardware grids; confirm not superseded. |
| `THEORY_DERIVATION_REPORT.md`, `GENERAL_K_VARIANCE_REPORT.md`, `CLIPPING_CORRECTION_REPORT.md` | Human-readable audit summaries of the three theory phases. Referenced only by `paper/DISCREPANCIES.md` (`GENERAL_K`/`CLIPPING` referenced nowhere). | Indirect | OPTIONAL — standalone provenance; not load-bearing (Decision 5). |
| `.gitkeep` | Keeps dir in tree. | — | KEEP |

### `results/figures/` — figure data

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `fig{1–6}_*.csv` | The exact plotted values for each figure. | Figs 1–6 | KEEP — traceability of every plotted number. |
| *(untracked)* `fig{1–6}_*.pdf`, `*.png` | Rendered figures; regenerate from CSV via `make_figures.py`. The paper's copies live in `paper/figures/`. | Figs 1–6 | (untracked; fine) |

### `results/hardware/` — the irreplaceable raw hardware record (~0.9 MB, ~200 files)

| Path (group) | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `*_counts.json` + `*.qasm` per campaign — `ce_*` (coherent error, 40), `hg_*` (GHZ-ladder grid, 38), `jr_*` (joint readout, 21), `ss_*` (same-session bracket, 18), `rf_*` (register fault, 17), `ro_*` (readout, 15), `cz_*` (CZ characterization, 15), `char_*` (device characterization, 11), `bell_swap_*`, `raw_output.json` | Submitted OpenQASM-3 circuits and their **raw shot counts** — the irreplaceable measurement record (cost real credits; committed verbatim before analysis). | §6 (all hardware numbers) | KEEP — the core data-availability payload. |
| `*_analysis.json` (`analysis`, `hardware_grid_analysis`, `same_session_analysis`, `coherent_error_analysis`, `characterization_analysis`, `joint_readout_analysis`, `readout_extension_analysis`, `register_fault_analysis`, `readout_rates`, `gate_fit`) | Recomputed per-campaign analysis (purities, bands, fits). | §6.2–§6.5 | KEEP |
| `locked_grid_predictions.json` (v1) + `locked_grid_predictions_v2.json` | Pre-registered locked predictions. v2 *extends* v1 with measured n=3/4 readout (derived from v1); v1 remains the n=2 anchor. | §6.2–§6.4 | KEEP both — v2 supersedes v1 only for n=3/4; both are cited provenance. |
| `locked_same_session.json`, `submission.json`, `job_id.txt` | Locked same-session prediction, submission manifest, job id. | §6.4 | KEEP |
| `*_REPORT.md` (10: `COHERENT_ERROR`, `SAME_SESSION`, `GRID`, `HARDWARE_GRID`, `JOINT_READOUT`, `READOUT_EXTENSION`, `REGISTER_FAULT`, `CHARACTERIZATION`, `CZ`, `REPORT`) | Human-readable per-campaign write-ups. Referenced by `grid_predictions.py`, `DISCREPANCIES.md`, `VERIFICATION.md`. | Indirect | OPTIONAL — rich provenance; not read by the paper build (Decision 5). |

---

## `tests/` — 20 test files (all pass; 229 tests)

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `test_theory.py`, `test_single_copy_law.py`, `test_general_k_variance.py`, `test_clipping.py` | Exercise `anrl/theory`. | §3 | KEEP |
| `test_benchmark.py`, `test_benchmark_sweep.py`, `test_scaling.py`, `test_budget.py`, `test_hardened.py`, `test_sweep_hardened.py`, `test_stress.py`, `test_states_pauli.py` | Exercise `anrl/benchmark` + `anrl/physics` (states/pauli). | §2–§5 | KEEP |
| `test_hardware.py`, `test_hardware_prediction.py`, `test_grid_predictions.py` | Exercise `anrl/hardware`. | §6 | KEEP |
| `test_figures.py` | Exercises `anrl/figures` (asserts the 6-figure set, parameter-free theory curves). | Figs | KEEP |
| `test_cgk_mechanism.py` | Exercises the CGK reconciliation (47 checks). | §7 | KEEP |
| `test_entanglement.py`, `test_witness.py`, `test_measurement.py` | Exercise the abandoned witness-line functions in `physics/{entanglement,witness,measurement}.py`. `purity` (live) aside, these test **no live code**. | No | CONSIDER CUTTING — only with the witness files (Decision 2). |
| `.gitkeep` | Keeps dir in tree. | — | KEEP |

---

## `paper/` — the manuscript and its provenance

| Path | What it is | Paper-referenced | Recommendation |
|---|---|---|---|
| `paper.tex` | **The single source of truth** (REVTeX, embedded bibliography). | — | KEEP |
| `paper.pdf` | Compiled manuscript, 35 pages. | — | KEEP |
| `refs.bib` | BibTeX references (all author-verified against arXiv). | — | KEEP |
| `figures/fig{1–6}_*.pdf` | The figure PDFs embedded by `paper.tex`. | Figs 1–6 | KEEP |
| `BUILD.md` | How to build the paper (Overleaf / local tectonic). | — | KEEP — useful build instructions. |
| `VERIFICATION.md` | Verification checklist + the "all 16 refs verified" closure note. | — | OPTIONAL — verification **provenance** (valuable to show rigor); not required to build (Decision 3). |
| `DISCREPANCIES.md` | The number-by-number correction audit (WRONG/CONFIRMED table + fixes). | — | OPTIONAL — strong provenance of the verification pass (Decision 3). |
| `archive/PAPER_draft.md`, `archive/OUTLINE_draft.md` | Historical Markdown draft + planning outline, headed "HISTORICAL DRAFT — superseded by paper.tex." | — | OPTIONAL — clearly labelled history; harmless. |

*(`VOICE_NOTES.md`, the internal prose-editing log, was deleted in the repo-final-trim pass.)*

---

## Untracked-but-present (a cloner won't see these; a local reader might)

- `.env` (gitignored) — real Open Quantum credentials. **Never commit.** Correctly ignored.
- `.DS_Store`, `.pytest_cache/`, `__pycache__/` — OS/tooling cruft; ignored, absent from a clone.
- `results/figures/*.pdf`, `*.png` — rendered figures; regenerate from the tracked CSVs.
- `results/hardware/{char_quotes.json, last_status.txt, prep_state.json}` — ephemeral hardware working files (credit quotes, status); deliberately untracked, non-scientific.

---

## Decisions for the author

1. **Prune the deep-learning dependency stack?** After the RL removal, **nothing in `anrl/`, `experiments/`, or `tests/` imports torch/torchrl/tensordict/gymnasium — `check_env.py` is the sole consumer.** Pruning them removes a ~2 GB install for zero functional loss. Exact edits: `pyproject.toml` lines 16, 21–24; `requirements.txt` lines 4–7, 17–19, 22; `check_env.py` lines 8, 42–45, 56–79 (drop the torch/torchrl checks and the CUDA/MPS section). *Recommendation:* **prune them** — the tradeoff is losing the historical "this exact RL env installed" record, which is no longer relevant to the paper.

2. **Cut the abandoned entanglement-witness code?** `physics/witness.py` and `physics/measurement.py` have **0 live callers** (only `test_witness`/`test_measurement`), and the `negativity`/`partial_transpose`/`pt_moment` functions in `physics/entanglement.py` are likewise dead. The catch: `entanglement.py` also defines **`purity()`, which 12 live benchmark/hardware files import**, and `physics/__init__.py` re-exports all of it. So a clean cut is a small refactor, not a delete: move `purity()` to `states.py`, delete `witness.py`/`measurement.py` and the dead functions in `entanglement.py`, update `physics/__init__.py`, and drop the three tests. *Recommendation:* **cut it** for a public repo (it's a visibly abandoned line, and `witness.py` even still says "no reinforcement-learning logic"), but only via that refactor — I did **not** touch it because it's coupled to live `purity`.

3. **Ship the paper's process/provenance docs?** `VERIFICATION.md` and `DISCREPANCIES.md` are genuine verification provenance — they show every number was checked and record what was corrected, which *strengthens* a reader's trust; I'd **keep** them (or move under `paper/archive/`). `VOICE_NOTES.md` (an internal prose-editing log with the least reader value) **was deleted** in the repo-final-trim pass. `BUILD.md` is plainly useful — keep. *Recommendation:* keep VERIFICATION + DISCREPANCIES as provenance.

4. **Confirm the canonical scaling / ζ files.** Two pairs have similar names: `scaling_crossover.json` (backs §2.4's "purity near 0.8"; not read by figures) vs `scaling_hardened.json` (the Fig-5 source); and `theory_zetas.json` (canonical, used by figures/tests) vs `theory_zetas_recomputed.json` (a verification recompute). Neither pair is strictly redundant, but the similar names invite confusion. *Recommendation:* keep all four, and consider a one-line note in each JSON's producing script clarifying which is which — **author should confirm** none is a stale leftover.

5. **The standalone `*_REPORT.md` audit files** (3 in `results/`, 10 in `results/hardware/`) are human-readable per-phase write-ups, not read by the paper build (a few are referenced by `DISCREPANCIES.md`/`VERIFICATION.md`/`grid_predictions.py`). They're informative provenance but add file count. *Recommendation:* **keep** — they cost little and make the raw hardware data legible to a reader; cut only if you want a leaner tree.
