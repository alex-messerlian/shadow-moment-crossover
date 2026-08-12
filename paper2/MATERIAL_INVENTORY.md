# Paper two: material inventory

The hardware case study, extracted from paper one's frozen manuscript so that paper one's
rewrite can delete it without losing anything.

**Status.** Source material only. Paper two is **not written**. Nothing here has been edited:
every file in `source/` is a verbatim copy carrying a provenance header with its source blob
and line range.

**Provenance.** Extracted at commit `f07f38b` from
`paper/paper.tex` blob `13590dbfe8119022ae95ff7a374910e30146ab8c` and
`paper/supplementary.tex` blob `2076cda654e58c43e1cc96dfc46ee239900dffec`, both byte-frozen.
Regenerate with `experiments/pass48_extract_paper2.py` (it re-checks the blobs and refuses to
run against modified text).

**Totals.** 15 fragments, 602 source lines, **4259 prose words**.

## Fragments

| File | Section | Source | Words | What it is |
|---|---|---|---|---|
| [`01_abstract_hardware_paragraph.tex`](source/01_abstract_hardware_paragraph.tex) | Abstract, final paragraph | `paper/paper.tex` L64-71 | 68 | The hardware result as the abstract states it: the 108-qubit backend, both locked predictions missed (0.8932 -> 0.7184 and 1.500 -> 1.344), and the disclaimer that the crossover was not demonstrated. |
| [`02_intro_hardware_sentences.tex`](source/02_intro_hardware_sentences.tex) | Section 1, final paragraph | `paper/paper.tex` L158-164 | 71 | The introduction's framing of the hardware test: the Peng et al. precedent, the locked-in-advance claim, and the shape-of-the-obstacle summary. |
| [`03_sec2_3_hardware_pointers.tex`](source/03_sec2_3_hardware_pointers.tex) | Section 2.3 | `paper/paper.tex` L243-246 | 41 | The sentence tying the destructive SWAP gate count to hardware affordability. Paper one keeps the gate count but must reword this pointer; paper two inherits the claim. |
| [`04_sec2_3_kge3_gate_ratios.tex`](source/04_sec2_3_kge3_gate_ratios.tex) | Section 2.3 | `paper/paper.tex` L255-262 | 87 | The k>=3 ancilla-based Hadamard-test gate ratios (~10x at k=3, ~16x at k=4) transpiled to the tested coupling map, and the Section 6.2 scoping note. NOTE: these counts are reproducible offline from the committed anrl/hardware/cepheus_metadata.json, so paper one KEEPS them as compilation facts; only the Section 6.2 pointer needs rewording. |
| [`05_sec5_2_readout_asymmetry.tex`](source/05_sec5_2_readout_asymmetry.tex) | Section 5.2 | `paper/paper.tex` L902-920 | 214 | The noise-model asymmetry paragraph, including the measured per-qubit readout rates fed back through both routes and the resulting conservative-upper-bound statement. Paper one loses the measured rates and must either cite a rate or drop the quantitative claim. |
| [`06_sec5_4_hardware_pointer.tex`](source/06_sec5_4_hardware_pointer.tex) | Section 5.4 | `paper/paper.tex` L1075-1083 | 83 | The 'entangling overhead is real but not the binding constraint' paragraph and its forward reference to the hardware section. The claim itself survives in paper one on the transpilation counts; the forward reference does not. |
| [`07_section6_full.tex`](source/07_section6_full.tex) | Section 6, all six subsections | `paper/paper.tex` L1094-1350 | 1691 | The hardware case study in full: platform and execution-path disclosure (6.1), the entangling-overhead measurement (6.2), the readout and SPAM characterization with the four-qubit table (6.3), cross-session variation with Figure 6 (6.4), the seven-mechanism elimination ledger as a longtable (6.5), and what remains (6.6). This is paper two's core. |
| [`08_sec7_hardware_comparison.tex`](source/08_sec7_hardware_comparison.tex) | Section 7 | `paper/paper.tex` L1417-1434 | 153 | Related Work: 'Crossovers and hardware comparison' -- the Pauli/Clifford ensemble crossover and the Peng et al. hardware comparison of the two routes. |
| [`09_sec7_bias_floor_on_hardware.tex`](source/09_sec7_bias_floor_on_hardware.tex) | Section 7 | `paper/paper.tex` L1435-1450 | 139 | Related Work: 'Statistical-to-bias-floor transitions on hardware' -- the March 2026 report of O(M^-1/2) decay saturating at a hardware floor, and how it differs from ours. |
| [`10_sec7_nisq_stability.tex`](source/10_sec7_nisq_stability.tex) | Section 7 | `paper/paper.tex` L1526-1536 | 47 | Related Work: 'NISQ device stability' -- Dasgupta and Humble over 22 months, and the calibration-drift analyses. This is paper two's direct literature. |
| [`11_sec8_hardware_limitations.tex`](source/11_sec8_hardware_limitations.tex) | Section 8 | `paper/paper.tex` L1554-1580 | 224 | Four Limitations items: the channel abstraction (which points at Section 6), the crossover not demonstrated on hardware, the computed-not-measured single-copy baseline, and the unverifiable Public Tier execution path. |
| [`12_sec8_readout_simulation.tex`](source/12_sec8_readout_simulation.tex) | Section 8 | `paper/paper.tex` L1607-1612 | 54 | The fifth hardware Limitations item: the readout sensitivity check is a simulation using the Section 6.3 rates. |
| [`13_sec9_hardware_conclusion.tex`](source/13_sec9_hardware_conclusion.tex) | Section 9 | `paper/paper.tex` L1654-1676 | 230 | The conclusion's two hardware paragraphs: the entangling-overhead accounting against the measured deficit, and 'what blocks the demonstration is readout fidelity and session-to-session variation' -- which is paper two's thesis sentence. |
| [`14_acknowledgements_platform.tex`](source/14_acknowledgements_platform.tex) | Acknowledgements | `paper/paper.tex` L1687-1696 | 58 | The platform paragraph: Open Quantum / Quantum Rings, Public Tier, the licence conditions requiring citation and anonymized data contribution, and the backend attribution. Paper two must carry this verbatim -- it is a licence obligation. |
| [`15_supplementary_S1_S6.tex`](source/15_supplementary_S1_S6.tex) | Supplementary S1-S6 | `paper/supplementary.tex` L52-230 | 1099 | The hardware supplement in full: S1 hardware protocol, S2 weight-resolved readout crosstalk, S3 the bracketed same-session experiment, S4 the three-session comparison, S5 the single-copy anchor, S6 cloud-QPU economics. S7 (the full crossover table) is theory and stays with paper one. |

## What paper one keeps

Two fragments are listed above but are **not** purely paper two's:

- `04_sec2_3_kge3_gate_ratios.tex` — the k>=3 gate ratios (~10x, ~16x) and the n=2 four-CZ /
  zero-routing counts are **transpilation facts**, reproducible offline from the committed
  `anrl/hardware/cepheus_metadata.json` with no credentials and no device access. Paper one
  keeps them in Section 2 as compilation costs; only the "Section 6.2" pointer is rewritten.
- `06_sec5_4_hardware_pointer.tex` — the claim that the entangling overhead is real but not the
  binding constraint rests on those same transpilation counts, so it survives in paper one. Only
  the closing forward reference to the hardware section goes.

Everything else in the list leaves paper one entirely.

## Backing artifacts (in this repository, not moved)

Both manuscripts cite the same public repository, so the hardware data stays where it is.

- `results/hardware/` — 192 files, 0.29 MB: raw OpenQASM 3 circuits as
  submitted, per-job measurement counts, the twirled coherent-error series, the calibration
  brackets, the three-session repeats, and the analysis JSONs.
- `experiments/` — 25 hardware scripts (explicit audited list, not a name match):
  `coherent_error_analysis.py`, `coherent_error_build.py`, `cz_characterization_analysis.py`, `device_characterization_analysis.py`, `grid_predictions.py`, `hardware_grid_analysis.py`, `hardware_grid_build.py`, `hardware_prediction.py`, `hardware_validation.py`, `hardware_validation_analysis.py`, `joint_readout_analysis.py`, `readout_extension_analysis.py`, `register_fault_analysis.py`, `register_fault_build.py`, `register_fault_lib.py`, `run_coherent_error.py`, `run_cz_characterization.py`, `run_device_characterization.py`, `run_hardware_grid.py`, `run_joint_readout.py`, `run_readout_extension.py`, `run_register_fault.py`, `run_same_session.py`, `same_session_analysis.py`, `same_session_lib.py`.
- `anrl/hardware/` — 13 modules: backend and credentials, calibration, the noise
  and readout models, SWAP-test and shadow circuit construction, Pauli twirling, shot budgeting,
  and the committed `cepheus_metadata.json` device description.

Three near-misses stay with **paper one**, and a name-based split would get them wrong:
- `experiments/run_clipping_crossover_grid.py`
- `experiments/run_clipping_paired_grid.py`
  — the PASS 36 clipping and paired-z-test grids, which are theory.
- `anrl/benchmark/readout_shadows.py`
- `anrl/benchmark/readout_correction.py`
  — readout *simulation* modules. They consume the measured per-qubit rates, but they implement
  paper one's Section 5.2 sensitivity check, not a hardware measurement.

## Figure

Figure 6 (`paper/figures/fig6_hardware.*`) is paper two's; its caption is inside
`07_section6_full.tex`. Figures 1-5 are paper one's and are untouched.

## Not extracted, deliberately

- Supplementary S7, the full crossover table: theory, stays with paper one.
- Section 7's "Collective measurement on noisy hardware" paragraph: despite the title it is the
  theoretical reconciliation with the noisy-purity-testing lower bound, so it stays with paper
  one. Paper one should rename the paragraph, since "on noisy hardware" will no longer be apt.
