# Joint (multi-qubit) readout characterization — testing cross-copy crosstalk

Diagnostic for the same-session falsification (collective SWAP below bands + reproducible non-monotonicity n=3<n=4). Hypothesis: the SWAP reads all 2n qubits at once and the purity is a parity sum over copy pairs (i, i+n); if readout crosstalk correlates a copy-A qubit with its copy-B partner, that lands on the parity and single-copy characterization is blind to it. Spend: 10 credits (balance 27 -> 17 full).

## State set and rationale (X-gate only, all 2n qubits measured together)

On the exact SWAP registers. n=3 {0,1,2,9,10,11} (pairs $0-$9, $1-$10, $2-$11), 7 states; n=4 {0,1,2,3,9,10,11,12}, 3 states.

| state | excited | probes |
|---|---|---|
| s0 | none | baseline joint readout — cross-copy correlation when both members idle |
| h0/h1/h2 | one copy-A qubit | does exciting a copy-A qubit corrupt its idle copy-B partner |
| copyA | all copy-A | aggregate copy-A -> copy-B (partner excited, member idle) |
| copyB | all copy-B | symmetric copy-B -> copy-A |
| all | all | cross-copy correlation in the excited context (higher flip rates) |

## n=3 results

* **Non-factorizability (TVD of measured all-idle vs independent product): 0.0094** — essentially factorizable; the joint readout is well described by independent per-qubit errors.
* **Cross-copy parity-pair correlation (mean Pearson): -0.0145** vs non-pair 0.0025 — negligible, and NOT stronger than random qubit pairs.

| parity pair | P(partner flips \| qubit flips) | marginal P(partner) | Pearson |
|---|---|---|---|
| 0-9 | 0.0091 | 0.0072 | +0.0048 |
| 1-10 | 0.0000 | 0.0792 | -0.0349 |
| 2-11 | 0.0179 | 0.0280 | -0.0133 |

* **Re-prediction:** independent model 0.606, joint model (with measured correlation) 0.6122, measured 0.3784. The joint prediction barely moves from independent — **the gap is NOT closed** (residual -0.228 -> -0.234).

## n=4 results

* **Non-factorizability (TVD of measured all-idle vs independent product): 0.0114** — essentially factorizable; the joint readout is well described by independent per-qubit errors.
* **Cross-copy parity-pair correlation (mean Pearson): -0.0054** vs non-pair -0.0012 — negligible, and NOT stronger than random qubit pairs.

| parity pair | P(partner flips \| qubit flips) | marginal P(partner) | Pearson |
|---|---|---|---|
| 0-9 | 0.0137 | 0.0140 | -0.0009 |
| 1-10 | 0.0769 | 0.0400 | +0.0193 |
| 2-11 | 0.0000 | 0.0176 | -0.0284 |
| 3-12 | 0.0000 | 0.0068 | -0.0117 |

* **Re-prediction:** independent model 0.5789, joint model (with measured correlation) 0.5797, measured 0.4204. The joint prediction barely moves from independent — **the gap is NOT closed** (residual -0.159 -> -0.159).

## Sensitivity / statistical power (is the null real?)

Two independent framings, both showing the null is not low-power:

**(a) Statistical resolution.** The SE of each parity-pair Pearson at 2500 shots (flip rates 1-8%) is ~0.021, so every measured pair value (mean -0.0145) sits within ~0.2-1.7 SE of zero, and all fall inside the non-pair spread (SD 0.023). A gap-closing correlation would show up at z~12.

**(b) Physical bracket (the decisive test).** Hold the measured single-qubit flip rates fixed and push each parity pair to its physical correlation extremes (Frechet bounds — only the correlation varies, marginals preserved exactly). Because max(both-flip prob) <= min(the two marginals), and the measured flip rates are tiny (n=3 per-pair caps [0.007, 0.014, 0.028]), there is almost no room to bend the parity:

| n | independent | measured-corr joint | full physical bracket (marginals fixed) | residual even at MAX correlation |
|---|---|---|---|---|
| 3 | 0.606 | 0.6122 | **[0.611, 0.669]** | measured 0.378 -> residual **-0.233** |
| 4 | 0.579 | 0.5797 | **[0.594, 0.632]** | measured 0.420 -> residual **-0.174** |

**Even the maximally-correlated readout that is consistent with the measured single-qubit rates leaves the ~0.23 (n=3) / ~0.17 (n=4) gap essentially untouched.** No physically-valid cross-copy readout correlation can close it. (An earlier draft quoted an injection curve reaching 0.41 at "rho=0.2"; that scheme silently inflated the marginals and is not physically realizable at the measured flip rates — the marginal-preserving bracket above is the correct power argument, and it makes the refutation stronger, not weaker.)

## Does it explain the non-monotonicity? NO

The n=3 parity-pair correlation (mean Pearson -0.0145) is NOT stronger than n=4 (-0.0054) — both are negligible. Cross-copy readout crosstalk cannot be the layout-dependent mechanism behind n=3 < n=4.

## Verdict — hypothesis REFUTED (model not tuned)

Cross-copy readout crosstalk is **ruled out**: the joint readout is factorizable (TVD ~0.01), the parity-pair correlations are negligible (~0.01) and no stronger than random pairs, adding them does not close the gap, and — decisively — even the maximally-correlated readout consistent with the measured single-qubit flip rates leaves the gap essentially untouched (physical bracket n=3 [0.611, 0.669], n=4 [0.594, 0.632]). The unexplained residual remains **~0.23 (n=3), ~0.16 (n=4)**, and the non-monotonicity is still unexplained.

**Now ruled out:** (1) calibration drift (same-session, ~1%/qubit), (2) single-copy readout (characterized, in the model), (3) gate count (n=4 has more CZ but is better), (4) cross-copy readout crosstalk (this run). **Remaining candidate (untested here):** gate-level coherent/crosstalk error DURING the circuit — e.g. ZZ coupling or coherent over-rotation on the cross-copy CZ gates of the SWAP body, which a readout experiment cannot probe. That is the next diagnostic, not a model tweak.

## Cost

* 10 credits (7 n=3 + 3 n=4). Balance: full=17, spark=0. Raw counts committed before analysis. Public Plan: publications must attribute Open Quantum.
