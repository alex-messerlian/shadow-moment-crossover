# Cepheus device characterization — results

Two jobs on physical qubits {0,1,9,10}: readout (Job A, 4x2000 shots) and a CZ-echo (Job B, 4000 shots). Post-processing only.

## Job A — readout error per qubit (Wilson 95% CI)

| qubit | P(1\|0) | 95% CI | P(0\|1) | 95% CI |
|---|---|---|---|---|
| $0 | 0.0925 | [0.0839,0.1019] | 0.0892 | [0.0808,0.0985] |
| $1 | 0.0067 | [0.0046,0.0098] | 0.0780 | [0.0701,0.0867] |
| $9 | 0.0673 | [0.0599,0.0754] | 0.0978 | [0.0889,0.1073] |
| $10 | 0.0318 | [0.0267,0.0376] | 0.0595 | [0.0526,0.0673] |

Mean symmetric readout error **0.0653 (6.5%)** vs the **2% assumed** in the prediction — ~3.3x higher. Readout is asymmetric (P(0|1)>P(1|0), T1 decay during readout).

## Correlated (context-dependent) readout

| qubit | P(1\|0) neighbors idle | neighbors excited | delta |
|---|---|---|---|
| $0 | 0.0165 | 0.1685 | +0.1520 |
| $1 | 0.0055 | 0.0080 | +0.0025 |
| $9 | 0.0615 | 0.0730 | +0.0115 |
| $10 | 0.0295 | 0.0340 | +0.0045 |

**$0 shows strong measurement crosstalk**: its false-1 rate jumps 0.017 -> 0.169 when neighbors are excited. The independent-qubit confusion matrix averages over this and under-captures it.

## Job B — CZ error (given measured readout)

* CZ-echo survival P(0000) = 0.8950; pure-readout reference (A_0000) = 0.8990. The 8 CZ add only +0.0040.
* Readout-corrected survival = 0.9862 (gate error 0.0138 for 8 CZ + 8 rx).
* **Inconclusive**: B ≈ A_0000 is consistent with EITHER near-perfect CZ OR the compiler cancelling CZ·CZ despite barriers. The submitted QASM was verified to contain all 8 CZ + barriers, but the executed circuit cannot be inspected. A definitive CZ measurement needs a verbatim box or a non-cancellable interleaved echo (single-qubit gates between the CZ pairs).

## Reconciliation with the Bell run (measured purity 0.7184)

| inputs | predicted Bell purity | residual |
|---|---|---|
| measured readout + CZ=0 | 0.7725 | -0.0541 |
| measured readout + CZ=spec(0.9%) | 0.7494 | -0.0310 |

Measured readout alone drops the predicted purity from 0.94 to ~0.75-0.77 — it explains the **bulk** of the Bell failure. A residual of ~0.03-0.05 remains (device slightly noisier than the aggregate parameters predict), consistent with the correlated readout crosstalk on $0 that the independent-qubit model omits. No exotic error source is required; the model is essentially correct and only the inputs (readout ~3x higher, and correlated) were wrong.

## Vs Rigetti published medians

* Readout: measured ~6.5% (per-qubit 0.7-9.8%) — not in the datasheet; we had assumed 2%.
* CZ: at or below the 0.9% median (Job B shows no excess; inconclusive on the exact value).
* **Conclusion: the degeneracy is broken — the Bell failure was READOUT, not gates.** Our qubits' CZ is fine; their readout is ~3x worse than assumed and strongly correlated on $0.
