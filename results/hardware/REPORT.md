# Cepheus hardware validation — result

Single job on Rigetti Cepheus-1-108Q (Public Compute, Standard queue), 5000 shots, physical qubits {0,1,9,10}, 4 CZ, no routing SWAPs.

## Measured purity vs locked prediction

* **Measured purity: 0.7184**  (95% bootstrap CI [0.6992, 0.7376], analytic SE 0.0098). Endianness-invariant (0.7184).
* Locked prediction: 0.9412 (Qiskit), 0.9401 (bias law); true Bell purity is exactly 1.0.
* **Deviation: -0.2228** — measured is LOWER than predicted.
* In predicted band (0.92, 0.96)? **False**. Prediction inside measured CI? False.

**Prediction NOT confirmed.** The device is noisier on our specific qubits than the published-median-based prediction. The measured distribution is a correctly-executed but noisy Bell-SWAP test: its four dominant outcomes are exactly the ideal support {0000,0011,1100,1111}, with ~25% of weight leaked into the other 12 outcomes by noise.

## Inverted effective noise vs Rigetti published spec

* Effective global-depolarizing g = **0.375** (the prediction implied g ~ 0.10-0.14).
* Consistent (p_ro, p2) level set for the measured purity:

  | assumed readout p_ro | implied CZ depol p2 | implied CZ avg err |
  |---|---|---|
  | 0.005 | 0.1030 | 0.0772 |
  | 0.010 | 0.0969 | 0.0727 |
  | 0.020 | 0.0846 | 0.0634 |
  | 0.030 | 0.0718 | 0.0539 |
  | 0.050 | 0.0453 | 0.0340 |

* At the assumed 2% readout, implied CZ avg error is 0.063 (~7x the 0.009 published median).
* Alternatively, if gates are AT spec, readout must be **7.4% per qubit** to explain the data. Readout was NOT in the datasheet (we assumed 2%); the prediction phase flagged it as the dominant unknown, and the hardware confirms readout is the leading suspect.

## Cost

* Estimated: 2 credits (quote). **Consumed: 2 spark credits.**
* Balance after: full=20, spark=23 (before: full=20, spark=25).
* Raw counts (irreplaceable): `results/hardware/raw_output.json`. Submitted circuit: `results/hardware/bell_swap_cepheus_q3.qasm`.

Public Plan: any publication must attribute Open Quantum (www.openquantum.com/citation).
