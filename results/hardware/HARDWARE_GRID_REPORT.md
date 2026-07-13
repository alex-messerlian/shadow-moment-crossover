# GHZ-ladder hardware experiment — the paper's central run

Rigetti Cepheus, Public Compute. Collective SWAP on GHZ n=2,3,4 (10k shots, zero-routing ladders) + a 15-basis single-copy anchor at n=2. Spend: 24 credits (44 -> 20; all 4 spark + 20 full).

## Locked predictions (frozen before submission, from hg_locked.json)

| n | CZ | collective band lo/mid/hi | readout pen | gate pen |
|---|---|---|---|---|
| 2 | 4 | 0.726/0.716/0.702 | 0.242 | 0.042 |
| 3 | 7 | 0.624/0.609/0.587 | 0.317 | 0.074 |
| 4 | 10 | 0.552/0.532/0.504 | 0.363 | 0.105 |

Single-copy anchor n=2: locked U-statistic **1.500 ± 0.028** (15 bases x 600 shots, biased protocol; hardware reproducing this validates the pipeline).

## Collective route — measured vs locked band

| n | measured | 95% CI | band | verdict |
|---|---|---|---|---|
| 2 | 0.7090 | [0.695,0.722] | 0.702-0.726 | **INSIDE** ✓ |
| 3 | 0.3168 | [0.299,0.336] | 0.587-0.624 | **OUTSIDE** by 0.270 (below) |
| 4 | 0.4308 | [0.413,0.448] | 0.504-0.552 | **OUTSIDE** by 0.073 (below) |

**n=2 is CONFIRMED; n=3 and n=4 FAIL (both below their bands). The degradation is NON-MONOTONIC (n=3 = 0.317 is worse than n=4 = 0.431), which is impossible under the static readout-scaling model since the n=4 qubit set contains the n=3 set.** The measured distributions concentrate on the ideal GHZ-SWAP support (~0.42 weight vs 0.06-0.12 uniform), so the circuits ran correctly — the low, non-monotonic purities are genuine device behavior, not a bug.

## Scaling test (the paper's claim) — NOT confirmed

| n | predicted (mid) | measured | deviation |
|---|---|---|---|
| 2 | 0.716 | 0.709 | -0.007 |
| 3 | 0.609 | 0.317 | -0.292 |
| 4 | 0.532 | 0.431 | -0.101 |

The predicted monotonic, readout-dominated degradation (0.716 -> 0.609 -> 0.532) is NOT observed. Measured is 0.709 -> 0.317 -> 0.431 — n=2 on target, then a large under-shoot with n=3 the worst point. The prediction failed for n>=3.

## Error decomposition (readout vs gate)

| n | measured deficit (1-purity) | model readout | model gate | model total | unexplained excess |
|---|---|---|---|---|---|
| 2 | 0.291 | 0.242 | 0.042 | 0.284 | +0.007 |
| 3 | 0.683 | 0.317 | 0.074 | 0.391 | +0.292 |
| 4 | 0.569 | 0.363 | 0.105 | 0.468 | +0.101 |

**At n=2 the decomposition holds and confirms the hardware finding: readout (0.242) dominates gates (0.042) by ~6x, and the model total (0.284) matches the measured deficit (0.291).** At n=3,4 a large UNEXPLAINED excess appears (+0.29 at n=3, +0.10 at n=4) — the static readout+gate model under-predicts the deficit. So readout-over-gate is confirmed only where the model matches (n=2); at larger n an unmodeled component dominates.

## Single-copy anchor n=2 — pipeline NOT validated

Measured U-statistic **1.344** (CI [1.266,1.433]) vs locked **1.500 ± 0.028** -> **MISMATCH** (deviation -0.156; CIs do not overlap). The single-copy prediction pipeline, run on the exact same 15 bases, does NOT reproduce the hardware result.

## Root cause and falsification verdict (stated plainly)

Three independent signals fail together: (1) collective n=3,4 below bands, (2) non-monotonic collective scaling, (3) single-copy anchor mismatch. The common cause is **calibration DRIFT**: the device parameters were characterized in earlier sessions (readout in readout-extension; CZ bounded from the Bell run), and the device has moved since. n=2 collective (qubits {0,1,9,10}) happened to still match, but the newer qubits and a different measurement session did not. **As pre-registered, we report this as a failed prediction and do NOT adjust the model to fit.** The scientific content: static single-session characterization has limited predictive power across time on this NISQ device; the clean readout-dominated scaling holds at n=2 but is confounded by temporal drift at n=3,4.

## Route comparison (copy-fair) — compromised by the drift

The paper copy-fair theory predicts single-copy wins with the gap narrowing 4.3x -> 4.0x -> 3.5x (crossover n*=8). BUT: the n=2 single-copy anchor MISMATCHED hardware, so the single-copy prediction pipeline is not hardware-validated here, and the collective n=3,4 purities are drift-affected. We therefore CANNOT ground the route comparison in this run's hardware as intended. The only clean hardware point is n=2 collective (0.709, in band); the single-copy side and n>=3 are not trustworthy this session. Reported honestly rather than as a confirmed trend.

## Cost

* 24 credits consumed (3 collective x 3 + 15 single x 1). Final balance: full=20, spark=0 (was full=40, spark=4).
* Raw counts: results/hardware/hg_*_counts.json (committed before analysis). Public Plan: publications must attribute Open Quantum.
