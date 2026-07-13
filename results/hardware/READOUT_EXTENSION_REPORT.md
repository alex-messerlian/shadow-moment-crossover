# Readout extension — n=3/n=4 characterization + grid re-lock

Seven X-only basis states (weights 0,2,4,4,6,6,8), 3000 shots each, on the 8 physical qubits of the n=4 GHZ SWAP circuit. Cost: 7 credits (spark 11->4, full unchanged 40).

## Physical qubits — they NEST

* n=3 GHZ SWAP: [0, 1, 2, 9, 10, 11]  ;  n=4 GHZ SWAP: [0, 1, 2, 3, 9, 10, 11, 12] (re-derived from the transpiler, zero routing).
* n=3 set is a SUBSET of n=4 -> one characterization covers both. Newly characterized: [2, 3, 11, 12] (previously assumed mean rates). Note: Haar n=4 maps to {0,1,2,9,10,11,18,19} — $18,$19 remain uncharacterized (mean-rate fallback, flagged).

## Measured readout per qubit (Wilson 95% CI)

| qubit | P(1\|0) | P(0\|1) | P(1\|0) by excitation weight w |
|---|---|---|---|
| $0 | 0.1618 [0.154,0.170] | 0.0913 | w0:0.0197, w4:0.2417, w6:0.224 |
| $1 | 0.0103 [0.008,0.013] | 0.0761 | w0:0.0117, w4:0.0103, w6:0.009 |
| $2 (NEW) | 0.0311 [0.028,0.035] | 0.0711 | w0:0.0277, w2:0.0347, w4:0.031 |
| $3 (NEW) | 0.0157 [0.013,0.018] | 0.0568 | w0:0.014, w2:0.0177, w4:0.0153 |
| $9 | 0.0614 [0.057,0.067] | 0.0987 | w0:0.0637, w2:0.058, w4:0.0627 |
| $10 | 0.0272 [0.024,0.031] | 0.0589 | w0:0.023, w2:0.0203, w4:0.0383 |
| $11 (NEW) | 0.0447 [0.041,0.049] | 0.0960 | w0:0.048, w2:0.04, w4:0.0457, w6:0.045 |
| $12 (NEW) | 0.0692 [0.065,0.074] | 0.0797 | w0:0.0663, w2:0.0637, w4:0.0777, w6:0.069 |

## The weight-correlation vs the grid's linear extrapolation (the tested assumption)

$0 is the only qubit with strong correlation, and it **SATURATES** — the grid extrapolated it linearly and badly overestimated at high w:

| w | measured P(1\|0) for $0 | grid linear extrap | error |
|---|---|---|---|
| 0 | 0.0197 | 0.0165 | -0.0032 |
| 4 | 0.2417 | 0.3205 | +0.0788 |
| 6 | 0.2240 | 0.4725 | +0.2485 |

The correlation rises steeply 0->2 (device-char: 0.017->0.169) then plateaus at ~0.22-0.24 for w>=4 — the linear model predicted 0.32 (w4) and 0.47 (w6), overestimating by up to +0.25. The other 7 qubits are essentially flat (no correlation). The new qubits $2 (0.031) and $3 (0.016) have LOWER readout than the assumed mean 0.050; $12 (0.069) is higher.

## Re-locked n=3/n=4 grid cells (v2)

| n | state | v1 purity | v2 purity | delta |
|---|---|---|---|---|
| 3 | ghz | 0.5776 | 0.6088 | +0.0312 |
| 3 | haar | 0.6170 | 0.6324 | +0.0154 |
| 4 | ghz | 0.4804 | 0.5321 | +0.0517 |
| 4 | haar | 0.3786 | 0.3977 | +0.0191 |

All cells move UP by +0.015 to +0.052 — the device is LESS noisy at n=3,4 than v1 assumed, because (a) the $0 correlation saturates (v1's linear extrapolation was too pessimistic) and (b) the new qubits $2,$3 have lower readout than the assumed mean. The biggest correction is n=4 GHZ (+0.052), which uses all 8 characterized qubits at the highest weights.

## n=2 regression check

* n=2 Bell: v1 model 0.7163, updated model 0.7159, **moved -0.0004** — unchanged (both reproduce the measured 0.7184, within its CI [0.699,0.738]). The device-char w<=2 rates for {0,1,9,10} are kept (they were validated by the Bell run); only the NEW high-w saturation and new-qubit rates enter. No regression.

## Cost

* 7 credits consumed (7 jobs x 1). Balance: full=40, spark=4 (was spark=11).
* Raw counts: results/hardware/ro_*_counts.json. Updated table: locked_grid_predictions_v2.json (v1 kept intact). Public Plan: publications must attribute Open Quantum.

(Job note: Cepheus was in a ~70-min calibration window mid-run; the driver polled gracefully and completed all 7 jobs once it resumed — no resubmissions.)
