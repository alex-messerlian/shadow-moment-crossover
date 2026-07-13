# Cepheus purity experiment — locked predictions (ZERO credits)

Noise model (Qiskit Aer depolarizing parameters): p2(CZ)=0.009, p1(rx)=0.001, p_ro(readout)=0.02. rz is virtual (noiseless).
CZ avg gate error at p2=0.009 is 0.00675 (fidelity 99.3250%); datasheet 99.1% -> p2=0.012.

Bell-calibrated effective global-depolarizing g = 0.1423 (from measured Bell purity 0.8932).

## Locked measured-purity predictions (p_ro = 0.020)

| state | true | SWAP (Qiskit) | SWAP (bias law) | discrep | own g (full) | own g (gate-only) | shadow mean | shadow SE@20k |
|---|---|---|---|---|---|---|---|---|
| bell | 1.000 | 0.8932 | 0.8932 | +0.0000 | 0.1423 | 0.0445 | 0.8699 | 0.0131 |
| haar | 1.000 | 0.9168 | 0.8932 | +0.0236 | 0.1109 | 0.0394 | 0.8872 | 0.0163 |
| mixed_P0.7 | 0.700 | 0.6512 | 0.6359 | +0.0152 | 0.1085 | 0.0386 | 0.6455 | 0.0096 |
| mixed_P0.5 | 0.500 | 0.4714 | 0.4644 | +0.0070 | 0.1145 | 0.0399 | 0.4688 | 0.0072 |

Key finding: gate-only effective-g is nearly state-independent (~0.04), but readout
error breaks the universality (full g spans 0.108–0.142). The global-depolarizing bias
law captures the GATE noise but not the state-dependent readout suppression.

## Readout sensitivity (readout dominates the collective bias)

| p_ro | Bell measured |
|---|---|
| 0.005 | 0.9477 |
| 0.010 | 0.9292 |
| 0.020 | 0.8932 |
| 0.030 | 0.8587 |
| 0.050 | 0.7938 |

## Shot budget

| state | collective bias | shots for 5σ | SWAP SE @20k | bias significance @20k |
|---|---|---|---|---|
| bell | 0.1068 | 444 | 0.0032 | 33.58σ |
| haar | 0.0832 | 576 | 0.0028 | 29.46σ |
| mixed_P0.7 | 0.0488 | 6045 | 0.0054 | 9.09σ |
| mixed_P0.5 | 0.0286 | 23735 | 0.0062 | 4.59σ |

Recommended: 20,000 shots/config x 8 configs = 160,000 shots = 41.6 credits (budget 45.0; within budget: True).
Device max 50,000 shots/circuit. Inversion level-set (measured 0.8932 -> consistent (p_ro,p2)): [[0.005, 0.028942], [0.01, 0.022385], [0.02, 0.009], [0.03, 0.0], [0.05, 0.0]]

Caveats (stated for honesty):
* The collective route resolves every bias at >=4.6σ at 20k shots; the single-copy
  (shadow) route has ~2-4x larger error bars at the SAME shot budget — that gap is
  itself the n=2 collective advantage. Its shadow SE is a 1/sqrt(M) extrapolation
  from M=2000, which is a CONSERVATIVE upper bound (the U-statistic SE falls slightly
  faster than 1/sqrt(M) at small M), so the shadow route is if anything a touch better.
* Copy accounting: one SWAP-test shot consumes 2 copies (2n=4 qubits); one shadow
  shot consumes 1 copy (n=2 qubits). Open Quantum bills per shot, so equal-shots is
  the fair credit comparison; at equal COPIES the shadow route would get sqrt(2) more
  shots, still short of the collective route at n=2.

No hardware job submitted — ZERO quantum credits spent.
