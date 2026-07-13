# Cepheus CZ characterization — depth sweep (result)

Non-cancellable X-twirl identity echo on physical {0,1,9,10}, 3000 shots/depth. Submitted CZ counts verified 0/8/16/24/32/48; all quoted+billed 1 credit (6 total).

## Survival vs CZ depth

| CZ | survival | SE | if CZ=0 | if CZ=0.1% | if CZ=spec 0.9% |
|---|---|---|---|---|---|
| 0 | 0.8993 | 0.0055 | 0.8141 | 0.8141 | 0.8141 |
| 8 | 0.9003 | 0.0055 | 0.8087 | 0.8018 | 0.7482 |
| 16 | 0.8863 | 0.0058 | 0.8049 | 0.7912 | 0.6896 |
| 24 | 0.8963 | 0.0056 | 0.8011 | 0.7807 | 0.6360 |
| 32 | 0.9023 | 0.0054 | 0.7973 | 0.7705 | 0.5872 |
| 48 | 0.9017 | 0.0054 | 0.7899 | 0.7503 | 0.5020 |

## Fit

* survival = A·B^nCZ: **A (intercept) = 0.8960 ± 0.0038**, B = 1.00010 ± 0.00016.
* Per-CZ decay (1−B) = -0.00010, 95% CI [-0.00041, +0.00021] — **consistent with zero.**
* Intercept 0.8960 **agrees** with the independently measured readout floor 0.899 (True) — the one clean consistency check.

## Interpretation — the echo was collapsed by the compiler (CZ NOT measured)

The survival is FLAT at the readout floor across a 6x range of nominal CZ depth. Two readings:
1. The per-CZ error is <0.02% (95%) — i.e. ~45x better than the 0.9% published median. Implausible for superconducting hardware.
2. The compiler resynthesized the identity echo and removed its gates, so every depth actually ran as ~(|+> prep, H unprep, readout) = the flat floor.

Reading 2 is correct, and the data proves it: the measured curve is flatter than even the **CZ=0** prediction (0.814->0.790), which still declines from the interior single-qubit (rx) gates. If the echo body had executed at all, those ~96 rx gates at depth-48 would cause visible decay. Zero decay => the whole echo body (CZ and rx) was optimized away. The interleaved X-twirl defeats Qiskit's optimizer (verified: 8/16/24/32/48 CZ preserved under opt3) but not the Braket/Rigetti compiler's full resynthesis. The verbatim box that would prevent this FAILED at execution on this platform (empty error), so we could not force the gates to run.

**We did not directly measure the CZ error.** What we did establish: the readout floor reproduces (intercept 0.896 = 0.899), and identity echoes are unusable for CZ characterization here without a working verbatim box.

## Reconciliation with the Bell run

Because the echo collapsed, a 'both-parameters-measured' reconciliation isn't available. The best indirect CZ estimate remains the Bell run itself: measured readout + CZ~spec(0.9%) predicts Bell purity ~0.749 vs the measured 0.7184 (residual ~0.03, attributed to the correlated readout crosstalk on $0). The tension — echo shows no CZ, Bell wants ~spec CZ — is resolved by the collapse: the Bell SWAP is NOT an identity circuit, so its CZ execute and contribute error, whereas the identity echo's CZ are optimized away.

## Recommended next step (not run here)

Measure CZ with a NON-identity observable the compiler can't collapse: e.g. interleaved randomized benchmarking with random recovery (output depends on CZ error, circuit is not globally identity), or resolve the verbatim-box execution failure with the platform. Do not infer a CZ number from this flat sweep.
