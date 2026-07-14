# Register-fault localization — global state (A) vs register-specific fault (B)

Bracketed same-session test on Cepheus-1-108Q to discriminate why the n=4 collective SWAP purity carries a persistent deficit. **Spend: 18 credits (balance 28 → 10 full, 0 spark).** Raw counts committed verbatim before analysis.

## Locked predictions (frozen from the opening calibration, BEFORE any SWAP)

Opening readout calibration on {0,1,2,3,9,10,11,12} (7 X-gate basis states, mean p10 = 0.038 — a **clean-readout** session). Predictions locked in `rf_locked.json`. (The lock's integrity does not rest on commit order: the predictions are a deterministic function of the *opening* calibration + fixed constants and never read the SWAP counts, and the cell layouts were pre-registered earlier in `a031c52`.)

| cell | register | predicted band (mid) |
|---|---|---|
| n3_std | {0,1,2,9,10,11} | 0.672–0.717 (**0.698**) |
| n4 | {0,1,2,3,9,10,11,12} | 0.572–0.627 (**0.604**) |
| n3_alt | {1,2,3,10,11,12} (incl. suspect {3,12}) | 0.649–0.691 (**0.674**) |

## Measured vs locked prediction (same session)

| cell | measured (95% CI) | band | verdict | deficit vs mid |
|---|---|---|---|---|
| n3_std | **0.581** [0.558, 0.604] | 0.672–0.717 | BELOW | −0.117 |
| n4 | **0.372** [0.346, 0.398] | 0.572–0.627 | BELOW | −0.232 |
| n3_alt | **0.581** [0.558, 0.603] | 0.649–0.691 | BELOW | −0.093 |

All three cells fell **below** their clean-readout predictions.

## Decoupling — the strict test did NOT fire; the raw gap did

The task's strict decoupling (n3_std matches its prediction while n4 falls below) **did not occur**: n3_std itself is below its prediction this session. But the **drift-/prediction-robust raw comparison** is decisive: n=4 measures **0.209 below** n=3_std (0.372 vs 0.581; CIs disjoint, significant). That reproduces n=4's persistent extra deficit relative to n=3.

## Localization — REFUTED: the fault is NOT the {3,12} pair

The key test: `n3_alt` = an n=3 SWAP on {1,2,3,10,11,12}, whose parity pairs (1,10)(2,11)(**3,12**) include the suspect pair. If {3,12} or its couplings carried the fault, this ladder would be anomalous at n=3.

**It is not.** `n3_alt` measured **0.58080** — *identical to 5 decimals* to standard `n3_std` (0.58080), from genuinely distinct data (different bitstring distributions, 5000 shots each). Swapping the suspect {3,12} pair into an n=3 ladder does **not** degrade it. The n=4 deficit is **not** localized to {3,12}; it is specific to the full n=4 register geometry.

## Within-session drift (opening → closing)

From the opening (7 states) and the **4 closing states obtained** (w0, w2, w4a, w4b — the closing bracket was truncated by a device calibration window; see below): mean |Δp10| = 0.012, mean |Δp01| = 0.007, 3/8 qubits with a significant shift. Most qubits are stable (~1%), though **q2's p10 drifted +5.2%** (a real shift). Even the worst-case drift is ~10× too small to explain a 0.2 purity deficit, so the deficits above are not a drift artifact. Caveat: **high-weight (w6/w8) closing readout — the regime most relevant to the 8-qubit n=4 measurement — is uncharacterized** because the closing bracket was truncated.

## Cross-session context — the actual purities are stable; the *prediction* is what moves

| session | n=3 std actual | n=4 actual | n4 − n3 gap |
|---|---|---|---|
| hardware-grid | 0.317 | 0.431 | +0.114 |
| same-session-grid | 0.378 | 0.420 | +0.042 |
| coherent-error | 0.587 | 0.382 | −0.205 |
| **register-fault (this)** | **0.581** | **0.372** | **−0.209** |

Two facts stand out. First, **the n4−n3 ordering flips sign** across the identical registers: in the first two sessions n=4 was *above* n=3 (+0.114, +0.042 — n=3 was the anomalous one then), and only in the last two is n=4 below n=3 (−0.205, −0.209). n=3 **healed** dramatically (0.317→0.587) while n=4 held ~0.38–0.43. So the n=4-below-n=3 deficit is **session-dependent, not a fixed property of the n=4 geometry** — the same n=4 register was the *better* one earlier.

Second, over the **last two sessions** the actual n=3 (~0.58) and n=4 (~0.38) purities are stable, and what changed between the coherent-error session (n=3 "matched" its 0.606 prediction) and this one (n=3 "below" its 0.698 prediction) is **only the prediction** — this session's readout calibration looked cleaner and predicted higher, while the actual purity held at ~0.58. **So "matches / misses prediction" is driven by calibration variability, not by the SWAP purity changing.**

## Verdict — NEITHER cleanly; a refined Hypothesis B

* **Hypothesis A (global device state, n=4 tracks readout) — REJECTED.** Readout was clean this session, yet every cell (including n=3) came in far below its clean-readout prediction. The deficit does **not** track the same-session readout quality.
* **Hypothesis B (register-specific fault) — PARTIALLY supported, but NOT as a standing geometry fault:**
  - The n=4-below-n=3 gap is real and significant **this session** (~0.21, disjoint CIs) and reproduces the coherent-error session — but it is **session-dependent, not fixed**: on the *identical* n=4 register, n=4 was *above* n=3 in the first two sessions (the ordering flipped). So B's core prediction — n=4 stays ~0.2 below *in every session* — is **contradicted by the full history**. The recent n=4 deficit is not a permanent property of the geometry.
  - It is **definitively NOT the {3,12} pair** (localization refuted: n3_alt with {3,12} equals standard n=3 exactly, 0.58080). Whatever recently degrades n=4 is not carried by that pair at the n=3 level; it emerges only in the full n=4 register.
* **Methodological finding:** the X-gate basis-state readout calibration **over-predicts** SWAP purity (all three cells below their clean-readout predictions this session), so the readout-based prediction is not a faithful session-to-session predictor of in-circuit SWAP purity. The robust, drift-immune signal is the **raw cross-register comparison**.

**Net:** neither A nor B as stated. Both registers' purities are **session/device-state-dependent** (n=3 healed over time, n=4 did not), they have **diverged in the last two sessions** (n=4 ~0.2 below n=3), and that recent divergence is **not the {3,12} pair** and **not a fixed n=4-geometry fault**. Whether the recent n=4 deficit is a transient device-state effect localized to the n=4 register or a slowly-developing register issue cannot be settled from four snapshots — it needs a same-session A/B/A/B/... time series and per-edge interleaved RB on the n=4 register.

**Elimination ledger (cumulative):** ruled out — within-session drift, single-copy readout, gate count, cross-copy readout crosstalk, coherent gate error, and now the **{3,12} pair specifically** and a **fixed n=4-geometry fault** (the ordering flipped across sessions). What stands: a **session-dependent, register-differentiated** deficit — the readout calibration over-predicts, and n=4 has recently run ~0.2 below n=3 for reasons not captured by any single-qubit/pair characterization. Next: a repeated same-session time series + per-edge interleaved RB, not a model tweak.

## Notes

* Closing bracket truncated: after the opening calibration, lock, all 3 SWAP cells, and 4 closing states, job C_w6a stuck ~2 h in a device calibration window ("Retrying 2 of 3 attempts"); it was charged (1 cr) but produced no counts, and C_w6b/C_w8 were not run. The 4 closing states still measure per-qubit p10 and p01 drift for all qubits.
* 18 credits: opening 7, SWAP 3×2 = 6, closing 4 + stuck C_w6a 1 = 5. Balance full=10, spark=0. Public Plan: publications must attribute Open Quantum.
