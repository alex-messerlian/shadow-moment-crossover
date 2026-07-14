# Randomized compiling (Pauli twirling) — testing coherent CZ error

Confirmatory test for the last hypothesis on the elimination ledger: is **coherent error in the SWAP body** (static ZZ, CZ over/under-rotation, simultaneous-gate crosstalk) the unmodeled mechanism behind the collective-purity deficit and the n=3<n=4 non-monotonicity? Twirling converts coherent error to stochastic; if coherent error is the mechanism the twirled purity should move toward the depolarizing-model prediction. **Spend: 19 credits (balance 27 → 8 full, 0 spark).**

## Method

For each CZ, insert a random 2-qubit Pauli `T` before it and its exact correction `C = CZ·T·CZ` after, so **C·CZ·T = CZ** exactly (verified numerically for all 16 Paulis + the native rz/rx decompositions). Each CZ's twirl is locally self-correcting: the ideal circuit is unchanged (every twirled circuit recovers SWAP purity **1.0** on GHZ in noiseless sim, max prob deviation ~1e-16), so **no measurement-frame correction is needed** and the standard sign-rule analysis applies directly. The coherent error on each CZ is independently Pauli-twirled toward a stochastic channel.

Circuits (GHZ-ladder registers, n=3 {0,1,2,9,10,11}, n=4 {0,1,2,3,9,10,11,12}): untwirled baseline, 12 (n=3) / 4 (n=4) independent twirls with explicit Pauli gates, and a **positive control**.

## Survival — does the twirl actually execute? YES

The platform's compiler resynthesized our circuits before (it ate the identity echo), and a *logically-trivial* twirl is exactly what a compiler could optimize away — so a null result would be ambiguous (coherent-error-ruled-out vs twirl-collapsed). The **positive control** (an *uncorrected* X before CZ(0,9): ideal purity 0, TVD 1.0 vs untwirled — disjoint bitstring support) resolves this:

* Hardware mass on the posctrl support **0.508** vs untwirled support **0.094** (5.4:1). The inserted mid-circuit gate around a CZ demonstrably executed → **the twirl physically survives**, not stripped/resynthesized away.

## The anomaly drifted — n=3 healed, n=4 persists

The same circuits give very different purities than in prior sessions (byte-identical QASM, so this is genuine cross-session device drift):

| register | hardware-grid | same-session-grid | **this session (untwirled)** | depol prediction |
|---|---|---|---|---|
| n=3 | 0.317 | 0.378 | **0.587** (CI 0.556–0.618) | 0.606 |
| n=4 | 0.431 | 0.420 | **0.382** (CI 0.346–0.419) | 0.579 |

* **n=3 healed**: drifted up +0.21 and now matches the depolarizing prediction (gap +0.019). The n=3 anomaly was **transient**.
* **n=4 persists**: ~0.38–0.43 across all three sessions, a stable ~0.15–0.20 deficit below prediction. **n=4 is the anomalous cell this session** — so it is the cell that actually tests the coherent-error hypothesis.

## Twirled vs untwirled — coherent error RULED OUT

| n | untwirled (same session) | twirled RC estimate | depol prediction | twirl − untwirled | physical scatter |
|---|---|---|---|---|---|
| 3 | 0.587 | 0.5504 ± 0.0044 (12 twirls) | 0.606 | −0.037 | 0 (shot-noise-limited) |
| 4 | 0.382 | 0.3687 ± 0.0076 (4 twirls) | 0.579 | −0.013 (z=−1.7, n.s.) | 0 (shot-noise-limited) |

* **At n=4 (the anomalous cell): twirling does NOT move the purity toward the prediction.** Twirled 0.369 ≈ untwirled 0.382 (shift −0.013, consistent with zero: z=−1.7 on the twirl SEM alone, ~−0.6 pooled), still **~0.21 below** the prediction 0.579 (z vs depol = −27.8). If coherent error were the ~0.20 mechanism, twirling would move it *up* toward 0.579 — there is **no significant upward move**.
* **No twirl-to-twirl scatter at either n.** Per-twirl purities cluster at the shot-noise floor (n=3 std 0.015 vs shot-noise 0.019; n=4 std 0.015 vs shot-noise 0.021); the physical (excess) scatter is 0 at both. Coherent error would make different Pauli frames sample the error differently → scatter. There is none. (The n=4 zero-scatter rests on 4 twirls — 3 dof, a weak null alone — but is corroborated by the independent 12-twirl zero-scatter at n=3.)
* **Masking bound — any hidden coherent component is ≤14% of the deficit.** A twirl adds ~28 (n=3) / ~39 (n=4) single-qubit Pauli gates (~0.1%/gate ⇒ a *downward* incoherent penalty ~0.03–0.04), which could in principle partly cancel an upward coherent→stochastic move. Crediting the full ~0.04 penalty, the largest upward coherent move the observed −0.013 net shift at n=4 could hide is ~**+0.027** — only ~14% of the 0.197 deficit. So coherent error cannot be the dominant mechanism; the small twirled<untwirled shift (−0.037 at n=3, −0.013 at n=4) is consistent with the added-gate penalty alone, **not** a coherent→stochastic conversion (which would show as a *net upward* move — absent here).
* Intra-run drift negligible (early−late twirls +0.004 at n=3, +0.024 at n=4) — the device was stable during the run, so the twirl cluster is a clean estimate.

## Non-monotonicity — a drift artifact, NOT restored by twirling

Historical n=3 (0.378) < n=4 (0.420). **This session the UNTWIRLED circuits already show n=3 (0.587) > n=4 (0.382)** — the ordering reversed *before any twirling*, because n=3 healed while n=4 stayed anomalous. Twirled gives the same ordering (0.550 > 0.369). So the non-monotonicity is a **cross-session drift artifact** (which register carries the deficit at a given time), **not** a coherent-error interference effect, and twirling did not restore it — drift did. The "cleanest confirmation" the design hoped for (twirling flips n3<n4 → n3>n4) did **not** occur; the flip was already present in the untwirled data.

## Verdict — coherent error RULED OUT (model not tuned)

The twirl provably executes (positive control), yet at the anomalous cell (n=4) it does not move the purity toward the depolarizing prediction and produces no twirl-to-twirl scatter at either n. **Coherent CZ error (ZZ, over-rotation, simultaneous-gate crosstalk) is not the dominant unmodeled mechanism.** The residual deficit is **incoherent** (twirl-invariant), device-state-dependent, and larger than the static readout+depolarizing model predicts.

**Elimination ledger — now ruled out:** (1) within-session drift, (2) single-copy readout, (3) gate count, (4) cross-copy readout crosstalk, (5) coherent gate error (this run). **What remains / what this run adds:**

* The deficit is a **large, incoherent, register- and time-dependent** error the static characterization underestimates. n=4's register (adds qubits {3,12}) carries a **persistent** such fault across all three sessions; n=3's was transient and has now healed.
* **Cross-session drift is large (~0.2 on n=3)** even though within-session drift is ~1%/qubit — the single most important operational finding for this device: collective-purity predictions from a prior session's characterization are not trustworthy, and the register that is "anomalous" changes over time.
* Remaining candidate mechanisms (untested here, all incoherent): under-modeled depolarizing/leakage on specific qubits or CZ edges (e.g. {3,12} or their couplings), or TLS-type transient faults — to be probed by per-edge interleaved randomized benchmarking, not a model tweak.

## Limitations

* Single session; the anomalous cell this session is n=4 (drift moved it from the pre-registered priority n=3). n=4 has only 4 twirls (the mean-shift conclusion is robust — twirled ≈ untwirled, far from prediction — but the scatter estimate is limited; it agrees with n=3's 12-twirl zero-scatter result).
* Twirling addresses coherent error commutable onto the CZ cycles; static ZZ during idle/1q layers is only partially twirled by per-CZ twirling.
* A twirl adds single-qubit gates (incoherent overhead); the comparison is twirled-with-overhead vs untwirled, which biases against seeing an upward move — yet even so no upward move toward the prediction is seen.

## Cost

19 credits (n=3: untwirled + posctrl + 12 twirls = 14; n=4: untwirled + 4 twirls = 5). Balance full=8, spark=0. Raw counts committed verbatim before analysis. Public Plan: publications must attribute Open Quantum.
