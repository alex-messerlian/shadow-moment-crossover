<!-- HISTORICAL DRAFT. Superseded by paper.tex. Do not edit. -->

> **⚠️ HISTORICAL DRAFT — superseded by [`paper.tex`](../paper.tex). Do not edit.**
> Kept for provenance only; the compiled paper is the single source of truth.

# The Sample-Complexity Transition in Single-Copy Estimation of Quantum State Moments

**Working title.** Alternative: "When Is Collective Quantum Measurement Worth the Noise? An Exact Variance Law and Its Crossover"

---

## The claim, in one paragraph (write this first; everything serves it)

Estimating nonlinear functionals of a quantum state — purity and the higher moments Tr(ρ^k) — from single-copy randomized measurements is known to be exponentially costly, and collective (multi-copy) measurement is known to fix this in principle. What has been missing is a *quantitative* account of when the trade is worth making on a noisy device, because collective measurement buys variance reduction at the price of a noise-induced bias. We derive an **exact, state-dependent variance law** for the single-copy U-statistic estimator via the Hoeffding decomposition, and from it identify a previously unnamed phenomenon: the **sample-complexity exponent α is not fixed at 1/2** but migrates continuously to 1 as the measurement budget falls below a threshold M* that grows exponentially in system size (M* ≈ 5.3^n for purity). We pair this with two exact, parameter-free bias laws for the collective route, yielding a crossover law that predicts, with no free parameters, the system size at which collective measurement overtakes single-copy. It reproduces 99% of measured crossovers within one qubit across four state ensembles, three of which it never saw. We then test it on a 108-qubit superconducting processor and report, with pre-registered predictions, why the crossover is not reachable on current hardware.

---

## 1. Introduction

- The task: estimating Tr(ρ^k) — purity, Rényi entropies, PT moments for entanglement.
- The known separation: single-copy vs collective measurement has a proven exponential gap [Huang et al. Science 2022; Chen-Cotler-Huang-Li FOCS 2022].
- The practical tension: collective measurement needs entangling gates between copies. The recent QML-advantage experiment on tens of noisy qubits explicitly **restricted itself to single-copy schemes** and left "a systematic multi-copy benchmark for future work" [2605.21346]. That is the question we answer.
- **What is missing in the literature:** variance *bounds* for shadow-based nonlinear estimation exist (shadow norms; the 4^n kernel / 2^n linear split). What does not exist is the **exact, state-dependent variance**, and consequently nobody has observed that the budget-scaling exponent itself is a function of (n, M).
- **Our contributions**, stated plainly:
  1. Exact Hoeffding variance for the k-th moment U-statistic under local Pauli shadows (k = 2, 3, 4), verified against brute force.
  2. The **α transition** — a new phenomenon — with the threshold M*(n) and its exponential scaling.
  3. Two exact, parameter-free collective bias laws distinguished by *channel geometry*.
  4. A parameter-free crossover law, validated out-of-sample.
  5. A hardware study with pre-registered predictions on Rigetti Cepheus-1-108Q, and an honest account of why the crossover is out of reach today.

## 2. Setting and estimators

- Local-Pauli classical shadows; the snapshot G = ⊗_q (3 U†|b⟩⟨b|U − I).
- The **copy-fair** single-copy estimator: the exact full U-statistic over distinct k-tuples. **Emphasize this** — a subsampled estimator artificially handicaps single-copy, and we show the choice materially changes the conclusion (this was a real trap; report it).
  - k=4 requires the Möbius inversion over the 15 set partitions of the 4-cycle, with the alternating (ABAB) term computed by tensor contraction. Give it; it is a small technical contribution.
- The collective route: the destructive (Bell-basis) SWAP test and its k-copy cyclic-permutation generalization. n CZ gates, no ancilla. State the parity sign rule.
- Copy-fairness accounting: single-copy spends M snapshots; the k-copy test spends M/k measurements.

## 3. The single-copy variance law  ← **the paper's core**

- Hoeffding decomposition of the k-th order U-statistic:
  `Var(U_M) = C(M,k)^{-1} Σ_{c=1..k} C(k,c)·C(M−k,k−c)·ζ_c`
  where ζ_c is the variance of the c-th order projection.
  - **Warn explicitly:** for k ≥ 3, ζ_2 is the two-argument projection, **not** the kernel variance (that is ζ_k). Conflating them is wrong by ~7× and it is an easy mistake. (We made it; say so in a footnote or not at all, but do not let a reader repeat it.)
- **The exact single-qubit second moment:** E[Tr(G·r)²] = 1/4 + (5/4)t² = (5/2)p − 1 (t = Bloch length, p = single-qubit purity). Exact. Grows 6× from maximally mixed to pure — this is the seed of the exponential and it explains *why the state ensemble matters* (random-mixed states are trivially easy; noisy-pure states are exponentially hard).
- **No closed form for ζ_1 at n ≥ 2.** A weight-only Pauli ansatz ζ_1 = Σ_P c_{|P|}⟨P⟩² **fails** (13% residual on a diverse state family). Caution: it appears to *hold* to 0.1% within a narrow single-parameter ensemble — a trap. Report this.
- **The α transition.** Two regimes:
  - M ≫ M*: linear term dominates → RMSE ∝ M^(−1/2), α = 1/2.
  - M ≪ M*: higher-order term dominates → RMSE ∝ M^(−1), α = 1.
  - Threshold **M* = ζ_2/(2ζ_1)** (k=2), growing as ≈ 5.3^n. Because M* explodes, *any fixed budget* is eventually below it, so large systems sit in the α = 1 regime.
- **Out-of-sample validation:** predicted vs measured α, 8/8 within 2·SE at k=2; 6/7 at k=3; 5/5 at k=4. No fitting to α data.
- Independent corroboration: M* derived from ζ_2/(2ζ_1) = 5.15–5.35 matches the empirically measured 5.343 obtained by a separate route.

## 4. The collective route: two exact bias laws

- **Global depolarizing** at rate g on the k-copy register:
  `measured = (1−g)·Tr(ρ^k) + g·2^{n(1−k)}`, so `bias = g·|Tr(ρ^k) − 2^{n(1−k)}|`.
  **Linear in g, no compounding across qubits.** (Because Tr(C_k) = 2^n: the cyclic permutation has a single cycle.)
- **Per-qubit channel E** (amplitude damping, dephasing) on every qubit of every copy:
  `measured = Tr(σ^k)` **exactly**, where σ = E^⊗n(ρ).
  The noise does not corrupt the measurement — it *relabels which state is being measured*. This is why no universal "effective attenuation rate" fits: the bias is however much the channel deforms that particular state's spectrum.
- Both verified to ~1e-15 against explicit construction, across ensembles, out to 3× the noise range in which they were derived.
- **Key structural point for the crossover:** the collective error is a **bias floor** — budget-independent. More shots do not reduce it. (Verified: the collective RMSE plateaus exactly at the predicted floor as budget grows.)

## 5. The crossover law

- Exponentially growing single-copy error vs bounded, budget-independent collective bias floor. Solve for n*.
- **Results:** 99% of measured crossovers within ±1 qubit, 88% exact, across 83 cells — including three ensembles (haar_pure, low_rank, ghz_noisy) the theory never saw.
- **Three qualitative predictions, all confirmed:**
  1. Higher noise → later crossover (bigger floor to climb over).
  2. Higher k → **later** crossover. *Mechanism:* Tr(ρ^k) shrinks with k, so the absolute bias floor is lower, so single-copy holds out longer. (Note: this is the opposite of a naive variance-compounding argument. Say so; it's the kind of detail that shows the mechanism is understood.)
  3. Larger budget → later crossover (single-copy improves; the floor does not).
- **The exponential wall.** Copy-fair single-copy purity RMSE on noisy-pure states: ~0.04 at n=2 → **11.98 at n=10**, growing ~2.5×/qubit, against a true purity of ~0.81. At n=10 the estimate is fifteen times larger than the quantity it estimates. Collective stays bounded. *This is the figure that carries the paper.*
- **The ensemble caveat, stated prominently:** random-mixed states have purity → 2^−n → 0, which makes the task trivial and single-copy look fine. Realistic NISQ states are noisy-pure with purity O(1). Choosing the wrong ensemble inverts the conclusion. (We got this wrong first; it is worth a paragraph, because others will too.)

## 6. Hardware: pre-registered test on Rigetti Cepheus-1-108Q

Frame honestly: **this section reports a failed prediction and its diagnosis.** That is its value.

- Setup: destructive SWAP test, GHZ ladders, zero routing SWAPs (n=2: 4 CZ on {0,1,9,10}).
- **Finding 1 — the entangling overhead is nearly free.** The circuit needs 4 CZ and no routing. Measured CZ error at or below the published median. *The overhead the field feared is not the problem.*
- **Finding 2 — readout is the wall, and it is not in the datasheet.** Measured readout error ~6.5%/qubit (~3× the assumed 2%), asymmetric, and **correlated**: qubit $0's P(1|0) rises 1.6% → 16.9% with excited neighbours. The correlation **saturates** rather than growing linearly (a linear extrapolation over-predicts by up to +0.25).
- **Finding 3 — the prediction fails, and drift is why.** With same-session bracketed calibration (within-session drift ~1%), measured purities fell below both opening and closing bands. Cross-session drift on byte-identical circuits is ~0.2 in purity (n=3: 0.317 → 0.378 → 0.587), **20× the within-session drift**.
- **The elimination ledger** (this is the section's real content — each with evidence):
  drift · single-copy readout · gate count · cross-copy readout crosstalk (ruled out by a *physical bound*: max correlation consistent with measured marginals leaves 0.23 unexplained) · coherent gate error (randomized compiling with a verified positive control moved nothing; zero twirl scatter) · the suspect {3,12} qubit pair (an n=3 ladder containing them was *identical* to one without) · a fixed register-geometry fault (the ordering flips across sessions).
- **Conclusion:** static device characterization cannot predict collective-measurement performance on this device, because the device is non-stationary at the relevant scale. **Cite the prior literature on NISQ instability [Dasgupta & Humble, arXiv 2105.09472; 2208.07219] — our contribution here is confirming this specifically for collective-measurement protocols, not discovering device drift.**
- **A practical finding worth stating:** single-copy shadows are *economically* infeasible on per-circuit-priced cloud QPUs — an unbiased estimate needs ~256 credits/cell because each random basis is a separate billable circuit. This is a real constraint on reproducing shadow-based results on commercial platforms.

## 7. Related work — be scrupulous here

- Shadow variance **bounds** for nonlinear functionals: [2106.10190] (kernel 4^{|AB|} / linear 2^{|AB|} split), [2102.10132].
- U-statistic moment estimators + Hoeffding: **Straeter, Tsesmelis & Kwek (arXiv 2606.28698, June 2026)** — same technique, continuous-variable homodyne, entanglement detection. **Cite prominently. Our difference: exact state-dependent variance for qubit Pauli shadows, and the α transition.**
- Shadow sample-complexity **crossovers**: [2601.00859] (Pauli vs Clifford ensembles — a *different* crossover).
- Statistical → bias-floor transitions on hardware: **[2603.12235]** ("Hardware Horizon", integrated photonics). Structurally adjacent; different platform, different task. **Cite and distinguish.**
- Purity sample-complexity lower bounds: [2410.12712].
- NISQ device instability: [2105.09472], [2208.07219], [2604.24397].
- The open problem we answer: [2605.21346] (QML advantage with tens of noisy qubits).

## 8. Limitations — write these ourselves, before a referee does

- ζ_1 has no closed form for n ≥ 2; the law is exact but its inputs are computed numerically.
- The k=3 α validation is 6/7; the miss sits on the 2σ boundary and is seed-sensitive.
- The noise model is a channel abstraction. Real hardware departs from it (§6) — that is a finding, not a hidden assumption.
- The hardware crossover was not demonstrated: it sits at n ≈ 5–8, and device non-stationarity dominates at those sizes.
- The single-copy hardware baseline at n ≥ 3 is predicted from measured device parameters, not directly measured, for the cost reason above. The n=2 anchor is what earns that.

## 9. Conclusion

- The exponent is not a constant. That is the finding.
- Practical guidance: for k-th moment estimation at n ≳ 6 under realistic noise, collective measurement is the right investment; below that, single-copy wins. The law tells you which side you are on.
- What has to improve for the crossover to be observable: readout fidelity and, above all, **device stability**.

---

## Figures (all exist, in results/figures/)

1. **Crossover map** — measured points with parameter-free theory curves overlaid. *The centrepiece.* Caption must state the curves are predictions, not fits.
2. **Crossover boundary** — predicted vs measured n*, 83 cells, 99% within ±1.
3. **The α transition** — measured α vs n with the derived curve. *This is the novel phenomenon; consider promoting it to Figure 1.*
4. **Out-of-ensemble validation** — predicted vs measured RMSE on three unseen ensembles, with the honest scatter shown.
5. **The exponential wall** — single-copy RMSE to n=10 on a log scale, with Tr(ρ²) ≈ 0.81 marked. Visceral.
6. **(New, needed) Hardware** — measured vs pre-registered bands across n = 2,3,4, plus the cross-session drift on byte-identical circuits.

## Venue

Given the crowded field and the honest hardware result: **arXiv immediately**, then Quantum Science and Technology, PRA, or a benchmarking-focused venue. Not PRX Quantum as it stands.

## What Alex must be able to derive on a whiteboard, unaided

1. The Hoeffding decomposition and why ζ_2/ζ_1 sets a budget threshold.
2. Why α migrates 1/2 → 1, and what M* means physically.
3. Both bias laws, and why depolarizing does *not* compound over qubits while a per-qubit channel does.
4. Why the parity sum makes cross-copy readout crosstalk invisible to single-copy characterization — and why the physical bound on P(both flip) kills that hypothesis.
5. Why n=3 being worse than n=4 rules out readout scaling and gate count.
6. Why higher k crosses *later* (smaller Tr(ρ^k) against a fixed bias floor).
