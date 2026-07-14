# Verification checklist

**Purpose.** Every quantitative claim in `PAPER.md` was written from the conversation record, not from the `anrl` repository, which lives on the author's machine and is not accessible to the drafting model. **One number in this project was previously misreported by the drafting model** (a hardware purity was stated as 0.9408 when the measured value was 0.7184). No number below should be trusted until it has been checked against the saved results.

**Instruction to Claude Code:** For each row, locate the authoritative value in the repository (in `results/`, `anrl/theory/`, or the committed hardware raw counts) and either confirm the paper's figure or supply the correct one. Where a claim is qualitative, confirm it against the saved analysis. **Report every discrepancy. Do not silently correct.**

---

## 🛑 SUBMISSION BLOCKER: references unverified

**This must be resolved before the paper is submitted anywhere.**

- **Reference [7]** (arXiv:2106.10190, "Experimental quantum state measurement with classical shadows"): the **title is now confirmed** and the paper does contain the second-order variance split we cite it for, but its **author list is still unconfirmed** and must be checked against **arxiv.org/abs/2106.10190** before submission. (That paper states the bound citing its own references, so our attribution is to a statement, not a derivation — reflected in the bibliography note.)
- The **co-author list of [4]** (Gong et al.) is unconfirmed and incomplete ("et al.") and must be completed.
- More broadly, **references [2], [3], [4], [9], and [11]–[16]** still have an author list, journal, volume, or page numbers **not** confirmed against the arXiv or journal record. Each is marked with a visible `% UNVERIFIED` comment in `refs.bib` and in the `thebibliography` block of `paper.tex`; [7] carries a `% PARTIAL` marker (title confirmed, authors not).
- **The author must verify all remaining references personally against arxiv.org.** Neither Claude Code nor the drafting model has done so reliably.

Confirmed to date: **[1]** (Huang–Kueng–Preskill, *Nature Physics* 16(10), 1050–1057), **[5]** (Danaci et al.), **[6]** (Cotler–Gong–Kannan, "Noisy Quantum Learning Theory"), **[8]** (Hu & You, *Phys. Rev. Research* 4, 013054), and **[10]** (Open Quantum white paper, required by the Public Tier licence). **[7]**'s title is confirmed; its authors are not.

---

## Section 2 — Estimators

| # | Claim in paper | Where to check | Status |
|---|---|---|---|
| 2.1 | Subsampled estimator inflates single-copy RMSE by 4–8× vs the exact full U-statistic | `fix-fair-estimator` phase report; `results/` sweep before/after | |
| 2.2 | Exact $k=4$ U-statistic matches brute-force enumeration to $10^{-14}$ | `harden-result` phase; `tests/` | |
| 2.3 | Destructive SWAP sign rule verified to $10^{-16}$ | `anrl/hardware/` tests | |
| 2.4 | Noisy-pure ensemble purity stays near 0.81 across sizes | `scaling-crossover` results JSON | |

## Section 3 — Variance law

| # | Claim | Where to check | Status |
|---|---|---|---|
| 3.1 | Lee/Hoeffding formula verified vs brute force at $k=2,3,4$; ratios within 3% | `theory-derivation` + `general-k-variance` reports | |
| 3.2 | General formula reduces to $k=2$ form at ratio 1.000000 | `general-k-variance` report | |
| 3.3 | Conflating $\zeta_2$ with kernel variance at $k\geq3$ is wrong by ~7× (ratio 0.135) | `general-k-variance` adversarial verification | |
| 3.4 | Single-qubit: $\mathbb{E}[\mathrm{Tr}(Gr)^2] = 1/4 + (5/4)t^2$; $\zeta_1 = (3/4)t^2 - (1/4)t^4$ | `anrl/theory/single_copy_law.py` tests | |
| 3.5 | Weight-only Pauli ansatz fails at $n=2$ with **13%** max residual on a diverse 25-state family | `theory-derivation` report | |
| 3.6 | Same ansatz appears to hold to **0.1%** within the narrow $q=0.1$ Haar family | `theory-derivation` report | |
| 3.7 | $\zeta_1 \approx 0.63\cdot(1.35)^n$, $\zeta_2 \approx 1.10\cdot(6.93)^n$, $M^* \approx 0.87\cdot(5.15)^n$ over $n=2..7$ | `results/theory_derivation.json` | |
| 3.8 | $M^* \approx 0.76\cdot(5.345)^n$ over $n=2..9$ | same | |
| 3.9 | Independently measured empirical $M^*$ base = **5.343** | earlier `crossover-theory` phase, `theory_zetas.json` | |
| 3.10 | **α table**: $n=2$ predicted 0.501 / measured 0.495 ± 0.013; $n=4$ predicted 0.527 / measured ~0.54; $n=9$ predicted 0.995 / measured 1.006 ± 0.023 | `results/` budget-scaling + theory-derivation JSON | |
| 3.11 | α validation: **8/8** within 2·SE at $k=2$; **6/7** at $k=3$; **5/5** at $k=4$ | `theory-derivation`, `general-k-variance` | |
| 3.12 | Two-term approximation manages only **5/8** at $k=2$, failing $n=6$ by ~6.8·SE | `theory-derivation` report | |
| 3.13 | $k=3$ ζ bases (1.30, 2.86, 14.9); $k=4$ (1.34, 2.61, 6.53, 36.2) | `general-k-variance` JSON | |

## Section 4 — Bias laws

| # | Claim | Where to check | Status |
|---|---|---|---|
| 4.1 | Global depolarizing: bias $= g\lvert \mathrm{Tr}(\rho^k) - 2^{n(1-k)}\rvert$, verified to ~$10^{-15}$ | `anrl/theory/` bias-law tests | |
| 4.2 | Per-qubit channel: measured $= \mathrm{Tr}(\sigma^k)$ exactly, verified to ~$10^{-15}$ | same | |
| 4.3 | Both hold at 3× the noise range they were derived in, on unseen ensembles | `theory-stress-test` report | |
| 4.4 | Original compounding form $[1-(1-g)^{kn}]$ overestimated depolarizing bias by **5–14×** | `budget-scaling` report | |

## Section 5 — Crossover

| # | Claim | Where to check | Status |
|---|---|---|---|
| 5.1 | **99% of measured crossovers within ±1 qubit, 88% exact, across 83 cells** | `results/figures/` fig2 tidy CSV; `crossover-theory` + `figures` reports | |
| 5.2 | Four ensembles, three unseen (haar_pure, low_rank, ghz_noisy) | `theory-stress-test` | |
| 5.3 | GHZ accuracy within ~1.7× of other ensembles | `theory-stress-test` report | |
| 5.4 | Collective RMSE plateaus at the predicted budget-independent floor | `budget-scaling` report (Prediction 2) | |
| 5.5 | Higher $k$ crosses **later**; $k=2$ at $n\approx6$–7, $k=3,4$ hold to $n\approx8$ | `sweep-corrected` report | |
| 5.6 | **Exponential wall**: single-copy purity RMSE 0.044 / 0.072 / 0.270 / 1.62 / 11.98 at $n=2,4,6,8,10$; growth ~2.5×/qubit; true purity ~0.81 | `harden-result` report; `results/figures/` fig5 CSV | |
| 5.7 | "Error fifteen times the quantity being estimated" at $n=10$ (11.98 / 0.81 ≈ 14.8) | arithmetic on 5.6 | |

**Note on 5.6:** an earlier phase reported growth of ~1.8×/qubit with values 0.05 → 0.11 → 0.30 → 0.88 → 1.23 at $n=4..8$. The paper quotes the *hardened* run (2.5×/qubit, out to $n=10$). **Confirm which run is canonical and make the paper consistent.** Do not mix the two.

## Section 6 — Hardware

| # | Claim | Where to check | Status |
|---|---|---|---|
| 6.1 | $n=2$ SWAP: 4 CZ, zero routing, physical $\{0,1,9,10\}$ | `hardware-circuits` report | |
| 6.2 | GHZ ladders map with $3n-2$ CZ, zero routing; ladders nest | `grid-predictions`, `readout-extension` | |
| 6.3 | Haar $n=4$ needs 46 CZ with 20 routing SWAPs | `grid-predictions` report | |
| 6.4 | Rigetti published: 99.1% median CZ, 99.9% single-qubit; readout not published | vendor spec / cited source | |
| 6.5 | **Bell prediction 0.9412; measured 0.7184, CI [0.6992, 0.7376], 5000 shots** | `results/hardware/analysis.json` — **check the CI bounds specifically** | |
| 6.6 | Effective $g = 0.375$ | same | |
| 6.7 | **Readout table**: $\$0$ (9.3%, 8.9%); $\$1$ (0.7%, 7.8%); $\$9$ (6.7%, 9.8%); $\$10$ (3.2%, 6.0%); mean ~6.5% | `results/hardware/CHARACTERIZATION_REPORT.md` | |
| 6.8 | Correlated readout: $\$0$ $P(1\vert0)$ 1.6% → 16.9% with excited neighbours | same | |
| 6.9 | Correlation **saturates**: measured 0.020, 0.169, 0.242, 0.224 at $w=0,2,4,6$; linear extrapolation predicts 0.473 at $w=6$, over by +0.25 | `readout-extension` report | |
| 6.10 | Correlated model closes the Bell residual: predicted **0.7163** vs measured 0.7184 | `grid-predictions` report (Step 1 gate) | |
| 6.11 | **Same-session table**: $n=2$ 0.686 [0.672,0.700] vs bands [0.718,0.742] / [0.704,0.728]; $n=3$ 0.378 [0.360,0.397] vs [0.597,0.634] / [0.593,0.631]; $n=4$ 0.420 [0.402,0.439] vs [0.506,0.553] / [0.520,0.568] | `results/hardware/SAME_SESSION_REPORT.md` | |
| 6.12 | Within-session drift ~1% per qubit | same | |
| 6.13 | **Cross-session**: $n=3$ 0.317 → 0.378 → 0.587; $n=4$ 0.431 → 0.420 → 0.382 | across `hardware-grid`, `same-session-grid`, `coherent-error` reports — **confirm each value's session** | |
| 6.14 | Joint readout TVD 0.0094 ($n=3$), 0.0114 ($n=4$); parity-pair Pearson −0.0145 / −0.0054 | `JOINT_READOUT_REPORT.md` | |
| 6.15 | **Physical bracket**: max realizable correlation gives [0.611, 0.669] at $n=3$; measured 0.378; residual 0.23 | same | |
| 6.16 | Twirled $n=4$ 0.369 vs untwirled 0.382 vs prediction 0.579; masking bound ≤ +0.027 (~14% of 0.197 deficit) | `coherent-error` report | |
| 6.17 | Positive control: 0.508 on posctrl support vs 0.094 on untwirled support (5.4:1) | same | |
| 6.18 | Localization: n3_alt (with $\{3,12\}$) = **0.58080**, n3_std = **0.58080**, distinct raw data | `register-fault` report | |
| 6.19 | Register-fault session: locked predictions n3_std 0.698, n4 0.604, n3_alt 0.674; measured 0.581 / 0.372 / 0.581 | same | |
| 6.20 | Ordering flips sign across sessions on identical $n=4$ register | `register-fault` adversarial finding | |
| 6.21 | Shadow route ≈ 256 credits/cell vs ~3 for collective at same shots | `hardware-grid` feasibility note | |
| 6.22 | Total spend ≈ 115 credits | sum the per-phase reports | |

## Section 8 — Limitations

| # | Claim | Where to check | Status |
|---|---|---|---|
| 8.1 | Out-of-ensemble median relative deviation **6.7%**; signed **+3.4%** | `clipping-correction` report | |
| 8.2 | Residual is finite-trial noise; measured converges to predicted within ±0.3% at high trials | same | |
| 8.3 | Estimator approximately Gaussian at $M\geq2000$ (kurtosis ≈ 3) | same | |
| 8.4 | ζ converged to better than 1% | `theory-derivation` convergence check | |

---

## Open Quantum licence compliance — MANDATORY, not optional

The Open Quantum white paper states, of the Public Tier: **"Public tier users are required to cite this paper in any publications resulting from their work on the platform."** All our hardware ran on the Public Tier. This is a licence condition.

| # | Requirement | Where handled in paper | Status |
|---|---|---|---|
| OQ.1 | Cite the platform paper: Wold, Armbruster & Kuhn, "Open Quantum: Democratizing Access to Quantum Computing Resources," Quantum Rings Inc. | Ref [10]; cited in §6.1, §6.7, Appendix C, Acknowledgements | |
| OQ.2 | Disclose that Public Tier use contributes anonymized aggregated circuits, results, and metadata to Open Quantum's common data repository | §6.1, Acknowledgements | |
| OQ.3 | Disclose the Public Tier execution path: routed via the Quantum Compute subnet (SN48) on Bittensor, operated by qBitTensor Labs, with validator spot-checks rather than direct vendor access | §6.1, Appendix C, Limitations | |
| OQ.4 | Confirm the white paper's canonical citation format and URL from the PDF the author holds | Ref [10] | |

**On OQ.3 — this is a real methodological caveat and it is new.** The Public Tier does not route jobs directly to Rigetti. Circuits go to distributed operators on a decentralized network, with validators spot-checking that execution occurred on the target QPU. We therefore cannot independently verify the execution path of any individual job. This **does not** affect the readout or gate characterizations, which are internally consistent. It **is** a candidate contributor to the session-to-session variability of §6.4 that we cannot exclude, and it is now listed in Limitations. A referee will ask about this; better that we raise it. Confirm the paper states it accurately against the white paper text.

## Citations — all need bibliographic confirmation

The drafting model supplied arXiv identifiers and titles from web search. **Author lists, journal references, years, and volume/page numbers for [1]–[16] must be confirmed against the actual arXiv/journal records.** Do not submit with any reference the author has not personally verified.

Specific items needing attention:

- **[7] and [8]** — the drafting model has arXiv IDs (2106.10190, 2102.10132) and a description of their content (the $4^{|AB|}$ kernel / $2^{|AB|}$ linear split), but **not confirmed titles or author lists.** Fetch and confirm, or drop.
- **[10]** — Open Quantum. Author names confirmed from the white paper PDF (Bob Wold, Omar Armbruster, Ryan Kuhn; Quantum Rings Inc., Broomfield, CO). Confirm the intended citation format and any DOI/URL from the source.
- **[14]** — Straeter, Tsesmelis, Kwek. Confirmed via arXiv abstract page (2606.28698, v1 27 June 2026). This is the closest methodological neighbour and **must be cited prominently and characterized accurately.** Read the paper before finalizing Section 7.
- **[5], [6], [15], [16]** — titles confirmed via search; author lists not. Fetch.
- **[11], [12]** — Dasgupta & Humble. Confirm exact titles and venues.

---

## Things the paper deliberately does NOT claim

Check that no draft revision reintroduces these:

1. **We do not claim to discover NISQ device drift.** It is documented [11, 12]. We claim to demonstrate it is the binding constraint for collective-measurement protocols specifically.
2. **We do not claim the Hoeffding-for-U-statistics technique is novel.** [14] applies it to quantum moment estimators in a different setting. We claim the *exact state-dependent* variance and the *exponent transition*.
3. **We do not claim to have demonstrated the collective advantage on hardware.** We did not. The crossover is out of reach at the sizes available.
4. **We do not claim the theory predicted the hardware.** It did not. Section 6 is a failed prediction with a diagnosis.
5. **We do not claim a closed form for $\zeta_1$ at $n \geq 2$.** There isn't one.

---

## Open drafting questions for the author

1. **Title.** The current title foregrounds the α transition. Confirm this is the intended emphasis over the crossover.
2. **Figure order.** The outline proposed promoting the α-transition figure (currently Fig. 3) to Figure 1, since it is the novel phenomenon. Decide.
3. **A hardware figure does not yet exist.** Section 6 needs one: measured vs pre-registered bands across $n = 2,3,4$, plus the cross-session drift on byte-identical circuits. This must be generated.
4. **Mentor authorship and affiliation.** Placeholders in the header.
5. **Venue.** The draft is written for arXiv plus a specialist venue (Quantum Science and Technology, PRA, or a benchmarking venue). It is not written as a PRX Quantum submission.
