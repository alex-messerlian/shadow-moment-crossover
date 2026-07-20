# PRIOR ART MAP — claim-by-claim novelty boundary

> **UPDATE 2026-07-17 — both CRITICAL citations are now IN the paper** (branch `add-citations`,
> `\bibitem`s [19] and [20], appended so [1]–[18] do not renumber). Every field was verified against
> **three** independent sources *plus the publisher's own typeset first page*:
>
> | | Paper A → **[19]** | Paper B → **[20]** |
> |---|---|---|
> | Authors | Xu-Jie Peng, Qing Liu, Lu Liu, Ting Zhang, You Zhou, and He Lu (**6**) | Adrian Skasberg Aasen and Martin Gärttner (**2**) |
> | Title | *Experimental shadow tomography beyond single-copy measurements* | *Limitations for adaptive quantum state tomography in the presence of detector noise* |
> | Ref | Phys. Rev. Applied **23**, 014075 (2025) | Phys. Rev. A **114**, 012414 (2026) |
> | DOI | 10.1103/PhysRevApplied.23.014075 | 10.1103/3q13-xpc8 |
> | arXiv | **none exists** (Crossref, OpenAlex, S2 all list DOI only) | 2601.04020 |
> | Sources | Crossref + OpenAlex + Semantic Scholar + APS typeset PDF | Crossref + arXiv meta + Semantic Scholar + APS typeset PDF |
>
> **Paper B is fully published**, not accepted-only: the APS page reads *"PHYSICAL REVIEW A 114, 012414
> (2026) … (Received 8 January 2026; accepted 12 June 2026; published 6 July 2026)"* — unlike refs
> [4]/[6], the volume and article number exist. ⚠️ **OpenAlex lists Paper B's first author as
> "Anonymous"** — the same APS metadata artifact that hit ref [4]. Crossref and arXiv both carry the
> real name. Do not "correct" it back.
>
> **Aasen's overlap is larger than the map first recorded.** Verified from the APS PDF: their rolling
> exponent *"decreases towards a = −0.5 as the number of measurements increases"* from *"the expected
> optimal ≈1/N scaling"* — i.e. **1 → 1/2 in the budget variable, the same two rates and the same
> direction as ours**. Only the mechanism differs (detector noise vs the ζ₂/ζ₁ balance). Claim 3 is
> weaker than "ALREADY-DONE in substance" implied; it is close to fully pre-empted as a *method*.

**Status: research deliverable. Nothing here was applied to `paper.tex`.** Written 2026-07-17 on
branch `priorart-sweep` (base `938b030`). Every load-bearing claim below was verified by me against
the **primary source** (LaTeX e-print where available), not from an agent summary or a snippet. Where
a fact rests on a single unverified pathway, it says so.

> **Read this first.** The headline is not that the novelty is smaller — it is that **the single
> biggest threat is a paper you already cite as [18], and it is worse than the Appendix D finding.**
> Elben's **main text**, with a figure, already states that the error decay rate depends on the
> measurement budget. See Claim 3.

---

## The table

| # | Claim | Verdict | Deciding evidence |
|---|---|---|---|
| 1 | Exact **state-dependent evaluation** of ζ₁, ζ₂ | **CLEAN** (premise needed a citation fix — done) | Nobody evaluates. HKP, Elben, Straeter all bound. Two papers *decline* the analysis in print. |
| 2 | Threshold **M\* = ζ₂/(2ζ₁)** and its base ≈ 5.3ⁿ | **CLEAN, but narrow** | No one states a threshold in the copy budget M. But M\* is one line from an identity three papers already have; only the **value and base** are yours. |
| 3 | **α-transition** as a diagnostic | **ALREADY-DONE in substance → rescope hard** | Elben **main text + Fig. 2**: "the error decay rate depends on number of measurements M". And arXiv:2601.04020 already fits a *rolling exponent*. |
| 4 | **Collective-vs-single-copy crossover** | **CLEAN** — the strongest claim you have | Elben has zero collective content. HKP explicitly disclaims collective protocols. Deside is an alternative *method*, not a crossover. |
| 5 | **Hardware confrontation**, pre-registered | **NEEDS-CITATION → rescope** | **PRApplied 23, 014075 (2025) already compares single-copy vs two-copy experimentally**, with a Fredkin gate. No crossover law. You are not first to the comparison — only to testing a *prediction*. |
| 6 | Higher-moment **k ≥ 3** qubit treatment | **NEEDS-CITATION → rescope to "exact, qubit, k=3,4"** | Elben App. D.2 does p₃ (bound, three regimes); Straeter does CV p₃; HKP flags k≥3 as future work. |

---

## Claim 1 — exact state-dependent evaluation of ζ₁, ζ₂ → **CLEAN**

**The decomposition itself was never yours, and is not even Elben's.** Verified from source:

- **Textbook.** Straeter introduces the general form as one that "admits the exact
  decomposition~\cite{Serfling1980,Lee1990}" (`hoeffding_variance.tex:22-23`). You already cite
  Serfling [17] and Hoeffding [16].
- **HKP [1] has it, and predates Elben by five months.** HKP supplementary Lemma (Variance)
  `lem:symm-full` (main.tex:1445-1452):
  `Var[ô(N,1)] = C(N,2)⁻¹ ( 2(N−2) Var[Tr(O_s ρ̂₁⊗ρ)] + Var[Tr(O_s ρ̂₁⊗ρ̂₂)] )` — an **equality**,
  identical to yours. arXiv dates: **HKP 2020-02-18** vs **Elben 2020-07-13** (verified from both
  abs pages).
- **Straeter [13] has it too**, `eq:hoeffding_p2`:
  `Var(p̂₂) = 4(T−2)/(T(T−1)) σ²_{2,1} + 2/(T(T−1)) σ²_{2,2}`. I verified symbolically that this
  equals the Hoeffding/Lee formula and therefore yours. *(Aside: Straeter's own `eq:hoeffding_general`
  is internally inconsistent with its k=2 case — it states the H-decomposition shape while defining
  σ as conditional-expectation variances. Their k=2 case is the correct one. Not your problem;
  don't cite it as support for the general form.)*

**What survives, and it survives cleanly: nobody EVALUATES the coefficients.**

- HKP bounds ζ₁, ζ₂ and never evaluates them; its central quantity, the shadow norm, is *defined*
  with a max over states, so it is structurally incapable of being state-dependent.
- Elben bounds via (D7)/(D12) and never evaluates either for any state.
- Straeter gives an exact state-dependent *expression* (`eq:sigma_bilinear`,
  σ²_{k,1} = **r**†**C r** − p²) but then only ever **bounds C** (`lem:entrywise`, `thm:gd_bounds`).
  Its only state-specific numbers come from a **jackknife**, i.e. estimated empirically from data
  "with no analytic input about the state".

**Two papers decline the analysis in print — these are your best citations, because they say the gap
is real:**

- **arXiv:2311.08108** (PRX 14, 031035), *Many-body entropies and entanglement from
  polynomially-many local measurements* — cites Elben's estimator, then: *"it is possible to derive
  explicit bounds on its variance, although it becomes increasingly involved for higher n."* It stops
  exactly where you start.
- **arXiv:2605.09958** (2026), *Quantum Nonlinear Properties from a Single Measurement Setting* —
  *"for general t ≥ 2, the statistical performance of these estimators is not yet fully understood."*
  A 2026 paper saying the question is open.

⚠️ **Wording caution.** Do not write "the exact variance" as the contribution — Straeter calls its own
decomposition "the exact Hoeffding variance formulas", and HKP has exact norm computations. Say
**"exact state-dependent evaluation of the coefficients"** or a referee will collide with their usage.

---

## Claim 2 — the threshold M\* = ζ₂/(2ζ₁) and base ≈ 5.3ⁿ → **CLEAN, but narrower than it reads**

**Nobody gives a threshold in the copy budget.** Elben's Lemma 1 `M ≥ 8 max{…}` contains no
threshold; its dominance condition `2^{1.5|AB|} > 2^{|AB|} p₂` **contains no M at all**. HKP: no
threshold of any kind (grep). The Nature Reviews toolbox: `M*` = 0 hits.

**But be honest about how much is left.** M\* is where Elben's own two terms cross:
`4(M−2)ζ₁ = 2ζ₂ ⟹ M−2 = ζ₂/(2ζ₁)` — one line of algebra from an identity that HKP, Elben and
Straeter all already have. And exponential-in-n crossing is *implicit* in Elben's bounds, whose terms
cross at `M ~ 2^{2|AB|}/p₂` — **base 4**. So the form and the exponential character are not new;
**the value and the base (5.3 vs the bound's 4) are.** The current text already says exactly this
after the last correction — keep it that way.

**Vocabulary near-misses that are NOT threats but a referee may pattern-match on:**

- **Yu et al., PRA 113, 032445 (2026)**, *Purity estimation for multiple quantum states with adaptive
  sampling* (DOI 10.1103/d52m-kz3x; **no arXiv preprint exists**). Its Eq. (34)
  `T_{k,m−1} = (c₂²/c₁)·d²/ln(1/δ)` **is** an explicit measurement-count threshold with an exponential
  base — but base **4** (d² = 4ⁿ), and it is a **Bernstein sub-Gaussian-vs-sub-exponential tail
  crossover**, separating "O(T^{−1/2}) sampling fluctuation" from a "T-independent large-deviation
  term". Not a U-statistic variance-regime threshold. Different object; worth one distinguishing
  sentence.
- **arXiv:2607.11369**, *Moment-based PPT criteria for random bipartite states* — gives "a threshold
  environment dimension s = λ_m d²". A threshold in **environment dimension for detection
  probability**, not in measurement budget. Different axis.

---

## Claim 3 — the α-transition as a diagnostic → **ALREADY-DONE in substance. Rescope hard.**

**🚩 SECOND-ELBEN FLAG — and it is Elben again, in the main text, with a figure.**

I verified this directly in Elben's `main.tex`, **before** the appendix boundary (char 32650 of
103246), i.e. this is main-text, not buried:

> *"our analysis reveals that the error decay rate depends on number of measurements M"*
>
> *"For intermediate M, the error decay rate is proportional to 1/M, while an even faster rate
> ∝ 1/M^{3/2} governs the error decay for small M."*
>
> *"Qualitatively similar results apply for estimating p₃ …, but there can be three decay regimes."*
>
> **Fig. 2 caption**: *"Statistical errors for the GHZ state. Dashed lines represent scalings of
> ∝ 1/M, and ∝ 1/√M."*

So: the budget-dependent decay rate is an Elben **main-text result with a supporting figure on a GHZ
state**. "The effective budget-scaling exponent is not a constant" is, as a *statement*, theirs.

**And the running-fit method is also published, six months before you.**

- **arXiv:2601.04020** — Aasen & Gärttner, *Limitations for adaptive quantum state tomography in the
  presence of detector noise*, **2026-01-07**, PRA (DOI 10.1103/3q13-xpc8). Verified: exists, abstract
  contains "gradual transition". Fig. 4 caption: *"Rolling power-law fit I ∝ N^{a} applied to the
  adaptive curves…, where a denotes the scaling exponent"*, and the text: *"The rolling fit of the
  scaling exponent … decreases towards a = −0.5 as the number of measurements increases."*
  **That is the running-exponent diagnostic, plotted, migrating.** Different domain (adaptive
  tomography infidelity, Bayesian simulation, noise-driven) — but the *method* is not new.
- **arXiv:2603.12235** — your [15] `hardwarehorizon2026`. Confirmed real: error obeys O(M^{−1/2})
  then saturates at a hardware floor.

**What actually survives for Claim 3** — and it is thin, so state it flatly:
fitting α **per (n, k)** on qubit shadow/purity data over a budget grid and comparing it, with no free
parameters, to a value computed from evaluated coefficients — and the **n-dependence** of that
migration. Elben draws two guide lines on one state; you predict the exponent quantitatively across
system sizes and moment orders. That is a real but modest contribution and must not be sold as
discovering that the exponent moves.

**MUST CITE: arXiv:2601.04020.** Not citing it is the single most likely referee ambush in the paper.

---

## Claim 4 — collective vs single-copy crossover → **CLEAN. This is your strongest claim.**

- **Elben does not treat collective measurement at all.** Its one "collective" hit (line ~251) is
  quasi-particle excitations. Verified.
- **HKP helps you**: its supplement (L1713) explicitly disclaims collective protocols as outside its
  lower bound — cite it as the gap you fill.
- **Deside, Haas & Cerf**, *Detecting strongly non-Gaussian entanglement*, **Phys. Rev. Research 8,
  033064 (2026)**, DOI 10.1103/297d-3tbj (arXiv:2504.15831, titled *"genuine"* on arXiv — renamed at
  publication; grep arXiv for "strongly" and you will find nothing). Interferes up to **3 copies**
  through a Fourier interferometer with photon-number-resolving detection. Its variance is **exact**:
  `Var[p_n^(k)] = (1−p_n²)/k` — *stronger than Elben, who only bounds*. **But: no crossover.** Zero
  occurrences of "single-copy", "crossover", "cheaper", "trade-off", "sample complexity", "scaling".
  Its only cost comparison is an **asymptotic divergence** argument — *"the randomized measurement
  toolbox breaks down in the CV regime"* — which is the logical opposite of a crossover: it says
  single-copy fails outright as d→∞, identifying no finite threshold. CV, simulation-only, no
  hardware, no pre-registration.
- **Straeter** compares single-copy vs collective only qualitatively, conceding Deside attains "a
  lower variance per detection event", traded against triple-coincidence rate. No crossover predicted.
- Chase over ~256–315 Elben-citing works: **zero** single-vs-collective crossover results.

**Caveat to record honestly:** Huang et al. (Science 2022) [2] and Chen et al. [3] prove *asymptotic
exponential separations* between learning with and without quantum memory. That is adjacent and makes
the *existence* of a collective advantage unsurprising — but it is a separation theorem for learning
tasks, not a parameter-free, budget-level crossover prediction for moment estimation. Say so
explicitly rather than letting a referee say it for you.

⚠️ **Unresolved (60-second check for the author):** Deside's published PRR full text is behind
Cloudflare (403); the "no crossover" finding is robust for **arXiv v1 (Apr 2025)** only, and the paper
was accepted Apr 2026. The v1→published abstract diff is 95% identical (only "genuine"→"strongly" and
grammar; the "extensive numerical simulations" clause survives), which is inconsistent with a new
theoretical result having been added. **If you have APS access, grep the published Sec. III.1 and
Discussion for "single-copy".**

---

## Claim 5 — hardware confrontation → **NEEDS-CITATION, and rescope the wording**

**🚩 You are not the first to compare the two routes on hardware.** Verified independently via
Crossref:

- **Peng, Liu, Liu, Zhang, Zhou & Lu**, *Experimental shadow tomography beyond single-copy
  measurements*, **Phys. Rev. Applied 23, 014075 (2025)**, DOI 10.1103/PhysRevApplied.23.014075.
  Demonstrates a **Fredkin gate** and reports a "significant reduction in sample complexity" for
  hybrid two-copy versus single-copy — **empirically, with no crossover law, no threshold, no
  exponent**. It is the experimental counterpart of your Section 5 and reads as motivation, not
  competition. **Note: Ting Zhang and He Lu are also authors of your Ref [7]** — the same group. A
  referee from that community will know this paper cold. It is currently uncited.

**What survives:** not "we tested the two routes on hardware" — that exists — but *"we locked a
parameter-free prediction in advance and confronted it with the device"*. Nothing in the ~256–315
Elben-citing works tests a variance/threshold/crossover **prediction** on hardware. Elben: simulation.
HKP: pure simulation. Deside: "extensive numerical simulations". Straeter: no hardware. Stricker et
al. is a hardware shadow-purity paper but tests tomography reconstruction, not a budget-scaling
prediction.

**Temper it with your own result.** Section 6 reports the n=2 anchor missing its locked prediction,
and Section 8 already says the computed points sit on a device the anchor shows the model does not
track. The defensible claim is *"we ran the confrontation and report what it did, including the
failure"* — which is a real contribution and is more credible than a success would have been. Do not
upgrade it.

---

## Claim 6 — higher moments k ≥ 3, qubits → **NEEDS-CITATION → rescope**

- **Elben App. D.2** (`sub:cubic`) treats **p₃ and Tr(ρ³)** with the full decomposition, Lemma 2
  (`eq:p3-sampling-rate`, a **three**-term `max{…}`), and numerics in D.3. Its **main text** names the
  three decay regimes. So "we treat k≥3" is not novel.
- **Straeter** does CV p₃ (the p₃-PPT criterion is its entire point), with 1/T, 1/T², 1/T³ terms.
- **HKP flags k≥3 as future work twice** (L695, L1540: *"leave a rigorous extension to future
  work"*) — a clean gap-citation for you.

**Rescope to:** the *exact evaluated* coefficients at k = 3, 4 for **qubit Pauli shadows**, and the
finding that even a correctly-formed two-term truncation fails there (15/20 vs 19/20) — which is a
statement nobody else is in a position to make, because nobody else evaluates the coefficients.

---

## Missing citations — papers to add

| Paper | Why | Priority |
|---|---|---|
| **arXiv:2601.04020** — Aasen & Gärttner, PRA, *Limitations for adaptive QST in the presence of detector noise* | Already does the **rolling power-law fit of the scaling exponent**, migrating toward −0.5. Directly pre-empts Claim 3's *method*. | **CRITICAL** |
| **arXiv:2311.08108** — PRX 14, 031035, *Many-body entropies… polynomially-many local measurements* | Cites Elben's estimator and **declines** the variance analysis ("becomes increasingly involved"). Best evidence the gap is real. | **HIGH** (helps you) |
| **arXiv:2605.09958** — *Quantum Nonlinear Properties from a Single Measurement Setting* | 2026: "statistical performance … not yet fully understood". Evidence of no scoop. | **HIGH** (helps you) |
| **Peng, Liu, Liu, Zhang, Zhou & Lu**, PRApplied **23**, 014075 (2025), DOI 10.1103/PhysRevApplied.23.014075 | *Experimental shadow tomography beyond single-copy measurements* — **already does the single-vs-two-copy comparison on hardware** (Fredkin gate). Directly bears on Claim 5. **Two of its authors also wrote your Ref [7].** | **CRITICAL** |
| **Deside, Haas & Cerf**, PRR 8, 033064 (2026), DOI 10.1103/297d-3tbj | The alternative collective route, with an **exact** variance. Straeter cites it; you compare against collective and should too. | **HIGH** |
| **arXiv:2304.12292** — *Enhanced Estimation of Quantum Properties with Common Randomized Measurements* | Nearest variance-machinery neighbour; unbiased multi-copy estimators with variance bounds. | MEDIUM |
| **Yu et al.**, PRA 113, 032445 (2026), DOI 10.1103/d52m-kz3x | Explicit measurement-count threshold with an exponential base (4ⁿ) — distinguish from M\*. | MEDIUM |
| **Stricker et al.**, PRX Quantum 3, 040310 (2022), arXiv:2206.00019 | Hardware shadow purity with the same U-statistic; the brief called it the closest to the threshold claim — it is **not** (zero threshold/crossover hits), but it is a hardware neighbour. | MEDIUM |
| **arXiv:2606.14204**, **arXiv:2607.11369** | Copy complexity under active memory; "threshold environment dimension". Adjacent axes. | LOW |

---

## Second-Elben flags

1. **🚩 Elben [18] main text + Fig. 2 — Claim 3.** Not the appendix: the budget-dependent decay rate
   is a headline result with a figure on a GHZ state. **Your thesis sentence and title both currently
   front exactly this.** This is the reframe's centre of gravity.
2. **🚩 arXiv:2601.04020 — Claim 3's method.** The rolling-exponent fit is published, Jan 2026.
2b. **🚩 PRApplied 23, 014075 (2025) — Claim 5.** The single-vs-two-copy comparison has already been
   done on hardware, by a group that overlaps your Ref [7]. It has no crossover law, so Claim 4 holds,
   but "we tested the routes on hardware" is not available as a novelty. Cite it and narrow to the
   *pre-registered prediction*.
3. **⚠️ HKP [1] — Claim 1's premise, not Claim 1.** It has the exact decomposition first (2020-02-18).
   Already fixed in Related work this pass; do not let it creep back.
4. **Not threats, confirmed:** Fu (2412.03381), the randomized-measurement toolbox review
   (2203.11374), Classical shadows with symmetries (2408.05279 — its "M" is the *number of
   observables*, not a copy budget), Stricker (2206.00019), Yu (PRA 113), Deside.

---

## The narrowed, defensible novelty

> The finite-M variance of the shadow purity/moment U-statistic is a textbook Hoeffding identity, and
> that it produces two budget-scaling regimes is established — stated in Huang, Kueng and Preskill's
> supplement and read off explicitly, with a figure, in Elben et al.'s main text. Both bound the two
> variance coefficients rather than evaluating them, and neither locates the budget at which the terms
> exchange dominance. **This work evaluates the coefficients state-by-state for qubit Pauli shadows,
> which fixes the threshold M\* = ζ₂/(2ζ₁) at a specific value with base ≈ 5.3ⁿ rather than the 4ⁿ the
> bounds imply, and carries the evaluation to k = 3 and 4, where even a correctly-formed two-term
> truncation fails. Evaluating the coefficients is what makes the second, parameter-free comparison
> possible: against the collective two-copy route, which neither of those works treats — Huang, Kueng
> and Preskill explicitly place it outside their analysis — yielding a crossover prediction we then
> confront with hardware and report as it came out, including the anchor it failed.**

Two sentences of ceded ground, then the contribution. The **crossover law** and the **pre-registered**
hardware confrontation are the load-bearing novelty; the exact evaluation is what earns them.

**Two hard constraints on the reframe:**
- **Claim 3 must not appear in the thesis or the title.** Elben's main text and Fig. 2 have it.
- **Claim 5 must be worded as testing a locked, parameter-free prediction** — not as comparing the two
  routes on hardware, which PRApplied 23, 014075 (2025) already did, with a Fredkin gate, by a group
  overlapping your Ref [7].

---

## Method notes / what I could not verify

- **Verified by me against primary sources:** HKP's `lem:symm-full` and its arXiv date; Elben's
  main-text regime sentences, Fig. 2 caption, App. D.1/D.2, Lemma 1/2 term counts (2 and 3), and its
  zero collective content; Straeter's `eq:hoeffding_p2` (symbolically equal to ours) and its
  Serfling/Lee attribution; the existence and abstracts of 2601.04020 and 2311.08108.
- **Rests on a single agent pathway (not independently re-read by me):** the Fu, symmetries, toolbox
  and Stricker reads; the Semantic Scholar / OpenAlex citation sweeps; the Yu (PRA 113) full text
  (publisher PDF via the Crossref similarity link; APS 403s automation).
- **Open:** Deside's *published* text (Cloudflare 403) — see the Claim 4 caveat.
- **⚠️ The citation sweep is a SCREEN, not a proof — do not overstate the negative.** Neither Semantic
  Scholar nor OpenAlex can resolve *"cites Elben's Appendix D specifically"*: contexts give the citing
  sentence, not the cited appendix, and **114 of 256 S2 records had no context sentence at all**.
  Coverage also differs (OpenAlex 315 vs S2 256); the delta was screened by abstract with the same
  signatures and turned up nothing, but that is not a proof. "Zero of ~256–315 do X" means *the screen
  found none* — closing it fully would need arXiv/Scholar full-text search. Write the reframe so it
  does not depend on an exhaustive-negative claim.
- **The brief's premises inverted again, twice.** It said 2206.00019 was "the closest to your
  threshold claim" — it has **zero** threshold/crossover hits. And its A1 premise (combine the SEs →
  3.1σ) double-counts. That is now the sixth and seventh inverted premise in this project; the habit of
  re-deriving every premise from source is earning its keep.


---

# PASS 7 (2026-07-20) — PRIOR ART FOR THE DRAFT CLOSED FORM (`DRAFT_asymptotic_section.tex`)

Verified against the two fetched papers (PDF text extracted locally, not from memory).

## Sources
- **HKP** = Huang, Kueng, Preskill, *Predicting Many Properties of a Quantum System from Very
  Few Measurements*, **arXiv:2002.08953** (the paper's ref `huang2020shadows`, bibitem at
  paper.tex:1632).
- **Hayashi** = Masahito Hayashi, *Finite-Sample Selected Covariance Spectra in Classical
  Shadows*, **arXiv:2606.00527**, submitted **30 May 2026**. NOT currently cited in paper.tex.

## 7.1 — The draft's closed form is a substitution, not a new derivation
- Reduction: `Tr(G rho) = 2^-n SUM_s <P_s> x_s`  (x_s = Tr(G P_s), reconstructed Pauli coeff)
  => `E[Tr(G rho)^2] = 4^-n SUM_{s,s'} <P_s><P_s'> E[x_s x_s']`.
- Substituting the known second moment `E[x_s x_s'] = 3^{|supp(s) cap supp(s')|} m_{s (-) s'}`
  (compatible, else 0) gives EXACTLY the draft's display, with `s TRIANGLE s'` identical to
  Hayashi/HKP's `s (-) s'` (cancellation string).
- **Numerically IDENTICAL:** `4^-n SUM <P_s><P_s'> E_Hayashi[x_s x_s']` vs the draft's Clifford-
  3-design TOTAL, on committed-seed states: max |diff| = 4.0e-15 (n=2), 8.9e-15 (n=3), 1.6e-14
  (n=4). The draft's closed form is a two-line corollary of the HKP/Hayashi second-moment
  formula. NOT new.

## 7.2 — Haar and Pauli shadows give identical second-moment statistics (k=2)
- Per-qubit second-moment tensor M[a,b] (a,b in {I,X,Y,Z}): **entrywise identical** for local
  Haar and local Pauli (random X/Y/Z) shadows, max |M_haar - M_pauli| = 1.8e-15.
- Full matrix `E[x_P x_Q]` under Haar = under Pauli = Hayashi's formula, to ~1e-13, at n=1,2,3.
- Consequence: `Tr(G rho)` is linear and `Tr(G_1 G_2) = 2^-n SUM_P x_P^(1) x_P^(2)` is bilinear
  in independent shadows, so BOTH `zeta_1` and `zeta_2` are functions of the second-moment
  matrix ONLY (no moments beyond 2nd). Verified: `zeta_2(Haar-M) = zeta_2(Pauli-M)` to 8.5e-14.
  Hence zeta_1, zeta_2, and M* = zeta_2/(2 zeta_1) are ENSEMBLE-INDEPENDENT for k=2.
- Bearing on the anchor disclosure (paper Sec 6.4, ~L1168-1178): the simulation (local Haar)
  and the hardware anchor (random Pauli-basis) are different *sampling* ensembles, but their
  k=2 second-moment statistics — and thus the anchor's locked 1.500 prediction — coincide. The
  "distinct ensembles" description is true of the sampling, not of the k=2 prediction. (Reported
  only; paper.tex not edited.)

## 7.3 — Attribution granularity
- **(a) HKP Lemma 4 (eq. S52) contains the FULL off-diagonal structure**, not just the diagonal
  Var[x_P]. Verbatim: "Fix two k-qubit Pauli observables P_p ..., P_q ... Then, the following
  formula is true for any state sigma: E_{U~Cl(2)^ox k} SUM_b <b|U sigma U^dag|b><b|U(D^-1_1/3)^ox
  k(P_p)U^dag|b><b|U(D^-1_1/3)^ox k(P_q)U^dag|b> = f(p,q) tr(sigma P_p P_q), where f(p,q) = 0
  whenever there exists an index i such that p_i != q_i and p_i, q_i != I. Otherwise f(p,q) = 3^s,
  where s is the number of non-identity Pauli indices that match." It fixes two DIFFERENT
  observables P_p, P_q; the diagonal `Sigma_PP = 3^{wt(P)} - m_P^2` is the special case.
- **(b) Hayashi presents his Proposition 4 as a RESTATEMENT of HKP**, not his own. Verbatim
  (Sec IV.D, "Exact covariance formula for qubit local Pauli shadows"): "For the uniform local
  Pauli protocol, the underlying second-moment mechanism already appears in the observable-wise
  variance analysis of Huang-Kueng-Preskill [11, Lemma 4]. We first recall the corresponding full
  covariance-matrix form in the uniform case, and then extend it to biased basis probabilities."
  Hayashi's OWN result is Theorem 4 (biased local Pauli, factor beta(P,Q)); Proposition 4 is the
  uniform special case he recalls, and Remark 7 states Theorem 4 reduces to Proposition 4 at
  p=1/3.
- **(c) Correct citation for the draft's closed form:** HKP [Lemma 4] is the origin (2020) of the
  uniform second-moment formula the draft substitutes; Hayashi Prop 4 is a compact restatement
  (2026), Hayashi Thm 4 a biased generalization the draft does not use. For the uniform case the
  draft needs, HKP alone is the correct primary citation; Hayashi is a valid secondary citation
  for the modern covariance-matrix form. Presenting the closed form as new is not supportable.
- **(d) The (5/4)^n rate / diagonal-sum asymptotics / M* threshold: NOT FOUND in either paper.**
  Neither Hayashi nor HKP contains "5/4", "(5/4)", "1.25", "5^n", "budget", "M*", or
  "zeta_2/(2 zeta_1)". The asymptotic growth of `SUM_s 3^|s| <P_s>^2` for depolarized Haar states
  (-> (1-q)^2 (5/4)^n), the OFF limit 2(1-q)^3, and the threshold base 28/5 are absent. The A1
  result (the base derivation) has no prior art found in these two sources.
- **(e) Single-copy vs collective CROSSOVER: NOT FOUND.** "crossover" appears in neither paper.
  HKP treats single-copy vs collective as a sample-complexity SEPARATION / lower bound (Thm 2:
  single-copy cost ~ ||O||^2_shadow; Thm 5: "does not apply to protocols where collective
  measurements are applied across many copies") — collective is more powerful, not a budget
  threshold where single-copy wins. HKP's "threshold" (once) is an experimental accuracy
  threshold. Hayashi's "collective" refers to the covariance-matrix's joint structure.

## What is and is not the paper's, under the corrected boundary
- **NOT the paper's (has prior art):** the closed-form second-moment identity for zeta_1 (draft
  Block A "Proposition"/the E[Tr(G rho)^2] display, and the "diagonal = weight-only ansatz"
  observation). Origin: HKP Lemma 4 (2020); restated Hayashi Prop 4 (2026). The n=1 reduction and
  the Clifford-3-design cross-check are independent re-verifications, not new theory.
- **The paper's own (no prior art found in HKP or Hayashi):** the asymptotic growth base
  (5/4)^n of the diagonal, the OFF limit 2(1-q)^3, the composite zeta_1 -> (1-q)^2(5/4)^n +
  2(1-q)^3 - (1-q)^4, the threshold base M* -> (28/5)^n, and the single-copy/collective crossover
  and exponential-wall framing (Secs 4-6). These use the second-moment formula as an input but
  are not stated in either source.


---

# PASS 9 (2026-07-20) — PRIOR ART FOR THE CLOSED-FORM ENSEMBLE-AVERAGED ZETAS

Verified against 5 fetched papers (PDF text extracted locally) + 2 web searches. The object
at risk: a closed-form HAAR-AVERAGED local-shadow variance for zeta_1, zeta_2 of a depolarized
Haar-random state, and the resulting analytic M*(n) with base 28/5 and prefactor 1/(2(1-q)^2).

## The closed forms (verified in 9.1, exact ensemble averages)
- zeta1 = 4^-n[ 1 + u^2(10^n-1)/(d+1) + 2u^2(4^n-1)/(d+1) + 2u^3(16^n-10^n-2*4^n+2)/((d+1)(d+2)) ] - tr2^2
- zeta2 = 4^-n[ 28^n + u^2(34^n-28^n)/(d+1) ] - tr2^2   (u=1-q, d=2^n, tr2=u^2+q(2-q)/d)
- Derived from HKP Lemma 4 (per-state) + Haar moments E[<P>^2]=1/(d+1),
  E[<Pu><Ps><Ps'>]=2/((d+1)(d+2)); counts sum_compat 3^|ov|=16^n (diag 10^n),
  sum_compat 9^|ov|=34^n (diag 28^n), all exact at n=2,3,4.
- Reproduce the PASS-8 exact evaluator over >=250 Haar states within 2 SEM (n=3..7).
- Asymptotics: zeta1/(5/4)^n -> (1-q)^2; zeta2/7^n -> 1 (so C_2=1 EXACTLY);
  M*(n) -> (28/5)^n / (2(1-q)^2), i.e. base 28/5 and prefactor 1/(2(1-q)^2), both EXACT.

## Term search across the 5 named sources (verbatim PDF extraction)
10^n, 16^n, 28^n, 34^n, 5/4, 28/5, and the threshold zeta_2/(2 zeta_1): **NOT FOUND in any** of
Elben 2007.06305, HKP 2002.08953, Hayashi 2606.00527, Gong 2410.12712, Aasen 2601.04020.

## Per-source (9.2a)
- **Elben 2007.06305** (Mixed-state entanglement from local randomized measurements, 2020-07-13,
  ref [18]). App. D eq. D15/D16: the EXACT variance decomposition into linear+quadratic (the
  Hoeffding/U-statistic structure) AND a variance UPPER BOUND
  Var[p2] <= 4*2^|AB| p2 / M + 4*2^{1.5|AB|} / M^2, plus the two 1/M vs 1/sqrt(M) regimes
  (Fig. D.1). NO Haar averaging (0 "Haar" in the paper), NO closed form, NO 28/5, NO threshold.
  => confirms the paper's "Elben bounds rather than evaluates, no threshold" is ACCURATE for
  Elben (unlike the PASS-8 finding that the same wording is FALSE when applied to HKP Lemma 4).
- **HKP 2002.08953** (ref [1]). Lemma 4 = per-state 2nd-moment identity (evaluation); shadow-norm
  bounds. No Haar-averaged closed form, no constants, no threshold.
- **Hayashi 2606.00527**. Prop 4 (restates HKP), Thm 4 (biased). 0 "Haar"; no closed form; no
  threshold.
- **Gong 2410.12712** (On the sample complexity of purity and inner product estimation, ref [4]).
  Sample-complexity LOWER BOUNDS using convex mixtures of Haar-random states as hard instances.
  No shadow-variance closed form, no constants, no threshold.
- **Aasen 2601.04020** (Limitations for adaptive quantum state tomography ..., ref [20]).
  Numerically Haar-averaged tomography-error curves. No purity-shadow variance closed form, no
  constants, no threshold.

## Web screen (9.2b) -- closest hits, NOT the object
- Hu, You et al., "Classical shadow tomography with locally scrambled quantum dynamics"
  (arXiv:2107.04817): ensemble-averaged shadow NORM via the entanglement-feature formalism -- a
  general framework for ensemble-averaged variances, but not the specific purity-estimator closed
  form, the (5/4)^n rate, the 28/5 base, or the threshold.
- arXiv:2202.03272 (Pauli-invariant unitary ensembles): ensemble-averaged shadow-variance
  framework, same character.
- **SCREEN, not proof:** 5 papers read verbatim + 2 web searches; a full exhaustive negative
  would require reading the locally-scrambled appendices.

## What is and is not new (9.2c)
- **Prior art:** the variance DECOMPOSITION (Elben, HKP), the per-state 2nd-moment IDENTITY (HKP
  Lemma 4), and the ensemble-averaged-variance FRAMEWORK (Hu-You locally-scrambled).
- **Not found in the sources searched (plausibly new, with caveats):**
  (i) the explicit closed-form ensemble-averaged zeta_1 -- but it is a routine Haar-average of HKP
      Lemma 4 using standard Haar moments;
  (ii) the closed-form zeta_2 -- same;
  (iii) the analytic M*(n) -- follows from (i),(ii); the threshold concept zeta_2/(2 zeta_1) is the
      paper's;
  (iv) the base 28/5 -- follows (= 7/(5/4));
  (v) the prefactor 1/(2(1-q)^2) -- follows from C_2=1 exactly.
- Two novelty claims have inverted in this project; (i),(ii) are elementary given HKP Lemma 4 and
  the Hu-You framework, so they should be framed as an explicit evaluation, not a new technique.
  (iii)-(v) -- the threshold, base, and prefactor -- were not found in any source and are the
  paper's genuine analytic contribution.
