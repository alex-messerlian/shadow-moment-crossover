# Voice-consistency pass — flagged passages

Every passage in `paper.tex` where the register drifts from finished-physics prose into
meta-commentary, importance-editorializing, or self-congratulatory honesty-framing. Each
entry: location, offending text, proposed replacement, reason. **No scientific claim,
number, equation, figure, caption, table, or citation changes** — these are prose-only.

The paper's second-person/lecture register was checked and is clean (no "you can see", "the
reader", "notice that", "recall that"). The dominant tic is a self-congratulatory
"we disclose X rather than hide it" frame that recurs seven times; the honest *content* is
kept in every case, only the framing is cut.

---

## §1 Introduction

**1. Line 134 — "the crux."**
Before: "…their task is testing and ours is estimation, a distinction that turns out to be the crux and that we take up in Section~7."
After: "…their task is testing and ours is estimation, a distinction we take up in Section~7."
Reason: "that turns out to be the crux" tells the reader the distinction is important instead of letting it be. The forward reference (a real signpost) is kept.

**2. Line 136 — filler "simply."**
Before: "The caution is reasonable. It has simply never been measured."
After: "The caution is reasonable. It has never been measured."
Reason: "simply" is rhetorical filler; the flat sentence is stronger.

**3. Line 145 — "not a detail."**
Before: "That framing conceals a structural feature of the estimator, and the feature turns out not to be a detail."
After: "That framing conceals a structural feature of the estimator."
Reason: "turns out not to be a detail" is a sentence about the next sentence (the Thesis paragraph, which states the feature). Cut the throat-clearing.

## §2 Setting and estimators

**4. Line 233 — "not decoration."**
Before: "That second identity is not decoration; it is what makes the depolarizing bias law of Section 4.1 come out linear in the noise rate."
After: "The second identity is what makes the depolarizing bias law of Section 4.1 come out linear in the noise rate."
Reason: "is not decoration; it" pre-defends the identity. The logical link (identity → §4.1 linearity) is genuine signposting and is kept.

**5. Line 252 — "matters for Section 6 / turns out to be."**
Before: "The gate count matters for Section 6: it is the reason the entangling overhead turns out to be affordable on real hardware."
After: "The gate count is the reason the entangling overhead is affordable on real hardware (Section 6)."
Reason: "matters for" and "turns out to be" are narrative padding around a forward reference; the reference is kept as a parenthetical.

**6. Line 283 — confessional meta-frame (§2.5).**
Before: "Choosing the wrong ensemble inverts the conclusion of the benchmark. We report this because we made the error ourselves, and correcting it reversed a headline result."
After: "Choosing the wrong ensemble inverts the conclusion of the benchmark. We made this error, and correcting it reversed a headline result."
Reason: the honest disclosure (we made the error; it reversed a headline result) is kept; the "We report this because" diary-frame is cut, matching the good register already used in §3.1 and §4.1.

## §3 The single-copy variance law

**7. Line 315 — "worth being clear about."**
Before: "One warning is worth being clear about, because the error is easy to make and it can occur quietly."
After: "One subtlety is easy to get wrong and can pass unnoticed."
Reason: "worth being clear about, because" announces the warning instead of stating it. The content (easy to make, quiet) is kept.

**8. Line 398 — "worth reporting because" (§3.4).**
Before: "The failure is worth reporting because the ansatz \emph{appears} to hold."
After: "The ansatz nonetheless \emph{appears} to hold."
Reason: "The failure is worth reporting because" is throat-clearing; "nonetheless" bridges from the preceding "This is false" finding.

**9. Line 454 — "load-bearing rather than decorative" (§3.5).**
Before: "The exact combinatorial structure is load-bearing rather than decorative. A two-term approximation of the form \(\mathrm{Var} \approx 4\zeta_1/M + \zeta_2/M^2\), which is the natural shortcut, manages only 5 of 8 at \(k = 2\) and fails at \(n = 6\) by nearly seven standard errors."
After: "The exact combinatorial structure is necessary, not a convenience. A two-term approximation of the form \(\mathrm{Var} \approx 4\zeta_1/M + \zeta_2/M^2\), the natural shortcut, manages only 5 of 8 at \(k = 2\) and fails at \(n = 6\) by nearly seven standard errors."
Reason: "load-bearing rather than decorative" editorializes; "necessary" states it flatly and the evidence (two-term fails) carries it. All numbers unchanged.

## §4 The collective route

**10. Line 553 — THE canonical example (§4.2).**
Before: "The statement is stronger than a bias formula, and the strength is the point. The noise does not corrupt the measurement. It relabels which state is being measured. The collective test performs exactly as designed; it simply answers a question about the damaged state \(\sigma\) rather than the intended state \(\rho\)."
After: "Under a per-qubit channel the noise does not corrupt the measurement; it relabels which state is measured. The collective test performs exactly as designed, returning the true \(k\)-th moment of the damaged state \(\sigma = \mathcal{E}^{\otimes n}(\rho)\) rather than of the intended \(\rho\)."
Reason: the task's own model fix. "and the strength is the point" is a sentence about the next sentence; the physics stated plainly is stronger. Content identical (\(\sigma\) and Tr(\(\sigma^k\)) are both already defined and boxed just above).

## §5 The crossover

**11. Line 665 — "sharper than it sounds" (§5.3, budget prediction).**
Before: "Confirmed, and the confirmation is sharper than it sounds: the collective RMSE plateaus exactly at the predicted floor as the budget grows, while the single-copy RMSE keeps falling. That is a direct test of the bias-versus-variance distinction on which the whole crossover rests, and it holds."
After: "Confirmed. The collective RMSE plateaus exactly at the predicted floor as the budget grows, while the single-copy RMSE keeps falling---a direct test of the bias-versus-variance distinction on which the whole crossover rests, and it holds."
Reason: "the confirmation is sharper than it sounds" tells the reader how to weigh the result; the result (plateaus exactly at the floor) speaks for itself.

**12. Line 671 — "the sharpest test…" (§5.3, higher-k prediction).**
Before: "Higher \(k\) moves the crossover later. This one is counterintuitive, and it is the sharpest test of whether the mechanism is understood rather than merely fitted. A naive variance argument says that higher moments compound the single-copy variance faster and should therefore cross earlier."
After: "Higher \(k\) moves the crossover later. This is counterintuitive: a naive variance argument says that higher moments compound the single-copy variance faster and should therefore cross earlier."
Reason: the task's named example. "the sharpest test of whether the mechanism is understood rather than merely fitted" is decoration; the counterintuitive result and its explanation follow immediately.

## §6 Hardware

**13. Line 724 — "its content and its value" (§6 opener).**
Before: "This section reports a prediction that failed, and the diagnosis of why it failed. That is its content and its value."
After: "This section reports a prediction that failed, and the diagnosis of why it failed."
Reason: the first sentence is the framing (kept — the failed-prediction framing is load-bearing). "That is its content and its value" tells the reader how to value the section.

**14. Line 736 — "turns out to matter" (§6.1).**
Before: "Readout error is not published, a fact that turns out to matter (Section 6.3)."
After: "Readout error is not published, a fact that matters (Section 6.3)."
Reason: "turns out to" is a narrative tic; "matters" is direct. Forward reference kept.

**15. Line 742 — "rather than bury them" (§6.1).**
Before: "Two properties of that tier are material to the interpretation of our results and we state them rather than bury them."
After: "Two properties of that tier are material to the interpretation of our results."
Reason: self-congratulatory honesty-frame; the disclosure (the two properties) follows in full.

**16. Line 788 / 792 — "for honesty about scope" / "and say so" (§6.2).**
Before: "For contrast and for honesty about scope: Haar-random states at \(n = 4\) require 46 CZ gates including 20 routing SWAPs on this topology. … We therefore restrict the hardware series to GHZ ladders and say so."
After: "By contrast, Haar-random states at \(n = 4\) require 46 CZ gates including 20 routing SWAPs on this topology. … We therefore restrict the hardware series to GHZ ladders."
Reason: "For contrast and for honesty about scope" and "and say so" are two frames around an honest scope statement; the scope statement (46 CZ, 20 SWAPs, restrict to GHZ) is kept verbatim.

**17. Line 1017 — "it is worth stating because" (§6.7).**
Before: "One further constraint emerged and it is worth stating because it affects reproducibility rather than physics."
After: "One further constraint emerged that affects reproducibility rather than physics."
Reason: "it is worth stating because" announces the constraint instead of stating it.

**18. Line 1032 — "rather than eliding it" (§6.7).**
Before: "We flag the substitution rather than eliding it."
After: "We flag the substitution."
Reason: the substitution is already disclosed in the preceding sentences; "rather than eliding it" is the self-congratulatory tag.

## §7 Related work (CGK reasoning left intact; only two editorializing tags cut)

**19. Line 1122 — "this is the crux."**
Before: "The relationship to our result turns on the task, and this is the crux."
After: "The relationship to our result turns on the task."
Reason: named example. The reasoning that follows (testing vs estimation) is the crux and is untouched.

**20. Line 1134 — "a strength rather than a coincidence."**
Before: "The two results nonetheless share a mechanism, and this is a strength rather than a coincidence."
After: "The two results nonetheless share a mechanism."
Reason: named example. The shared-mechanism argument that follows is untouched.

## §8 Limitations

**21. Line 1188 — "rather than leave them to be found" (section opener).**
Before: "We state these ourselves rather than leave them to be found."
After: (removed)
Reason: self-congratulatory frame; a Limitations section's bullets are the disclosure.

**22. Line 1204 — "a finding rather than a concealed assumption."**
Before: "Section 6 documents exactly where a real device departs from them, and that departure is a finding rather than a concealed assumption."
After: "Section 6 documents exactly where a real device departs from them."
Reason: the "finding rather than a concealed assumption" tag editorializes; the plain statement is stronger.

---

## Considered and deliberately kept (not flagged)

- §3.1 "We made this error, and it produced an approximation that failed at \(k \geq 3\)…" — honest disclosure stated as a finding; the target register, kept.
- §4.1 "We initially assumed exactly that compounding form, and it overestimated the bias by a factor of five to fourteen…" — same; kept.
- §5.3 "We predicted the wrong direction before running this, and the law corrected us." — crisp, honest, kept.
- §6.3 "Had we extrapolated rather than measured… we would have recorded a failed prediction that we had manufactured ourselves." — kept.
- §6.6 "This is not a new observation about NISQ devices in general, and we do not claim it as one." — honest scoping (priority disclaimer), kept.
- §8 "The single miss sits on the two-sigma boundary… We report it rather than round it to 7 of 7." — the most concrete of the honesty-frames (it names the specific rounding resisted); kept as characterful content.
- §6.1 platform disclosure, §6.6 conclusion scope, and the §7 CGK reasoning — kept intact; these reason rather than editorialize.
