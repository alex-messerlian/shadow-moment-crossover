# The single-copy variance theorem generalizes to k=3 and k=4

Zero credits (local computation). The k=2 (purity) variance law is exact and proven; this task tests whether the general Hoeffding/Lee U-statistic variance holds for the higher moments Tr(ρ³) and Tr(ρ⁴):

> Var(U_M) = C(M,k)⁻¹ · Σ_{c=1..k} C(k,c)·C(M−k, k−c)·ζ_c,  ζ_c = Var[h_c], h_c = c-th Hoeffding projection.

**Verdict up front: it generalizes.** The formula verifies against brute force at k=3 and k=4 across three ensembles, and its out-of-sample α predictions match the measured budget-scaling values (k=3: 6/7 within 2·SE, k=4: 5/5). Nothing was tuned. Code: `anrl/theory/general_k.py` (+ `tests/test_general_k_variance.py`); reproduce with `experiments/general_k_variance.py`.

## Step 1 — the brute-force reference is the exact estimator

The brute-force variance is measured on the **exact full U-statistic** `anrl.benchmark.moment_ustats.exact_moment_ustatistic` (the Möbius/power-sum estimator, verified elsewhere against brute-force enumeration to ~1e-13) — **not** a subsampled tuple sum. This avoids the trap of comparing against a subsample (which has strictly larger variance).

## Step 2 — the ζ_c are computed correctly and converge

The intermediate projections are the trap: for k≥3, **ζ₂ is the two-argument projection, not the kernel variance** (the kernel variance is ζ_k). I compute each ζ_c state-agnostically by the projection's definition — h_c(g_1..g_c) = mean over the k! orderings of Re Tr of the product of the c fixed dense G's and (k−c) copies of ρ (the inner expectation is exact since E[G]=ρ and the trace is multilinear; only the outer c-tuple average is Monte Carlo). This works for **any** state, including low-rank, where the noisy-pure closed forms do not apply.

**Convergence (k=3, n=3), ζ_c stable from 200k → 1.8M outer samples:**

| state | ζ₁ | ζ₂ | ζ₃ |
|---|---|---|---|
| noisy_pure | 1.470 → 1.470 | 28.50 → 28.52 | 1943 → 1940 (0.2%) |
| ghz | 1.021 → 1.022 | 18.28 → 18.24 | 1738 → 1721 (1%) |
| low_rank | 1.060 → 1.062 | 21.94 → 21.81 | 1822 → 1820 |

All ζ_c stabilize to ≲1%. **Cross-check:** the state-agnostic nested-MC matches the existing noisy-pure closed-form estimator (`estimate_hoeffding_components`) to 0.997–1.006 at k=3 (n=3) and k=4 (n=2,3) — confirming the existing code is correct (not the conflation bug) and both methods agree.

## Step 3 — the general formula verifies at k=3 and k=4

Lee formula (fed the converged ζ_c) vs the brute-force variance of the exact estimator (24000 reps, 8-batch SE), across states and M (ratio = brute / formula):

| k, n | state | M=20 | M=40 |
|---|---|---|---|
| k=3, n=3 | noisy_pure | 1.023 | 1.024 |
| | ghz | 0.992 | 0.995 |
| | low_rank | 0.997 | 1.000 |
| k=4, n=2 | noisy_pure | 1.019 | 0.992 |
| | ghz | 1.029 | 1.018 |
| | low_rank | 0.985 | 0.982 |

All ratios within ~3%; z-scores within ±1.1 (k=4) and ±2.9 (the one outlier, k=3 noisy_pure, is brute-force MC scatter — the ζ's are converged). A wrong projection (e.g. the ζ₂-vs-kernel-variance conflation) would be off by tens of percent, not ~2%. **The formula holds at k=3 and k=4, for all three ensembles including low-rank.**

## Step 4 — physics: ζ scalings and out-of-sample α

**ζ_c exponential bases (noisy-pure, q=0.1):**

| | ζ₁ | ζ₂ | ζ₃ | ζ₄ |
|---|---|---|---|---|
| k=3 | 1.30ⁿ | 2.86ⁿ | 14.9ⁿ | — |
| k=4 | 1.34ⁿ | 2.61ⁿ | 6.53ⁿ | 36.2ⁿ |

Each projection has a distinct exponential base, growing with c; the kernel variance ζ_k has the steepest growth. The α transition (0.5 → 1.0) occurs as the budget crosses the threshold where the ζ_k term (RMSE ~ 1/M) overtakes the ζ₁ term (RMSE ~ 1/√M) — the same mechanism as k=2, now with the higher-c terms.

**Out-of-sample α (exact formula with the recomputed ζ_c, over the actual saved budgets; no fitting to α):**

| k | n range | α within 2·SE of measured |
|---|---|---|
| 3 | 2–8 | **6/7** (only n=5 misses, at ~2.15·SE — right at the boundary and seed-sensitive: it lands 2.11–2.15·SE across RNG seeds, so effectively a marginal in/out) |
| 4 | 2–6 | **5/5** |

The predictions track the full transition (e.g. k=3: α 0.50 → 0.81 → 1.18 across n=2→8; k=4: 0.49 → 0.79 across n=2→6), matching the measured budget-scaling values.

## Verdict

**The single-copy variance theorem generalizes across the moment family (k=2, 3, 4).** The exact Hoeffding/Lee formula, with the correctly-computed projections ζ_c (ζ₂ = two-argument projection, ζ_k = kernel variance), matches the brute-force variance of the exact estimator at k=3 and k=4 for noisy-pure, GHZ, and low-rank states, and predicts the measured α transition out-of-sample (11/12 cells within 2·SE). No scoping to purity is needed — the theorem is not purity-specific.

**Integrated:** `anrl/theory/general_k.py` (`hoeffding_components_mc` — state-agnostic ζ_1..ζ_k for k∈{2,3,4}) + exports, with 6 passing tests in `tests/test_general_k_variance.py` locking the Lee-formula-vs-brute-force match, the nested-MC-vs-closed-form agreement, and the projection definitions. Numbers: `results/general_k_variance.json`.
