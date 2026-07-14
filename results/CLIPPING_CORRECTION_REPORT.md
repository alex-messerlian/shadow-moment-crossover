# Does estimator clipping explain the out-of-ensemble RMSE gap?

Zero credits (local computation). Hypothesis under test: the ~7% predicted-vs-measured RMSE gap in the out-of-ensemble stress test is caused by the pipeline **clipping** purity estimates to [0,1], which lowers the measured RMSE while the theory predicts the unclipped variance. Verdict up front: **the pipeline does not clip, clipping does not explain the gap, and the gap is finite-sample estimation noise on both sides (a ~144-trial measured RMSE plus predicted-side ζ Monte-Carlo error) — the theory's RMSE point prediction is exact.** Nothing was tuned to fit.

Reproduce: `experiments/clipping_investigation.py` (Steps 1–4) and `tests/test_clipping.py`. Code: `anrl/theory/clipping.py`.

## Step 1 — ground truth about our pipeline

**Does the codebase clip? NO.** The copy-fair single-copy estimator returns the raw unbiased U-statistic with no projection to any range:
- `anrl.benchmark.shadows.full_purity_ustatistic` → `(pair_trace.sum() − trace)/(M(M−1))`
- `anrl.benchmark.budget.moment_ustat_linear` → `(tr_s2 − c)/(M(M−1))`, `…/(M(M−1)(M−2))`
- `run_stress_test._measure_worker` computes `(moment_ustat_linear(…) − truth)²` — **no clip**; the predicted RMSE is the raw Hoeffding `exact_single_copy_rmse`. **Both sides are unclipped.** (The repo's `np.clip` calls are on sampling probabilities and the collective route's p₊, never the single-copy estimate.)

**The signed gap (from `results/stress_test.json`, 36 cells):** median |rel err| = **6.7%** (matches the ~7% recollection), but median **signed** rel err = **+3.4%** with 24/36 cells over. So the theory over-predicts *in the median*, consistent with the clipping hypothesis's direction — but the 6.7% is mostly **symmetric scatter** (−18% … +30%), not a strong systematic.

## Step 2 — the clipped-RMSE closed form (derived + verified)

For X ~ N(μ, σ²), Y = clip(X, a, b), with α=(a−μ)/σ, β=(b−μ)/σ:

> E[(Y−μ)²] = (a−μ)²Φ(α) + (b−μ)²(1−Φ(β)) + σ²[(Φ(β)−Φ(α)) + αφ(α) − βφ(β)]

(the three terms: mass below a → a, mass above b → b, and the in-range truncated variance, using ∫z²φ = Φ − zφ). Implemented as `clipped_rmse` / `clipped_mse`.

**Verification:** matches Gaussian Monte Carlo to <0.1% across μ ∈ {0.05…1.0}, σ ∈ {0.02…0.3}, including the μ=0,1 boundaries (12 tests). Against the *actual* clipped U-statistic it matches to ~3% for near-μ=1 pure states (the small overshoot is genuine non-Gaussianity of the estimator at the boundary) and to ~0.5% for interior μ. It reproduces the reported sandbox effect (haar n=4, M=200: unclipped ≈0.43, clipped ≈0.29 = 0.71×; the exact μ=1 identity is RMSE_clipped = σ/√2).

## Step 3 — does clipping close the gap? NO — it worsens it

Re-scoring the stress test with the clipping-corrected prediction against the (unclipped) measured RMSE:

| | raw (unclipped) prediction | clipping-corrected prediction |
|---|---|---|
| median \|rel err\|, all cells | **6.7%** | **13.1%** |
| haar_pure (μ=1) | 5.8% | **28.2%** |
| low_rank (μ≈0.45) | 5.2% | 5.2% |
| ghz_noisy (μ≈0.69) | 9.7% | 9.7% |

Because the measured RMSE is unclipped, clipping the *prediction* pulls the μ=1 pure-state cells ~30% below the measured value; interior-μ cells are unaffected (σ too small to reach a boundary). **Clipping does not explain the gap — applying it opens it.** (This is expected: our pipeline doesn't clip, so the unclipped prediction already matches.)

## Step 4 — what the residual actually is

The predicted RMSE = √(exact Hoeffding variance) is the **true** RMSE — it needs no distributional assumption (for an unbiased estimator, RMSE = √Var, and the Hoeffding formula gives Var exactly). So the gap must be finite-sample estimation error. Direct checks on three representative cells (20k trials):

* **ζ is converged** — σ(60k samples) vs σ(200k) differ by ≤ 0.4%. Under-convergence ruled out (and it would push the *opposite* sign).
* **The estimator is ~Gaussian at M ≥ 2000** — skewness ~0.15–0.34, kurtosis ~3.1–3.2 (Gaussian = 3). So the Gaussian approximation holds at the stress-test budgets; the finite-trial RMSE bias is only ~0.2%.
* **The measured RMSE converges to the predicted with more trials** — e.g. haar n4k2: 0.0848 (144 trials) → 0.0807 (20k), predicted 0.0804; ghz n5k3: 0.1055 → 0.1134, predicted 0.1130; low_rank n4k3 → 0.0549 = predicted. **Gap at 20k trials: ±0.3%.** The theory RMSE is essentially exact.
* **The gap magnitude = finite-trial noise.** The stress test estimates each cell's RMSE from only ~144 trials (non-det) / 36 (ghz). The relative SE of an RMSE from N trials, near-Gaussian, is √((κ−1)/(4N)) ≈ **5.9%** (N=144) / **11.8%** (N=36) — and the stress test's own bootstrap CI half-width is **6.5%** of the measured value. This *is* the observed 6.7% median gap.

The 68% CI covers the prediction in only 17/36 = 47% of cells (below the nominal 68%) because the bootstrap CI captures only the **measured**-side noise, not the **predicted**-side ζ MC error — so the prediction lands just outside a too-narrow CI more often. This is a limitation of the stress test's error bars, not a theory gap; the direct high-trial convergence (±0.3%) is the definitive check.

## Verdict

- **The pipeline does not clip.** Both predicted and measured RMSE are unclipped.
- **Clipping does not explain the gap** — and cannot, since our measured RMSE is unclipped; applying a clipping correction *worsens* the fit (6.7% → 13.1%).
- **There is no real ~7% systematic.** The median signed gap is only +3.4% — and only ~1.3–1.8σ once the cell correlations are accounted for (the 36 cells share measurement states across k and budget, giving ~9–18 effective independent units, not 36), so it is within noise (the finite-trial RMSE Jensen bias is ~0.2%). The ~6.7% |gap| is **finite-sample estimation noise on both sides**: the measured RMSE (~6% rel SE at ~144 trials/cell, matching the 6.5% bootstrap CI half-width) plus the predicted-side ζ Monte-Carlo error (from 60k samples × 4 component states). The theory's RMSE point prediction is **exact** — the measured RMSE converges to it (±0.3%) at high trials.
- The clipped-RMSE formula is nonetheless correct and is shipped for pipelines that *do* clip (e.g. real experiments); it is **not** applied to anrl's own predictions, which need no such correction.

**Integrated:** `anrl/theory/clipping.py` (`clipped_rmse`, `clipped_mse`) + package exports, with 14 passing tests in `tests/test_clipping.py`. Consolidated numbers in `results/clipping_correction.json`.
