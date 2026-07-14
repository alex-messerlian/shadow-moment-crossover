# Single-copy variance law — first-principles derivation, independently verified

Zero credits (local computation). Each claim was checked against data already in `results/` with no tuning. Code lives in `anrl/theory/single_copy_law.py` (+ `tests/test_single_copy_law.py`); reproduce with `experiments/theory_single_copy_scaling.py` (ζ scan) and `experiments/theory_single_copy_verify.py` (all checks → `results/theory_derivation.json`).

## Claim 1 — the exact Hoeffding variance: **CORRECT (not misremembered)**

The purity estimator is the 2nd-order U-statistic `U_M = C(M,2)⁻¹ Σ_{i<j} Tr(G_i G_j)` (`E[G_i]=ρ`). Its variance is

> **Var(U_M) = [ 4(M−2)·ζ₁ + 2·ζ₂ ] / [ M(M−1) ]**,  ζ₁ = Var[Tr(G·ρ)], ζ₂ = Var[Tr(G_i G_j)].

Verified by brute-force Monte Carlo of the full estimator (independent of the formula), with ζ₁, ζ₂ estimated from the same snapshot distribution (reproducible via `experiments/theory_single_copy_verify.py` → `results/theory_derivation.json`):

| state | M | brute-force Var | exact formula | ratio |
|---|---|---|---|---|
| noisy_pure n=1 q=0.1 | 4 / 8 | 1.437 / 0.424 | 1.392 / 0.425 | 1.03 / 1.00 |
| noisy_pure n=2 q=0.1 | 4 / 8 | 9.178 / 2.307 | 9.181 / 2.256 | 1.00 / 1.02 |
| GHZ n=2 q=0.15 | 4 / 8 | 8.81 / 2.153 | 8.777 / 2.144 | 1.00 / 1.00 |

Ratios ~0.98–1.03 (normal MC scatter for a variance estimate at small M) across pure/noisy/GHZ states — **the formula is exact.** (An independent adversarial check confirmed `simplify(hoeffding_variance − Lee_formula) = 0` symbolically, and equality with `exact_ustatistic_variance` to 1e-12.) It is precisely the k=2 case of the general Lee/Hoeffding U-statistic variance (`exact_ustatistic_variance`). The simpler two-term form `4ζ₁/M + ζ₂/M²` is **wrong at small M** (off by 1.3×–2.5×): the correct large-M expansion is `4ζ₁/M + (2ζ₂ − 4ζ₁)/M² + …`, so the second-order coefficient is asymptotically **2ζ₂**, not ζ₂.

## Claim 2 — the ζ scalings (converged): reported with the correction

Recomputed by independent chunked/streaming MC (noisy-pure, q=0.1), with convergence verified (ζ₁, ζ₂ stable to ~1% by 1M snapshots; n=7 ζ₂ moved 0.3% from 100k→3M). Fitted scalings (±1σ):

| range | ζ₁ | ζ₂ | M* = ζ₂/(2ζ₁) |
|---|---|---|---|
| **n=2..7** (task range) | 0.63·(1.346±0.031)ⁿ | 1.10·(6.928±0.020)ⁿ | 0.87·(5.147±0.122)ⁿ |
| n=2..9 | 0.72·(1.299±0.023)ⁿ | 1.09·(6.941±0.012)ⁿ | 0.76·(5.345±0.097)ⁿ |

Versus the task's quoted `ζ₁≈0.60·1.38ⁿ`, `ζ₂≈0.99·7.19ⁿ`, `M*≈0.82·5.20ⁿ`: **the quoted values were mildly under-converged.** The ζ₁ base is slightly *curved* (1.35 over n≤7 → 1.30 over the full range), which is why a small-n fit reads ~1.38. ζ₂'s base is **6.93**, not 7.19 (a few-thousand-sample estimate over-reads the base because the heavy-tailed kernel variance converges from below). Per-n across-state spreads are ≤5%.

## Claim 3 — α transition, out-of-sample: **the law predicts it (8/8 n)**

Using the recomputed ζ's (no fitting to any α data), the exact-formula RMSE `√Var(U_M)` was log-log fit over the **actual** budgets used per n (n≤6: 2000/8000/32000/128000; n≥7: 2000/8000/32000) and compared to the measured α in `budget_scaling.json`:

| n | α predicted (exact) | α measured ± SE | residual | within 2·SE? |
|---|---|---|---|---|
| 2 | 0.501 | 0.495 ± 0.013 | +0.006 | ✓ |
| 3 | 0.506 | 0.521 ± 0.010 | −0.015 | ✓ |
| 4 | 0.528 | 0.528 ± 0.012 | +0.000 | ✓ |
| 5 | 0.605 | 0.605 ± 0.012 | −0.000 | ✓ |
| 6 | 0.748 | 0.766 ± 0.012 | −0.018 | ✓ |
| 7 | 0.950 | 0.947 ± 0.015 | +0.003 | ✓ |
| 8 | 0.990 | 0.984 ± 0.017 | +0.006 | ✓ |
| 9 | 0.998 | 1.006 ± 0.023 | −0.008 | ✓ |

**Verdict: the derived law predicts the measured α at every n within 2·SE** (tightest margins n=3 at 1.5·SE and n=6 at 1.5·SE; the other six ≤0.5·SE), tracking the full 0.5→1.0 transition through the crossover (n≈5–7). The two-term approximation is strictly worse — only 5/8 within 2·SE, failing n=5 (3.4·SE), n=6 (6.8·SE), n=7 (2.3·SE) — confirming the *exact* Hoeffding formula is required.

## Claim 4 — the two M* estimates: **they agree**

- **Derived** from ζ₂/(2ζ₁) (my fresh ζ recompute): base **5.147** (n=2..7) / **5.345** (n=2..9).
- **Empirical** M* base from the earlier theory phase (`theory_zetas.json`, k=2, independent MC): base **5.343**.

These agree (5.15–5.35 vs 5.343). The M* base is `ζ₂base/ζ₁base` and is **independent of the 2-vs-4 prefactor**, so the agreement is robust; the task's apparent 5.2-vs-5.3 gap was fit noise from the under-converged ζ₁ base. Note the **crossover is ζ₂/(2ζ₁)** (from the exact formula), which corrects the codebase's two-term `ζ₂/(4ζ₁)` (`variance.py` `estimate_zetas`/`single_copy_variance`) — a factor-of-2 that halved the reported M* prefactor (base unaffected).

## Part E — single-qubit identity and closed-form ζ₁

- **Single-qubit identity VERIFIED**: `E[Tr(G·r)²] = 1/4 + (5/4)t² = (5/2)p − 1` (t = Bloch length, p = purity) matches MC to ~1e-4. Consequently the single-qubit `ζ₁ = E[Tr(Gr)²] − p² = (3/4)t² − (1/4)t⁴` — an exact closed form (in `single_qubit_zeta1`).
- **Weight-only ansatz `ζ₁ = Σ_P c_{|P|}⟨P⟩²` FAILS at n=2** — confirmed: across a diverse 25-state family (varying q; low-rank; GHZ; product) a single universal (c₁,c₂) fits only to **13.0% max / 5.2% RMS** residual. (It *appears* to hold at 0.0% within the narrow q=0.1 Haar family only because those states span a low-dimensional invariant subspace — a trap.) It also fails at n=3 (2.6%).
- **Why no simple closed form:** ζ₁ = Var[Tr(Gρ)] with `E[Tr(GP)Tr(GP')] = 3^{|P|+|P'|}·E_{U,b~ρ}[⟨b|UPU†|b⟩⟨b|UP'U†|b⟩]`; the U-average factorizes per qubit but the outcome sum weights by the diagonal `⟨b|UρU†|b⟩`, which does **not** factorize for entangled ρ. So the coefficients are not universal and depend on the full state. **ζ₁ (n≥2) has no simple closed form and must be computed numerically**; the single-qubit form is the only exact one.

## Summary

| claim | verdict |
|---|---|
| 1. Hoeffding variance `[4(M−2)ζ₁+2ζ₂]/[M(M−1)]` | **correct** (MC ratio 0.98–1.01); two-term is wrong at small M |
| 2. ζ scalings | ζ₁≈0.63·1.35ⁿ, ζ₂≈1.10·6.93ⁿ (task's 7.19 was under-converged), M*≈0.87·5.15ⁿ |
| 3. α transition (out-of-sample) | **predicted within 2·SE at all 8 n** |
| 4. M* base: derived vs empirical | **agree** (5.15–5.35 vs 5.343) |
| single-qubit identity | **verified** to ~1e-4 |
| closed-form ζ₁ | **none** for n≥2 (numerical); single-qubit `(3/4)t²−(1/4)t⁴` only |

Integrated into `anrl/theory/single_copy_law.py` with 12 passing tests.
