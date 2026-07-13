# Locked grid predictions — Cepheus (measured parameters, ZERO credits)

Parameters: measured correlated readout ($0 P(1|0) 1.6%->16.9% with neighbor excitation, others per-qubit measured); CZ error 0.9% median, band 0.5%-1.5%; p1=0.001.

## Step 1 — gate: reproduce the measured Bell purity 0.7184

* Correlated readout + spec CZ: **0.7163** (residual +0.0021).
* Independent readout (old model): 0.7494 (residual -0.0310).
* The correlated-readout model **closes the ~0.03 residual to +0.002** — within the Bell measurement's own CI [0.699, 0.738]. The model reproduces the data; the gate is passed.

## Step 2 — locked grid (measured purity band from CZ 0.5%-1.5%)

| n | state | CZ(dev) | SWAP purity lo/mid/hi | SWAP SE@10k | shadow purity | shadow SE@10k | raw-SE ratio | gate pen | readout pen |
|---|---|---|---|---|---|---|---|---|---|
| 2 | ghz | 4 | 0.726/0.716/0.702 | 0.0070 | 0.721 | 0.0170 | 2.43x | 0.042 | 0.242 |
| 2 | haar | 4 | 0.827/0.817/0.801 | 0.0058 | 0.735 | 0.0154 | 2.67x | 0.038 | 0.145 |
| 3 | ghz | 7 | 0.592/0.578/0.557 | 0.0082 | 0.550 | 0.0206 | 2.53x | 0.074 | 0.348 |
| 3 | haar | 15(+4 route) | 0.649/0.617/0.573 | 0.0079 | 0.508 | 0.0210 | 2.67x | 0.139 | 0.244 |
| 4 | ghz | 10 | 0.497/0.480/0.456 | 0.0088 | 0.419 | 0.0215 | 2.45x | 0.105 | 0.414 |
| 4 | haar | 46(+20 route) | 0.447/0.379/0.299 | 0.0093 | 0.257 | 0.0213 | 2.3x | 0.387 | 0.234 |

### Route comparison — two metrics (both reported)

**(a) Which route wins — the paper's copy-fair RMSE (the 'predicted crossover').** Using the saved theory components (results/theory_zetas.json, q=0.1, a common copy budget of 20,000), the SINGLE-COPY route wins at every n we test — consistent with the expectation and the paper's theory. The sustained crossover is **n* = 8**, so n=2,3,4 are below it. The single-copy advantage NARROWS with n as the theory says:

| n | single RMSE | collective RMSE | winner | gap (coll/single) |
|---|---|---|---|---|
| 2 | 0.0143 | 0.0610 | single | 4.27x |
| 3 | 0.0178 | 0.0712 | single | 4.0x |
| 4 | 0.0218 | 0.0762 | single | 3.5x |
| 5 | 0.0259 | 0.0788 | single | 3.04x |
| ... | | | | |
| 8 | (crossover) | | collective | |

The gap ratio shrinks (n=2 -> 4: 4.27 -> 3.5), i.e. single-copy's lead erodes toward the crossover at n*=8.

**(b) Raw statistical error at 10k shots (the hardware cost metric).** At EQUAL shots the collective SWAP SE (~0.006-0.009) is ~2.3-2.7x smaller than the shadow SE (~0.015-0.022). This does NOT contradict (a): the raw equal-shots SE ignores both the copy cost (a SWAP shot consumes 2 copies vs 1 for a shadow) AND the noise bias (deviation from the true purity 1.0), both of which the copy-fair RMSE folds in. The collective RMSE is bias-dominated (the depolarizing bias), which is why single-copy wins the RMSE race below n*. The two metrics answer different questions: raw precision per circuit execution (collective) vs accuracy in estimating the true purity at a fixed copy budget (single-copy, below the crossover).

### Readout vs gate penalty (the hardware finding)

Readout penalty scales with the 2n measured qubits (GHZ SWAP: 0.24 -> 0.35 -> 0.41 for n=2,3,4); gate penalty scales with the CZ count (Haar n=4: 46 CZ incl. 20 routing SWAPs -> gate penalty 0.39, the largest single contribution in the grid). GHZ maps with zero routing (CZ = 3n-2); Haar routes heavily at n>=3.

## Analytic bias law vs gate-level simulation

Single global-depolarizing g calibrated at n=2 GHZ (g_ref=0.378). Discrepancy = bias law - sim:

| n | state | sim purity | bias law (single g) | discrepancy | effective g (this cell) |
|---|---|---|---|---|---|
| 2 | ghz | 0.716 | 0.716 | +0.000 | 0.378 |
| 2 | haar | 0.817 | 0.716 | -0.100 | 0.244 |
| 3 | ghz | 0.578 | 0.669 | +0.091 | 0.483 |
| 3 | haar | 0.617 | 0.669 | +0.052 | 0.438 |
| 4 | ghz | 0.480 | 0.645 | +0.165 | 0.554 |
| 4 | haar | 0.379 | 0.645 | +0.267 | 0.663 |

The analytic law agrees at n=2 GHZ by calibration (disc ~0) but **DIVERGES** elsewhere: it already misses Haar at n=2 (state-dependent readout), and the discrepancy grows with n (up to ~0.17 at n=4 GHZ). The effective g is NOT constant across cells (it ranges widely), so a single global-depolarizing g does not describe the device — readout scales with 2n and is state-dependent + correlated, which the depolarizing law cannot capture. Unlike the old single-point 0.0011 agreement, the law does NOT track the gate-level simulation across the grid.

## Step 3 — budget

* 12 cells x 10,000 shots = 120,000 shots = **31.2 credits** on Rigetti Public Compute (26/100k). Available: 51 (11 spark + 40 full). Within budget.

## Caveats (stated for honesty)

* **CZ error is bounded, not measured** (identity echoes were resynthesized away by the compiler). The 0.5%-1.5% band carries that uncertainty; the mid (0.9% median) is the point estimate.
* **Readout-vs-CZ identifiability**: the Bell number alone has a mild degeneracy (independent readout at CZ=1.5% also lands in the Bell CI). The correlated model is the physically-justified correction because the $0 correlation was measured independently and it closes the residual at the median CZ; the Bell closure is robust to the linear-interpolation form (the Bell true-outcome distribution puts weight on neighbor-excitation w in {0,2} — the two measured endpoints — so it does not rely on assumed linearity).
* **n=3,4 readout is partly assumed**: only {0,1,9,10} have measured readout; the extra ladder qubits take the mean measured rates with no correlation, and the $0 correlation model is extrapolated to w>=3. These are the main untested assumptions in the larger cells.

ZERO credits spent — local simulation only. No grid predictions were locked to an assumed CZ split; the CZ uncertainty is carried as an explicit band.
