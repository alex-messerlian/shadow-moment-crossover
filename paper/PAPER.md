# Sample-complexity transition in shadow-based estimation of quantum state moments and the crossover to collective measurement

**Alexander Messerlian**

 *[affiliation]*

*Mentor: [Ziwei surname], [affiliation]*

---

## Abstract

Nonlinear functionals of a quantum state, such as the purity $\mathrm{Tr}(\rho^2)$ and the higher moments $\mathrm{Tr}(\rho^k)$, can be estimated from single-copy randomized measurements at a cost that grows exponentially with system size, or from collective measurements on $k$ copies at $O(1)$ variance but at the price of entangling gates and the noise they carry. Deciding which route is cheaper on a given device requires a quantitative account of both costs, and the single-copy side of that account has been available only as worst-case bounds.

We derive the exact, state-dependent variance of the $k$-th moment U-statistic estimator under local-Pauli classical shadows via the Hoeffding decomposition, and verify it against brute-force enumeration for $k = 2, 3, 4$. The law exposes a phenomenon that fixed-exponent bounds cannot see: the sample-complexity exponent $\alpha$, defined by $\mathrm{RMSE} \propto M^{-\alpha}$ in the measurement budget $M$, is not the constant $1/2$. It migrates continuously to $1$ as $M$ falls below a threshold $M^*$ that grows exponentially in the number of qubits, with $M^* = \zeta_2 / (2\zeta_1) \approx 5.3^n$ for purity. Because $M^*$ diverges exponentially, any fixed budget eventually sits in the $\alpha = 1$ regime, and the estimator's scaling behavior changes character as the system grows. Predicted and measured $\alpha$ agree at every point we tested, with no fitted parameters.

For the collective route we give two exact, parameter-free bias laws, distinguished by the geometry of the noise channel rather than by its rate. Under global depolarizing noise at rate $g$, the measured value is $(1-g)\mathrm{Tr}(\rho^k) + g\,2^{n(1-k)}$, linear in $g$ with no compounding across qubits. Under any per-qubit channel $\mathcal{E}$, the $k$-copy test returns exactly $\mathrm{Tr}(\sigma^k)$ with $\sigma = \mathcal{E}^{\otimes n}(\rho)$: the noise does not corrupt the measurement, it relabels which state is measured. Both are exact to machine precision across state ensembles and out to three times the noise range in which they were derived. The collective error is a bias floor and is therefore independent of the shot budget.

Combining the two gives a parameter-free crossover law. It predicts the system size at which collective measurement overtakes single-copy for 99% of measured crossovers within one qubit across 83 cells and four state ensembles, three of which the theory never saw. We then test the law on a 108-qubit superconducting processor with predictions registered before submission. The prediction fails, and we report the failure and its diagnosis: the entangling overhead the field has been cautious about is nearly free on this device, readout error is roughly three times the published value and strongly correlated, and cross-session drift on byte-identical circuits exceeds every modeled error source by a factor of twenty. Seven candidate mechanisms are eliminated with quantitative bounds. Static device characterization cannot predict collective-measurement performance on this hardware.

---

## 1. Introduction

At ten qubits, the standard single-copy method for estimating the purity of a quantum state returns an error fifteen times larger than the quantity it is estimating. The estimate is not noisy. It is not an estimate.

That failure matters because the quantity is not exotic. The moments $\mathrm{Tr}(\rho^k)$ of an unknown state are the entry point to a large fraction of what one wants to know about it. The second moment is the purity. Logarithms of the moments give the R\'enyi entropies. Moments of the partial transpose furnish PPT entanglement criteria and bounds on negativity. Purity appears directly in error-mitigation protocols such as virtual distillation. Any experiment that wants to characterize the state its device actually prepared, rather than the state it intended to prepare, needs these numbers.

Two routes exist to obtain them. The single-copy route measures one copy of the state at a time in randomized local bases, builds classical shadows [1], and forms an unbiased U-statistic from the snapshots. It uses no entangling gates and is therefore hardware-friendly. The collective route holds $k$ copies simultaneously and measures a joint observable, the cyclic-permutation operator, whose expectation is exactly $\mathrm{Tr}(\rho^k)$. It has $O(1)$ variance but requires entangling operations across copies.

The asymptotic separation between the two is settled. Single-copy strategies require exponentially more samples than collective ones for a family of learning tasks [2, 3], and the lower bound for purity estimation without quantum memory is $\Omega(\max\{1/\varepsilon^2, 2^{n/2}/\varepsilon\})$ [4]. What is not settled is the practical question that an experimentalist actually faces: on this device, at this noise level, at this system size, which route costs less? The collective route buys variance reduction and pays in noise-induced bias. Whether that trade is worth making depends on numbers, and the numbers have not been computed.

The gap is visible in recent work. A May 2026 study of quantum machine learning advantage on tens of noisy qubits [5] declined to give its measure-first protocol multi-copy access, citing both a theoretical result specific to their learning task—even multiple noiseless copies do not let a measure-first protocol solve it efficiently—and the resource demands of multi-copy entangled measurements, which increase qubit count, two-qubit depth, and routing overhead. Their learning task is not moment estimation, so their open questions are not ours; but their caution about the resource cost of multi-copy measurement is a premise we can now test quantitatively. The caution is reasonable. It has never been measured.

The reason the numbers have not been computed is that the single-copy side of the ledger has been available only as worst-case bounds. The shadow-norm analysis of second-order functionals splits the variance into a kernel term bounded by $4^{|AB|}$ and linear terms bounded by $2^{|AB|}$ [7, 8]. Bounds of this kind are state-independent by construction, and they express sample complexity as a fixed power of the budget. That framing conceals a structural feature of the estimator, and the feature turns out not to be a detail.

**Thesis.** The sample-complexity exponent for shadow-based moment estimation is not a constant. It is a function of system size and measurement budget, changing character at a threshold that grows exponentially in the number of qubits, and the exact variance law that produces it, paired with two exact bias laws for the collective route, predicts without any free parameter the system size at which collective measurement becomes the cheaper option.

**Roadmap.** Section 2 fixes the estimators and shows that two seemingly innocuous choices, the tuple-sampling scheme and the state ensemble, each invert the conclusion of the benchmark if made carelessly. Section 3 derives the exact variance law from the Hoeffding decomposition, gives the exact single-qubit projection variance, and establishes the $\alpha$ transition together with its out-of-sample validation. Section 4 derives two exact collective bias laws and shows that the relevant distinction is the geometry of the noise channel, not its rate. Section 5 combines them into a parameter-free crossover law and validates it against 83 measured crossovers across four state ensembles. Section 6 reports a pre-registered test on a 108-qubit superconducting processor: the prediction fails, and the diagnosis, which eliminates seven candidate mechanisms with quantitative bounds, is the section's content. Section 7 positions the work against the recent literature, Section 8 states what the work does not establish, and Section 9 concludes.

---

## 2. Setting and estimators

### 2.1 Local-Pauli classical shadows

Let $\rho$ be an unknown $n$-qubit state. A single snapshot is produced by drawing an independent Haar-random single-qubit unitary $U_q$ for each qubit, applying $\bigotimes_q U_q$, measuring in the computational basis to obtain a bitstring $b$, and forming

$$G = \bigotimes_{q=1}^{n} \left( 3\, U_q^\dagger \lvert b_q \rangle \langle b_q \rvert U_q - \mathbb{I} \right).$$

The snapshot is an unbiased estimator of the state, $\mathbb{E}[G] = \rho$. A budget of $M$ snapshots consumes $M$ copies of $\rho$.

### 2.2 The estimator must be the exact U-statistic

The $k$-th moment is estimated by the U-statistic over distinct $k$-tuples of snapshots,

$$U_M = \binom{M}{k}^{-1} \sum_{i_1 < \cdots < i_k} h(G_{i_1}, \ldots, G_{i_k}), \qquad h(g_1, \ldots, g_k) = \frac{1}{k!} \sum_{\pi \in S_k} \mathrm{Re}\,\mathrm{Tr}\!\left( g_{\pi(1)} \cdots g_{\pi(k)} \right),$$

which is exactly unbiased: $\mathbb{E}[U_M] = \mathrm{Tr}(\rho^k)$.

The tuples must be formed exhaustively, not sampled. Forming tuples from already-collected snapshots is classical post-processing and consumes no additional copies of the state, so subsampling them saves nothing physical while inflating the variance. In our sweeps a subsampled estimator inflated the single-copy RMSE by a factor of about five at $n = 2$, growing to more than fiftyfold at $n = 4$, relative to the exact one. That factor is larger than the effect the benchmark is trying to measure, and it is enough to reverse the benchmark's conclusion. Any comparison between single-copy and collective measurement must use the exact estimator on the single-copy side, or it is measuring an implementation choice rather than physics.

For $k = 2$ the full U-statistic reduces to an $O(M^2)$ closed form via power sums. For $k = 4$ it does not, because the snapshots do not commute. We give an exact construction in Appendix A by M\"obius inversion over the 15 set partitions of the four cyclic slots, with the alternating (ABAB) partition, which has no power-sum representation, evaluated by a tensor contraction. The construction matches brute-force enumeration to $10^{-14}$.

### 2.3 The collective route

Let $C_k$ be the cyclic permutation operator on $k$ registers of dimension $d = 2^n$. It satisfies

$$\mathrm{Tr}\!\left( C_k\, \rho^{\otimes k} \right) = \mathrm{Tr}(\rho^k), \qquad \mathrm{Tr}(C_k) = d,$$

the second identity because the cyclic permutation on $k$ registers has a single cycle. That second identity is not decoration; it is what makes the depolarizing bias law of Section 4.1 come out linear in the noise rate.

We use the destructive SWAP test rather than the ancilla-based controlled-SWAP construction, because the latter decomposes into many two-qubit gates and would be dominated by gate noise on current hardware. Two copies of the state are prepared on $2n$ qubits. For each pair $(i, i+n)$ a CNOT is applied with control $i$ and target $i+n$, followed by a Hadamard on qubit $i$, and all $2n$ qubits are measured. The purity is recovered from the bitstrings by the parity rule

$$\mathrm{Tr}(\rho^2) = \sum_{\text{bitstrings}} (-1)^{\,\#\{\,i \,:\, a_i = b_i = 1\,\}} \, P(\text{bitstring}),$$

where $a_i, b_i$ are the outcomes on the two copies of qubit $i$. The circuit uses exactly $n$ two-qubit gates for an $n$-qubit state, requires no ancilla, and has depth two after state preparation. We verified the sign rule against exact simulation to $10^{-16}$ (Appendix B). The gate count matters for Section 6: it is the reason the entangling overhead turns out to be affordable on real hardware.

### 2.4 Copy-fair accounting

A single-copy protocol at budget $M$ consumes $M$ copies of $\rho$. A $k$-copy collective test at budget $M$ consumes $M/k$ measurements, each using $k$ copies. All comparisons in this paper hold the total number of state copies fixed.

### 2.5 The state ensemble is not a free choice

The variance of shadow-based purity estimation scales with the purity of the state being measured, a fact we derive exactly in Section 3.3. The consequence is a trap. A random Ginibre state under heavy depolarizing noise has purity approaching $2^{-n}$, which tends to zero with system size. Estimating a quantity that is approximately zero is easy for any estimator, and single-copy estimation of such states shows no exponential blow-up at all. That absence is an artifact of the ensemble, not a property of the task.

Realistic NISQ states are noisy-pure: a prepared pure state subject to modest depolarizing noise, with purity of order one. We therefore use

$$\rho = (1-q)\lvert \psi \rangle \langle \psi \rvert + q\,\mathbb{I}/2^n, \qquad \lvert \psi \rangle \sim \text{Haar}, \qquad q \in [0.05,\, 0.3],$$

for which the purity stays near $0.8$ at every size tested. Choosing the wrong ensemble inverts the conclusion of the benchmark. We report this because we made the error ourselves, and correcting it reversed a headline result.

---

## 3. The single-copy variance law

### 3.1 The Hoeffding decomposition gives the variance exactly

For a symmetric kernel $h$ of order $k$, define the $c$-th order projection and its variance,

$$h_c(g_1, \ldots, g_c) = \mathbb{E}_{G_{c+1}, \ldots, G_k}\!\left[ h(g_1, \ldots, g_c, G_{c+1}, \ldots, G_k) \right], \qquad \zeta_c = \mathrm{Var}\!\left[ h_c(G_1, \ldots, G_c) \right].$$

The variance of the U-statistic is then exactly

$$\boxed{\;\mathrm{Var}(U_M) = \binom{M}{k}^{-1} \sum_{c=1}^{k} \binom{k}{c} \binom{M-k}{\,k-c\,} \zeta_c \;}$$

For $k = 2$ this reduces to

$$\mathrm{Var}(U_M) = \frac{4(M-2)\,\zeta_1 + 2\,\zeta_2}{M(M-1)}, \qquad \zeta_1 = \mathrm{Var}\!\left[\mathrm{Tr}(G\rho)\right], \quad \zeta_2 = \mathrm{Var}\!\left[\mathrm{Tr}(G_i G_j)\right].$$

We verified the general formula against brute-force Monte Carlo of the exact estimator at $k = 3$ and $k = 4$, on noisy-pure, GHZ, and low-rank states, with all ratios within 3%, and confirmed that it reduces to the $k=2$ form at a ratio of $1.000000$. The law is exact, not asymptotic, and it holds at every budget.

One warning is worth making explicit, because the error is easy to make and it is silent. For $k = 2$ the second-order projection *is* the kernel, so $\zeta_2$ is the kernel variance. For $k \geq 3$ this is false: $\zeta_2$ is the two-argument projection, and the kernel variance is $\zeta_k$. Substituting one for the other is wrong by roughly a factor of seven and corrupts any variance estimate at higher moments while appearing to work at $k=2$. We made this error, and it produced an approximation that failed at $k \geq 3$ for reasons that looked like physics and were not.

### 3.2 Two terms compete, and their ratio is a budget threshold

The exact variance contains two competing terms with different budget dependence. At large $M$ the linear term dominates,

$$\mathrm{Var}(U_M) \approx \frac{4\zeta_1}{M} \quad \Longrightarrow \quad \mathrm{RMSE} \propto M^{-1/2}, \qquad \alpha = \tfrac{1}{2},$$

which is the familiar statistical scaling that a fixed-exponent bound predicts. At small $M$ the higher-order term dominates,

$$\mathrm{Var}(U_M) \approx \frac{2\zeta_2}{M^2} \quad \Longrightarrow \quad \mathrm{RMSE} \propto M^{-1}, \qquad \alpha = 1.$$

The two terms are equal at

$$\boxed{\;M^* = \frac{\zeta_2}{2\zeta_1}\;}$$

This threshold is the object the rest of the section is about. Which side of $M^*$ a given experiment sits on determines the character of its scaling, and the projection variances that set $M^*$ are state-dependent. To see what that dependence is, we compute them exactly for one qubit.

### 3.3 The single-qubit second moment, exactly

For a single qubit in state $r$ with Bloch length $t$ and purity $p = (1+t^2)/2$,

$$\boxed{\;\mathbb{E}\!\left[ \mathrm{Tr}(G r)^2 \right] = \tfrac{1}{4} + \tfrac{5}{4} t^2 = \tfrac{5}{2} p - 1\;}$$

and the single-qubit first projection variance follows as

$$\zeta_1 = \tfrac{3}{4} t^2 - \tfrac{1}{4} t^4.$$

We verified this numerically across the full Bloch range. The identity is exact, and two consequences follow from it.

The second moment grows by a factor of six from a maximally mixed qubit, where it equals $1/4$, to a pure qubit, where it equals $3/2$. Compounded over $n$ qubits, that factor is the origin of the exponential cost, and it is state-dependent rather than universal. This is the analytic reason why the ensemble choice of Section 2.5 is not cosmetic: a low-purity ensemble is not a harder instance of the same problem, it is a different and much easier problem. Independent numerical work has observed that shadow-based purity estimation scales exponentially in $n$ and linearly in the purity of the state [9]; the identity above is the single-qubit mechanism behind that observation.

The identity also passes the boundary check. At $t = 0$ it gives $\zeta_1 = 0$, which is correct: for a maximally mixed qubit, $\mathrm{Tr}(Gr) = 1/2$ deterministically, with no variance to speak of.

### 3.4 There is no weight-only closed form at $n \geq 2$

The single-qubit result invites a generalization. If the second moment carries a factor of $3$ per qubit in the support of a Pauli string, then $\zeta_1$ ought to decompose over Pauli strings with a coefficient depending only on Pauli weight,

$$\zeta_1 \stackrel{?}{=} \sum_P c_{|P|} \langle P \rangle^2 .$$

This is false. Tested against a diverse family of 19 states at $n = 2$, varying depolarizing rate, rank, and structure, the ansatz leaves a residual of up to 13%. The reason is structural: the local-shadow second moment $\mathbb{E}[\mathrm{Tr}(GP)\,\mathrm{Tr}(GP')]$ carries a factor $3^{|P| + |P'|}$ multiplied by a term that does not factorize for entangled states, so weight alone does not determine the coefficient.

The failure is worth reporting because the ansatz *appears* to hold. Tested only within a narrow single-parameter family, Haar-pure states at a fixed depolarizing rate of $q = 0.1$, it fits to 0.1%, because that family spans a low-dimensional invariant subspace. A closed form validated on one ensemble is not a closed form. For $n \geq 2$ the projection variances must be computed numerically, and the variance law is exact while its inputs are not analytic.

### 3.5 The $\alpha$ transition

Computing the projection variances numerically for noisy-pure states gives clean exponential scalings. Over $n = 2$ to $7$,

$$\zeta_1 \approx 0.63 \cdot (1.35)^n, \qquad \zeta_2 \approx 1.10 \cdot (6.93)^n, \qquad M^* \approx 0.87 \cdot (5.15)^n,$$

and over the wider range $n = 2$ to $9$, $M^* \approx 0.76 \cdot (5.345)^n$. The base of $\zeta_1$ is mildly curved, reading approximately $1.35$ at small $n$ and $1.30$ over the full range, which accounts for the difference between the two fits.

The kernel variance grows roughly five times faster per qubit than the linear variance. Their ratio, the threshold $M^*$, therefore diverges exponentially. This is the entire mechanism, and its consequence is immediate: at any fixed budget, increasing $n$ eventually pushes $M$ below $M^*$, the higher-order term takes over, and the effective exponent migrates from $1/2$ toward $1$.

Defining $\alpha$ as the local slope $-\,\mathrm{d}\log \mathrm{RMSE} / \mathrm{d}\log M$ over the budget range actually used in the experiment:

| $n$ | $M^*(n)$ | predicted $\alpha$ | measured $\alpha$ |
|---|---|---|---|
| 2 | ~22 | 0.501 | $0.495 \pm 0.013$ |
| 4 | ~604 | 0.528 | $0.528 \pm 0.012$ |
| 9 | $\approx 2.3 \times 10^{6}$ | 0.998 | $1.006 \pm 0.023$ |

The predictions are computed from projection variances estimated from shadow snapshots, fed into the exact variance formula. Nothing is fitted to the $\alpha$ data. Across every system size and moment order for which we have measurements, the derived law matches: 8 of 8 within two standard errors at $k = 2$, 6 of 7 at $k = 3$, where the single miss sits on the two-sigma boundary and is sensitive to the random seed, and 5 of 5 at $k = 4$.

The exact combinatorial structure is load-bearing rather than decorative. A two-term approximation of the form $\mathrm{Var} \approx 4\zeta_1/M + \zeta_2/M^2$, which is the natural shortcut, manages only 5 of 8 at $k = 2$ and fails at $n = 6$ by nearly seven standard errors.

The threshold base is also stable across independent Monte Carlo samplings. Derived here from $\zeta_2/(2\zeta_1)$ on a fresh estimate of the projection variances, it is 5.15 to 5.35; an earlier, independently sampled run of the same $\zeta_2/(2\zeta_1)$ construction gave 5.343. This agreement is a reproducibility check on the projection-variance estimate, not corroboration by a methodologically independent route.

### 3.6 Why a changing exponent matters

A fixed-exponent bound of the form $\mathrm{RMSE} \leq C_n / \sqrt{M}$ is not wrong. It is a bound, and bounds hold. But it describes the wrong regime for the systems people are actually running. At nine qubits, with the budgets accessible on current hardware, the estimator sits in the $\alpha = 1$ regime, where additional shots buy error reduction faster than $1/\sqrt{M}$ but from a variance that is already exponentially large. The improved rate does not rescue the protocol, because the exponential prefactor still dominates. What the transition establishes is that the shape of the sample-complexity curve is not the shape a fixed-exponent analysis assumes, and that any extrapolation of measured scaling to larger systems which presumes $\alpha = 1/2$ will be wrong in a predictable direction.

Having characterized the single-copy cost exactly, we turn to the collective route, where the error has a different character entirely.

---

## 4. The collective route: two exact bias laws

The collective route's error under noise is a bias, not a variance. It is therefore independent of the shot budget: taking more shots does not reduce it, a prediction we test directly in Section 5.3 and which holds. The size of that bias is given exactly by one of two laws, and which law applies is determined by the geometry of the noise channel rather than by its rate.

### 4.1 Global depolarizing noise: linear in $g$, with no compounding

Let the $kn$-qubit collective register be subject to global depolarizing noise at rate $g$, so that $\rho^{\otimes k} \mapsto (1-g)\rho^{\otimes k} + g\,\mathbb{I}/d^k$ with $d = 2^n$. Then

$$\mathrm{Tr}\!\left( C_k \left[ (1-g)\rho^{\otimes k} + g\,\tfrac{\mathbb{I}}{d^k} \right] \right) = (1-g)\,\mathrm{Tr}(\rho^k) + g\,\frac{\mathrm{Tr}(C_k)}{d^k} = (1-g)\,\mathrm{Tr}(\rho^k) + g\, 2^{\,n(1-k)},$$

using $\mathrm{Tr}(C_k) = d$ from Section 2.3. Hence

$$\boxed{\;\text{bias} = g \left| \mathrm{Tr}(\rho^k) - 2^{\,n(1-k)} \right| \;}$$

The bias is linear in $g$ and does not compound across qubits. That result contradicts the natural intuition, which is that noise acting on $kn$ qubits should accumulate as $1 - (1-g)^{kn}$. We initially assumed exactly that compounding form, and it overestimated the bias by a factor of five to fourteen. The error is structural rather than numerical: a per-qubit compounding law was applied to a channel that acts globally. The single-cycle property of $C_k$ is what collapses the compounding, and any treatment that ignores the channel's geometry will get this wrong by an exponential factor.

### 4.2 Per-qubit channels: the noise relabels the state

Let $\mathcal{E}$ be any single-qubit channel applied to every qubit of every copy. Applying the same channel to every qubit of every copy is identical to applying $\mathcal{E}^{\otimes n}$ to each copy independently, so

$$\mathcal{E}^{\otimes nk}\!\left( \rho^{\otimes k} \right) = \left( \mathcal{E}^{\otimes n}(\rho) \right)^{\otimes k} = \sigma^{\otimes k}, \qquad \sigma \equiv \mathcal{E}^{\otimes n}(\rho),$$

and the cyclic test of a product of identical states returns the $k$-th moment of that state:

$$\boxed{\;\text{measured} = \mathrm{Tr}(\sigma^k), \qquad \sigma = \mathcal{E}^{\otimes n}(\rho) \;}$$

so that the bias is $\left| \mathrm{Tr}(\sigma^k) - \mathrm{Tr}(\rho^k) \right|$.

The statement is stronger than a bias formula, and the strength is the point. The noise does not corrupt the measurement. It relabels which state is being measured. The collective test performs exactly as designed; it simply answers a question about the damaged state $\sigma$ rather than the intended state $\rho$.

This explains a puzzle in the data that a rate-based model cannot. No universal effective attenuation rate fits the measured biases, because the bias is not a rate. It is however much the channel happens to deform that particular state's spectrum, and two states of identical purity can be deformed by different amounts.

### 4.3 Verification

Both laws are exact identities rather than approximations, and we verified them accordingly: against explicit construction of $C_k$ and the noise-damaged $k$-copy state, at $n = 2, 3$ and $k = 2, 3, 4$, for global depolarizing, amplitude damping, and dephasing, agreeing to approximately $10^{-15}$. They continue to hold at three times the noise rates in which they were derived, and on state ensembles they had never been tested against, including Haar-pure, low-rank, and GHZ-noisy states. An exact identity is not expected to degrade under extrapolation, and it does not.

With the single-copy variance and the collective bias both known exactly, the crossover follows by setting one against the other.

---

## 5. The crossover

### 5.1 The law

The single-copy error grows exponentially in $n$ and falls with the budget. The collective error is a bias floor: bounded above by $\mathrm{Tr}(\rho^k)$, and independent of the budget. The crossover size $n^*$ is where the former rises above the latter,

$$\underbrace{\sqrt{\mathrm{Var}(U_M)}}_{\text{exponential in } n,\ \text{falls with } M} \;=\; \underbrace{\text{bias}(g, k, n)}_{\text{bounded, budget-independent}}$$

Every quantity on both sides comes from Sections 3 and 4. Nothing is fitted, and the law has no free parameters.

### 5.2 Validation on 83 cells and four ensembles

Across 83 cells spanning moment order $k \in \{2,3,4\}$, three noise models (depolarizing, amplitude damping, dephasing), noise rates from 0 to 0.3, system sizes $n = 2$ to $10$, four budget multipliers, and four state ensembles, the law predicts 99% of measured crossovers within one qubit and 88% exactly.

Three of the four ensembles, Haar-pure, low-rank, and GHZ-noisy, were never used in deriving the law. GHZ states in particular are maximally non-Haar-typical, and we included them specifically because an ensemble-tuned theory ought to fail on them. The theory's accuracy on GHZ is within a factor of about 1.7 of its accuracy elsewhere, which is the same order of magnitude rather than a breakdown. A law that transferred to structured, low-rank, and pure states without refitting is not a curve fit to the ensemble it was built on.

### 5.3 Three qualitative predictions, and the one we got wrong

Higher noise moves the crossover later. A larger bias floor takes longer for the exponentially growing single-copy error to climb over. Confirmed.

A larger budget moves the crossover later. The single-copy error falls with budget while the bias floor does not. Confirmed, and the confirmation is sharper than it sounds: the collective RMSE plateaus exactly at the predicted floor as the budget grows, while the single-copy RMSE keeps falling. That is a direct test of the bias-versus-variance distinction on which the whole crossover rests, and it holds.

Higher $k$ moves the crossover later. This one is counterintuitive, and it is the sharpest test of whether the mechanism is understood rather than merely fitted. A naive variance argument says that higher moments compound the single-copy variance faster and should therefore cross earlier. The data say the opposite, and the law explains why: $\mathrm{Tr}(\rho^k)$ shrinks with $k$, so the absolute bias floor that the single-copy error must climb over is lower, and single-copy therefore holds out to larger $n$. Purity ($k=2$) crosses at $n \approx 6$ to $7$ under realistic noise, while $k = 3$ and $k = 4$ hold out to $n \approx 8$. The signal-against-a-fixed-floor argument dominates the variance-compounding argument. We predicted the wrong direction before running this, and the law corrected us.

### 5.4 The exponential wall

The practical statement of the result is a single sequence of numbers. Using the exact copy-fair estimator on noisy-pure states at a fixed copy budget, the single-copy purity RMSE runs

$$0.043 \;\to\; 0.072 \;\to\; 0.270 \;\to\; 1.62 \;\to\; 11.98$$

at $n = 2, 4, 6, 8, 10$, growing by roughly a factor of 2.5 per qubit and accelerating. The true purity of these states is approximately 0.81.

At ten qubits the single-copy estimate carries an error fifteen times larger than the quantity it is estimating. The collective route over the same range stays bounded, because its error is a bias floor and the floor cannot exceed $\mathrm{Tr}(\rho^k)$.

This number speaks directly to the resource-cost concern behind the choice in [5] to keep its measure-first protocol single-copy—though that learning task is not moment estimation, and the open questions it poses are not ours. The entangling overhead that motivates such caution is real, and it is not the binding constraint. The binding constraint is that single-copy estimation of nonlinear functionals does not survive to the system sizes they are entering. Whether the collective route survives on a real device is a separate question, and it is the one we take up next.

---

## 6. Hardware: a pre-registered test on Rigetti Cepheus-1-108Q

This section reports a prediction that failed, and the diagnosis of why it failed. That is its content and its value.

### 6.1 Platform, and a disclosure about the execution path

All measurements were performed on Rigetti Cepheus-1-108Q, a 108-qubit superconducting processor built from twelve interconnected nine-qubit chiplets. Published median fidelities at the time of the experiments were 99.1% for the two-qubit CZ gate and 99.9% for single-qubit gates. Readout error is not published, a fact that turns out to matter (Section 6.3).

Access was obtained through the Open Quantum platform operated by Quantum Rings [10], which provides a unified API to QPUs from multiple vendors. We used the Public Tier throughout. Two properties of that tier are material to the interpretation of our results and we state them rather than bury them.

First, the Public Tier does not route jobs directly to the vendor. It is powered by the Quantum Compute subnet (SN48) on the Bittensor network, operated by qBitTensor Labs: circuits are routed to distributed operators who execute them on the target hardware, and validators perform spot checks to confirm that circuits were executed on the appropriate target QPU [10]. We therefore cannot independently verify the execution path of any individual job beyond the platform's own validation. This does not affect Sections 6.2 and 6.3, which are internally consistent characterizations, but it is a candidate contributor to the session-to-session variability reported in Section 6.4 that we are not able to rule out, and we list it as such in Section 8.

Second, use of the Public Tier carries two licence conditions: publications resulting from work on the tier are required to cite the platform paper [10], which we do, and circuits, results, and metadata are contributed in anonymized aggregated form to a common repository maintained by Open Quantum [10].

Jobs were submitted as OpenQASM 3 with physical-qubit addressing, which the platform required because it rejects non-contiguous virtual registers.

### 6.2 The entangling overhead is nearly free

The destructive SWAP test at $n = 2$ transpiles onto physical qubits $\{0, 1, 9, 10\}$ using four CZ gates and zero routing SWAPs. The GHZ ladders at $n = 3$ and $n = 4$ likewise map with zero routing, at $3n - 2$ CZ gates, and the ladders nest. Measured CZ error on these qubits is at or below the published median of 0.9%, and gate error accounts for roughly 0.042 of the purity deficit at $n = 2$ against a total deficit of 0.29.

The resource-demand concern cited in [5]—that collective measurement demands significant qubit count, two-qubit depth, and routing overhead—does not materialize for the destructive SWAP test on a square-lattice device. Four two-qubit gates and no routing is not an overhead problem, and the gates contribute about a seventh (roughly 14%) of the observed error. Whatever is degrading the collective measurement on this hardware, it is not the entangling overhead.

For contrast and for honesty about scope: Haar-random states at $n = 4$ require 46 CZ gates including 20 routing SWAPs on this topology. The favorable transpilation is a property of structured states on a matched topology, not of the SWAP test in general. We therefore restrict the hardware series to GHZ ladders and say so.

### 6.3 Readout is the wall, and it is not in the datasheet

Our first prediction, computed from the published gate fidelities and an assumed 2% readout error, was a measured Bell-state purity of 0.9412. The device returned

$$0.7184 \quad [0.6992,\, 0.7376] \;\; \text{(95\% bootstrap CI, 5000 shots)},$$

against a true purity of exactly 1. The prediction failed by more than twenty standard errors.

The circuit was not at fault. The four dominant measured outcomes are exactly the ideal support of the Bell SWAP test, with noise leaking approximately 25% of the weight into the remaining twelve outcomes. This is a correctly executed circuit on a noisier device than we modeled. Inverting the bias law of Section 4.1 gives an effective global depolarizing rate of $g = 0.375$, nearly five times what the published specifications imply.

A single measurement cannot separate gate error from readout error, because both enter the same number. We therefore measured them separately. Readout characterization, using computational basis states prepared with X gates only and no two-qubit gates, gave:

| qubit | $P(1 \mid 0)$ | $P(0 \mid 1)$ |
|---|---|---|
| $q_0$ | 9.3% | 8.9% |
| $q_1$ | 0.7% | 7.8% |
| $q_9$ | 6.7% | 9.8% |
| $q_{10}$ | 3.2% | 6.0% |

We write $q_N$ for the physical qubit addressed as `$N` in the platform's OpenQASM 3 syntax. The mean is approximately 6.5% per qubit, roughly three times the value we assumed, and it is asymmetric in the direction expected from $T_1$ decay during readout. Feeding the measured readout back into the model moves the predicted Bell purity from 0.94 to approximately 0.75, against the measured 0.7184. The device model was correct and the inputs were wrong, and the input that was wrong is precisely the one the vendor does not publish.

Readout error on this device is also correlated. Qubit $q_0$'s $P(1|0)$ rises from 1.6% when its neighbours are idle to 16.9% when they are excited. Including this measurement crosstalk closes the residual: the corrected model predicts 0.7163 against a measured 0.7184.

The correlation saturates rather than accumulating. Measured across excitation weights $w = 0, 2, 4, 6$, qubit $q_0$'s $P(1|0)$ runs $0.020,\, 0.169,\, 0.242,\, 0.224$: a steep rise from $w = 0$ to $w = 2$, then a plateau. A linear extrapolation from the first two points predicts 0.473 at $w = 6$, overestimating the measured value by 0.25. Had we extrapolated rather than measured, our $n = 4$ predictions would have been pessimistic by five points, and we would have recorded a failed prediction that we had manufactured ourselves.

### 6.4 The prediction fails, and drift is why

With every parameter on the GHZ ladder measured rather than assumed, we ran a bracketed same-session experiment: opening readout calibration, predictions locked and committed to version control from that calibration, SWAP measurements, closing readout calibration. Within-session drift measured approximately 1% per qubit, so the device was stable while the experiment ran.

| $n$ | measured | opening band | closing band |
|---|---|---|---|
| 2 | 0.686 [0.672, 0.700] | 0.718 -- 0.742 | 0.704 -- 0.728 |
| 3 | 0.378 [0.360, 0.397] | 0.597 -- 0.634 | 0.593 -- 0.631 |
| 4 | 0.420 [0.402, 0.439] | 0.506 -- 0.553 | 0.520 -- 0.568 |

All three cells fall below both bands rather than between them, so drift within the session does not account for the gap. The degradation is also non-monotonic: $n = 3$ is worse than $n = 4$, despite the $n = 4$ register containing the $n = 3$ register. Neither readout error, which scales with the $2n$ measured qubits, nor CZ count, which grows with $n$, can produce that ordering.

The cross-session picture explains it. Running the same byte-identical circuits, on the same physical qubits, in different sessions:

| register | session 1 | session 2 | session 3 |
|---|---|---|---|
| $n = 3$ | 0.317 | 0.378 | 0.587 |
| $n = 4$ | 0.431 | 0.420 | 0.382 |

Within-session drift is approximately 1%. Cross-session drift is approximately 0.2 in purity, twenty times larger. The $n = 3$ anomaly healed across sessions and the $n = 4$ anomaly persisted, after which the ordering between them flipped. The non-monotonicity that sent us hunting for a physical mechanism was a drift artifact, and identifying it as such required running the same circuit three times over several days, which is not standard practice.

### 6.5 The elimination ledger

Each candidate mechanism was eliminated with a quantitative argument, not by absence of evidence.

| mechanism | how it was ruled out |
|---|---|
| Within-session drift | Bracketed calibration measured approximately 1% per qubit, against a 0.2 discrepancy. |
| Single-copy readout error | Directly characterized. Feeding measured values into the model does not close the gap at $n \geq 3$. |
| Gate count | Cannot produce $n=3$ worse than $n=4$, since CZ count is monotone in $n$. |
| Cross-copy readout crosstalk | Ruled out by a physical bound. Joint readout on the full $2n$ register is factorizable (TVD 0.009 at $n=3$, 0.011 at $n=4$), and parity-pair correlations are negligible and no stronger than non-pair correlations. Holding the measured single-qubit flip rates fixed and pushing every copy-pair to its maximum physically realizable correlation, which is bounded because $P(\text{both flip}) \leq \min$ of the two marginals and the measured marginals are small, the entire achievable range of predicted purity is $[0.611, 0.669]$ at $n = 3$. The measurement is 0.378. Even maximal correlation leaves 0.23 unexplained. There is no room. |
| Coherent gate error | Randomized compiling. Pauli-twirled SWAP circuits, with a positive control confirming that the inserted gates physically execute (5.4:1 mass on the disjoint support), moved the purity by $-0.013$ against a 0.197 deficit, and twirl-to-twirl scatter was shot-noise limited with zero physical scatter. A masking bound places any hidden coherent contribution at no more than 14% of the deficit. The residual is incoherent. |
| A specific bad qubit pair ($\{3, 12\}$) | Localization test. An $n=3$ ladder that includes the suspect pair measured 0.58080, identical to the standard $n=3$ ladder's 0.58080, from genuinely distinct raw data. Adding the suspect qubits changes nothing. |
| A fixed register-geometry fault | The $n=4$ versus $n=3$ ordering flips sign across sessions on identical registers. The deficit is session-dependent, not geometric. |

### 6.6 What remains, and what we conclude

What is left is a large, incoherent, session-dependent fault that appears and disappears on the same physical qubits running byte-identical circuits. We do not identify it. Localizing it would require per-edge interleaved randomized benchmarking across many sessions, which is a research program rather than a section, and we do not claim to have done it.

The conclusion the data support is narrower than "we found a new error mechanism" and more useful than "the device is noisy." Static device characterization cannot predict collective-measurement performance on this hardware, because the device is not stationary at the relevant scale.

This is not a new observation about NISQ devices in general, and we do not claim it as one. Temporal and spatial instability of superconducting processors is documented, with month-to-month Hellinger distances between characterizations exceeding 0.2 [11, 12], and calibration-drift analyses identify gate error as the dominant cross-device mismatch and readout as secondary [13]. Our contribution is to demonstrate, with pre-registered predictions and a seven-mechanism elimination, that this instability is the binding constraint specifically for collective-measurement protocols, at magnitudes that dwarf every error source such protocols are conventionally modeled with. A protocol whose entire advantage rests on a bounded, characterizable bias floor cannot be deployed on a device whose bias floor moves by 0.2 between sessions.

### 6.7 A practical finding about cloud QPU economics

One further constraint emerged and it is worth stating because it affects reproducibility rather than physics. The single-copy shadow route requires many independent random measurement bases. On a per-circuit-priced cloud platform without circuit batching, each basis is a separately billable job. An unbiased shadow purity estimate at the precision needed for this comparison requires on the order of a few hundred independent random measurement bases, each a separately billable job -- one to two orders of magnitude more than the roughly 3 credits per cell the collective route used at the same shot count.

Single-copy classical shadows are therefore economically infeasible on per-circuit-priced cloud QPUs at the scale required to reproduce their own sample-complexity claims. This is the reason our hardware single-copy baseline at $n \geq 3$ is computed from independently measured device parameters rather than measured directly, with the $n = 2$ point measured directly as an anchor. We flag the substitution rather than eliding it.

---

## 7. Related work

**Classical shadows and variance bounds.** The classical-shadow framework [1] gives sample-complexity bounds via the shadow norm. For second-order nonlinear functionals, the variance splits into a kernel term bounded by $4^{|AB|}$ and linear terms bounded by $2^{|AB|}$ [7], with related analyses in [8]. These are worst-case, state-independent bounds. Our contribution is the exact, state-dependent variance, from which the exponent transition follows. A fixed-power bound cannot exhibit that transition, which is why it has not previously been reported.

**U-statistics and the Hoeffding decomposition for quantum moments.** Straeter, Tsesmelis and Kwek [14] construct unbiased U-statistic estimators for the partial-transpose moments $p_2$ and $p_3$ from randomized homodyne data in continuous-variable systems and derive their variance via a Hoeffding decomposition, obtaining sample-complexity bounds for a $p_3$-PPT entanglement criterion. This is the closest methodological neighbour to Section 3. The differences are the platform (continuous-variable homodyne rather than qubit Pauli shadows), the target (entanglement detection rather than moment estimation and the collective-measurement trade-off), and the result (bounds rather than the exact state-dependent variance, and consequently no exponent transition). We do not claim novelty for the Hoeffding technique itself.

**Crossovers in shadow sample complexity.** A crossover in shot cost between Pauli and Clifford shadow ensembles for multipartite entanglement witnesses has been reported [15], with Pauli favored for local witnesses (cost $\sim 4^n$) and Clifford for global ones (cost $\sim 2^{N-n}$). That is a crossover in the choice of measurement ensemble, not between single-copy and collective measurement, and its mechanism is unrelated to ours.

**Statistical-to-bias-floor transitions on hardware.** A March 2026 study of shadow tomography on an integrated photonic processor [16] reports that reconstruction error initially follows $O(M^{-1/2})$ in a variance-dominated regime and then saturates at a hardware-determined floor, a transition the authors term a Hardware Horizon. The structure of that argument, statistical error meeting an irreducible hardware bias floor, is adjacent to Section 5, and the two should be distinguished carefully. Their transition is between statistical scaling and a bias floor at fixed system size. Ours is a change in the statistical exponent itself as a function of system size, occurring before any hardware bias enters. Both effects are real and they are not the same effect.

**Sample-complexity lower bounds for purity.** Gong et al. [4] improve the lower bound for purity estimation without quantum memory to $\Omega(\max\{1/\varepsilon^2,\, 2^{n/2}/\varepsilon\})$. Our exact variance is consistent with this bound and supplies the state-dependent constant it leaves open.

**Collective measurement on noisy hardware.** The exponential single-copy versus collective separation is established in [2, 3]. A recent study of quantum machine learning on tens of noisy qubits [5] keeps its measure-first protocol single-copy, partly to avoid the resource cost of multi-copy entangled measurements; that cost concern, though raised for a different learning task, is one our moment-estimation results speak to directly. See also [6]. Section 5 quantifies the separation under noise, and Section 6 tests the collective route on hardware.

**NISQ device stability.** Dasgupta and Humble [11, 12] quantify the temporal and spatial instability of superconducting devices over 22 months, and related work analyzes calibration drift across devices [13]. Section 6.6 confirms their conclusions in the specific setting of collective-measurement protocols and claims no priority over them.

---

## 8. Limitations

We state these ourselves rather than leave them to be found.

**The variance law's inputs are numerical.** The law is exact and the single-qubit projection variance has a closed form. For $n \geq 2$, $\zeta_1$ has no weight-only closed form (Section 3.4) and the projections must be estimated numerically. A fully analytic $M^*(n)$ does not follow from this work.

**The $k=3$ validation is 6 of 7.** The single miss sits on the two-sigma boundary and is sensitive to the random seed. We report it rather than round it to 7 of 7.

**The noise model is a channel abstraction.** Global depolarizing and independent per-qubit channels are idealizations. The bias laws are exact given the channel, and they say nothing about whether the channel describes any particular device. Section 6 documents exactly where a real device departs from them, and that departure is a finding rather than a concealed assumption.

**The crossover was not demonstrated on hardware.** It sits at $n \approx 5$ to $8$ under realistic noise, and device non-stationarity dominates at those sizes on the hardware available to us. Section 6 is a test of the model, not a demonstration of the advantage.

**The hardware single-copy baseline at $n \geq 3$ is computed, not measured**, for the economic reason given in Section 6.7. The $n = 2$ point is measured directly and is what earns the computed points whatever credibility they have.

**The execution path on the Public Tier is not independently verifiable by us.** Jobs were routed through a decentralized network with validator spot-checking rather than direct vendor access (Section 6.1). We cannot exclude this as a contributor to the session-to-session variability of Section 6.4. It does not affect the readout and gate characterizations, which are internally consistent, but a replication with direct vendor access would be worth running.

**Statistical caveats.** The out-of-ensemble RMSE validation carries a median relative deviation of 6.7% between predicted and measured values. We investigated this and found it to be finite-trial noise on both sides rather than a systematic error: the estimator is approximately Gaussian at the budgets used, the projection variances are converged to better than 1%, and the measured RMSE converges to the predicted value to within $\pm 0.3\%$ as the trial count increases. The 6.7% figure matches the intrinsic RMSE noise at the trial counts used, and the signed deviation is +3.4%, which is within noise once cell correlations are accounted for.

---

## 9. Conclusion

We began with a number: at ten qubits, single-copy estimation of purity returns an error fifteen times the quantity being estimated. That number is not an accident of implementation, and this paper explains where it comes from.

The sample-complexity exponent for shadow-based moment estimation is not a constant. It is a function of system size and measurement budget, and it changes character at a threshold that grows exponentially in the number of qubits. We derived this from the exact Hoeffding decomposition of the U-statistic estimator, verified the derivation against brute-force enumeration at three moment orders and four state ensembles, and confirmed its prediction of the exponent out of sample with no fitted parameters. The mechanism is a competition between two terms in the estimator's variance whose ratio diverges exponentially, and it is invisible to any analysis that fixes the exponent in advance.

Paired with two exact bias laws for the collective route, distinguished by the geometry of the noise channel rather than by its rate, the variance law predicts without free parameters the system size at which collective measurement becomes the cheaper option. It gets 99% of measured crossovers within one qubit.

The practical guidance follows. For estimating $\mathrm{Tr}(\rho^k)$ beyond roughly six qubits under realistic noise, single-copy classical shadows do not work, and the law says which side of the boundary a given experiment is on. Collective measurement does work in principle, and the entangling overhead that has made the field cautious about it turns out, on the hardware we tested, to be nearly free: four two-qubit gates and no routing for a two-qubit destructive SWAP test, contributing about a seventh of the observed error.

What blocks the demonstration is neither variance nor gate error. It is readout fidelity, which on the device we used is three times the published figure and strongly correlated with neighbouring excitations, and above all it is device stability. Cross-session drift on byte-identical circuits exceeded every modeled error source by a factor of twenty and defeated seven successive attempts to attribute it to a characterizable mechanism. A protocol whose advantage rests on a bounded, characterizable bias floor cannot be fielded on a device whose floor moves by 0.2 between sessions.

That is the actionable conclusion for hardware developers, and it is more specific than a call for better gates. Two-qubit gate fidelity, the figure of merit the field advertises, was not the limitation. Readout fidelity, which vendors do not publish, and run-to-run stability, which vendors do not characterize, are what stand between current devices and a collective-measurement advantage that the theory says is waiting at six qubits.

Three questions follow from this work. Whether $\zeta_1$ admits a closed form under a richer ansatz than Pauli weight, which would make $M^*(n)$ fully analytic. Whether the same variance law, applied to the partial-transpose moments, gives an exact sample-complexity account of entanglement negativity estimation, which is the application that motivated us. And whether the drift we measured is a property of this device, this platform, or this class of hardware, which requires a multi-session, multi-device study that we did not have the access to run.

---

## Acknowledgements

Quantum hardware access was provided by the Open Quantum platform operated by Quantum Rings Inc. [10], on the Public Tier, whose licence conditions require citation of the platform paper and contribution of anonymized aggregated circuit data to a common repository. All hardware experiments ran on Rigetti Computing's Cepheus-1-108Q processor. [Mentor acknowledgement.]

## Data and code availability

All code, raw measurement counts, and analysis are available at [repository URL]. Raw hardware counts were committed to version control before any analysis was performed, and every locked prediction was committed before the corresponding measurement was submitted.

---

## References

[1] H.-Y. Huang, R. Kueng, and J. Preskill, "Predicting many properties of a quantum system from very few measurements," *Nature Physics* **16**, 1050 (2020). arXiv:2002.08953.

[2] H.-Y. Huang, M. Broughton, J. Cotler, S. Chen, J. Li, M. Mohseni, H. Neven, R. Babbush, R. Kueng, J. Preskill, and J. R. McClean, "Quantum advantage in learning from experiments," *Science* **376**, 1182 (2022). arXiv:2112.00778.

[3] S. Chen, J. Cotler, H.-Y. Huang, and J. Li, "Exponential separations between learning with and without quantum memory," *FOCS* (2022). arXiv:2111.05881.

[4] W. Gong et al., "On the sample complexity of purity and inner product estimation." arXiv:2410.12712.

[5] O. Danaci, Y. J. Patel, R. Molteni, E. van Nieuwenburg, V. Dunjko, and J. A. Krzywda, "Evidence of quantum machine learning advantage with tens of noisy qubits" (2026). arXiv:2605.21346.

[6] "Noisy quantum learning theory" (2026). arXiv:2512.10929.

[7] Shadow-norm analysis of second-order functionals. arXiv:2106.10190.

[8] Hamiltonian-driven shadow tomography. arXiv:2102.10132.

[9] "Estimating the coherence of noise in mid-scale quantum systems." arXiv:2409.02110.

[10] B. Wold, O. Armbruster, and R. Kuhn, "Open Quantum: Democratizing Access to Quantum Computing Resources," Quantum Rings Inc., Broomfield, CO. Available at www.openquantum.com.

[11] S. Dasgupta and T. S. Humble, "Stability of noisy quantum computing devices." arXiv:2105.09472.

[12] S. Dasgupta and T. S. Humble, "Assessing the stability of noisy quantum computation." arXiv:2208.07219.

[13] "Few-shot cross-device transfer for quantum noise modeling on real hardware" (2026). arXiv:2604.24397.

[14] M. Straeter, M. Tsesmelis, and L.-C. Kwek, "Detecting entanglement of non-Gaussian continuous-variable states from single-copy homodyne measurements" (2026). arXiv:2606.28698.

[15] "Sample complexity for embedded multipartite entanglement witness via Pauli and Clifford classical shadows" (2026). arXiv:2601.00859.

[16] "Transition from statistical to hardware-limited scaling in photonic quantum state reconstruction" (2026). arXiv:2603.12235.

---

# Appendix A. The exact fourth-moment U-statistic

The full U-statistic for $\mathrm{Tr}(\rho^4)$ requires the sum over all distinct ordered 4-tuples of snapshots,

$$S = \sum_{i \neq j \neq k \neq l} \mathrm{Tr}(G_i G_j G_k G_l).$$

Because the snapshots do not commute, this does not reduce to matrix power sums. It is obtained by M\"obius inversion over the partition lattice of the four cyclic slots.

For each set partition $P$ of $\{0,1,2,3\}$, let $T(P)$ be the sum of $\mathrm{Tr}(G_{a_0} G_{a_1} G_{a_2} G_{a_3})$ over all index assignments in which slots in the same block share an index and different blocks are unconstrained. The M\"obius function of the partition lattice is

$$\mu(P) = \prod_{b \in P} (-1)^{|b|-1} (|b|-1)! ,$$

and the all-distinct sum is

$$S = \sum_{P} \mu(P)\, T(P),$$

summed over the 15 set partitions of four elements.

Most $T(P)$ reduce to matrix power sums built from $S_1 = \sum_i G_i$, $P_2 = \sum_i G_i^2$, $P_3 = \sum_i G_i^3$, and $P_4 = \sum_i G_i^4$. The exception is the alternating partition $\{0,2\}\{1,3\}$, which requires

$$A = \sum_{i,j} \mathrm{Tr}(G_i G_j G_i G_j),$$

and this has no power-sum representation. It is computed exactly by a tensor contraction. Define the $d^2 \times d^2$ matrix

$$X_{(a,b),(c,d)} = \sum_i (G_i)_{ab} (G_i)_{cd} = \sum_i \mathrm{vec}(G_i)\,\mathrm{vec}(G_i)^{\!\top},$$

reshape it to $X_{abcd}$, and contract

$$A = \sum_{a,b,c,d} X_{abcd}\, X_{bcda}.$$

Total cost is $O(M d^2 + d^4)$. For larger $n$ the per-qubit factorization $\mathrm{Tr}(G_i G_j) = \prod_q \mathrm{Tr}(G_i^{(q)} G_j^{(q)})$ can be used inside the same identity. The complete $k=4$ estimator matches brute-force enumeration over all distinct 4-tuples to $10^{-14}$.

# Appendix B. The destructive SWAP test

Two copies of an $n$-qubit state occupy qubits $0 \ldots n-1$ (copy A) and $n \ldots 2n-1$ (copy B). For each $i \in \{0, \ldots, n-1\}$, apply a CNOT with control $i$ and target $i+n$, then a Hadamard on qubit $i$. Measure all $2n$ qubits. For each measured bitstring, let $a_i$ and $b_i$ be the outcomes on qubits $i$ and $i+n$. The purity is

$$\mathrm{Tr}(\rho^2) = \sum_{\text{bitstrings}} (-1)^{\,\#\{i \,:\, a_i = b_i = 1\}} \; P(\text{bitstring}).$$

The circuit uses exactly $n$ two-qubit gates, no ancilla, and has depth two after state preparation. The sign rule is invariant under bitstring reversal, so the endianness convention of the backend does not affect the result. We verified the rule against exact simulation for pure and mixed states to $10^{-16}$.

# Appendix C. Hardware protocol

**Platform.** Open Quantum (Quantum Rings Inc.) [10], Public Tier, Standard queue, Rigetti Cepheus-1-108Q. Jobs were submitted as OpenQASM 3 with physical-qubit addressing (`$0`, `$1`, and so on), which the platform required because it rejects non-contiguous virtual registers. The Public Tier routes jobs through the Quantum Compute subnet (SN48) on the Bittensor network, operated by qBitTensor Labs, with validator spot-checking of execution on the target QPU [10]; see Section 6.1 for the interpretive consequences.

**Registers.** GHZ ladders, verified by re-transpilation to require zero routing SWAPs:

| $n$ | physical qubits | CZ gates |
|---|---|---|
| 2 | $\{0, 1, 9, 10\}$ | 4 |
| 3 | $\{0, 1, 2, 9, 10, 11\}$ | 7 |
| 4 | $\{0, 1, 2, 3, 9, 10, 11, 12\}$ | 10 |

The ladders nest, so a single readout characterization on the $n=4$ register covers all three.

**Discipline.** Every prediction was computed, printed, and committed to version control before the corresponding measurement job was submitted. Raw counts were committed verbatim before any analysis. Where a prediction failed, the model was not adjusted to fit. Every hardware job was quote-gated against a hard credit ceiling before charging.

**Total hardware expenditure.** Approximately 115 platform credits across nine experimental campaigns.
