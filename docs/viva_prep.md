# Viva preparation (plan.md §21)

Draft answers to the 20 likely questions, grounded in the actual results (`results/summary/`).
Every group member should be able to give each answer in their own words, without reading
from this page. Numbers below match `results/summary/report_numbers.json` at the time of
writing; if the sweep is re-run, regenerate this intuition by skimming the updated tables
rather than trusting the exact figures quoted here verbatim.

---

**1. What is the difference between error mitigation and error correction?**

Error correction encodes one logical qubit redundantly across many physical qubits and
actively fixes errors on every shot, which needs extra qubits and, for most useful codes,
fault-tolerant hardware we don't have yet. Error mitigation needs no extra qubits: it runs
a handful of extra circuits (calibration circuits, or the same circuit at amplified noise)
and corrects the *average* classically afterward. It never touches an individual shot —
only the statistic computed from many shots — which is why we always talk about mitigating
"the expectation value," never "the measurement."

**2. Why must global-folding scale factors be odd? What is U(U†U)^k?**

Folding replaces the circuit U with U_λ = U·(U†U)^k for λ = 2k+1. U† is the exact inverse
of U, so U†U is the identity on paper — the ideal output of U_λ is identical to U's — but on
real hardware every one of those extra gates adds real noise. Each (U†U) pair adds 2 units
of noise per unit of λ, so only odd λ (1, 3, 5, …) are reachable by this construction;
there's no way to get an even multiple by appending whole U†U blocks to a single U.

**3. Why doesn't ZNE remove readout error in your setup?**

Folding only repeats the *gates*; the measurement happens exactly once, after the folding,
at every λ. Since the readout error is applied identically regardless of λ, the Richardson
extrapolation — which relies on seeing the *signal* shrink predictably with λ — never sees
the readout contribution change, so it cannot be extrapolated away. We confirmed this
directly: dividing the exact readout attenuation (1 − 2p_ro)ⁿ out of the infinite-shot ZNE
result removes most of the remaining bias, which is exactly the part that is gate noise
(§6, RQ1 "readout vs. gate noise").

**4. Derive the Richardson coefficients for λ = (1, 3, 5). Why does ZNE increase variance, and where does the factor of about 2.3 come from?**

γᵢ = Πⱼ≠ᵢ λⱼ/(λⱼ−λᵢ). For i=1 (λ=1): γ₁ = (3·5)/((3−1)(5−1)) = 15/8 = 1.875. For i=2
(λ=3): γ₂ = (1·5)/((1−3)(5−3)) = 5/(−4) = −1.25. For i=3 (λ=5): γ₃ = (1·3)/((1−5)(3−5)) =
3/8 = 0.375. They sum to 1 (needed so a constant signal extrapolates to itself). Treating
each E(λᵢ) as an independent ±1-valued average, Var(E₀) = Σγᵢ²(1−Eᵢ²)/N, so the standard
deviation scales by √Σγᵢ² = √(1.875² + 1.25² + 0.375²) = √5.21875 ≈ 2.285. Measured in our
sweep, the ratio runs a little higher than this (2.07×–2.70× across noise levels; theory
2.28×), which we attribute to the folded circuits' shots not behaving as perfectly
independent as the simple model assumes.

**5. Why did you use `optimization_level=0`?**

Because a transpiler at any higher optimization level would notice that U† immediately
followed by U is the identity and delete the pair — exactly the gates we added on purpose
to scale the noise up. At level 0 nothing is cancelled, so the folded circuit really does
execute λ times the gates of the original.

**6. Why can REM produce negative probabilities, and what did you do about it?**

The assignment matrix A is estimated from a finite number of calibration shots, and the
measured distribution p̃ is also from finite shots, so A⁻¹p̃ is an estimate, not an exact
inverse — it can push some entries below zero. We call this a "quasi-probability" vector
and record its negative mass directly (`rem_negative_mass`; worst case observed was about
0.019 for full REM in our sweep). We compute expectation values from this raw,
possibly-negative vector, because that's linear and unbiased; we only ever project it onto
the nearest valid probability distribution (Euclidean projection onto the simplex) when we
need an actual distribution, for Hellinger fidelity or total variation distance.

**7. Why does full REM need 2ⁿ circuits? What does tensored REM assume, and when does it fail?**

Full calibration measures the readout response to every one of the 2ⁿ computational basis
states, one calibration circuit per state, so the assignment matrix's every column is
measured directly. Tensored REM instead assumes each qubit's readout error is independent
of every other qubit's, which lets it build each qubit's own 2×2 matrix from just 2
circuits (all-0 and all-1) and combine them with a Kronecker product — 2 circuits at any n,
instead of 2ⁿ. It would fail, or at least need to be augmented, if readout errors were
correlated between qubits (crosstalk during measurement), which our noise model doesn't
include — a model limitation we state explicitly, since it's exactly the assumption that
makes tensored REM look as good as full REM here.

**8. What does the depolarizing parameter mean in Qiskit? How did you map the course's "% error" to it?**

`depolarizing_error(λ, k)` on k qubits implements E(ρ) = (1−λ)ρ + λ·Tr(ρ)·I/2^k: with
probability λ the state is replaced by the maximally mixed state on those k qubits. We
mapped the course's "X% single-qubit error" directly to `depolarizing_error(X/100, 1)` on
every `ry`/`rz` gate, "Y% two-qubit error" to `depolarizing_error(Y/100, 2)` on every `cx`,
and "Z% readout error" to a symmetric bit-flip readout matrix with flip probability Z/100 on
every qubit — logged as a fixed decision in `DECISIONS.md`.

**9. Why did you choose parity as the observable? Why the |E| ≥ 0.3 threshold, and what bias does it introduce?**

Parity ⟨Z⊗…⊗Z⟩ is a global observable: a bit flip on *any* qubit, from *any* source, flips
its sign, so it is maximally sensitive to every noise channel we model — it makes mitigation
effects clearly visible rather than diluted. The threshold exists because a random
multi-qubit parity is very often close to zero, where the shot-noise floor (σ ≈ 0.031 at
1024 shots) swamps the signal and relative error becomes undefined. Rejecting low-|E|
instances guarantees a measurable signal, but it means our circuit instances are not a
uniform sample of random circuits — we explicitly list this as a selection bias.

**10. Explain your encoding. Why scale features to [0, π]? What limits expressivity?**

Each feature x_k is encoded as `ry(x_k)` on qubit k, starting from |0⟩, giving
⟨Z_k⟩ = cos(x_k) before the ansatz runs. [0, π] is exactly half a Bloch meridian: the
minimum feature value lands on |0⟩ and the maximum on |1⟩, with no wrap-around ambiguity
(unlike [0, 2π), where two different feature values could map to states that are hard to
distinguish downstream). By Schuld, Sweke & Meyer (ref. 15), a single encoding layer
restricts the whole model to a trigonometric polynomial with frequencies {−1, 0, +1} per
feature — genuinely limited expressivity, not just an implementation detail, which is why
"data re-uploading" (repeating the encoding between ansatz layers) is listed as future work.

**11. Why was PCA or scaling fitted only on training data? What is data leakage?**

Data leakage is when information from the test set influences model fitting, even
indirectly — for example, if a scaler's mean and variance are computed from the full
dataset, the training process has implicitly "seen" the test distribution, which inflates
reported accuracy relative to what the model would achieve on genuinely unseen data. We fit
`StandardScaler`, `PCA` and `MinMaxScaler` on the 80-sample training split only, and only
*transform* the 20-sample test split with those already-fitted parameters (clipping it
afterward, since a fitted transform can push a test point slightly outside the training
range). A dedicated test (`test_preprocessing_fit_on_train_only_no_leakage`) proves this by
showing that fitting the same scaler on train+test gives different parameters.

**12. Why might accuracy barely change under noise while expectation error grows?**

Accuracy only depends on the *sign* of the expectation value (`sign(E_hat)`), not its
magnitude. Depolarizing noise shrinks |E| toward zero roughly uniformly — it attenuates the
margin, but it mostly does not flip the sign of a confidently-classified point. With only 20
test samples, accuracy also moves in coarse 5% steps. In our data, unmitigated margin
retention falls from 0.997 (ideal) to 0.411 (high noise) while accuracy only drifts from
0.920 to 0.900 over the same range — see figures B4 (mean |E| error vs. noise) and B5 (a
scatter of E_hat against E_exact, where unmitigated points are visibly shrunk toward the
origin along the same sign as the exact value, not flipped across it).

**13. Is there any quantum advantage in your results?**

No, and none is claimed. The exact (noiseless) VQC's test accuracy, averaged across
qubit counts, layers and seeds, is comparable to Logistic Regression's and SVM's on the
identical preprocessed features and the identical train/test split — sometimes a little
higher, sometimes a little lower, with no systematic direction. This is a fairness
comparison under identical conditions, run at a scale (2–4 qubits, 100 Iris samples) far
too small to say anything about asymptotic advantage either way.

**14. How did you make the comparison with LR and SVM fair?**

All three models see exactly the same stratified 80:20 train/test split (same indices,
derived from the same seed) and exactly the same preprocessed features — the 2 PCA
components or the 4 raw features, after the identical train-only-fitted scaling. No model
saw the test labels during fitting, and no hyperparameter was tuned using the test set (the
LR and SVM hyperparameters are fixed in the config, not searched).

**15. Are 5 seeds enough? How did you compute the confidence intervals, and what can the tests detect?**

Five seeds is small by statistical standards, and we say so explicitly as a limitation. For
each (n, L, noise, method) we report the mean, the sample standard deviation (ddof=1), the
median, and a 95% confidence interval as mean ± t₀.₉₇₅,₄·std/√5, with t = 2.776. Per
condition we use a paired t-test rather than a Wilcoxon signed-rank test, because Wilcoxon
on 5 pairs has a minimum achievable two-sided p-value of 0.0625 — it can never reach the
conventional 0.05 threshold at this sample size. Pooling 30 paired seeds (3 qubit counts × 2
depths × 5 seeds) per noise level lets us run a genuine Wilcoxon test with real power, which
is what we use for the headline significance claims in §6.

**16. What happens at high noise, and why might ZNE struggle there?**

At the optional `high` stress level (3%/5%/5% gate/gate/readout error), every method's
error is larger than at moderate noise, but every mitigated method still beats the
unmitigated baseline, and `zne_rem` remains the strongest. ZNE's own extrapolator
diagnostics do show more strain at high noise: more exponential fits fail (the folded
signal crosses zero or gets too small to take a reliable logarithm of), even though
Richardson estimates themselves rarely leave the physical [−1, 1] range in our grid. This is
consistent with (but a smaller-scale echo of) the theoretical results that mitigation's
sampling cost must eventually grow without bound as noise or depth increases (refs 11, 12).

**17. What is your research gap and your proposed improvement? What did the data show?**

The gap: prior work mostly studies readout mitigation and extrapolation-based mitigation
separately, often at hardware scale, and rarely connects expectation-value gains to
downstream QML metrics measured against classical baselines on identical inputs. We propose
two lower-cost variants and test both: tensored REM (2 calibration circuits instead of 2ⁿ),
which matched full REM's accuracy to within a few thousandths on average; and
variance-optimal ZNE shot allocation (shots proportional to |γᵢ| instead of split evenly),
which cut the Richardson estimate's empirical standard deviation to within a few percent of
the ~11% reduction theory predicts, at an unchanged total shot budget.

**18. How would results differ on real hardware?**

Our noise model is incoherent (depolarizing) and Markovian (memoryless); it omits coherent
errors, crosstalk between qubits, T1/T2 relaxation, drift in the noise parameters over time,
and leakage out of the computational subspace. On real hardware, ZNE's assumption that
folding scales noise cleanly is less exact (coherent errors don't necessarily scale the way
depolarizing noise does), REM's assumption that readout noise is stable over the calibration
period can be violated by drift, and the deepest folded circuits (up to 5× our base depth)
could exceed the device's coherence time outright. We would expect both methods to help
less, and to need re-calibration more often, than this simulation suggests.

**19. Why not use PEC (probabilistic error cancellation)?**

PEC can be unbiased in principle, but its sampling overhead grows exponentially with the
noise strength and circuit size, and it requires an accurate noise tomography of the device
first — both of which make it impractical at the breadth of conditions (120 Track A
conditions, 4 noise levels) this project sweeps. It remains listed as future work for a
smaller, more targeted follow-up study.

**20. What do Hellinger fidelity and TVD measure, and how do they differ?**

Both compare two probability distributions over the same 2ⁿ outcomes. Hellinger fidelity,
(Σₓ√(pₓqₓ))², is 1 for identical distributions and 0 for distributions with no overlapping
support; it is more sensitive to differences concentrated on a few high-probability outcomes.
Total variation distance, ½Σₓ|pₓ−qₓ|, is 0 for identical distributions and 1 for disjoint
support, and treats every outcome's absolute probability difference equally regardless of
its size. We report both: REM visibly improves both metrics over the unmitigated baseline
(figures A2 and `table_fidelity.md`), which is additional evidence that it is correcting the
whole output distribution, not just the one number (parity) we optimize the other analyses
around.
