# Literature notes (plan.md §18)

Reading notes for every paper in the review table, one per paper. Each note is grounded in
that paper's own abstract (fetched from arXiv, Nature, PRX, PRL, npj Quantum Information or
Reviews of Modern Physics, as linked). Nothing below is drawn from memory alone; anything
the group has not personally read beyond the abstract is marked accordingly. Group members
should read at least the introduction and results of each paper before the viva and replace
any remaining `TODO(team)` marks with their own notes.

Citation style: numbered, IEEE-like, matching `report/report_template.md`'s reference list.

---

## [1] J. Preskill, "Quantum Computing in the NISQ era and beyond," *Quantum* 2, 79 (2018). arXiv:1801.00862.

**Theme:** (a) NISQ and mitigation foundations.

**Problem addressed:** What near-term ("Noisy Intermediate-Scale Quantum", NISQ) devices can
and cannot be expected to do, and what the path to useful quantum computing looks like
before fault tolerance.

**Method:** A position/survey paper, not an experiment: it defines the NISQ regime (devices
of roughly 50–100 qubits, with no error correction), and argues about near-term
applications, noise, and the gap to fault-tolerant quantum computing.

**Setting:** Conceptual / no specific hardware experiment.

**Key finding:** NISQ devices may be able to perform tasks beyond the reach of classical
computers, but gate noise limits the depth and width of circuits that can be run reliably
without mitigation or correction; this motivates mitigation specifically because full error
correction is out of reach at this scale.

**Relevance to us:** Frames the whole project. Our entire testbed exists because of the gap
this paper names: devices too small and too noisy for error correction, but too useful to
ignore. `TODO(team)`: read past the abstract for Preskill's specific remarks on readout
error and near-term algorithm classes, and cite a page/section if quoted directly.

---

## [2] K. Temme, S. Bravyi, J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Phys. Rev. Lett.* 119, 180509 (2017). arXiv:1612.02058.

**Theme:** (a) foundations, (b) ZNE.

**Problem addressed:** How to get accurate expectation values from a noisy, short-depth
circuit without quantum error correction.

**Method:** Introduces two mitigation schemes: (i) extrapolation to the zero-noise limit via
Richardson's method, and (ii) probabilistic error cancellation (quasi-probability
resampling of randomized circuits).

**Setting:** Theoretical derivation with simulated short-depth circuit examples.

**Key finding:** Expectation values can be corrected by running the circuit at several
amplified noise strengths and extrapolating back to zero noise (Richardson), or by
compensating errors in the classical average using a quasi-probability decomposition of the
inverse noise channel (PEC).

**Relevance to us:** This is the origin of the Richardson-extrapolation ZNE method we
implement in `mitigation/zne.py`, and it is also where probabilistic error cancellation
(out of our scope, §10.6) comes from.

---

## [3] Y. Li, S. C. Benjamin, "Efficient variational quantum simulator incorporating active error minimization," *Phys. Rev. X* 7, 021050 (2017). arXiv:1611.09301.

**Theme:** (b) ZNE.

**Problem addressed:** Whether useful quantum simulation can run on near-term, imperfectly
controlled hardware rather than waiting for fault tolerance.

**Method:** A variational quantum/classical loop in which every quantum operation is
presumed noisy; errors are deliberately amplified and the result is extrapolated to the
zero-error limit.

**Setting:** Theoretical/simulated variational quantum eigensolver context.

**Key finding:** Independently of [2], this paper proposes the same core idea — scale the
noise up on purpose, then extrapolate back down — specifically for variational algorithms.

**Relevance to us:** A second, independent origin of the extrapolation idea, which shows the
Track A "fold and extrapolate" method and the Track B VQC noisy-inference pipeline rest on
an idea developed twice, in the same year, for exactly our two use cases (an expectation
value and a variational model).

---

## [4] S. Endo, S. C. Benjamin, Y. Li, "Practical quantum error mitigation for near-future applications," *Phys. Rev. X* 8, 031027 (2018). arXiv:1712.09271.

**Theme:** (a) foundations.

**Problem addressed:** Both extrapolation and quasi-probability mitigation assume the noise
model is known exactly, which is unrealistic; this paper asks what to do about an
imperfectly known noise model.

**Method:** A protocol for systematically measuring the effect of errors in order to design
practical mitigation circuits, plus an exponential-fit variant of extrapolation.

**Setting:** Theoretical, with numerical demonstration.

**Key finding:** Practical mitigation needs to account for imperfect knowledge of the error
model; an exponential extrapolation form is proposed as more physically motivated than a
polynomial one for some noise channels.

**Relevance to us:** This is the direct justification for our `exp` extrapolator in
`mitigation/zne.py` (§10.4): depolarizing noise decays a traceless observable toward 0
exponentially, not polynomially, which is exactly the physical motivation this paper gives
for preferring an exponential fit in some regimes. Our results (`table_extrapolators.md`)
let us check when the exponential form actually helps.

---

## [5] A. Kandala, K. Temme, A. D. Córcoles, A. Mezzacapo, J. M. Chow, J. M. Gambetta, "Error mitigation extends the computational reach of a noisy quantum processor," *Nature* 567, 491–495 (2019).

**Theme:** (b) ZNE, on hardware.

**Problem addressed:** Whether the Richardson-extrapolation idea from [2] actually works on
real superconducting hardware, not just in theory.

**Method:** Extrapolation of results from a set of experiments run at deliberately varied
noise strength, demonstrated on an IBM superconducting processor, applied to variational
optimization of Hamiltonians for quantum chemistry and magnetism.

**Setting:** Real hardware (IBM superconducting qubits), single- and two-qubit calibration
experiments plus variational chemistry/magnetism problems.

**Key finding:** Error mitigation measurably extends the accuracy achievable on noisy
hardware with no additional qubits or hardware changes, using only extra circuit executions
and classical post-processing.

**Relevance to us:** The hardware proof that the simulation story we build (Track A) is not
just a simulation artifact — gains of the same qualitative kind (extrapolation recovering
otherwise-inaccessible accuracy) have been measured on real devices. `TODO(team)`: note the
specific noise-scaling method used (this paper predates unitary/gate folding, [6]) if
discussed in the viva.

---

## [6] T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, W. J. Zeng, "Digital zero noise extrapolation for quantum error mitigation," *IEEE QCE* 306–316 (2020). arXiv:2005.10921.

**Theme:** (b) ZNE.

**Problem addressed:** How to scale noise up on digital gate-based hardware in a controlled,
reproducible way (as opposed to the analog pulse-stretching used in earlier ZNE
demonstrations), and how to extrapolate well.

**Method:** Introduces unitary folding (circuit-level: replace U by U(U†U)^k) and
parameterized/local noise scaling as digital noise-scaling methods, and reviews
extrapolation choices.

**Setting:** Theoretical with simulated and small hardware benchmarks.

**Key finding:** Unitary folding is a reliable, purely digital way to scale noise without
needing pulse-level hardware control.

**Relevance to us:** This is, directly, our `fold_global` function (§10.4): "the same noise-
scaling method" named in the plan is this paper's global unitary folding. Our tests that
check the folded statevector equals the original, and that gate counts scale exactly by λ,
are checking the exact property this paper's construction guarantees.

---

## [7] S. Bravyi, S. Sheldon, A. Kandala, D. C. McKay, J. M. Gambetta, "Mitigating measurement errors in multiqubit experiments," *Phys. Rev. A* 103, 042605 (2021). arXiv:2006.14044.

**Theme:** (c) readout mitigation.

**Problem addressed:** Full 2ⁿ×2ⁿ readout calibration does not scale; this paper asks how to
mitigate measurement error on larger devices.

**Method:** Two schemes based on tensor-product and correlated Markovian noise models;
error rates are extracted from calibration data and the inverse noise matrix is applied to
the measured probability vector. Demonstrated on graph states up to 12 qubits and 20-qubit
random Clifford states.

**Setting:** Real IBM Quantum hardware.

**Key finding:** A tensor-product (per-qubit) calibration model is often an adequate
approximation to the full correlated noise model, at a fraction of the calibration cost.

**Relevance to us:** This is the direct origin of our tensored REM (§10.3, Improvement 1):
the same tensor-product assumption, the same per-qubit assignment-matrix construction, and
the same basic trade (scalability against the possibility of missing correlated errors).
Our `table_rem_full_vs_tensored.md` is a small-scale repeat of exactly the comparison this
paper makes.

---

## [8] P. D. Nation, H. Kang, N. Sundaresan, J. M. Gambetta, "Scalable mitigation of measurement errors on quantum computers," *PRX Quantum* 2, 040326 (2021). arXiv:2108.12518.

**Theme:** (c) readout mitigation.

**Problem addressed:** Even the tensor-product model in [7] requires building or inverting a
matrix; this paper asks how to mitigate readout error without ever forming the assignment
matrix or its inverse, for much larger qubit counts.

**Method:** "M3" (matrix-free measurement mitigation): works directly in the subspace
spanned by the noisy bitstrings actually observed, with a matrix-free iterative solver that
converges in O(1) steps and handles both correlated and uncorrelated readout noise.

**Setting:** Real IBM Quantum hardware, scaled to far more qubits than full calibration
could reach.

**Key finding:** Readout mitigation can be made to scale to large qubit counts by working in
the (small) subspace of observed outcomes rather than the full 2ⁿ-dimensional space.

**Relevance to us:** This is the forward path from our Improvement 1. Our tensored REM is
one scalable approximation; M3 is the more general one and is listed explicitly as future
work (§14.3) once our qubit counts outgrow what even tensored calibration can handle
comfortably.

---

## [9] R. LaRose, A. Mari, S. Kaiser, P. J. Karalekas, A. A. Alves, P. Czarnik, M. El Mandouh, M. H. Gordon, Y. Hindy, A. Robertson, P. Thakre, M. Wahl, D. Samuel, R. Mistri, M. Tremblay, N. Gardner, N. T. Stemen, N. Shammah, W. J. Zeng, "Mitiq: A software package for error mitigation on noisy quantum computers," *Quantum* 6, 774 (2022). arXiv:2009.04417.

**Theme:** (b) ZNE, reference implementation.

**Problem addressed:** Reproducible, backend-agnostic software for applying error
mitigation, so results across groups and hardware are comparable.

**Method:** Describes Mitiq, a Python toolkit implementing ZNE, probabilistic error
cancellation and Clifford data regression against a common backend interface.

**Setting:** Software library paper; benchmarked against multiple backends and simulators.

**Key finding:** The same mitigation building blocks (noise scaling, extrapolation) we
implement by hand are the ones a widely used community library exposes, which lets our
implementation be checked against theirs (Appendix A.5, optional).

**Relevance to us:** Mitiq's `RichardsonFactory` and `scale_noise=fold_global` are the
reference we would cross-check our Richardson extrapolator and folding against, if the
optional Appendix A.5 cross-check is done. Not required for the core results.

---

## [10] Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. van den Berg, S. Rosenblatt, H. Nayfeh, Y. Wu, M. Zaletel, K. Temme, A. Kandala, "Evidence for the utility of quantum computing before fault tolerance," *Nature* 618, 500–505 (2023).

**Theme:** (b) ZNE, large-scale hardware.

**Problem addressed:** Whether mitigated, noisy quantum computation can be useful (produce
results beyond leading classical approximation methods) before fault tolerance is achieved.

**Method:** Mitigated expectation-value experiments on a 127-qubit superconducting
processor, at circuit volumes compared against classical tensor-network methods (matrix
product states and isometric tensor network states).

**Setting:** Real large-scale superconducting hardware.

**Key finding:** For strongly entangled circuits, the mitigated quantum results stayed
accurate where the tested classical approximation methods broke down, which the authors
present as evidence for the practical utility of mitigated near-term quantum computation.

**Relevance to us:** The highest-profile demonstration that the strategy we study at
2–6-qubit scale (mitigate rather than correct) is the same strategy used at the frontier of
current hardware, orders of magnitude larger than our testbed. `TODO(team)`: this paper's
central claim (comparison against classical methods) was later debated in the literature;
if cited in the report beyond this note, say only what the paper itself claims.

---

## [11] R. Takagi, S. Endo, S. Minagawa, M. Gu, "Fundamental limits of quantum error mitigation," *npj Quantum Information* 8, 114 (2022). arXiv:2109.04457.

**Theme:** (d) fundamental limits of mitigation.

**Problem addressed:** Does mitigation have a fundamental cost, and is any particular method
provably optimal?

**Method:** A general framework relating mitigation to sampling cost, applied to local
depolarizing noise.

**Setting:** Theoretical.

**Key finding:** The number of samples needed for mitigation grows exponentially with
circuit depth under local depolarizing noise, and probabilistic error cancellation is
provably optimal in some of the settings studied.

**Relevance to us:** The theoretical grounding for why our own costs grow the way they do:
our shot/circuit counts already scale with 2ⁿ (REM) and with the fold depth (ZNE); this
paper is the reason to expect that growth is not just an engineering inconvenience of our
particular methods but a fundamental feature of mitigation itself.

---

## [12] Y. Quek, D. Stilck França, S. Khatri, J. J. Meyer, J. Eisert, "Exponentially tighter bounds on limitations of quantum error mitigation," *Nature Physics* 20, 1648 (2024). arXiv:2210.11505.

**Theme:** (d) fundamental limits of mitigation.

**Problem addressed:** Sharpening the bounds in papers like [11]: how quickly do mitigation
costs really grow with circuit depth and noise?

**Method:** Frames mitigation as a statistical-inference problem and derives sampling lower
bounds from that framing.

**Setting:** Theoretical.

**Key finding:** A superpolynomial number of samples is required in the worst case even at
circuit depths comparable to current experiments, and the depth at which noise effectively
scrambles the circuit (so that mitigation becomes intractable) can be exponentially smaller
than earlier estimates suggested.

**Relevance to us:** Directly supports the "ZNE breaks down at high noise" hypothesis (H4):
our optional `high` stress level is exactly probing the regime this paper says mitigation
should start failing, and our Richardson-out-of-range and exp-fit-failure counts
(`table_extrapolators.md`) are an empirical, small-scale echo of this theoretical limit.

---

## [13] Z. Cai, R. Babbush, S. C. Benjamin, S. Endo, W. J. Huggins, Y. Li, J. R. McClean, T. E. O'Brien, "Quantum error mitigation," *Rev. Mod. Phys.* 95, 045005 (2023). arXiv:2210.00921.

**Theme:** (a)-(d), comprehensive review.

**Problem addressed:** A unifying review of the whole quantum-error-mitigation field.

**Method:** Survey, from basic concepts and motivation through implementation details of
specific techniques (ZNE, PEC, readout mitigation, symmetry verification, and more).

**Setting:** Review article.

**Key finding:** Organizes the field into a small number of recurring ideas (scale noise and
extrapolate; invert a known noise channel at a sampling cost; use known symmetries to
reject bad shots) that recur across otherwise different-looking methods.

**Relevance to us:** Used to frame our own taxonomy in the report's background section
(§16.2): REM is a readout-mitigation technique, ZNE is an extrapolation technique, and our
"out of scope" list (§10.6: PEC, CDR, dynamical decoupling, symmetry verification) follows
this review's categorization.

---

## [14] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," *Nature* 567, 209–212 (2019).

**Theme:** (e) QML encoding and noise.

**Problem addressed:** Whether a quantum computer's larger Hilbert space can be used as a
feature space for classification.

**Method:** Two methods: a variational quantum classifier (parametrized circuit trained like
an SVM) and a quantum kernel estimator, both implemented on a superconducting processor.

**Setting:** Real hardware, small supervised classification tasks.

**Key finding:** Both quantum feature-map approaches can be implemented and trained on
near-term hardware for small classification problems.

**Relevance to us:** The variational quantum classifier in this paper is the architectural
category our Track B VQC belongs to (angle encoding plus a trainable ansatz, trained to
match ±1 labels). We do not claim the encoding advantage this paper investigates; our
encoding is the much simpler per-qubit angle encoding of §13.5, chosen for its minimal
circuit depth rather than for any feature-space argument.

---

## [15] M. Schuld, R. Sweke, J. J. Meyer, "The effect of data encoding on the expressive power of variational quantum machine learning models," *Phys. Rev. A* 103, 032430 (2021). arXiv:2008.08605.

**Theme:** (e) QML encoding and noise.

**Problem addressed:** How the choice of data encoding (how classical features are loaded
into gate angles) limits what functions a variational quantum model can represent.

**Method:** Shows the model output can be written as a (partial) Fourier series in the input
data, where the accessible frequencies come directly from the encoding gates.

**Setting:** Theoretical.

**Key finding:** With a single encoding layer, the accessible frequency spectrum is limited
(for a single ry per feature per encoding layer, the frequencies are {−1, 0, +1} in that
feature); repeating the encoding ("data re-uploading") is needed to access richer frequency
spectra and, with enough repetitions, to become a universal function approximator.

**Relevance to us:** The direct theoretical justification for the limitation named in §13.5
and §16.5: our VQC uses a single `ry(x_k)` encoding layer per qubit, so by this paper's
result our model is restricted to frequencies {−1, 0, +1} per feature. Data re-uploading
(repeating the encoding between ansatz layers) is listed as future work (§14.3) precisely
because this paper shows it is the lever that would remove that restriction.

---

## [16] S. Wang, E. Fontana, M. Cerezo, K. Sharma, A. Sone, L. Cincio, P. J. Coles, "Noise-induced barren plateaus in variational quantum algorithms," *Nature Communications* 12, 6961 (2021).

**Theme:** (e) QML encoding and noise.

**Problem addressed:** Whether hardware noise itself (not just the ansatz structure) can
destroy the trainability of a variational quantum model.

**Method:** A rigorous proof, for local Pauli noise, that the cost-function gradient vanishes
exponentially in the number of qubits when the ansatz depth grows linearly with qubit count.

**Setting:** Theoretical.

**Key finding:** Noise alone, independent of barren plateaus caused by circuit expressivity,
can make training intractable at scale.

**Relevance to us:** Explains why we train the VQC noiselessly and exact (§13.8) rather than
training under noise: training under noise, at any scale beyond our tiny circuits, risks
exactly the vanishing-gradient failure mode this paper proves. Noise-aware training is
listed as future work (§14.3), with this paper as the reason it is nontrivial.

---

## [17] J. A. Smolin, J. M. Gambetta, G. Smith, "Efficient method for computing the maximum-likelihood quantum state from measurements with additive Gaussian noise," *Phys. Rev. Lett.* 108, 070502 (2012). arXiv:1106.5458.

**Theme:** (c) readout mitigation / projection to physical states.

**Problem addressed:** Measured data can produce an unphysical (e.g., negative-eigenvalue)
estimate; how to efficiently find the nearest physical state.

**Method:** Change basis to get a candidate (possibly unphysical) estimate, then find the
nearest physical state under the 2-norm, in at most O(d⁴) + O(d³) time (O(d³) for a Pauli
measurement basis).

**Setting:** Theoretical, with a concrete efficient algorithm.

**Key finding:** Euclidean projection onto the nearest physical state can be done efficiently
and is the nearest-point projection in exactly the sense our `project_to_simplex` uses.

**Relevance to us:** The same "find the nearest valid point when my raw estimate is
unphysical" problem, specialized by us to probability distributions instead of density
matrices. Our `project_to_simplex` (§10.2) implements Euclidean projection onto the
probability simplex in the same spirit as this paper's projection onto the physical-state
set, used only for distribution metrics (Hellinger fidelity, TVD) and never for the
expectation value itself, which is taken from the unprojected quasi-probabilities.

---

## Dataset citation (not counted as a reviewed paper)

**R. A. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of
Eugenics* 7(2), 179–188 (1936).**

Source of the Iris dataset (collected by Edgar Anderson; used by Fisher to illustrate linear
discriminant analysis): 50 samples each of *Iris setosa*, *versicolor* and *virginica*, with
four measurements per flower (sepal length/width, petal length/width). Track B uses the
`versicolor`/`virginica` subset (100 samples), excluding `setosa` because it is linearly
separable from the other two and would make the classification task trivial (§13.1).

---

## Synthesis (for report §2)

**On mitigation foundations and ZNE (themes a, b; refs 1–6, 9, 10, 13).** The field's two
core mitigation ideas — scale noise up and extrapolate back to zero (refs 2, 3), or invert a
known noise channel at a sampling cost (ref 2's second method, surveyed in ref 13) — were
both proposed in 2017 and demonstrated on hardware within two years (ref 5). Digital unitary
folding (ref 6) turned the "scale noise up" idea into something implementable purely at the
gate level, which is exactly our `fold_global`. By 2023 the same basic extrapolation
machinery was being run at 127-qubit scale (ref 10) — a four-order-of-magnitude jump from
our testbed, using the same conceptual tool.

**On readout mitigation (theme c; refs 7, 8, 17).** Readout mitigation has its own
scalability story, independent of ZNE: full calibration (§10.2) is exact but exponential;
the tensor-product approximation (ref 7) trades a (usually small) accuracy loss for linear
cost, which is exactly what our Improvement 1 tests; and matrix-free methods (ref 8) push
scalability further still by avoiding the assignment matrix altogether. The projection step
needed when an inverted estimate is unphysical (ref 17) recurs in both the density-matrix
and the probability-vector settings.

**On fundamental limits (theme d; refs 11, 12).** Both papers prove, by different routes,
that mitigation's sampling cost cannot stay bounded as circuits grow — it must eventually
grow exponentially with depth. Our optional `high` noise stress level and our tracking of
Richardson out-of-range estimates and failed exponential fits are a small, empirical window
onto the same phenomenon these papers bound theoretically.

**On QML encoding and noise (theme e; refs 14–16).** Variational quantum classifiers (ref
14) are trainable on near-term hardware for small problems, but two separate limitations
apply to them: the encoding strategy bounds what functions the model can represent at all
(ref 15), and noise on the hardware can destroy trainability even for an otherwise
expressive ansatz (ref 16). Both limitations are directly visible in our design choices: a
single encoding layer (hence the Fourier-frequency limit of ref 15) and noiseless training
(avoiding the barren-plateau mechanism of ref 16).

## Research gap

Most of the work above evaluates mitigation's effect on expectation values in physics or
chemistry tasks (refs 5, 10), often at hardware scale, or studies mitigation's fundamental
limits in the abstract (refs 11, 12). Reported gains are usually expressed as estimator bias
rather than in cost-normalized terms, and readout mitigation (refs 7, 8) and
extrapolation-based mitigation (refs 2, 3, 6) are mostly studied separately rather than
side by side under one controlled protocol. Fewer studies provide small, fully controlled,
cost-normalized side-by-side comparisons of REM, ZNE and their combination that also test
whether expectation-value gains carry through to downstream QML classification metrics
against classical baselines (ref 14's variational classifier is trained and evaluated
noiselessly; this project instead trains noiselessly but evaluates under all four noise
levels and all four mitigation methods). This project addresses that gap within the scope of
a controlled simulation study: a single paired protocol across both a physics-style testbed
and a QML task, reporting accuracy, cost and their trade-off together, and testing two
lower-cost variants (tensored REM, following the direction of ref 7; and variance-optimal
ZNE shot allocation, which to the team's knowledge is not addressed directly in the papers
reviewed above).
