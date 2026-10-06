# Group 20 — Quantum Circuit Noise and Error Mitigation: Implementation Plan (v2)

This document is the complete specification for the project. Hand it to Claude Code (or any developer) and implement it end to end. Decisions marked **FIXED** must not change without asking the team. Items marked **OPTIONAL** are done only after every required part is complete and passing.

## Changelog: v2 vs. v1

v2 aligns the plan with the course's Standard Requirements sheet and the 20-mark rubric. If you already implemented parts of v1, apply these changes:

1. **Noise levels now match the course standard.**
   - Low is 0.1% / 0.5% / 1%.
   - Moderate is 1% / 2% / 3%.
   - A new optional high-noise stress level is added (Section 8).
2. **The Iris binary classification track (Track B) is now REQUIRED.** It uses a stratified 80:20 split, Logistic Regression and SVM baselines, and the full set of classification metrics (Section 13).
3. **New required packages:** scikit-learn, plus pylatexenc for circuit diagrams.
4. **New metric:** success probability (Section 11).
5. **Tensored REM is now a required method.** It is one of the two proposed improvements. The other is a variance-optimal shot allocation experiment for ZNE (Section 14).
6. **New sections:**
   - Problem statement, objectives and RQs (Section 1)
   - Literature review and research gap (Section 18)
   - Rubric-mapped report outline (Section 19)
   - Screenshots and evidence (Section 20)
   - Viva prep (Section 21)
   - Fair-comparison and no-quantum-advantage rules (Section 0)

---

## 0. Instructions for Claude Code

1. Read this entire file before writing any code.
2. Implement the phases in the order given in Section 17. After each phase, run that phase's acceptance checks and the full test suite (`pytest -q`). Do not move on while anything fails.
3. Use only the libraries listed in Section 3. The following are forbidden:
   - `qiskit.ignis` (removed)
   - `qiskit.execute` (removed in Qiskit 1.0)
   - `qiskit.providers.aer` (old import path; use `qiskit_aer`)
   - `CompleteMeasFitter` and any other deprecated measurement-fitter API
4. Never transpile folded circuits with `optimization_level > 0`. It cancels the U†U pairs and silently destroys ZNE.
5. All randomness must go through the helpers in `src/qem/seeds.py`. Re-running any sweep must reproduce its CSV byte-for-byte, excluding timing columns.
6. Do not hardcode or "expect" numeric results. Sections 16.3 and 16.4 list hypotheses. The report must state what the data actually shows.
7. **Never state or imply quantum advantage.** Any comparison between the VQC and classical models must use the identical split, features and preprocessing, and must be described as a comparison only.
8. **Never invent literature content or citations.** Follow Section 18.5.
9. If anything is ambiguous or conflicts with installed library versions, stop and report the conflict rather than improvising. Record every resolution in `DECISIONS.md` with the date and a one-line reason.
10. Commit at least once per phase, with messages like `phase-3: noise models and executor`.
11. Generate every number in the report from the summary CSVs by code. Never hand-type a result.

---

## 1. Problem statement, objectives, research questions

### 1.1 Problem statement

Current (NISQ) quantum processors are noisy. Gate errors and measurement (readout) errors bias the outputs of quantum circuits, and the bias grows with circuit width and depth. Quantum error mitigation reduces this bias through extra circuit executions and classical post-processing, without needing extra qubits, but it costs extra circuits, shots and depth, and it has known limits.

This project investigates three things:

1. Whether selected error-mitigation methods improve the output quality of a noisy quantum circuit.
2. What each method costs.
3. Whether improvements in expectation values carry through to a downstream quantum machine-learning (QML) classification task.

### 1.2 Objectives

- **O1:** Build a controlled, reproducible noisy-simulation testbed in Qiskit Aer, using the course-standard noise levels.
- **O2:** Implement readout error mitigation (full and tensored) and zero-noise extrapolation (global folding with Richardson extrapolation), alone and combined.
- **O3:** Quantify how effective each method is (expectation-value error, distribution fidelity, success probability) and what it costs (circuits, shots, depth, two-qubit gates, time), across qubits, depth and noise.
- **O4:** Evaluate the same methods on a variational quantum classifier (VQC) for Iris binary, against Logistic Regression and SVM baselines.
- **O5:** Propose cheaper mitigation variants and test them with data.

### 1.3 Research questions

- **RQ1:** Do REM, ZNE and their combination reduce expectation-value error and improve the output distribution of a noisy circuit across noise levels, qubit counts and depths?
- **RQ2:** What is the resource cost (circuits, shots, depth, two-qubit gates) of each method, and how does it scale with qubit count?
- **RQ3:** Do expectation-value improvements translate into better classification metrics for a VQC on Iris binary? How does the VQC compare with LR and SVM on identical inputs?
- **RQ4:** Can cheaper variants (tensored REM, variance-optimal shot allocation for ZNE) keep most of the benefit at lower cost?

### 1.4 Two experimental tracks

| Track | What | Answers |
|---|---|---|
| **A: Mitigation testbed** | Layered expectation-value circuit, 2/4/6 qubits, depths 2/4, 4 noise levels, 5 methods | RQ1, RQ2, RQ4 |
| **B: QML application** | Iris binary VQC (2/4 qubits, depths 2/4), 4 noise levels, 4 methods, plus LR and SVM baselines | RQ3 |

### 1.5 Rubric coverage map (20 marks)

| Rubric component | Marks | Covered by (plan section) | Report section | Evidence produced |
|---|---|---|---|---|
| Problem understanding and objectives | 2 | 1.1–1.3 | §1 | Problem statement, 5 objectives, 4 RQs |
| Literature review | 3 | 18 | §2 | At least 8 papers in a comparison table, plus an explicit research gap |
| Algorithm/circuit design | 3 | 7, 10, 13.4–13.7 | §3 | Circuit diagrams (D1–D5), encoding, ansatz, parameter tables, mitigation algorithms |
| Qiskit implementation | 4 | 3–6, 9, 17, 20 | §4 | Working code, passing tests, screenshots, configs, seeds, versions |
| Experiments and visualization | 3 | 12, 13.12, 16 | §5 | 120 Track A plus 80 Track B controlled conditions, tables, about 25 figures |
| Comparative analysis | 2 | 15, 16.2 | §6 | Comparison against no-mitigation and against LR/SVM; noise and resource effects |
| Research gap and proposed improvement | 2 | 14, 18.4 | §7 | Limitations, plus 2 improvements tested with data |
| Report | 1 | 19, 21 | Whole report, viva | Clear write-up; prepared viva answers |

### 1.6 Course standard coverage map

| Standard requirement | How it is satisfied |
|---|---|
| Python 3.x, Qiskit, Qiskit Aer, NumPy, Pandas, Matplotlib, scikit-learn | All required (Section 3) |
| QML dataset; fit scaling/PCA only on training data | Iris binary (13.1). Scalers and PCA are fitted on the training split only (13.3) and asserted by tests. |
| Stratified 80:20 split; same split for classical and quantum | 13.2. LR, SVM and VQC use identical index arrays, asserted by tests. |
| 5 seeds; mean ± std | Both tracks (Sections 6, 15) |
| 1024 shots per circuit execution | Every execution. The one documented exception is Section 14.2, which keeps the total budget equal. |
| Qubits 2 and 4 | Track B uses 2 and 4. Track A uses 2 and 4, plus 6 because the group sheet asks for it. |
| Depths 2 and 4 | Both tracks |
| Noise Levels 0, 1, 2 | ideal / low / moderate with exactly the standard values (Section 8) |
| Noise Level 3 (optional) | Included as the `high` stress level and reported separately (Section 8) |
| Depolarizing plus readout noise model | Yes. Thermal relaxation is not added, and this is documented. |
| Classical baselines | LR and SVM for classification (13.9). The exact statevector serves as the exact-solver reference for Track A. |
| Classification metrics | Accuracy, precision, recall, F1, confusion matrix (13.11) |
| Quantum metrics | Hellinger fidelity, success probability, expectation-value error (Section 11, 13.11). Energy error and approximation ratio are not applicable here (no energy or optimization task), and the report says so. |
| Resource metrics | Qubits, depth, gate count, two-qubit gate count, shots, per method (Section 11, `table_resources.md`) |
| Reproducibility | Versions, seeds, dataset source, preprocessing parameters and noise parameters are all saved (Sections 6, 12, 13.12) |
| Research reporting | Tables, plots, limitations, research gap, fair comparison, no quantum-advantage claims (Sections 0, 19) |
| Minimum experiment set (6) | Track A: 6 qubit/depth settings × 4 noise levels = 24. Track B: 4 × 4 = 16. |

---

## 2. Specification reconciliation

| Parameter | Group sheet 1 | Group sheet 2 | Course standard | Adopted (**FIXED**) |
|---|---|---|---|---|
| Qubits | 2, 4 | 2, 4, 6 | 2 and 4 where feasible | Track A: 2, 4, 6. Track B: 2, 4 |
| Depth | not specified | 2, 4 | 2, 4 | 2, 4 (ansatz layers) |
| Noise levels | none / low / medium | ideal / low / moderate | L0 ideal; L1 0.1%/0.5%/1%; L2 1%/2%/3%; L3 optional 3–5% gate | ideal, low, moderate, high (values in Section 8) |
| Noise model | — | — | Depolarizing plus readout | Depolarizing plus readout |
| Shots | 1024 | 1024 | 1024 | 1024 per circuit execution |
| Seeds | — | 5 | 5 | 0, 1, 2, 3, 4 |
| Dataset | "select a circuit" | Iris binary or small expectation-value circuit | Iris / Wine / WDBC | Both: expectation-value circuit (Track A) and Iris binary (Track B) |
| Split | — | — | Stratified 80:20 | Stratified 80:20 |
| Baseline | before vs. after | no-mitigation baseline | LR + SVM | No-mitigation (both tracks), plus LR and SVM (Track B) |
| Metrics | error, distribution, overhead | accuracy/fidelity before and after, overhead | classification + quantum + resource metrics | Union of all three |

**Noise-parameter interpretation (FIXED, record in DECISIONS.md):** "X% single-qubit error" means `depolarizing_error(X/100, 1)`, "Y% two-qubit error" means `depolarizing_error(Y/100, 2)`, and "Z% readout error" means a symmetric bit-flip probability of Z/100 on every qubit.

Ask the instructor to confirm that Track A's 6-qubit runs and the inclusion of Track B are acceptable. Both can be switched off in config without code changes.

---

## 3. Environment

- **Python:** 3.11 (3.10–3.12 is acceptable).
- **Hardware:** Any laptop. No GPU and no IBM Quantum account are needed; everything is simulated locally.

`requirements.txt`:

```
qiskit>=1.2
qiskit-aer>=0.15
numpy
scipy
pandas
matplotlib
scikit-learn
pylatexenc        # required by QuantumCircuit.draw("mpl")
pyyaml
tqdm
pytest
```

`requirements-optional.txt`:

```
mitiq             # ZNE cross-check only (Appendix A)
jupyter           # exploration notebook only
```

Setup:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Use a minimal `pyproject.toml` (setuptools, package `qem` under `src/`).

At the start of every sweep, `run_*.py` writes `pip freeze` to `results/environment.txt` and the Python version to `results/python_version.txt`.

If mitiq conflicts with the installed Qiskit, skip it and log the conflict in `DECISIONS.md`.

---

## 4. Repository layout

```
qem-group20/
├── README.md
├── DECISIONS.md
├── plan.md
├── pyproject.toml
├── requirements.txt
├── requirements-optional.txt
├── .gitignore                         # .venv/, __pycache__/, *.pyc, .pytest_cache/
├── config/
│   ├── experiment.yaml                # full config (both tracks + improvement)
│   └── smoke.yaml                     # tiny config for quick checks
├── src/qem/
│   ├── __init__.py
│   ├── config.py
│   ├── seeds.py
│   ├── circuits.py                    # ansatz, measurement, folding, calibration circuits, stats
│   ├── observables.py
│   ├── noise.py
│   ├── execution.py
│   ├── metrics.py
│   ├── mitigation/
│   │   ├── __init__.py
│   │   ├── readout.py                 # full + tensored REM, simplex projection
│   │   ├── zne.py                     # extrapolators, variance, shot allocation
│   │   └── pipeline.py                # apply all methods to a set of counts -> estimates
│   ├── experiment.py                  # Track A condition + sweep
│   ├── improvement.py                 # Section 14.2 experiment
│   ├── qml/
│   │   ├── __init__.py
│   │   ├── data.py                    # load, split, preprocess (train-only fit)
│   │   ├── classical.py               # LR, SVM
│   │   ├── vqc.py                     # encoding + ansatz circuit, training
│   │   └── evaluate.py                # noisy inference + mitigation + metrics
│   ├── analysis.py
│   └── plotting.py
├── scripts/
│   ├── run_sweep.py                   # Track A
│   ├── run_qml.py                     # Track B
│   ├── run_improvement.py             # Section 14.2
│   ├── analyze.py
│   ├── make_plots.py
│   ├── make_design_figures.py         # circuit diagrams D1–D5
│   └── smoke_test.py
├── tests/
│   ├── test_seeds.py
│   ├── test_config.py
│   ├── test_circuits.py
│   ├── test_observables.py
│   ├── test_noise.py
│   ├── test_readout.py
│   ├── test_zne.py
│   ├── test_metrics.py
│   ├── test_experiment.py
│   ├── test_qml_data.py
│   ├── test_qml_vqc.py
│   ├── test_qml_evaluate.py
│   └── test_improvement.py
├── results/
│   ├── environment.txt
│   ├── python_version.txt
│   ├── logs/                          # sweep logs + pytest output (evidence)
│   ├── raw/                           # Track A: runs.csv, counts/, circuits/, calibration/
│   ├── qml/                           # Track B: qml_runs.csv, splits/, preprocessing/, models/, predictions/
│   ├── improvement/                   # zne_allocation.csv
│   ├── summary/                       # all summary CSVs and markdown tables
│   └── figures/
├── docs/
│   └── viva_prep.md
└── report/
    ├── report.md
    ├── literature_notes.md            # per-paper reading notes (team-written)
    └── screenshots/
        └── README.md                  # checklist of required screenshots
```

---

## 5. Configuration

`config/experiment.yaml` (**FIXED** values):

```yaml
experiment:
  name: group20_qem
  shots: 1024
  seeds: [0, 1, 2, 3, 4]

noise:
  one_qubit_gates: [ry, rz]
  two_qubit_gates: [cx]
  levels:                         # order matters for tables/plots
    ideal:    {p1: 0.0,   p2: 0.0,   p_ro: 0.0}    # course Level 0
    low:      {p1: 0.001, p2: 0.005, p_ro: 0.01}   # course Level 1
    moderate: {p1: 0.01,  p2: 0.02,  p_ro: 0.03}   # course Level 2
    high:     {p1: 0.03,  p2: 0.05,  p_ro: 0.05}   # course Level 3 (stress test, reported separately)
  headline_levels: [low, moderate] # used for headline averages

track_a:
  enabled: true
  qubits: [2, 4, 6]
  depths: [2, 4]
  noise_levels: [ideal, low, moderate, high]
  methods: [none, rem, rem_tensored, zne, zne_rem]
  circuit:
    type: layered_ry_rz_cx
    entanglement: linear
    angle_low: 0.0
    angle_high: 6.283185307179586
    observable: parity
    min_abs_exact_expectation: 0.3
    max_resample_attempts: 20000

track_b:
  enabled: true
  dataset: iris_binary
  classes: {positive: 1, negative: 2}     # versicolor = +1, virginica = -1
  test_size: 0.2
  stratified: true
  qubits: [2, 4]                          # = number of features after preprocessing
  depths: [2, 4]
  noise_levels: [ideal, low, moderate, high]
  methods: [none, rem, zne, zne_rem]
  feature_range: [0.0, 3.141592653589793]
  observable: parity
  training:
    optimizer: COBYLA
    min_maxiter: 300
    maxiter_per_param: 25
  classical:
    logistic_regression: {C: 1.0, max_iter: 1000}
    svm: {kernel: rbf, C: 1.0, gamma: scale}

rem:
  calibration_shots: 1024
  cond_threshold: 1.0e8

zne:
  folding: global
  scale_factors: [1, 3, 5]
  extrapolators: [richardson, linear, exp]
  primary_extrapolator: richardson

improvement:
  shot_allocation:
    enabled: true
    noise_levels: [low, moderate]
    total_shots: 3072
    repetitions: 50

simulator:
  method: density_matrix
  transpile_basis: [ry, rz, cx]
  optimization_level: 0

plots:
  example_condition: {n: 4, depth: 4, noise: moderate, seed: 0}
  qml_confusion_condition: {n: 4, depth: 2}
  dpi: 200

output:
  dir: results
```

`config/smoke.yaml` is the same as above except for these overrides:

- Track A: `qubits: [2]`, `depths: [2]`, `seeds: [0]`
- Track B: `qubits: [2]`, `depths: [2]`, `seeds: [0]`, `training.min_maxiter: 50`, `maxiter_per_param: 5`
- Improvement: `repetitions: 3`

`config.py` loads the YAML into frozen dataclasses and validates it. Every failure raises `ValueError` with a clear message. The checks are:

- Every qubit count is at least 2.
- Every depth is at least 1.
- Every noise probability is in [0, 0.5).
- Scale factors are odd positive integers, with `scale_factors[0] == 1`.
- Shots are greater than 0.
- Method names are valid.
- Every noise level referenced anywhere is defined.
- `test_size` is in (0, 1).

---

## 6. Seeding and reproducibility (**FIXED**)

```python
import hashlib

def derive_seed(*parts) -> int:
    s = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(s).hexdigest()[:8], 16) % (2**31 - 1)
```

| Purpose | Seed |
|---|---|
| Track A circuit angles | `derive_seed("circuit", n, L, seed)` |
| Track A noiseless shot reference | `derive_seed("sim", n, L, "ideal_shots", seed)` |
| Track A base / fold3 / fold5 | `derive_seed("sim", n, L, noise, seed, "base" \| "fold3" \| "fold5")` |
| Track A full / tensored calibration | `derive_seed("sim", n, L, noise, seed, "cal" \| "cal_tensored")` |
| Track B split | `derive_seed("qml_split", seed)` (independent of n, so both qubit counts share one split per seed) |
| Track B PCA | `derive_seed("qml_pca", seed)` |
| Track B VQC weight init | `derive_seed("qml_init", n, L, seed)` |
| Track B LR / SVM | `derive_seed("qml_lr", n, seed)`, `derive_seed("qml_svm", n, seed)` |
| Track B executions | `derive_seed("qml_sim", n, L, noise, seed, "base" \| "fold3" \| "fold5" \| "cal")` |
| Improvement repetitions | `derive_seed("alloc", n, L, noise, seed, rep, "uniform" \| "optimal", scale)` |
| Transpiler | `derive_seed("transpile", ...same parts..., purpose)` |

The comparison is paired by construction:

- In Track A, all methods for one (n, L, noise, seed) post-process the same base counts, and ZNE variants share the same folded counts.
- In Track B, all methods share the same counts per test sample, and LR, SVM and VQC share the same split and features.

---

## 7. Track A circuit design

### 7.1 Ansatz (**FIXED**)

The circuit has n qubits and L layers. Layer ℓ consists of:

1. `ry(theta[ℓ,q])` on every qubit q.
2. `rz(phi[ℓ,q])` on every qubit q.
3. A CNOT chain `cx(0,1), cx(1,2), …, cx(n−2,n−1)`.

Angles are drawn uniformly from [0, 2π), in this order: all theta of layer 0, then all phi of layer 0, then layer 1, and so on.

The circuit therefore has 2nL one-qubit gates and (n−1)L CNOTs:

| n | L | 1-qubit gates | CNOTs |
|---|---|---|---|
| 2 | 2 | 8 | 2 |
| 2 | 4 | 16 | 4 |
| 4 | 2 | 16 | 6 |
| 4 | 4 | 32 | 12 |
| 6 | 2 | 24 | 10 |
| 6 | 4 | 48 | 20 |

"Depth" in this project means the number of layers L. Also record the transpiled depth, excluding measurements and barriers:

```python
qc.depth(filter_function=lambda inst: inst.operation.name not in ("barrier", "measure"))
```

### 7.2 Construction functions (`circuits.py`)

```python
build_unitary(n, layers, rng) -> tuple[QuantumCircuit, np.ndarray]   # no clbits, no measurements; angles shape (L, 2, n)
with_measurements(unitary) -> QuantumCircuit    # QuantumCircuit(n, n); compose; barrier; measure(range(n), range(n))
                                                # single classical register; NEVER measure_all()
fold_global(unitary, scale) -> QuantumCircuit   # Section 10.4
calibration_circuits(n) -> list[QuantumCircuit]           # 2^n circuits (Section 10.2)
tensored_calibration_circuits(n) -> list[QuantumCircuit]  # 2 circuits (Section 10.3)
circuit_stats(qc) -> dict   # {"depth","cx","n1q","total_gates"} on transpiled circuit, excluding barrier/measure
```

### 7.3 Instance selection by rejection sampling (**FIXED**)

```python
@dataclass
class CircuitInstance:
    n: int; layers: int; seed: int
    unitary: QuantumCircuit; angles: np.ndarray
    attempts: int; E_exact: float

select_instance(n, layers, seed, threshold=0.3, max_attempts=20000) -> CircuitInstance
```

The function creates `rng = default_rng(derive_seed("circuit", n, layers, seed))`. It then repeatedly draws fresh angles from the same stream and returns the first instance with |E_exact| ≥ 0.3. If no instance qualifies within `max_attempts`, it raises `RuntimeError`.

**Rationale (for the report):** A random multi-qubit parity is often close to 0. At 1024 shots the shot-noise standard deviation is about 0.031, so effects near 0 are invisible, and relative error is undefined. The threshold guarantees a measurable signal. It also introduces a selection bias, which must be listed as a limitation.

Save each instance to `results/raw/circuits/n{n}_L{L}_s{seed}.json`. Include the angles, the number of attempts, E_exact, and the QASM 2 string from `qiskit.qasm2.dumps(with_measurements(unitary))`.

### 7.4 Observables and the noiseless reference (`observables.py`)

**Bit ordering (critical):** Qiskit is little-endian.

- In a count key `s`, qubit q is `s[n-1-q]`.
- In an integer index `i = int(s, 2)`, qubit q is `(i >> q) & 1`.
- `Statevector.probabilities()` uses the same indexing.

All functions in `observables.py` operate on probability vectors of length 2^n:

```python
counts_to_probvec(counts, n) -> np.ndarray       # strip spaces, zfill(n), fill missing with 0, normalize
parity_signs(n) -> np.ndarray                    # (-1)^popcount(i), cached
parity_from_probs(p) -> float                    # works on quasi-probabilities too
z_signs(n, q) -> np.ndarray                      # (-1)^((i>>q)&1)
magnetization_from_probs(p, n) -> float          # (1/n) Σ_q <Z_q>
success_probability(p, target_index) -> float    # p[target_index]
exact_reference(unitary) -> dict
    # sv = Statevector(unitary); probs = sv.probabilities()
    # E_exact = parity_from_probs(probs); cross-check vs sv.expectation_value(SparsePauliOp("Z"*n)).real (1e-9)
    # M_exact; probs; target_index = argmax(probs); P_succ_exact = probs[target_index]
```

**Observables:**

- **Primary:** the parity P = ⟨Z⊗…⊗Z⟩. It is global, so every gate error and every readout flip can affect it, which makes it maximally noise-sensitive. A symmetric readout flip with probability p multiplies it by exactly (1−2p)^n.
- **Secondary:** the magnetization M = (1/n)Σ_q⟨Z_q⟩.
- **Success probability:** P_succ = p̂(x*), where x* = argmax_x p_exact(x) is the most likely ideal bitstring. Also report the ratio P_succ_hat / P_succ_exact.

**References:**

- `E_exact` (statevector, infinite shots) is the ground truth for every error.
- `E_ideal_shots` (noiseless, 1024 shots) is the shot-noise floor.

---

## 8. Noise models (`noise.py`) (**FIXED**)

```python
build_noise_model(level, one_q_gates, two_q_gates) -> NoiseModel | None
```

- If all probabilities are 0, return `None`.
- If `p1 > 0`, apply `depolarizing_error(p1, 1)` to `["ry", "rz"]` with `add_all_qubit_quantum_error`.
- If `p2 > 0`, apply `depolarizing_error(p2, 2)` to `["cx"]`.
- If `p_ro > 0`, apply `ReadoutError([[1-p_ro, p_ro], [p_ro, 1-p_ro]])` with `add_all_qubit_readout_error`.

Imports: `from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError`.

| Level | Course name | p1 (1q depolarizing) | p2 (2q depolarizing) | p_ro (readout flip) | Role |
|---|---|---|---|---|---|
| ideal | Level 0 | 0 | 0 | 0 | sanity check |
| low | Level 1 | 0.001 | 0.005 | 0.01 | headline |
| moderate | Level 2 | 0.01 | 0.02 | 0.03 | headline |
| high | Level 3 (optional) | 0.03 | 0.05 | 0.05 | stress test, reported separately |

**Depolarizing semantics (for the report):** Qiskit's `depolarizing_error(λ, k)` is E(ρ) = (1−λ)ρ + λ·Tr(ρ)·I/2^k. The probability of a non-identity Pauli is λ(4^k−1)/4^k.

Helper: `readout_attenuation(p_ro, n) = (1 - 2*p_ro)**n`.

**Critical:** noise is attached by gate name. Any gate other than ry, rz or cx runs noise-free without any warning. The executor asserts this (Section 9).

Thermal relaxation (T1/T2) and crosstalk are not modeled. This is documented as a limitation.

---

## 9. Execution layer (`execution.py`)

```python
class Executor:
    def __init__(self, cfg): ...   # cache AerSimulator(method="density_matrix", noise_model=nm) per noise level
    def run(self, circuits, noise_level, shots, seed) -> tuple[list[dict[str, int]], float]:
        # 1. tcircs = transpile(circuits, basis_gates=["ry","rz","cx"], optimization_level=0, seed_transpiler=...)
        # 2. assert all op names in {"ry","rz","cx","measure","barrier"}
        # 3. t0 = perf_counter(); result = sim.run(tcircs, shots=shots, seed_simulator=seed).result()
        # 4. counts = [normalize(result.get_counts(i)) for i in range(len(tcircs))]   # strip spaces, zfill(n)
        # 5. return counts, perf_counter() - t0
```

The ideal level uses an `AerSimulator(method="density_matrix")` with no noise model, so every level goes through the same code path.

**Recommended diagnostic: `exact_noisy_expectation(unitary, noise_level)`**

1. Copy the unitary, call `save_density_matrix()` on it, and run it on the noisy density-matrix simulator.
2. Compute `p_gate = DensityMatrix(...).probabilities()`. This captures gate noise only.
3. Apply readout with `A_true = kron(A_{n−1}, …, A_0)`, where `A_q = [[1−p, p], [p, 1−p]]`.
4. `E_noisy_exact = parity_from_probs(A_true @ p_gate)`.

This separates each method's bias from shot noise.

---

## 10. Mitigation methods

All methods operate on probability vectors of length 2^n. `mitigation/pipeline.py` exposes:

```python
apply_methods(p_by_scale: dict[int, np.ndarray], A_full, A_tensored, methods, n, target_index) -> dict[method, Estimates]
# Estimates: E_hat, M_hat, P_succ_hat, distribution (or None), E_by_scale, E_richardson, E_linear, E_exp, extras
```

Both tracks call this same function, so the mitigation code is shared.

### 10.1 `none` (baseline)

Uses the base noisy counts at λ=1. E_hat, M_hat and P_succ_hat come from `p_noisy`, and the distribution is `p_noisy`.

### 10.2 `rem`: readout error mitigation, full calibration (**FIXED**)

**Calibration circuits:** for each j in 0..2^n−1, build `QuantumCircuit(n, n)`. Apply `ry(pi)` on every qubit q where `(j >> q) & 1 == 1`, then a barrier, then measure all qubits. Use `ry(pi)` rather than `x` so the circuit stays in the noisy basis. This preparation gate picks up 1-qubit depolarizing noise, which is realistic and should be documented.

Run all 2^n circuits at 1024 shots with seed purpose `"cal"`.

**Assignment matrix:** `A[i, j] = counts_j[i] / shots`, where column j is the prepared state and row i the measured state. Assert that columns sum to 1 (within 1e-12). Save to `results/raw/calibration/full_n{n}_L{L}_{noise}_s{seed}.npy`.

**Inversion** (`apply_rem(p_noisy, A) -> RemResult(quasi, projected, negative_mass, cond)`):

1. Compute `cond = np.linalg.cond(A)`.
2. If cond < 1e8, solve `q = solve(A, p_noisy)`. Otherwise use `lstsq` and log a warning.
3. Record `negative_mass = Σ|q_i| over q_i < 0`.
4. Compute expectations (E, M, P_succ) **from the unprojected q**. This keeps them linear and avoids projection bias. Store both the raw E_hat and the clipped `E_hat_clipped`. The raw value is primary.
5. Compute the distribution as `projected = project_to_simplex(q)`, and use it for Hellinger fidelity and TVD.

**Simplex projection** (Euclidean):

1. Sort v descending into u, and take `css = cumsum(u)`.
2. Find `rho = max{j : u_j − (css_j − 1)/j > 0}`.
3. Set `tau = (css_rho − 1)/rho`.
4. Return `max(v − tau, 0)`.

### 10.3 `rem_tensored`: scalable REM (**FIXED**, proposed improvement 1)

Run 2 calibration circuits: all-|0⟩ (no gates) and all-|1⟩ (`ry(pi)` on every qubit). Use seed purpose `"cal_tensored"`.

For each qubit q, compute marginals from the counts:

- `P_q(0|0)` = fraction of all-0 shots where bit q reads 0.
- `P_q(1|1)` = fraction of all-1 shots where bit q reads 1.

Build `A_q = [[P_q(0|0), 1−P_q(1|1)], [1−P_q(0|0), P_q(1|1)]]` and combine them as `A = kron(A_{n−1}, kron(…, A_0))`. Qubit n−1 is the most significant bit, so A_{n−1} goes leftmost.

Then apply exactly the same inversion and projection as Section 10.2. Save to `results/raw/calibration/tensored_….npy`.

This needs 2 calibration circuits instead of 2^n. It assumes readout errors are uncorrelated across qubits, which holds in our noise model by construction. That favors the tensored method, so state it as a limitation and see Appendix A.4.

### 10.4 `zne`: zero-noise extrapolation (**FIXED**)

**Global unitary folding:** for λ = 2k+1, the folded circuit is U_λ = U(U†U)^k.

```python
def fold_global(u, scale):
    assert scale >= 1 and scale % 2 == 1
    k = (scale - 1) // 2
    folded = u.copy()
    for _ in range(k):
        folded.barrier()
        folded.compose(u.inverse(), inplace=True)
        folded.barrier()
        folded.compose(u, inplace=True)
    return folded
```

`inverse()` maps `ry(θ)` to `ry(−θ)` and `rz(φ)` to `rz(−φ)`, and leaves `cx` as `cx`. Measurements are added after folding and are never folded. Readout error therefore does not scale with λ, so ZNE alone cannot remove readout bias. This motivates `zne_rem`.

**Runs:** λ=1 reuses the base counts. λ=3 and λ=5 each get their own seed. Compute E(λ), M(λ) and P_succ(λ).

**Extrapolators** (`zne.py`), each returning the value at λ=0:

- **`richardson` (primary):** E0 = Σ γ_i E(λ_i), where γ_i = Π_{j≠i} λ_j/(λ_j − λ_i). For λ = (1, 3, 5), γ = (1.875, −1.25, 0.375); they sum to 1, and √Σγ² ≈ 2.285 is the shot-noise std amplification. `richardson_coeffs(scales)` must compute these generally, not hardcode them.
- **`linear`:** least-squares fit of a + bλ; returns a.
- **`exp`:** E(λ) = b·e^(−cλ), fitted by linear regression of ln|E| on λ. It is only valid if every E(λ) has the same sign and |E| > 1e-6; otherwise return NaN. Returns sign·b.

Record E at λ = 1, 3, 5, all three extrapolations, and `extrapolation_out_of_range = |E_richardson| > 1`.

- E_hat is the raw `E_richardson`. Also store `E_hat_clipped`.
- Extrapolate M and P_succ the same way (Richardson).
- The ZNE standard deviation is `zne_std = sqrt(Σ γ_i² (1 − E_i²)/shots)`.

ZNE produces no distribution, so Hellinger fidelity and TVD are NaN for `zne` and `zne_rem`. This is by design, and the report must say so.

### 10.5 `zne_rem`: combined

Apply full REM (same A) to the probability vector at each λ. Compute the quasi-probability expectations, then Richardson-extrapolate (also record linear and exp). Overhead is 3 + 2^n circuits.

### 10.6 Out of scope

The report should give one sentence on each of these as further work:

- **PEC:** exponential sampling cost and needs noise tomography.
- **CDR:** needs training circuits.
- **Dynamical decoupling:** targets idle and coherent errors, which are not in this model.
- **Symmetry verification.**

---

## 11. Track A metrics (`metrics.py`)

**Per-row effectiveness metrics:**

| Metric | Definition | Methods |
|---|---|---|
| `E_hat` | Estimated parity (raw) | all |
| `abs_error` | \|E_hat − E_exact\| | all |
| `signed_error` | E_hat − E_exact | all |
| `rel_error` | abs_error/\|E_exact\| (valid since \|E_exact\| ≥ 0.3) | all |
| `abs_error_none` | abs_error of `none`, same condition | all |
| `improvement_pct` | 100·(1 − abs_error/abs_error_none) | all except none |
| `error_reduction_factor` | abs_error_none/abs_error (inf if abs_error < 1e-12) | all except none |
| `M_hat`, `M_abs_error` | Magnetization estimate and error | all |
| `P_succ_hat`, `P_succ_exact`, `P_succ_ratio` | Success probability, its exact value, and their ratio | all (Richardson for ZNE variants) |
| `hellinger_fidelity` | (Σ_x √(p_x q_x))², estimated vs. exact distribution | none, rem, rem_tensored |
| `tvd` | ½Σ\|p_x − q_x\| | none, rem, rem_tensored |
| `shot_noise_floor` | √((1 − E_exact²)/shots) | all |
| `est_std` | none: √((1 − E_hat²)/shots); zne: zne_std; otherwise NaN (bootstrap is optional) | — |

Implement Hellinger fidelity and TVD manually on full 2^n vectors. Cross-check them against `qiskit.quantum_info.hellinger_fidelity` in a test.

**Resource metrics** (course standard: qubits, depth, gate count, 2-qubit gate count, shots):

| Metric | none | rem | rem_tensored | zne | zne_rem |
|---|---|---|---|---|---|
| `n_qubits` | n | n | n | n | n |
| `n_circuits` | 1 | 1 + 2^n | 3 | 3 | 3 + 2^n |
| `total_shots` | 1024 | 1024(1 + 2^n) | 3072 | 3072 | 1024(3 + 2^n) |
| `base_depth` | d | d | d | d | d |
| `max_depth` | d | d | d | ≈5d | ≈5d |
| `base_cx` (2-qubit gates) | c | c | c | c | c |
| `total_cx` (summed over executions) | c | c | c | 9c | 9c |
| `total_1q`, `total_gates` | computed | computed | computed | computed | computed |
| `time_quantum_s`, `time_classical_s` | measured | measured | measured | measured | measured |

Store the actual values from `circuit_stats` on the circuits that were executed, and test that they match the formulas. Two notes for the report:

- Simulator time does not equal hardware time; circuits and shots are the hardware-relevant cost.
- A REM calibration can be reused across many circuits on a device, so the per-estimate figure is a worst case.

**Aggregation:** per (n, L, noise, method) across the 5 seeds, report mean, std (ddof=1), median, and a 95% CI of mean ± 2.776·std/√5. At the ideal level, improvement_pct is dominated by shot noise. Aggregate it by median, flag it, and exclude both `ideal` and `high` from headline averages; `high` is reported separately.

---

## 12. Track A grid, run procedure, schema

**Grid:** 3 n × 2 L × 4 noise × 5 seeds = **120 conditions**. With 5 methods, `runs.csv` has **600 rows**.

**Circuits per condition:** 1 base + 2 folded + 2^n full calibration + 2 tensored calibration. That is 9 for n=2, 21 for n=4 and 69 for n=6, for about 3,960 noisy executions plus 30 noiseless references in total. Expected runtime is minutes; do not treat this as a requirement.

**Procedure** (`experiment.run_condition`):

```
for n, L, seed:
    inst = select_instance(n, L, seed)          # shared across all noise levels (paired design)
    ref = exact_reference(inst.unitary); save circuit JSON
    E_ideal_shots = parity(run([measured U], "ideal", seed "ideal_shots"))
    for noise in [ideal, low, moderate, high]:
        base  = run([measured U], noise, seed "base")
        f3    = run([measured fold_global(U,3)], noise, seed "fold3")
        f5    = run([measured fold_global(U,5)], noise, seed "fold5")
        A     = assignment matrix from run(calibration_circuits(n), noise, seed "cal")
        A_t   = tensored matrix from run(tensored_calibration_circuits(n), noise, seed "cal_tensored")
        est   = apply_methods({1: base, 3: f3, 5: f5}, A, A_t, methods, n, ref.target_index)
        rows += one row per method; save counts JSON to results/raw/counts/{run_id}.json
        (recommended) E_noisy_exact diagnostic
```

`run_id = f"n{n}_L{L}_{noise}_s{seed}"`.

**CLI:** `python scripts/run_sweep.py --config config/experiment.yaml [--smoke] [--only-n 2 4] [--resume]`

- Shows a tqdm progress bar.
- Logs to `results/logs/sweep.log`.
- Writes rows in a deterministic sort order: n, L, noise order, seed, method order.

**`runs.csv` columns** (exactly this order; NaN where not applicable):

```
run_id, n_qubits, depth_layers, noise_level, p1, p2, p_ro, seed, method, circuit_attempts,
E_exact, E_ideal_shots, E_noisy_exact, E_hat, E_hat_clipped, abs_error, signed_error, rel_error,
abs_error_none, improvement_pct, error_reduction_factor,
M_exact, M_hat, M_abs_error,
P_succ_exact, P_succ_hat, P_succ_ratio,
E_lambda1, E_lambda3, E_lambda5, E_richardson, E_linear, E_exp, extrapolation_out_of_range,
hellinger_fidelity, tvd, rem_negative_mass, rem_condition_number,
est_std, shot_noise_floor,
n_circuits, total_shots, base_depth, max_depth, base_cx, total_cx, total_1q, total_gates,
time_quantum_s, time_classical_s
```

For `zne_rem`, the `E_lambda*` values are REM-corrected. Write floats with `float_format="%.10g"`.

---

## 13. Track B: QML application (Iris binary VQC) (**REQUIRED**)

### 13.1 Dataset

- **Source:** `sklearn.datasets.load_iris()`, the bundled copy of the UCI Iris data (Fisher, 1936). Record this as the dataset source in the report.
- **Binary task:** versicolor (target 1) → y = +1 and virginica (target 2) → y = −1. That is 100 samples (50 per class) with 4 features.
- **Rationale:** setosa vs. rest is linearly separable and trivial. Versicolor and virginica overlap, so noise effects on accuracy are measurable.

### 13.2 Split (**FIXED**)

```python
train_test_split(X, y, test_size=0.2, stratify=y, random_state=derive_seed("qml_split", seed))
```

This gives 80 train samples (40/40) and 20 test samples (10/10).

- Save the indices to `results/qml/splits/s{seed}.json`.
- The identical indices are used by LR, SVM and VQC, at both n=2 and n=4.

### 13.3 Preprocessing: fit on the training data only (**FIXED**)

- **n = 4:** fit `StandardScaler` on X_train, then fit `MinMaxScaler(feature_range=(0, π))` on the scaled training data. Transform the test set and apply `np.clip(…, 0, π)`.
- **n = 2:** fit `StandardScaler` on X_train, then `PCA(n_components=2, random_state=derive_seed("qml_pca", seed))` on the scaled training data, then `MinMaxScaler((0, π))` on the PCA training output. Transform the test set and clip it.

Save every fitted parameter to `results/qml/preprocessing/n{n}_s{seed}.json`:

- scaler `mean_` and `scale_`
- PCA `components_` and `explained_variance_ratio_`
- MinMax `data_min_` and `data_max_`

These files are required for reproducibility.

### 13.4 Feature-to-qubit mapping

Feature k goes to qubit k. The number of qubits equals the number of features (2 after PCA, or 4).

### 13.5 Encoding (explain this in the report)

The encoding is angle encoding: `ry(x_k)` on qubit k, applied to |0⟩. This gives cos(x_k/2)|0⟩ + sin(x_k/2)|1⟩, with ⟨Z_k⟩ = cos x_k before the ansatz.

Mapping features to [0, π] uses half of a Bloch meridian. The minimum feature value maps to |0⟩ and the maximum to |1⟩, with no wrap-around ambiguity.

With a single encoding layer, the model output is a trigonometric polynomial with frequencies {−1, 0, +1} in each feature (Schuld et al., 2021). This limits expressivity, so data re-uploading is listed as future work.

### 13.6 Ansatz and parameters

After the encoding comes the same layered block as Track A (`ry(w)`, `rz(w)`, CX chain), with L ∈ {2, 4} trainable layers. Use `ParameterVector("x", n)` for the features and `ParameterVector("w", 2nL)` for the weights, ordered like Track A's angles.

| n | L | Trainable params | 1-qubit gates (incl. encoding) | CNOTs |
|---|---|---|---|---|
| 2 | 2 | 8 | 10 | 2 |
| 2 | 4 | 16 | 18 | 4 |
| 4 | 2 | 16 | 20 | 6 |
| 4 | 4 | 32 | 36 | 12 |

The encoding gates are `ry`, so they receive noise and are folded in ZNE along with the rest of the circuit. This is correct: the whole bound unitary is folded.

### 13.7 Readout and decision rule

The model output is f(x) = ⟨Z^{⊗n}⟩ ∈ [−1, 1]. The prediction is ŷ = +1 if f ≥ 0, else −1; a tie goes to +1.

### 13.8 Training (`vqc.py`)

- **Training is noiseless and exact.** This is a deliberate choice: it isolates the effect of noise and mitigation at inference. Noise-aware training is future work.
- **Loss:** MSE = (1/N_train)·Σ(f(x_i) − y_i)².
- **Optimizer:** `scipy.optimize.minimize(method="COBYLA", options={"maxiter": max(300, 25·n_params)})`.
- **Initial weights:** `w0 ~ U[0, 2π)` from `default_rng(derive_seed("qml_init", n, L, seed))`.
- **Evaluation:** batched with `qiskit.primitives.StatevectorEstimator`, using a PUB `(circuit_without_measurements, SparsePauliOp("Z"*n), values)`. If that is unavailable, fall back to a `Statevector` loop.
  - **Parameter order pitfall:** `qc.parameters` is sorted by name, so all `w[...]` come before `x[...]`. Build value arrays in `list(qc.parameters)` order, or bind with a dict, and assert the order.
- **Logging:** inside the objective, log every loss value as the loss history.
- **Saving:** write the final weights, loss history, iteration count, train accuracy (exact) and training time to `results/qml/models/n{n}_L{L}_s{seed}.json`.
- **Caching:** skip training if a model file exists and its stored config hash matches; `--retrain` forces retraining.

### 13.9 Classical baselines (`classical.py`)

- `LogisticRegression(C=1.0, max_iter=1000, random_state=derive_seed("qml_lr", n, seed))`
- `SVC(kernel="rbf", C=1.0, gamma="scale", random_state=derive_seed("qml_svm", n, seed))`

**Inputs:** exactly the same preprocessed features as the VQC at that n (2 PCA features or 4 features), and the same split.

The hyperparameters above are fixed, with no tuning on the test set. Optionally, 5-fold CV on the training data only (Appendix A).

Classical results do not depend on L or noise. They have one row per (model, n, seed).

### 13.10 Noisy evaluation with mitigation (`evaluate.py`)

For each (n, L, seed):

1. Load the trained weights. For each of the 20 test samples, bind x and w to get the unitary U_x (no measurements).
2. Compute the exact reference from the statevector: E_exact(x), the exact prediction, and the exact distribution. This produces the `noise_level = "exact"` row, with `method = "none"`.
3. For each noise level in [ideal, low, moderate, high]:
   - **base:** `with_measurements(U_x)` for all 20 samples in one executor call, with seed purpose `"base"`.
   - **fold3, fold5:** `with_measurements(fold_global(U_x, 3 or 5))` for all samples, with seed purposes `"fold3"` and `"fold5"`.
   - **Calibration:** run `calibration_circuits(n)` once and share it across all 20 samples (seed `"cal"`). This demonstrates amortization.
   - **Estimates:** for each sample, call `apply_methods(...)` to get E_hat per method, then predictions.
   - **Metrics:** compute them per method (13.11).
   - **Predictions:** save per-sample predictions to `results/qml/predictions/n{n}_L{L}_{noise}_s{seed}.csv`, with columns: sample_idx, y_true, E_exact, and E_hat_{method} and yhat_{method} for each method.

Total cost is about 5,600 circuit executions. Expected runtime is minutes.

**Per-sample overhead** is reported two ways:

- Unamortized: same as Track A.
- Amortized: the calibration cost divided by 20 test samples (rem: 1 + 2^n/20).

### 13.11 Track B metrics

Compute these with `sklearn.metrics`, using `pos_label=+1` and `zero_division=0`.

**Classification metrics:**

- accuracy, precision, recall, F1
- the confusion matrix, from `confusion_matrix(y_true, y_pred, labels=[+1, −1])`. This returns `[[tp, fn], [fp, tn]]`. Extract the four counts explicitly and store them as columns.

**Quantum-side metrics** (VQC rows only):

- `mean_abs_E_error`: mean over the test set of |E_hat − E_exact|.
- `margin_retention`: mean(E_hat·sign(E_exact)) / mean(|E_exact|). This measures how much noise shrinks the decision margins; 1 means no shrinkage.
- `mean_hellinger`: Hellinger fidelity against the exact distribution, for none and rem only.
- `agreement_with_exact`: the fraction of test predictions equal to the exact-VQC predictions.

**Resource metrics:** n_qubits, n_params, base_depth, base_cx, base_1q, n_circuits_per_sample (both unamortized and amortized), and total_shots_eval.

### 13.12 Track B procedure, schema, outputs

**CLI:** `python scripts/run_qml.py --config config/experiment.yaml [--smoke] [--retrain]`

The script runs these stages in order:

1. Data, split and preprocessing.
2. Classical baselines.
3. VQC training (cached).
4. Exact evaluation.
5. Noisy evaluation.

**`results/qml/qml_runs.csv` columns:**

```
model, n_qubits, depth_layers, noise_level, seed, method, n_train, n_test, n_params,
accuracy, precision, recall, f1, tp, fn, fp, tn,
mean_abs_E_error, margin_retention, mean_hellinger, agreement_with_exact,
train_accuracy_exact, train_loss_final, train_iterations, train_time_s,
n_circuits_per_sample, n_circuits_per_sample_amortized, total_shots_eval,
base_depth, base_cx, base_1q, time_eval_s
```

| Model | Rows | Columns that are "na" |
|---|---|---|
| logreg, svm | 2 n × 5 seeds = 20 | depth_layers, noise_level, method, and the quantum columns |
| vqc exact | 2 × 2 × 5 = 20 | — |
| vqc noisy | 2 n × 2 L × 4 noise × 5 seeds × 4 methods = 320 | — |
| **Total** | **360** | |

### 13.13 Fairness rules (state these in the report)

- LR, SVM and VQC use the same split, the same preprocessed features and the same test set.
- No model sees the test data during fitting.
- No hyperparameter tuning is done on the test set.
- Classical models are compared against the exact VQC and against the noisy VQC.
- Any accuracy differences are reported descriptively, with seed variability. No claim of quantum advantage is made.

---

## 14. Proposed improvement experiments (rubric: "research gap and proposed improvement")

### 14.1 Improvement 1: tensored REM (runs inside the Track A grid)

- **Motivation:** full REM needs 2^n calibration circuits, which is exponential and becomes infeasible beyond about 10–15 qubits.
- **Proposal:** use per-qubit tensored calibration, which needs only 2 circuits.
- **Evidence to produce:**
  - abs_error, Hellinger fidelity and P_succ_ratio for rem_tensored vs. rem, paired by condition.
  - Overhead of 3 vs. 1 + 2^n circuits.
  - Figure A12 and table `table_rem_full_vs_tensored.md`.
- **Expected:** near-identical accuracy under our uncorrelated readout model. Verify this; do not assume it.
- **Limitation to state:** the model is uncorrelated by construction, which favors the tensored method. Appendix A.4 optionally tests correlated readout.

### 14.2 Improvement 2: variance-optimal shot allocation for ZNE (`improvement.py`)

- **Motivation:** Richardson extrapolation amplifies shot noise. Uniform allocation (1024 shots per scale) is not optimal for a fixed total budget.
- **Theory:** minimize Var(E0) = Σ γ_i² σ_i²/N_i subject to Σ N_i = N. The optimum is N_i ∝ |γ_i|·σ_i.
  - Assuming σ_i ≈ 1 (a prior-free choice, so no pilot runs are needed), N_i ∝ |γ_i| = (1.875, 1.25, 0.375)/3.5.
  - For N = 3072 that is about (1646, 1097, 329) shots for λ = (1, 3, 5).
  - Theoretical variance: uniform gives Σγ²/1024 = 5.21875/1024 ≈ 0.00510, and optimal gives (Σ|γ|)²/3072 = 12.25/3072 ≈ 0.00399.
  - That is about 21.8% lower variance, or about 11.5% lower std, at the same total shots.
- **Implementation:**
  - `optimal_allocation(gammas, sigmas, total) -> list[int]`, using largest-remainder rounding so the counts sum exactly to `total`, with a minimum of 1 per scale.
  - Optional variant: a pilot that spends 10% of the budget estimating σ_i.
- **Protocol:**
  - Conditions: n ∈ {2, 4, 6}, L ∈ {2, 4}, noise ∈ {low, moderate}, seeds 0–4. Use the same circuit instances as Track A.
  - Run R = 50 repetitions per condition. Each repetition runs λ = 1, 3, 5 under both uniform (1024 each) and optimal allocations, with distinct seeds `derive_seed("alloc", …, rep, scheme, scale)`.
  - Compute the Richardson E0 for both schemes.
- **Outputs:**
  - `results/improvement/zne_allocation.csv`, one row per (condition, scheme), with columns: mean_E0, bias, empirical_std, rmse, theoretical_std, total_shots.
  - Figure I1: empirical std, uniform vs. optimal, per condition, with the theoretical ratio line.
  - Table `table_zne_allocation.md`.
- **Deviation note (state it in the report):** this experiment deliberately departs from the course rule of 1024 shots per execution, while keeping the total budget equal (3 × 1024). It is reported separately from the main grid.

### 14.3 Future directions (text only, justified in the report)

- Noise-aware training of the VQC.
- Data re-uploading encoding.
- M3 (matrix-free measurement mitigation; Nation et al., 2021) for larger n.
- PEC on small circuits.
- Adaptive selection of the mitigation method from calibration-estimated noise and shot budget.
- Validation on real IBM hardware.

---

## 15. Analysis outputs (`analysis.py`, `scripts/analyze.py`)

```
python scripts/analyze.py --config config/experiment.yaml
```

All outputs go to `results/summary/`.

**Track A:**

1. **`summary_track_a.csv`:** grouped by (n, L, noise, method), giving mean, std, median and CI95 of abs_error, signed_error, improvement_pct, error_reduction_factor, hellinger_fidelity, tvd, P_succ_ratio, M_abs_error and est_std, plus the overhead columns and timing means. That is 3×2×4×5 = **120 rows**.
2. **`table_error.md`:** rows (n, L); columns noise × method, mean ± std abs_error to 3 decimals, with the best method per noise level in bold. High noise goes in a separate block.
3. **`table_fidelity.md`:** Hellinger fidelity, TVD and P_succ_ratio for none, rem and rem_tensored.
4. **`table_overhead.md`:** method × n, showing n_circuits, total_shots, max_depth/base_depth and total_cx/base_cx.
5. **`table_resources.md`** (course resource metrics for both tracks): qubits, depth, gate count, 2-qubit gate count and shots per estimate, per method and per (n, L).
6. **`table_extrapolators.md`:** abs_error for Richardson, linear and exp, the number of out-of-range estimates and the number of exp NaNs, per noise level.
7. **`table_rem_full_vs_tensored.md`.**

**Track B:**

8. **`summary_track_b.csv`:** grouped by (model, n, L, noise, method), giving mean, std and CI95 of accuracy, precision, recall, F1, mean_abs_E_error, margin_retention, mean_hellinger and agreement_with_exact. That is 4 classical + 4 exact + 64 noisy = **72 rows**.
9. **`table_qml_classification.md`:** for each (n, L), rows LR, SVM, VQC exact, and VQC at each noise level × method; columns accuracy, precision, recall and F1 (mean ± std).
10. **`table_qml_confusion.md`:** confusion matrices summed over seeds for the condition set in the config.

**Improvement:**

11. **`table_zne_allocation.md`.**

**Statistics:**

12. **`stats_tests.csv`:**
    - **Track A, per condition:** `ttest_rel` on abs_error, method vs. none, across 5 seeds.
    - **Track A, pooled:** Wilcoxon signed-rank over the 30 pairs per noise level (3 n × 2 L × 5 seeds), per method.
    - **Track B:** `ttest_rel` on accuracy and on mean_abs_E_error, method vs. none, per (n, L, noise). Also VQC exact vs. LR and vs. SVM per n, paired by seed since the split is shared.
    - Report t, p and the mean difference throughout.
    - **Note for the report:** with n = 5, the minimum two-sided Wilcoxon p is 0.0625, so per-condition Wilcoxon tests cannot reach 0.05. Small-sample results are therefore descriptive.

---

## 16. Figures, analysis questions, hypotheses

### 16.1 Figures (`plotting.py`, `scripts/make_plots.py`, `scripts/make_design_figures.py`)

General rules:

- matplotlib only.
- Fixed method colors: none = gray, rem = blue, rem_tensored = light blue, zne = orange, zne_rem = green. LR and SVM are black dashed and black dotted lines.
- 200 dpi PNG.
- Every axis labeled, every figure titled, and no legend covering data.

**Design figures** (rubric: circuit diagram). Use `qc.draw("mpl")`, which requires pylatexenc.

| ID | File | Content |
|---|---|---|
| D1 | `fig_D1_trackA_circuit.png` | Track A ansatz, n=4, L=2, with measurements |
| D2 | `fig_D2_folded_circuit.png` | n=2, L=1 unitary folded at λ=3, showing U, U†, U and barriers |
| D3 | `fig_D3_calibration_circuits.png` | All 4 full-calibration circuits for n=2, plus the 2 tensored ones |
| D4 | `fig_D4_vqc_circuit.png` | VQC, n=4, L=2, with parameter labels x[k] and w[k], encoding block marked |
| D5 | `fig_D5_mitigation_pipeline.png` | Flow diagram drawn with matplotlib patches: circuit → noise → {base, folded, calibration} runs → {none, REM, ZNE, ZNE+REM} → metrics |

**Track A:**

| ID | File | Content |
|---|---|---|
| A1 | `fig_A1_error_vs_noise.png` | Grid: rows L, columns n. X: noise level. Grouped bars by method (mean abs_error ± std). Dashed line for the shot-noise floor. |
| A2 | `fig_A2_fidelity_before_after.png` | Same grid, Hellinger fidelity for none / rem / rem_tensored |
| A3 | `fig_A3_success_probability.png` | Same grid, P_succ_ratio for all methods |
| A4 | `fig_A4_error_vs_qubits.png` | Moderate noise. X: n. Lines per method. One panel per L. |
| A5 | `fig_A5_zne_extrapolation_example.png` | Example condition. E(λ) points, raw and REM-corrected. Richardson, linear and exp curves on λ ∈ [0, 5.5]. Stars at λ=0. E_exact line. |
| A6 | `fig_A6_overhead_circuits.png` | X: n. Y: circuits per estimate (log2). Lines per method. |
| A7 | `fig_A7_depth_overhead.png` | Transpiled depth at λ = 1, 3, 5 per (n, L) |
| A8 | `fig_A8_distributions_n{n}.png` | Moderate noise, L=4, seed 0. Exact vs. none vs. rem vs. rem_tensored bars. Top 16 states for n=6. |
| A9 | `fig_A9_improvement_heatmap.png` | Panels: noise ∈ {low, moderate, high} × method ∈ {rem, rem_tensored, zne, zne_rem}. Each an n×L heatmap of median improvement %, annotated. |
| A10 | `fig_A10_cost_benefit.png` | X: total_shots (log). Y: mean error_reduction_factor. Marker by method, color by noise, labeled by n. |
| A11 | `fig_A11_bias_variance.png` | Box plots of signed_error per method at moderate noise |
| A12 | `fig_A12_rem_full_vs_tensored.png` | Scatter of abs_error, full vs. tensored, with a y=x line, colored by n |
| A13 | `fig_A13_noisy_exact_vs_sampled.png` | Recommended: E_noisy_exact vs. E_hat(none) |

**Track B:**

| ID | File | Content |
|---|---|---|
| B1 | `fig_B1_qml_accuracy_vs_noise.png` | Panels n×L. X: noise (exact, ideal, low, moderate, high). Lines per method. LR and SVM as horizontal lines. |
| B2 | `fig_B2_qml_f1_vs_noise.png` | Same layout, F1 |
| B3 | `fig_B3_qml_confusion_matrices.png` | Summed over seeds for the configured (n, L): LR, SVM, VQC exact, VQC moderate none, VQC moderate zne_rem, VQC high none, VQC high zne_rem |
| B4 | `fig_B4_qml_expectation_error.png` | mean_abs_E_error vs. noise per method |
| B5 | `fig_B5_qml_margin_scatter.png` | E_exact vs. E_hat per test sample, moderate and high noise, none vs. zne_rem. y=x line and decision-boundary lines at 0. |
| B6 | `fig_B6_qml_training_curves.png` | Loss vs. objective evaluations, one line per seed, panels n×L |

**Improvement:**

| ID | File | Content |
|---|---|---|
| I1 | `fig_I1_zne_allocation.png` | Empirical std, uniform vs. optimal, per condition, with the theoretical 0.885 ratio reference |

### 16.2 Questions the report must answer, with evidence

1. **RQ1:** Does each method reduce error at the low and moderate noise levels? By what median improvement %? What happens at high noise?
2. How does effectiveness change with n and L?
3. Which error source dominates, readout or gate noise? Compare rem, zne and zne_rem.
4. **RQ2:** What does each method cost (circuits, shots, depth, 2-qubit gates, time)? Does full REM scale as 2^n?
5. What happens at the ideal noise level? REM should be unchanged, since A = I. ZNE should be slightly worse, since it amplifies shot noise.
6. Does ZNE increase variance by about the predicted factor of 2.3?
7. Do the extrapolators differ? How often is the result out of range, and how often does the exp fit fail? When does ZNE break down (high noise, n=6, L=4)?
8. How large is the REM negative quasi-probability mass?
9. Does REM improve distribution fidelity and success probability, not just expectation values?
10. **RQ3:** Does mitigation improve VQC accuracy, F1 and expectation error? Why might accuracy change less than expectation error does (look at margin retention in B5)?
11. How does the VQC compare with LR and SVM on identical inputs? Answer descriptively, with no advantage claims.
12. **RQ4:** Does tensored REM match full REM at a fraction of the cost? Does optimal allocation reduce ZNE variance as predicted?
13. What are the limitations? See Section 19, report §7.

### 16.3 Hypotheses for Track A (verify; do not assume)

- **H1:** Unmitigated error grows with noise level, n and L.
- **H2:** REM removes most of the readout contribution and improves fidelity and success probability. It leaves the gate-noise bias.
- **H3:** ZNE removes much of the gate-noise bias but not the readout bias, because measurements are not folded.
- **H4:** zne_rem has the lowest error at low and moderate noise. At high noise, ZNE degrades: E(5) approaches 0 and extrapolation becomes unstable.
- **H5:** ZNE has higher variance. At the ideal level it is no better than none.
- **H6 (overhead):** REM uses 1 + 2^n circuits, tensored REM 3, ZNE 3 (with 9× total CX and about 5× max depth), and zne_rem 3 + 2^n.

### 16.4 Hypotheses for Track B and the improvements

- **H7:** Exact VQC accuracy is comparable to or below LR and SVM. No advantage is expected or claimed.
- **H8:** Noise shrinks the margins (margin_retention < 1) while mostly preserving sign. As a result, accuracy degrades less than expectation error does, and mitigation restores margins more visibly than accuracy, except at high noise and near the decision boundary.
- **H9:** Tensored REM matches full REM within shot noise.
- **H10:** Optimal allocation reduces the ZNE std by about 11% at equal budget.

---

## 17. Implementation phases and acceptance criteria

**Phase 1: Scaffolding.**

- **Build:** the repository, `pyproject.toml`, requirements, `config.py`, `seeds.py`, `.gitignore`, the README skeleton, `DECISIONS.md` (with the noise-interpretation entry), and `report/screenshots/README.md` (Section 20).
- **Tests:**
  - `test_seeds`: determinism, distinctness, range.
  - `test_config`: valid configs load, invalid ones raise.
- **Acceptance:** `pip install -e .` succeeds, `import qem` works, and `pytest -q` passes.

**Phase 2: Circuits and observables.**

- **Tests:**
  - Gate counts match the table in 7.1.
  - The same seed gives the same angles, and a different seed gives different angles.
  - `select_instance` returns |E_exact| ≥ 0.3.
  - The parity computed from probabilities matches the SparsePauliOp result.
  - **Bit-order test:** `ry(pi)` on qubit 0 only gives key "0…01" with probability 1, ⟨Z_0⟩ = −1, and every other ⟨Z_q⟩ = +1.
  - `counts_to_probvec` handles missing keys, keys with spaces and short keys.
  - Parity is +1 for |00⟩, −1 for |01⟩ and +1 for |11⟩.
  - `success_probability` returns the correct entry.

**Phase 3: Noise and execution.**

- **Tests:**
  - The ideal level returns None.
  - Running the |0…0⟩ circuit at the ideal level gives a single key.
  - **Readout attenuation:** with readout-only noise at p_ro = 0.03 and n = 4, the parity of |0000⟩ at 50,000 shots is within 4σ of 0.94^4.
  - Moderate gate noise lowers |parity| for a nontrivial instance.
  - The executor raises on an `h` gate.
  - The same seed gives identical counts.
  - The noise model values match the table in Section 8 exactly.

**Phase 4: REM, full and tensored.**

- **Tests:**
  - Calibration circuit j yields key `format(j, f"0{n}b")` at the ideal level.
  - A = I exactly at the ideal level, for both full and tensored REM.
  - With readout-only noise and 50,000 calibration shots, A is within 0.01 of the true tensor matrix, for both methods.
  - With readout-only noise, full and tensored REM both recover parity within 4σ, while none does not.
  - The simplex projection returns a valid distribution and leaves a valid one unchanged.
  - Matrix columns sum to 1.
  - The kron ordering is verified by a 2-qubit asymmetric example.

**Phase 5: ZNE.**

- **Tests:**
  - The folded statevector equals the original (fidelity > 1 − 1e-9) at λ = 3 and 5.
  - CX and 1-qubit gate counts scale exactly by λ.
  - The Richardson coefficients are (1.875, −1.25, 0.375) and sum to 1.
  - Richardson is exact on quadratics.
  - The linear extrapolator is exact on lines.
  - The exp extrapolator is exact on exponentials and returns NaN on a sign change.
  - Under gate-only noise at 50,000 shots, the ZNE error is below the none error for a fixed n=4, L=4 instance.
  - `optimal_allocation` sums exactly to the total and gives each scale at least 1 shot; for (1,3,5) with N = 3072 it returns (1646, 1097, 329) within ±1.

**Phase 6: Track A runner and smoke test.**

- **Tests:**
  - The smoke run produces 4 noise levels × 5 methods = **20 rows** in the exact column order.
  - Required columns are non-NaN per method.
  - Two runs are identical (excluding timing columns).
  - At the ideal level, rem and rem_tensored give E_hat equal to none (within 1e-12).
- **Acceptance:** `python scripts/run_sweep.py --smoke` passes.

**Phase 7: Track A full sweep.**

- **Acceptance:**
  - 600 rows.
  - 120 counts files and 30 circuit files.
  - `environment.txt` written.
  - Log free of errors.
- **Spot check (logged, not asserted):** at moderate noise, none's mean abs_error is greater than at low noise.

**Phase 8: Track B data and classical baselines.**

- **Tests:**
  - The split is 80/20 with 40/40 and 10/10 per class.
  - The scaler and MinMax were fitted on 80 samples (`n_samples_seen_ == 80`), and PCA was fitted on training data only. Refitting on train+test must give different parameters, which proves no leakage.
  - LR, SVM and VQC receive identical index arrays and feature matrices.
  - Test features are clipped to [0, π].
  - Preprocessing JSON is written.
- **Logged:** LR and SVM test accuracy per seed.

**Phase 9: Track B VQC training.**

- **Tests:**
  - Parameter counts match the table in 13.6.
  - Bound-circuit statevector parity equals the StatevectorEstimator output (within 1e-9).
  - The parameter-order assertion holds.
  - Training lowers the loss below its initial value.
  - The same seed gives identical weights.
  - The model cache works.
- **Logged:** exact train and test accuracy.

**Phase 10: Track B noisy evaluation.**

- **Tests:**
  - At the ideal level, rem predictions equal none predictions.
  - The confusion-matrix counts sum to 20.
  - The smoke run gives the expected row count.
- **Acceptance:** the full run writes `qml_runs.csv` with **360 rows** and the prediction CSVs.

**Phase 11: Improvement experiment (14.2).**

- **Tests:** the smoke run works, and the CSV schema is correct.
- **Acceptance:** the full run writes `zne_allocation.csv` with 3×2×2×5×2 = **120 rows**.

**Phase 12: Analysis, tables, figures.**

- **Acceptance:**
  - All summary files from Section 15 exist, with 120 and 72 rows in the two summary CSVs.
  - Figures D1–D5, A1–A12 (plus A13 if done), B1–B6 and I1 all exist and are non-empty.
  - Each figure has been visually checked: labels readable, nothing clipped, legends not covering data.

**Phase 13: Literature review, report, evidence, viva prep.**

- **Build:** Sections 18–21.
- **Acceptance:**
  - Every rubric row in 1.5 has its evidence present.
  - Every question in 16.2 is answered with a reference to a figure or table.
  - Every hypothesis is marked supported, partially supported or not supported.
  - `pytest -q` output is saved to `results/logs/pytest_output.txt`.

**Phase 14: OPTIONAL extras (Appendix A).**

---

## 18. Literature review (rubric: 3 marks; at least 5 papers, plus the research gap)

### 18.1 Requirements

- Review **at least 5** papers; the target is 8–10.
- Cover the themes:
  - (a) NISQ and mitigation foundations
  - (b) ZNE
  - (c) readout mitigation
  - (d) fundamental limits of mitigation
  - (e) QML encoding and noise

### 18.2 Candidate papers (bibliographic details verified; the team must read them)

| # | Reference | Theme | Relevance to us |
|---|---|---|---|
| 1 | J. Preskill, "Quantum Computing in the NISQ era and beyond," *Quantum* 2, 79 (2018) | a | Defines the NISQ setting and motivates mitigation |
| 2 | K. Temme, S. Bravyi, J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Phys. Rev. Lett.* 119, 180509 (2017) | a, b | Proposes ZNE (Richardson) and probabilistic error cancellation |
| 3 | Y. Li, S. C. Benjamin, "Efficient variational quantum simulator incorporating active error minimization," *Phys. Rev. X* 7, 021050 (2017) | b | Independent extrapolation-based error minimization |
| 4 | S. Endo, S. C. Benjamin, Y. Li, "Practical quantum error mitigation for near-future applications," *Phys. Rev. X* 8, 031027 (2018) | a | Practical mitigation and its overheads |
| 5 | A. Kandala et al., "Error mitigation extends the computational reach of a noisy quantum processor," *Nature* 567, 491–495 (2019) | b | ZNE demonstrated on superconducting hardware |
| 6 | T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, W. J. Zeng, "Digital zero noise extrapolation for quantum error mitigation," *IEEE QCE* (2020) | b | Unitary/gate folding, which is exactly our noise-scaling method |
| 7 | S. Bravyi, S. Sheldon, A. Kandala, D. C. McKay, J. M. Gambetta, "Mitigating measurement errors in multiqubit experiments," *Phys. Rev. A* 103, 042605 (2021) | c | Scalable readout-mitigation models (tensored/correlated) |
| 8 | P. D. Nation, H. Kang, N. Sundaresan, J. M. Gambetta, "Scalable mitigation of measurement errors on quantum computers," *PRX Quantum* 2, 040326 (2021) | c | M3, a matrix-free REM; our future direction |
| 9 | R. LaRose et al., "Mitiq: A software package for error mitigation on noisy quantum computers," *Quantum* 6, 774 (2022) | b | Reference implementation and benchmarking of ZNE |
| 10 | Y. Kim et al., "Evidence for the utility of quantum computing before fault tolerance," *Nature* 618, 500–505 (2023) | b | Large-scale hardware use of ZNE |
| 11 | R. Takagi, S. Endo, S. Minagawa, M. Gu, "Fundamental limits of quantum error mitigation," *npj Quantum Information* 8, 114 (2022) | d | Lower bounds on mitigation cost |
| 12 | Y. Quek, D. Stilck França, S. Khatri, J. J. Meyer, J. Eisert, "Exponentially tighter bounds on limitations of quantum error mitigation," *Nature Physics* 20 (2024) | d | Mitigation cost can grow exponentially with depth and size |
| 13 | Z. Cai et al., "Quantum error mitigation," *Rev. Mod. Phys.* 95, 045005 (2023) | a–d | Comprehensive review; use it to frame the taxonomy |
| 14 | V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," *Nature* 567, 209–212 (2019) | e | Variational classifier and quantum kernel on hardware |
| 15 | M. Schuld, R. Sweke, J. J. Meyer, "Effect of data encoding on the expressive power of variational quantum-machine-learning models," *Phys. Rev. A* 103, 032430 (2021) | e | Justifies our analysis of angle-encoding expressivity |
| 16 | S. Wang et al., "Noise-induced barren plateaus in variational quantum algorithms," *Nature Communications* 12, 6961 (2021) | e | How noise degrades variational models |
| 17 | J. A. Smolin, J. M. Gambetta, G. Smith, *Phys. Rev. Lett.* 108, 070502 (2012) | c | Projection to valid probabilities (our simplex step) |
| — | R. A. Fisher, "The use of multiple measurements in taxonomic problems," *Annals of Eugenics* 7(2), 179–188 (1936) | dataset | Iris dataset source; cite it but do not count it as a reviewed paper |

### 18.3 Review table format (in report §2)

Columns: Paper | Year | Problem addressed | Method | Setting (simulation or hardware, scale) | Key finding | Limitation / relevance to this project.

Follow the table with 2–3 paragraphs synthesizing the themes.

### 18.4 Research gap

This is a candidate statement. The team must confirm it against what they actually read and adjust it.

> Most prior work evaluates mitigation by its effect on expectation values in physics or chemistry tasks, often at hardware scale. Reported gains are usually expressed as estimator bias rather than in cost-normalized terms. Fewer studies provide small, fully controlled, cost-normalized side-by-side comparisons of REM, ZNE and their combination that also test whether expectation-value gains carry through to downstream QML classification metrics against classical baselines. This project addresses that gap within the scope of a controlled simulation study. It also evaluates two lower-cost variants: tensored REM and variance-optimal ZNE shot allocation.

### 18.5 Rules

1. Each paper must be read by a team member, at minimum the abstract, introduction and results. Write notes in `report/literature_notes.md`.
2. Never attribute a finding that is not in the paper. Cite with DOI or arXiv ID.
3. Claude Code may draft table entries only from text the team pastes in or that Claude Code fetched from the paper's page. Anything unverified gets a `TODO(team): verify` marker.
4. Use a consistent citation style throughout, either IEEE or APS.

---

## 19. Report outline (`report/report.md`), mapped to the rubric

| Report section | Rubric component (marks) | Contents |
|---|---|---|
| Front matter | — | Title: *Effectiveness and Cost of Quantum Error Mitigation: From Noisy Circuits to a Variational Classifier*. Group 20 member names and USNs. 150–200-word abstract. |
| §1 Introduction, problem and objectives | Problem understanding (2) | Problem statement (1.1), objectives (1.2), RQs (1.3), scope and two tracks |
| §2 Literature review | Literature review (3) | Review table (18.3), synthesis, research gap (18.4) |
| §3 Design | Algorithm/circuit design (3) | Track A ansatz (D1, gate-count table). VQC encoding and ansatz (D4, 13.5–13.7, parameter table). Noise models table. REM (D3, assignment matrix, inversion, projection). ZNE (D2, folding, Richardson γ, variance factor). Pipeline (D5). |
| §4 Implementation | Qiskit implementation (4) | Repository structure, key code snippets (folding, REM inversion, noise model, VQC binding), configuration, seeding scheme, software versions, screenshots (Section 20), how to reproduce |
| §5 Experiments and results | Experiments and visualization (3) | Experiment grids for both tracks. Tables from 15. Figures A1–A12, B1–B6 and I1, each with a 2–4 sentence interpretation. High-noise stress results reported separately. |
| §6 Comparative analysis | Comparative analysis (2) | Methods vs. no-mitigation. VQC vs. LR and SVM, framed descriptively under the fairness statement (13.13). Effects of noise, n and L. Resource and overhead effects (A6, A7, A10, `table_resources.md`). Answers to 16.2. Hypothesis verdicts. |
| §7 Limitations, research gap and proposed improvement | Gap and improvement (2) | Limitations list (below). Improvement 1 (tensored REM) and Improvement 2 (shot allocation), each with evidence. Future directions (14.3). |
| §8 Conclusion | — | Direct answers to RQ1–RQ4, and a recommendation on which method to use when |
| References | — | All cited papers plus the Qiskit, Aer and scikit-learn documentation |
| Appendix | — | Environment versions, a sample circuit as QASM, full tables, preprocessing parameters, DECISIONS.md summary |

**Mandatory statements in the report:**

- The fairness statement (13.13).
- "No quantum advantage is claimed."
- "Energy error and approximation ratio are not applicable (no energy-estimation or optimization task)."
- The 14.2 shot-budget deviation note.

**Limitations checklist** (each must be covered, with evidence where possible):

- Mitigation is not correction; it improves average expectation values only.
- **ZNE:**
  - Amplifies variance (about 2.3×).
  - The extrapolation model can mismatch the true noise and go out of range.
  - Assumes noise scales faithfully with folding.
  - Does not remove readout error.
  - Folding makes circuits up to 5× deeper.
  - Breaks down at high noise.
- **REM:**
  - Full calibration needs 2^n circuits.
  - Assumes readout noise is stable and independent of state preparation.
  - Produces negative quasi-probabilities.
  - Does nothing for gate noise.
  - The tensored variant assumes uncorrelated readout, and our model is uncorrelated by construction.
- **Simulation only:** the noise model is incoherent and Markovian, and omits T1/T2, crosstalk, drift, coherent errors and leakage.
- **Statistics:** small n and 5 seeds limit statistical power; the test set has only 20 samples, so accuracy moves in 5% steps.
- **Selection bias:** Track A instances are selected by |E_exact| ≥ 0.3.
- **Training:** the VQC is trained noise-free, and one encoding layer limits expressivity.
- **Runtime:** simulator time does not equal hardware time.

---

## 20. Implementation evidence and screenshots (rubric: "screenshots")

Claude Code saves these text logs automatically:

- `results/logs/pytest_output.txt`, via `pytest -q | tee …`
- `results/logs/sweep.log`
- `results/logs/qml.log`
- `results/logs/improvement.log`
- `results/environment.txt`

`report/screenshots/README.md` lists the screenshots the team must capture manually. Claude Code cannot take them; save each to `report/screenshots/` with its listed name:

1. `01_tests_passing.png`: terminal showing `pytest -q` passing.
2. `02_track_a_sweep.png`: `run_sweep.py` in progress or just completed.
3. `03_track_b_run.png`: `run_qml.py` output, with training logs and accuracies.
4. `04_improvement_run.png`: `run_improvement.py` output.
5. `05_project_tree.png`: the repository structure in the IDE.
6. `06_key_code_folding.png`: `fold_global` in the editor.
7. `07_key_code_rem.png`: `apply_rem` in the editor.
8. `08_key_code_noise.png`: `build_noise_model` in the editor.
9. `09_results_csv.png`: `runs.csv` or a summary table open.
10. `10_circuit_diagram.png`: the D1 or D4 figure open.
11. `11_config.png`: `experiment.yaml` showing the fixed parameters.

---

## 21. Viva preparation (rubric: "response to questions")

`docs/viva_prep.md` holds a list of likely questions, each with an answer of 3–6 sentences. Claude Code drafts the answers after the results exist, citing specific numbers and figures. The team reviews every answer and should be able to explain each one without notes.

1. What is the difference between error mitigation and error correction?
2. Why must global-folding scale factors be odd? What is U(U†U)^k?
3. Why doesn't ZNE remove readout error in your setup?
4. Derive the Richardson coefficients for λ = (1, 3, 5). Why does ZNE increase variance, and where does the factor of about 2.3 come from?
5. Why did you use `optimization_level=0`?
6. Why can REM produce negative probabilities, and what did you do about it?
7. Why does full REM need 2^n circuits? What does tensored REM assume, and when does it fail?
8. What does the depolarizing parameter mean in Qiskit? How did you map the course's "% error" to it?
9. Why did you choose parity as the observable? Why the |E| ≥ 0.3 threshold, and what bias does it introduce?
10. Explain your encoding. Why scale features to [0, π]? What limits expressivity?
11. Why was PCA or scaling fitted only on training data? What is data leakage?
12. Why might accuracy barely change under noise while expectation error grows? (Margins, B5.)
13. Is there any quantum advantage in your results? (No, and explain why.)
14. How did you make the comparison with LR and SVM fair?
15. Are 5 seeds enough? How did you compute the confidence intervals, and what can the tests detect?
16. What happens at high noise, and why does ZNE break down?
17. What is your research gap and your proposed improvement? What did the data show?
18. How would results differ on real hardware?
19. Why not use PEC?
20. What do Hellinger fidelity and TVD measure, and how do they differ?

---

## 22. Suggested team split (4 members)

| Member | Responsibility | Phases | Literature themes |
|---|---|---|---|
| A | Circuits, observables, seeding, Track A runner | 1, 2, 6, 7 | (a) foundations |
| B | Noise, executor, REM (full and tensored), ZNE, shot allocation | 3, 4, 5, 11 | (b) ZNE, (c) readout |
| C | Track B: data, classical baselines, VQC, noisy evaluation | 8, 9, 10 | (e) QML |
| D | Analysis, figures, report assembly, screenshots, viva doc | 12, 13 | (d) limits |

- Every member reads at least 2–3 papers and writes their review-table rows.
- Every member reviews another member's tests.
- Everyone prepares for all viva questions.

---

## 23. Deliverables checklist

- [ ] The repository with the layout from Section 4, where `pytest -q` passes and the output is saved.
- [ ] Track A: `results/raw/runs.csv` (600 rows), plus counts, circuits and calibration files.
- [ ] Track B: `results/qml/qml_runs.csv` (360 rows), plus splits, preprocessing, models and predictions.
- [ ] Improvement: `results/improvement/zne_allocation.csv` (120 rows).
- [ ] `results/summary/`: both summary CSVs, all tables and `stats_tests.csv`.
- [ ] `results/figures/`: D1–D5, A1–A12, B1–B6, I1.
- [ ] `results/environment.txt` and the log files.
- [ ] `report/report.md`, complete per Section 19, with every rubric row's evidence present.
- [ ] `report/literature_notes.md`, with at least 5 papers (target 8–10) read and noted.
- [ ] `report/screenshots/`, with all 11 screenshots.
- [ ] `docs/viva_prep.md`, reviewed by every member.
- [ ] A README with install and reproduce steps.
- [ ] `DECISIONS.md`.

---

## 24. Common pitfalls (check every one)

1. **`optimization_level > 0` on folded circuits:** the folds cancel and ZNE silently does nothing.
2. **Gates outside {ry, rz, cx}:** they run noise-free silently. Keep the executor assertion.
3. **Bit ordering:** qubit q is `key[n-1-q]`, equivalently bit `(index >> q) & 1`.
4. **`measure_all()`:** it creates a second register, which puts spaces in count keys. Use explicit measurement into a single register.
5. **Missing count keys:** always convert counts to full 2^n vectors.
6. **Assignment matrix orientation:** A[i, j] = P(measured i | prepared j). Columns sum to 1.
7. **Kron order for tensored REM:** `kron(A_{n−1}, …, A_0)`.
8. **Expectation values from a projected distribution:** this adds bias. Use the unprojected quasi-probabilities.
9. **One simulator seed for all λ:** this correlates the samples. Use a derived seed per purpose.
10. **ZNE distributions:** ZNE produces none, so report no Hellinger fidelity or TVD for the ZNE methods.
11. **Improvement % at the ideal level:** it is noise-dominated. Use the median, flag it, and keep it out of headlines. Report high noise separately.
12. **Readout error and `save_density_matrix`:** the saved density matrix excludes readout error, so apply A_true afterwards.
13. **Data leakage:** fit the scaler, PCA and MinMax on training data only, and clip the test features.
14. **Mismatched splits:** classical and quantum models must use identical splits and features.
15. **Parameter order:** `qc.parameters` is name-sorted (w before x). Bind with a dict or an asserted order.
16. **Tie rule and positive label:** E = 0 maps to +1, and `pos_label=+1` everywhere.
17. **Confusion-matrix layout:** with `labels=[+1, −1]` the matrix is `[[tp, fn], [fp, tn]]`.
18. **COBYLA callbacks:** behavior varies across SciPy versions, so log the loss inside the objective function instead.
19. **Circuit drawing:** `qc.draw("mpl")` requires pylatexenc.
20. **Hand-typed numbers:** the report uses generated tables only.
21. **Unpinned optional packages:** if mitiq breaks Qiskit, uninstall it.

---

## Appendix A: OPTIONAL extras (Phase 14, in this order)

1. **`E_noisy_exact` diagnostic and figure A13**, if not already done.
2. **Bootstrap std** for rem, rem_tensored and zne_rem: 200 multinomial resamples, with seeds from `derive_seed("boot", run_id, method)`.
3. **Equal-shot-budget baseline, `none_3072`:** unmitigated with 3072 shots. It shows that ZNE's gain is reduced bias and not just more shots.
4. **Correlated-readout stress test for tensored REM:**
   - Add a correlated 2-qubit readout error on qubit pairs (0,1), (2,3) and so on, via `NoiseModel.add_readout_error(ReadoutError(4x4 probabilities), [q0, q1])`. Use correlated readout only; do not mix it with the all-qubit readout error.
   - Compare full REM and tensored REM.
   - If the installed Aer does not support this, skip it and record that in DECISIONS.md.
5. **Mitiq cross-check:** for (n=4, L=4, moderate, seed 0), compare our Richardson result with `mitiq.zne.execute_with_zne(circuit, executor, factory=RichardsonFactory([1,3,5]), scale_noise=fold_global)`. They should agree within about 3× the combined zne_std.
6. **5-fold CV** for LR and SVM hyperparameters, on training data only.
7. **Generalization datasets:** Wine binary (classes 0 vs. 1) and WDBC reduced (`load_breast_cancer`), each with PCA to 2 and 4 components fitted on training data, run through the same Track B pipeline. Use separate output files.
8. **`notebooks/exploration.ipynb`** reproducing A5, A8 and B5 interactively.

---

## Appendix B: Formula quick reference

| Quantity | Formula |
|---|---|
| Parity from probabilities | E = Σ_i (−1)^popcount(i)·p_i |
| ⟨Z_q⟩ | Σ_i (−1)^((i>>q)&1)·p_i |
| Success probability | P_succ = p̂(x*), where x* = argmax p_exact |
| Readout attenuation of parity | (1 − 2p_ro)^n |
| Depolarizing channel (Qiskit) | E(ρ) = (1−λ)ρ + λ·I/2^k |
| Assignment matrix | A[i, j] = P(measure i \| prepare j) |
| REM | q = A⁻¹p̃; E = Σ_i (−1)^popcount(i)·q_i |
| Tensored REM | A = A_{n−1} ⊗ … ⊗ A_0 |
| Global folding | U_λ = U(U†U)^((λ−1)/2), λ odd |
| Richardson coefficients | γ_i = Π_{j≠i} λ_j/(λ_j − λ_i); for (1, 3, 5): (1.875, −1.25, 0.375) |
| ZNE estimate | E0 = Σ γ_i·E(λ_i) |
| ZNE std | √(Σ γ_i²(1 − E_i²)/N) |
| Optimal shot allocation | N_i ∝ \|γ_i\|·σ_i; Var_opt = (Σ\|γ_i\|σ_i)²/N |
| Shot-noise floor | √((1 − E²)/N) |
| Hellinger fidelity | (Σ_x √(p_x q_x))² |
| TVD | ½Σ_x \|p_x − q_x\| |
| Precision / recall / F1 | TP/(TP+FP); TP/(TP+FN); 2PR/(P+R) |
| Margin retention | mean(E_hat·sign(E_exact)) / mean(\|E_exact\|) |
| Improvement % | 100·(1 − err_method/err_none) |
| 95% CI (5 seeds) | mean ± 2.776·std/√5 |
