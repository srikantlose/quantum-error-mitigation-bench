# Group 20 — Quantum Circuit Noise and Error Mitigation: Implementation Plan

This document is the complete specification for the project. It is meant to be handed to Claude Code (or any developer) and implemented end to end. Decisions marked **FIXED** must not be changed without asking the team. Items marked **OPTIONAL** are done only after every required part is complete and its tests pass.

---

## 0. Instructions for Claude Code

1. Read this entire file before writing any code.
2. Implement the phases in the order given in Section 15. After each phase, run that phase's acceptance checks and the full test suite (`pytest -q`). Do not move on while anything fails.
3. Use only the libraries listed in Section 3. Do not use any of the following:
   - `qiskit.ignis` (removed)
   - `qiskit.execute` (removed in Qiskit 1.0)
   - `qiskit.providers.aer` (old import path; use `qiskit_aer`)
   - `CompleteMeasFitter` or any other deprecated measurement-fitter API
4. Never transpile folded circuits with `optimization_level > 0`. Optimization cancels the U†U pairs and silently destroys zero-noise extrapolation.
5. Every random number must come from the seeding helpers in `src/qem/seeds.py`. Running the full sweep twice must produce a byte-identical `results/raw/runs.csv`.
6. Do not hardcode or "expect" numeric results. Section 14.3 lists hypotheses. The report must state what the data actually shows, even when it contradicts a hypothesis.
7. If anything here is ambiguous or conflicts with the installed library versions, stop and report the conflict rather than improvising. Record every resolution in `DECISIONS.md` with the date and a one-line reason.
8. Commit at least once per phase, with messages like `phase-3: noise models and executor`.
9. All tables in the report must be generated from the summary CSVs by code. Never type result numbers by hand.

---

## 1. Project summary

**Research problem:** Does a selected error-mitigation method improve the output quality of a noisy quantum circuit?

**Research objective:** Test simple error-mitigation techniques against a no-mitigation baseline. Measure how effective they are (error and fidelity before and after) and what they cost (extra circuits, shots, depth, and time).

**Experimental tasks**, as given in the brief, and where each is handled:

| # | Task | Where in this plan |
|---|------|--------------------|
| 1 | Select a circuit | Section 7 |
| 2 | Establish a noiseless reference | Sections 7.4 and 11 |
| 3 | Add noise | Section 8 |
| 4 | Apply a suitable error-mitigation technique | Section 10 |
| 5 | Compare results before and after mitigation | Sections 11 to 14 |

**Expected results** required by the brief:

- An error comparison.
- An output distribution analysis.
- The circuit overhead of each method.
- Accuracy and fidelity before and after mitigation.
- A discussion of mitigation's limitations and its effectiveness versus cost.

**Approach in one paragraph:** We build a parameterized, layered expectation-value circuit on 2, 4, and 6 qubits at depths of 2 and 4 layers. We simulate it with Qiskit Aer under three noise levels: ideal, low, and moderate. Each level combines a depolarizing gate-noise model with a readout-error model. Every circuit is run at 1024 shots with 5 seeds. We compare four methods on the same raw data:

- `none`: the baseline, no mitigation.
- `rem`: readout error mitigation using a full calibration matrix.
- `zne`: zero-noise extrapolation using global unitary folding and Richardson extrapolation.
- `zne_rem`: both combined.

We then measure error in the expectation value, distribution fidelity, and overhead.

---

## 2. Specification reconciliation

The two specification sheets differ slightly. The adopted values below are a superset that satisfies both sheets.

| Parameter | Sheet 1 (table) | Sheet 2 (spreadsheet) | Adopted (**FIXED**) |
|---|---|---|---|
| Qubits | 2, 4 | 2, 4, 6 | 2, 4, 6 |
| Circuit depth | not specified | 2, 4 | 2, 4 (number of ansatz layers) |
| Noise levels | no, low, medium | ideal, low, moderate | ideal, low, moderate (medium = moderate) |
| Shots | 1024 | 1024 | 1024 per circuit execution |
| Seeds | not specified | 5 | 5 (seeds 0–4) |
| Dataset / circuit | "select a circuit" | Iris binary or small expectation-value circuit | Expectation-value circuit (core); Iris binary classifier is OPTIONAL (Appendix A) |
| Baseline | before vs after mitigation | no-mitigation baseline | `none` method on identical raw counts |
| Metrics | error comparison, output distribution, circuit overhead | accuracy/fidelity before and after, overhead | Union of both (Section 11) |
| Analysis | limitations of mitigation | effectiveness and cost | Both (Section 14) |

In the core experiment, "accuracy" means estimation accuracy: the absolute error of the expectation value against the exact noiseless value. Classification accuracy appears only in the optional Iris extension.

The team should confirm with the instructor that 6 qubits and 5 seeds are expected. If only 2 and 4 qubits are required, set `qubits: [2, 4]` in the config; no code changes are needed.

---

## 3. Environment

- **Python:** 3.11 (3.10–3.12 are acceptable).
- **Hardware:** Any laptop. No GPU and no IBM Quantum account are needed; everything is simulated locally.

`requirements.txt`:

```
qiskit>=1.2
qiskit-aer>=0.15
numpy
scipy
pandas
matplotlib
pyyaml
tqdm
pytest
```

OPTIONAL extras, in `requirements-optional.txt`:

```
mitiq            # ZNE cross-check only (Phase 10)
scikit-learn     # Iris extension only (Appendix A)
jupyter          # exploration notebook only
```

Set up the environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Use a minimal `pyproject.toml` (setuptools, package `qem` under `src/`) so `import qem` works everywhere.

At the start of every sweep, `run_sweep.py` must write `pip freeze` output to `results/environment.txt` and the Python version to `results/python_version.txt`.

If `mitiq` cannot be installed alongside the installed Qiskit version, skip it. It is only used for an optional cross-check. Record the conflict in `DECISIONS.md`.

---

## 4. Repository layout

```
qem-group20/
├── README.md                     # how to install, run, reproduce
├── DECISIONS.md                  # log of any deviations/resolutions
├── plan.md                       # this file
├── pyproject.toml
├── requirements.txt
├── requirements-optional.txt
├── .gitignore                    # .venv/, __pycache__/, *.pyc, .pytest_cache/
├── config/
│   ├── experiment.yaml           # full sweep
│   └── smoke.yaml                # tiny sweep for quick checks
├── src/qem/
│   ├── __init__.py
│   ├── config.py                 # load + validate YAML into dataclasses
│   ├── seeds.py                  # deterministic seed derivation
│   ├── circuits.py               # ansatz, measurement, folding, calibration circuits, stats
│   ├── observables.py            # counts -> prob vectors, parity, magnetization, exact refs
│   ├── noise.py                  # Aer noise models
│   ├── execution.py              # Executor: transpile + run + timing
│   ├── metrics.py                # error, fidelity, TVD, improvement, shot-noise floor
│   ├── mitigation/
│   │   ├── __init__.py
│   │   ├── readout.py            # REM: assignment matrix, inversion, simplex projection
│   │   └── zne.py                # extrapolators, variance estimate
│   ├── experiment.py             # run one condition, run the sweep
│   ├── analysis.py               # aggregation, CIs, tables
│   └── plotting.py               # all figures
├── scripts/
│   ├── run_sweep.py
│   ├── analyze.py
│   ├── make_plots.py
│   └── smoke_test.py
├── tests/
│   ├── test_seeds.py
│   ├── test_circuits.py
│   ├── test_observables.py
│   ├── test_noise.py
│   ├── test_readout.py
│   ├── test_zne.py
│   ├── test_metrics.py
│   └── test_experiment.py
├── results/
│   ├── environment.txt
│   ├── python_version.txt
│   ├── logs/sweep.log
│   ├── raw/
│   │   ├── runs.csv
│   │   ├── counts/               # one JSON per condition (all counts used)
│   │   ├── circuits/             # angles + QASM per circuit instance
│   │   └── calibration/          # assignment matrices (.npy)
│   ├── summary/
│   │   ├── summary.csv
│   │   ├── table_error.md
│   │   ├── table_fidelity.md
│   │   ├── table_overhead.md
│   │   ├── table_extrapolators.md
│   │   └── stats_tests.csv
│   └── figures/
└── report/
    └── report.md
```

---

## 5. Configuration

`config/experiment.yaml` (**FIXED** values):

```yaml
experiment:
  name: group20_qem
  qubits: [2, 4, 6]
  depths: [2, 4]                 # number of ansatz layers L
  noise_levels: [ideal, low, moderate]
  seeds: [0, 1, 2, 3, 4]
  shots: 1024
  methods: [none, rem, zne, zne_rem]

circuit:
  type: layered_ry_rz_cx
  entanglement: linear           # cx(0,1), cx(1,2), ..., cx(n-2,n-1)
  angle_low: 0.0
  angle_high: 6.283185307179586  # 2*pi
  observable: parity             # primary observable Z⊗Z⊗...⊗Z
  min_abs_exact_expectation: 0.3 # rejection threshold (Section 7.3)
  max_resample_attempts: 20000

noise:
  one_qubit_gates: [ry, rz]
  two_qubit_gates: [cx]
  levels:
    ideal:    {p1: 0.0,   p2: 0.0,  p_ro: 0.0}
    low:      {p1: 0.001, p2: 0.01, p_ro: 0.01}
    moderate: {p1: 0.005, p2: 0.03, p_ro: 0.03}

rem:
  calibration: full              # full | tensored (tensored is OPTIONAL extra)
  calibration_shots: 1024
  cond_threshold: 1.0e8          # above this, use lstsq instead of solve

zne:
  folding: global
  scale_factors: [1, 3, 5]
  extrapolators: [richardson, linear, exp]   # exp is computed but may return NaN
  primary_extrapolator: richardson

simulator:
  method: density_matrix
  transpile_basis: [ry, rz, cx]
  optimization_level: 0

plots:
  example_condition: {n: 4, depth: 4, noise: moderate, seed: 0}
  dpi: 200

output:
  dir: results
```

`config/smoke.yaml` is identical except for:

```yaml
qubits: [2]
depths: [2]
seeds: [0]
```

`config.py` loads the YAML into frozen dataclasses and validates it:

- Every `qubits` value is at least 2.
- Every depth is at least 1.
- Every noise probability lies in [0, 0.5).
- Every scale factor is an odd positive integer, because global folding requires it.
- `scale_factors[0] == 1`.
- `shots > 0`.
- Every listed method is one of `none`, `rem`, `zne`, `zne_rem`.

Validation failures raise `ValueError` with a clear message.

---

## 6. Seeding and reproducibility (**FIXED**)

`src/qem/seeds.py`:

```python
import hashlib

def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed from any sequence of hashable parts."""
    s = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(s).hexdigest()[:8], 16) % (2**31 - 1)
```

How seeds are used:

| Purpose | Seed | Notes |
|---|---|---|
| Circuit angles | `derive_seed("circuit", n, L, seed)` → `np.random.default_rng(...)` | Depends only on (n, L, seed). The same circuit instance is used across all noise levels and methods. |
| Noiseless shot reference | `derive_seed("sim", n, L, "ideal_shots", seed)` | |
| Base noisy run (λ=1) | `derive_seed("sim", n, L, noise, seed, "base")` | Shared by all four methods. |
| Folded λ=3 run | `derive_seed("sim", n, L, noise, seed, "fold3")` | |
| Folded λ=5 run | `derive_seed("sim", n, L, noise, seed, "fold5")` | |
| REM calibration runs | `derive_seed("sim", n, L, noise, seed, "cal")` | |
| Transpiler | `derive_seed("transpile", n, L, noise, seed, purpose)` | Irrelevant at optimization level 0 but set anyway. |

The design is paired by construction. All four methods for a given (n, L, noise, seed) post-process the same base counts, and `zne` and `zne_rem` share the same folded counts. Differences between methods therefore come only from the mitigation, not from different shot samples.

---

## 7. Circuit design

### 7.1 Ansatz (**FIXED**)

The circuit has n qubits and L layers. Each layer ℓ = 0..L−1 contains, in this order:

1. `ry(theta[ℓ, q])` on every qubit q.
2. `rz(phi[ℓ, q])` on every qubit q.
3. A CNOT chain: `cx(0,1)`, `cx(1,2)`, …, `cx(n−2, n−1)`.

All angles are drawn uniformly from [0, 2π) using the circuit RNG. Angles are drawn in a fixed order: all `theta` for layer 0, then all `phi` for layer 0, then layer 1, and so on.

This gives 2·n·L one-qubit gates and (n−1)·L CNOTs per circuit:

| n | L | 1-qubit gates | CNOTs |
|---|---|---|---|
| 2 | 2 | 8 | 2 |
| 2 | 4 | 16 | 4 |
| 4 | 2 | 16 | 6 |
| 4 | 4 | 32 | 12 |
| 6 | 2 | 24 | 10 |
| 6 | 4 | 48 | 20 |

The term "depth" in this project means the number of layers L. We also record the transpiled circuit depth, computed with measurements and barriers excluded:

```python
qc.depth(filter_function=lambda inst: inst.operation.name not in ("barrier", "measure"))
```

### 7.2 Construction functions (`circuits.py`)

```python
build_unitary(n: int, layers: int, rng: np.random.Generator) -> tuple[QuantumCircuit, np.ndarray]
    # returns circuit with NO classical bits and NO measurements, plus angles array shape (L, 2, n)

with_measurements(unitary: QuantumCircuit) -> QuantumCircuit
    # QuantumCircuit(n, n); compose unitary; barrier; measure(range(n), range(n))
    # Single classical register so count keys have no spaces. Do NOT use measure_all().

fold_global(unitary: QuantumCircuit, scale: int) -> QuantumCircuit
    # Section 10.3; returns unitary-only folded circuit (no measurements)

calibration_circuits(n: int) -> list[QuantumCircuit]
    # Section 10.2; 2^n circuits, index j prepares basis state |j>

tensored_calibration_circuits(n: int) -> list[QuantumCircuit]   # OPTIONAL
    # 2 circuits: all-|0>, all-|1>

circuit_stats(qc: QuantumCircuit) -> dict
    # {"depth": ..., "cx": ..., "n1q": ..., "total_gates": ...}
    # Computed on transpiled circuit, excluding barrier and measure.
```

### 7.3 Instance selection by rejection sampling (**FIXED**)

```python
@dataclass
class CircuitInstance:
    n: int
    layers: int
    seed: int
    unitary: QuantumCircuit
    angles: np.ndarray
    attempts: int
    E_exact: float           # exact parity expectation

select_instance(n, layers, seed, threshold, max_attempts) -> CircuitInstance
```

Selection procedure:

1. Create `rng = np.random.default_rng(derive_seed("circuit", n, layers, seed))`.
2. Repeat until `max_attempts` is reached. On each attempt, draw a fresh set of angles from the same RNG stream, build the unitary, and compute the exact parity. If `|E_exact| >= threshold` (0.3), return the instance.
3. If no attempt succeeds, raise `RuntimeError`.

Rationale for the report: a random multi-qubit parity is often close to 0. In that case, noise and mitigation effects are invisible under shot noise (σ ≈ 0.031 at 1024 shots), and relative error is undefined. The threshold guarantees a measurable signal. This is a selection bias, and it must be listed as a limitation.

Save each instance to `results/raw/circuits/n{n}_L{L}_s{seed}.json`. The file contains the angles (nested lists), attempts, E_exact, and the QASM 2 string from `qiskit.qasm2.dumps(with_measurements(unitary))`.

### 7.4 Observables and the noiseless reference (`observables.py`)

**Bit ordering (critical):** Qiskit is little-endian. In a count key `s` of length n, qubit q is the character `s[n-1-q]`. The integer index `i = int(s, 2)` has qubit q at bit `(i >> q) & 1`. `Statevector.probabilities()` uses the same index convention. Every function in this file must work on probability vectors of length 2^n, indexed this way.

```python
counts_to_probvec(counts: dict[str, int], n: int) -> np.ndarray
    # strip spaces, zfill(n), fill missing keys with 0, divide by total shots

parity_signs(n) -> np.ndarray          # s[i] = (-1)^popcount(i), cached
parity_from_probs(p) -> float          # sum_i s[i] * p[i]   (works for quasi-probs too)
z_signs(n, q) -> np.ndarray            # (-1)^((i>>q)&1)
magnetization_from_probs(p, n) -> float  # (1/n) * sum_q <Z_q>

exact_reference(unitary) -> tuple[float, float, np.ndarray]
    # sv = Statevector(unitary); probs = sv.probabilities()
    # E_exact = parity_from_probs(probs)
    # cross-check: sv.expectation_value(SparsePauliOp("Z"*n)).real must match within 1e-9
    # M_exact = magnetization_from_probs(probs, n)
    # return E_exact, M_exact, probs
```

There are two observables:

- **Primary:** the parity P = ⟨Z⊗Z⊗…⊗Z⟩. It is a global observable, so every gate error and every readout bit flip can affect it. It is therefore maximally noise-sensitive, which makes mitigation effects clearly visible. A useful exact fact for tests and the report is that symmetric readout error with flip probability p multiplies the parity by exactly (1−2p)^n.
- **Secondary:** the mean single-qubit magnetization M = (1/n) Σ_q ⟨Z_q⟩.

There are also two noiseless references:

- `E_exact`: the exact value from the statevector, with infinite shots. All errors are measured against this.
- `E_ideal_shots`: a noiseless run at 1024 shots. This shows the irreducible shot-noise floor, and every noisy method should be compared against it.

---

## 8. Noise models (`noise.py`) (**FIXED**)

```python
build_noise_model(level: NoiseLevelConfig, one_q_gates, two_q_gates) -> NoiseModel | None
```

The model is built as follows:

- If all of p1, p2, and p_ro are 0, return `None` (ideal).
- If `p1 > 0`, add `depolarizing_error(p1, 1)` to `one_q_gates` (`["ry", "rz"]`) with `add_all_qubit_quantum_error`.
- If `p2 > 0`, add `depolarizing_error(p2, 2)` to `two_q_gates` (`["cx"]`) with `add_all_qubit_quantum_error`.
- If `p_ro > 0`, add `ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])` with `add_all_qubit_readout_error`. Row k of this matrix gives P(measured | prepared k).

Imports: `from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError`.

Noise levels:

| Level | p1 (1-qubit depolarizing) | p2 (2-qubit depolarizing) | p_ro (readout flip) |
|---|---|---|---|
| ideal | 0 | 0 | 0 |
| low | 0.001 | 0.01 | 0.01 |
| moderate | 0.005 | 0.03 | 0.03 |

Semantics, for the report: Qiskit's `depolarizing_error(λ, k)` implements E(ρ) = (1−λ)ρ + λ·Tr(ρ)·I/2^k. The probability that a non-identity Pauli is applied is λ(4^k−1)/4^k.

Helper:

```python
readout_attenuation(p_ro, n) -> float   # (1 - 2*p_ro)**n
```

**Critical:** Noise is attached by gate name. If a transpiled circuit contains any gate other than `ry`, `rz`, or `cx`, that gate gets no noise, silently. The executor must assert this (Section 9).

Thermal relaxation (T1/T2) and crosstalk are out of scope. Mention them in the limitations.

---

## 9. Execution layer (`execution.py`)

```python
class Executor:
    def __init__(self, cfg): ...
        # build and cache one AerSimulator(method="density_matrix", noise_model=nm) per noise level
    def run(self, circuits: list[QuantumCircuit], noise_level: str, shots: int, seed: int
            ) -> tuple[list[dict[str, int]], float]:
        # 1. tcircs = transpile(circuits, basis_gates=["ry","rz","cx"],
        #                       optimization_level=0, seed_transpiler=<derived>)
        # 2. assert every op name in every tcirc is in {"ry","rz","cx","measure","barrier"}
        # 3. t0 = perf_counter(); result = sim.run(tcircs, shots=shots, seed_simulator=seed).result()
        # 4. counts = [normalize(result.get_counts(i)) for i in range(len(tcircs))]
        # 5. return counts, perf_counter() - t0
```

`normalize` removes spaces from keys and applies `zfill(n)` to each key.

The ideal level uses an `AerSimulator(method="density_matrix")` with no noise model, so every level goes through the same code path.

**OPTIONAL but recommended diagnostic** `exact_noisy_expectation(unitary, noise_level)`. It separates the bias of each method from shot noise:

1. Copy the unitary and call `save_density_matrix()` on it (available after importing `qiskit_aer`).
2. Run it on the density-matrix simulator with the noise model.
3. Take `p_gate = DensityMatrix(...).probabilities()`.

`save_density_matrix` captures gate noise only; readout error applies only at measurement. To include readout, apply the true assignment matrix: A_true = A_{n−1} ⊗ … ⊗ A_0, where each A_q = [[1−p, p], [p, 1−p]]. Then `p_noisy_exact = A_true @ p_gate`, and `E_noisy_exact = parity_from_probs(p_noisy_exact)`. Store this in the column `E_noisy_exact`.

---

## 10. Mitigation methods

All methods operate on probability vectors of length 2^n.

### 10.1 `none` (baseline)

Estimates come straight from the base noisy counts at λ=1: `E_hat = parity_from_probs(p_noisy)`, with the distribution `p_noisy`.

### 10.2 `rem`: readout error mitigation, full calibration (**FIXED**)

**Calibration circuits.** For each j in 0..2^n−1, create a `QuantumCircuit(n, n)`. For each qubit q where `(j >> q) & 1 == 1`, apply `ry(pi)` on qubit q. Use `ry(pi)`, not `x`, so the circuit stays inside the noisy basis and needs no transpile changes. Then add a barrier and measure every qubit q into classical bit q.

Run all 2^n circuits at `calibration_shots` (1024) with the "cal" seed.

The `ry(pi)` preparation gate picks up 1-qubit depolarizing noise (p1 ≤ 0.005). Real calibrations have this same state-preparation error, so keep it and document it.

**Assignment matrix.** `A[i, j] = counts_j[i] / shots`, where column j is the prepared state j and row i is the measured state i. Every column sums to 1; assert this to within 1e-12. Save the matrix to `results/raw/calibration/n{n}_L{L}_{noise}_s{seed}.npy`.

**Inversion** (`apply_rem(p_noisy, A) -> RemResult`):

1. `cond = np.linalg.cond(A)`.
2. If `cond < cond_threshold`, compute `q = np.linalg.solve(A, p_noisy)`. Otherwise use `q = np.linalg.lstsq(A, p_noisy, rcond=None)[0]` and log a warning.
3. `q` is a quasi-probability vector. It sums to 1 but may contain negative entries. Record `negative_mass = sum(|q_i| for q_i < 0)`.
4. **Expectation values come from the unprojected `q`:** `E_hat = parity_from_probs(q)`. This is linear and does not add the bias that projection would. Store both raw and clipped values (`E_hat_clipped = clip(E_hat, -1, 1)`). Raw is the primary value.
5. **The distribution comes from the projected `q`:** `p_rem = project_to_simplex(q)`, which is used for fidelity and TVD.

**Euclidean projection onto the probability simplex** (`project_to_simplex(v)`):

```
u = sort(v, descending)
css = cumsum(u)
rho = max { j in 1..len(v) : u_j - (css_j - 1)/j > 0 }
tau = (css_rho - 1)/rho
return max(v - tau, 0)
```

Properties to test: the output is non-negative and sums to 1, and a valid distribution is returned unchanged.

`RemResult` fields: `quasi`, `projected`, `negative_mass`, `cond`.

**OPTIONAL `rem_tensored` (Phase 10):** Run two calibration circuits, all-|0⟩ and all-|1⟩. From their marginals, build per-qubit matrices `A_q = [[P(0|0), P(0|1)], [P(1|0), P(1|1)]]`. Combine them as `A = kron(A_{n-1}, kron(…, A_0))`. The kron order matters: the most significant bit is qubit n−1, so A_{n−1} goes leftmost. This assumes readout errors are uncorrelated, which holds in our noise model. It needs only 2 circuits instead of 2^n. Report it as the scalable alternative.

### 10.3 `zne`: zero-noise extrapolation (**FIXED**)

**Global unitary folding.** For a scale factor λ = 2k+1, the folded circuit is U_λ = U·(U†·U)^k:

```python
def fold_global(u: QuantumCircuit, scale: int) -> QuantumCircuit:
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

`u.inverse()` turns `ry(θ)` into `ry(−θ)` and `rz(φ)` into `rz(−φ)`, and leaves `cx` as `cx`, so every gate stays in the noisy basis. Measurements are added after folding and are never folded. Because measurements are not folded, readout error does not scale with λ. ZNE alone therefore cannot remove readout bias, which motivates `zne_rem`.

Properties to test:

- The folded circuit's statevector equals the original's (state fidelity > 1 − 1e-9).
- The folded CNOT count is exactly λ × the base CNOT count.
- The folded 1-qubit gate count is exactly λ × the base count.

**Runs.** λ=1 reuses the base counts and is not re-run. λ=3 and λ=5 are run with their own seeds. Compute `E(λ) = parity_from_probs(p_λ)` and likewise `M(λ)`.

**Extrapolators** (`zne.py`), each returning the estimate at λ=0:

- **`richardson` (primary):** Fit the exact polynomial of degree m−1 through all m points. The estimate is E0 = Σ_i γ_i·E(λ_i), where γ_i = Π_{j≠i} λ_j / (λ_j − λ_i). For λ = (1, 3, 5), γ = (15/8, −5/4, 3/8) = (1.875, −1.25, 0.375). These sum to 1, and √(Σγ²) ≈ 2.285 is the approximate factor by which shot-noise standard deviation is amplified.
- **`linear` (secondary):** A least-squares fit E(λ) = a + b·λ, returning a.
- **`exp` (secondary):** Fit E(λ) = b·e^(−cλ) by linear regression of ln|E(λ)| on λ. This form is physically motivated, since depolarizing noise decays a traceless observable toward 0. It is valid only if every E(λ) has the same sign and |E(λ)| > 1e-6; otherwise return NaN. The estimate is `sign·b`.

`richardson_coeffs(scales)` must be general for any distinct scales, not hardcoded.

Record `E_lambda1`, `E_lambda3`, `E_lambda5`, `E_richardson`, `E_linear`, `E_exp`, and `extrapolation_out_of_range = |E_richardson| > 1`. Store `E_hat = E_richardson` (raw, unclipped) and `E_hat_clipped`. Apply the same procedure to M.

**Estimated standard deviation of the ZNE estimate:** `zne_std = sqrt(sum_i γ_i^2 * (1 - E_i^2) / shots)`. This treats each E(λ_i) as an independent ±1-valued average.

ZNE produces no distribution. Distribution metrics (Hellinger fidelity, TVD) for `zne` and `zne_rem` are NaN by design. Say so explicitly in the report.

### 10.4 `zne_rem`: combined

Apply REM (Section 10.2), using the same assignment matrix A built for that condition, to the probability vector at each λ. Compute `E_rem(λ) = parity_from_probs(q_λ)` from the unprojected quasi-probabilities, then Richardson-extrapolate (with linear and exp also recorded). Overhead is 3 noisy circuits plus 2^n calibration circuits.

### 10.5 Out of scope

Probabilistic error cancellation (PEC), Clifford data regression (CDR), dynamical decoupling, and symmetry verification are out of scope. They may be mentioned in the report as further work, with one sentence each on why they were not used. PEC has exponential sampling overhead and needs noise tomography. CDR needs training circuits. Dynamical decoupling targets idle and coherent errors, which this noise model does not include.

### 10.6 OPTIONAL Mitiq cross-check (Phase 10)

For the single condition (n=4, L=4, moderate, seed 0), compare our Richardson result against Mitiq. Use `mitiq.zne.execute_with_zne(circuit, executor, factory=RichardsonFactory([1, 3, 5]), scale_noise=fold_global)`, with an executor that wraps `Executor.run` and returns the parity. The two results should agree within about 3× the combined `zne_std`; exact equality is not expected because the shot seeds differ. Record the outcome in `DECISIONS.md`.

---

## 11. Metrics (`metrics.py`)

**Per-row metrics**, one row per condition and method:

| Metric | Definition | Applies to |
|---|---|---|
| `E_hat` | Estimated parity (raw) | all |
| `abs_error` | \|E_hat − E_exact\| | all |
| `signed_error` | E_hat − E_exact (bias direction) | all |
| `rel_error` | abs_error / \|E_exact\| (valid since \|E_exact\| ≥ 0.3) | all |
| `abs_error_none` | abs_error of `none` for the same condition | all (copied) |
| `improvement_pct` | 100·(1 − abs_error / abs_error_none) | all except `none` |
| `error_reduction_factor` | abs_error_none / abs_error (inf if abs_error < 1e-12) | all except `none` |
| `M_hat`, `M_abs_error` | Same for magnetization | all |
| `hellinger_fidelity` | (Σ_x √(p_x·q_x))² against the exact distribution | `none`, `rem` |
| `tvd` | ½·Σ_x \|p_x − q_x\| against the exact distribution | `none`, `rem` |
| `shot_noise_floor` | √((1 − E_exact²)/shots) | all |
| `est_std` | none: √((1−E_hat²)/shots); zne: zne_std; rem and zne_rem: NaN unless bootstrap enabled | per method |

Implement Hellinger fidelity and TVD manually on full 2^n vectors. The Hellinger definition above matches `qiskit.quantum_info.hellinger_fidelity`; add a unit test that cross-checks the two on count dictionaries.

**Overhead metrics**, per row:

| Metric | none | rem | zne | zne_rem |
|---|---|---|---|---|
| `n_circuits` (executions per estimate) | 1 | 1 + 2^n | 3 | 3 + 2^n |
| `total_shots` | 1024 | 1024·(1 + 2^n) | 3072 | 1024·(3 + 2^n) |
| `base_depth` | d | d | d | d |
| `max_depth` (largest circuit executed) | d | d | ≈5d | ≈5d |
| `base_cx` | c | c | c | c |
| `total_cx` (summed over executed circuits) | c | c | 9c | 9c |
| `total_1q` (summed, including calibration preps) | computed | computed | computed | computed |
| `time_quantum_s` | simulation time of the circuits this method needs | | | |
| `time_classical_s` | post-processing time (inversion, extrapolation) | | | |

The table formulas are expectations. Store the actual values computed by `circuit_stats` on the executed circuits, and add a test asserting they match the formulas.

Notes for the report:

- Simulator time is not representative of hardware time. Circuits and shots are the hardware-relevant overhead.
- In practice, one REM calibration can be reused for many circuits on the same device, so its cost amortizes. The per-estimate number reported here is a worst case.

**Aggregation**, per (n, L, noise, method) over 5 seeds: report mean, std (ddof=1), median, and a 95% confidence interval of mean ± t·std/√5 with t = 2.776 (t_{0.975, df=4}).

When noise is `ideal`, abs_error_none is just shot noise, so `improvement_pct` is unstable. Aggregate it by median, flag it in tables, and exclude the ideal level from headline averages.

---

## 12. Experiment grid and run procedure

**Grid:** n ∈ {2, 4, 6} × L ∈ {2, 4} × noise ∈ {ideal, low, moderate} × seed ∈ {0..4} gives 90 conditions. With 4 method rows per condition, `runs.csv` has **360 rows**.

**Executed circuits per condition:** 1 base, 2 folded, and 2^n calibration circuits. That is 7 for n=2, 19 for n=4, and 67 for n=6. In total that is about 2,790 noisy circuit executions plus 30 noiseless reference runs. Expected runtime is a few minutes on a laptop; do not treat this as a requirement.

**Per-condition procedure** (`experiment.run_condition`):

```
for n in qubits:
  for L in depths:
    for seed in seeds:
      inst = select_instance(n, L, seed, ...)                 # same instance for all noise levels
      E_exact, M_exact, p_exact = exact_reference(inst.unitary)
      save circuit JSON
      ideal_counts = executor.run([with_measurements(inst.unitary)], "ideal", shots, seed("ideal_shots"))
      E_ideal_shots = parity(ideal_counts)
      for noise in noise_levels:
        base   = executor.run([with_measurements(U)], noise, shots, seed("base"))
        f3, f5 = executor.run([with_measurements(fold_global(U,3)),
                               with_measurements(fold_global(U,5))], noise, shots, seed("fold"))
                 # NOTE: run fold3 and fold5 as separate calls with seeds "fold3" and "fold5"
        cal    = executor.run(calibration_circuits(n), noise, cal_shots, seed("cal"))
        A      = build_assignment_matrix(cal, n, cal_shots)
        -> compute rows for none, rem, zne, zne_rem
        -> (optional) E_noisy_exact diagnostic
        -> save all counts for this condition to results/raw/counts/{run_id}.json
```

The circuit instance is shared across noise levels for paired comparison. Calibration is redone for every (n, L, noise, seed), which keeps the overhead accounting honest and the samples independent.

The run ID for a condition is `run_id = f"n{n}_L{L}_{noise}_s{seed}"`.

**CLI** (`scripts/run_sweep.py`):

```
python scripts/run_sweep.py --config config/experiment.yaml [--smoke] [--only-n 2 4] [--resume]
```

- `--smoke` loads `config/smoke.yaml`.
- `--resume` skips run IDs already present in `runs.csv`.
- A tqdm progress bar runs over conditions.
- Logs go to `results/logs/sweep.log` with timestamps, per-condition timing, cond(A) warnings, and out-of-range extrapolation warnings.
- Rows are written sorted by (n, L, noise order, seed, method order), so the CSV is deterministic.

**`runs.csv` schema.** Use exactly this column order. Fields that do not apply to a method are left empty (NaN).

```
run_id, n_qubits, depth_layers, noise_level, p1, p2, p_ro, seed, method, circuit_attempts,
E_exact, E_ideal_shots, E_noisy_exact, E_hat, E_hat_clipped, abs_error, signed_error, rel_error,
abs_error_none, improvement_pct, error_reduction_factor,
M_exact, M_hat, M_abs_error,
E_lambda1, E_lambda3, E_lambda5, E_richardson, E_linear, E_exp, extrapolation_out_of_range,
hellinger_fidelity, tvd,
rem_negative_mass, rem_condition_number,
est_std, shot_noise_floor,
n_circuits, total_shots, base_depth, max_depth, base_cx, total_cx, total_1q,
time_quantum_s, time_classical_s
```

For `zne_rem`, the `E_lambda*` columns hold the REM-corrected values.

Floats are written with `float_format="%.10g"`. Timing columns are excluded from the byte-identical reproducibility check, because wall time varies between runs. The test compares all other columns.

---

## 13. Analysis outputs (`analysis.py`, `scripts/analyze.py`)

```
python scripts/analyze.py --runs results/raw/runs.csv --out results/summary
```

Files produced:

1. **`summary.csv`:** grouped by (n_qubits, depth_layers, noise_level, method). For each of abs_error, signed_error, improvement_pct, error_reduction_factor, hellinger_fidelity, tvd, M_abs_error, and est_std, it gives mean, std, median, ci95_low, and ci95_high. It also includes the overhead columns (constant per group, so take the first value) and the mean of each timing column. That is 3×2×3×4 = **72 rows**.
2. **`table_error.md`:** rows (n, L); columns are noise × method with mean ± std of abs_error to 3 decimals; the best method per noise level is bolded.
3. **`table_fidelity.md`:** rows (n, L); columns are noise × {none, rem} Hellinger fidelity and TVD, mean ± std.
4. **`table_overhead.md`:** rows are methods; columns per n give n_circuits, total_shots, max_depth / base_depth, and total_cx / base_cx.
5. **`table_extrapolators.md`:** at low and moderate noise, the mean abs_error for Richardson, linear, and exp (zne and zne_rem), plus the count of out-of-range Richardson estimates and NaN exp fits.
6. **`stats_tests.csv`:** paired comparisons of each method against `none`.
   - **Per condition:** `scipy.stats.ttest_rel` on abs_error across the 5 seeds. Report t, p, and the mean difference. Note in the report that a Wilcoxon signed-rank test with n=5 has a minimum two-sided p of 0.0625, so it cannot reach 0.05 per condition. That is why a paired t-test is used per condition.
   - **Pooled:** a Wilcoxon signed-rank test over all 30 pairs per noise level (3 n × 2 L × 5 seeds), one test per method and per non-ideal noise level.

---

## 14. Figures, analysis questions, and hypotheses

### 14.1 Figures (`plotting.py`, `scripts/make_plots.py`)

```
python scripts/make_plots.py --runs results/raw/runs.csv --summary results/summary/summary.csv --out results/figures
```

Figure rules:

- Use matplotlib only.
- Each method keeps the same color in every figure: none = gray, rem = blue, zne = orange, zne_rem = green.
- Save PNGs at 200 dpi.
- Every axis is labeled, every figure has a title, and legends never cover data.

| ID | File | Content |
|---|---|---|
| F1 | `fig_error_vs_noise.png` | Grid of 2 rows (L=2, 4) × 3 columns (n=2, 4, 6). X axis: noise level. Grouped bars by method, showing mean abs_error with std error bars. A dashed horizontal line marks the mean shot-noise floor. |
| F2 | `fig_fidelity_before_after.png` | Same grid. Bars for none vs rem Hellinger fidelity, with std error bars. |
| F3 | `fig_error_vs_qubits.png` | Moderate noise. X axis: n. One line per method, showing mean abs_error with error bars. One panel per L. |
| F4 | `fig_zne_extrapolation_example.png` | For the example condition from the config: scatter of E(λ) at λ = 1, 3, 5 for raw and REM-corrected values. Curves on λ ∈ [0, 5.5] for the Richardson polynomial, the linear fit, and the exp fit. Stars at λ=0 for each extrapolated value. A horizontal line for E_exact. |
| F5 | `fig_overhead_circuits.png` | X axis: n. Y axis: circuits per estimate, on a log2 scale. One line per method. Shows the 2^n growth of full REM. If the tensored variant was run, add `rem_tensored` as a dashed line. |
| F6 | `fig_depth_overhead.png` | Grouped bars of transpiled depth at λ = 1, 3, 5 for each (n, L). |
| F7 | `fig_distributions_n{n}.png` | One file each for n = 2, 4, 6, at moderate noise, L=4, seed 0. Bars for exact, noisy (none), and rem distributions over basis states. For n=6, show only the 16 states with the highest exact probability. |
| F8 | `fig_improvement_heatmap.png` | Panels for noise ∈ {low, moderate} × method ∈ {rem, zne, zne_rem}. Each panel is an n × L heatmap of median improvement_pct, with the values annotated. |
| F9 | `fig_cost_benefit.png` | Scatter. X axis: total_shots (log). Y axis: mean error_reduction_factor. Marker shape by method, color by noise (low, moderate), and point label n. |
| F10 | `fig_bias_variance.png` | Box plots of signed_error per method at moderate noise, pooled over n, L, and seeds. Shows bias (offset from 0) against variance (spread). |
| F11 | `fig_noisy_exact_vs_sampled.png` (OPTIONAL) | E_noisy_exact against E_hat for `none`, separating the noise bias from shot noise. |

### 14.2 Questions the report must answer explicitly, with evidence

1. Does each method reduce error at the low and moderate noise levels? By how much (median improvement %)?
2. How does effectiveness change with the number of qubits and with depth?
3. Which error source dominates: readout or gate noise? Compare `rem`, `zne`, and `zne_rem`.
4. What does each method cost in circuits, shots, depth, and time? Does full REM grow as 2^n in practice?
5. What happens at the ideal noise level? This is the sanity check: REM should leave the estimate unchanged because A = I, and ZNE should, if anything, slightly increase the error because it amplifies shot noise.
6. Does ZNE increase the variance? Is the increase roughly the predicted factor of √5.22 ≈ 2.3 in standard deviation?
7. Do the extrapolators differ? How often does Richardson go out of range, and how often does the exp fit fail?
8. Are REM quasi-probabilities negative, and by how much (negative mass by n and noise level)?
9. Does REM improve distribution fidelity (Hellinger and TVD), not just the expectation value?
10. What are the limitations? See Section 16.5.

### 14.3 Hypotheses (verify; do not assume)

- **H1:** Unmitigated error grows with noise level, n, and L.
- **H2:** REM removes most of the readout contribution. It improves Hellinger fidelity but leaves the gate-noise bias.
- **H3:** ZNE removes much of the gate-noise bias but not the readout bias, because measurements are not folded.
- **H4:** `zne_rem` has the lowest absolute error at low and moderate noise.
- **H5:** ZNE has a higher variance than `none`. At the ideal level, ZNE is no better than `none` and is possibly worse.
- **H6:** In overhead, REM uses 1 + 2^n circuits, ZNE uses 3 circuits with 9× total CNOTs and about 5× maximum depth, and `zne_rem` uses the sum of both.

---

## 15. Implementation phases and acceptance criteria

### Phase 1: Scaffolding

Build the repository structure, `pyproject.toml`, the requirements files, `config.py`, `seeds.py`, `.gitignore`, a README skeleton, and an empty `DECISIONS.md`.

Tests:

- `test_seeds.py`: `derive_seed` is deterministic, different parts give different seeds, and outputs lie in [0, 2^31−1).
- Loading the config succeeds, and invalid configs raise errors.

Acceptance: `pip install -e .` works, `python -c "import qem"` works, and `pytest -q` passes.

### Phase 2: Circuits, observables, exact reference

Implement `circuits.py`, except the calibration and folding functions (those are written in Phases 4 and 5), and `observables.py`.

Tests in `test_circuits.py` and `test_observables.py`:

- The gate counts match the table in Section 7.1 for all six (n, L) combinations.
- The same (n, L, seed) produces identical angles, and a different seed produces different angles.
- `select_instance` returns |E_exact| ≥ 0.3, and attempts ≥ 1.
- The `exact_reference` parity from probabilities matches the `SparsePauliOp` expectation within 1e-9.
- **Bit-order test:** a circuit with `ry(pi)` on qubit 0 only, run noiselessly, gives the key `"0…01"` with probability 1, and `<Z_0> = −1` while every other `<Z_q> = +1`.
- `counts_to_probvec` handles missing keys, keys with spaces, and short keys.
- Parity from counts for |00⟩ is +1, for |01⟩ is −1, and for |11⟩ is +1.

Acceptance: all tests pass.

### Phase 3: Noise and execution

Implement `noise.py` and `execution.py`.

Tests in `test_noise.py`:

- The ideal level returns `None`.
- An ideal run of the basis state |0…0⟩ gives exactly one key.
- **Readout attenuation:** with a readout-only noise model (p_ro = 0.03, n = 4), the measured parity of |0000⟩ at 50,000 shots is within 4σ of (1−0.06)^4.
- Moderate gate noise reduces |parity| for a nontrivial instance compared with ideal (fixed seed, 20,000 shots).
- The executor's operation assertion fails if a circuit contains an `h` gate, which proves the guard works.
- The same seed gives identical counts.

Acceptance: all tests pass.

### Phase 4: REM

Implement `calibration_circuits`, `build_assignment_matrix`, `apply_rem`, and `project_to_simplex`.

Tests in `test_readout.py`:

- Calibration circuit j produces key `format(j, f"0{n}b")` under ideal noise.
- Under ideal noise, A is exactly the identity.
- Under readout-only noise with 50,000 calibration shots, A is within 0.01 (elementwise) of the true tensor-product matrix.
- With readout-only noise, REM recovers the parity of a fixed instance to within 4 × the shot-noise σ of E_exact, while `none` does not.
- The simplex projection outputs a valid distribution and leaves a valid input unchanged (within 1e-12).
- The columns of A sum to 1.

Acceptance: all tests pass.

### Phase 5: ZNE

Implement `fold_global` and `zne.py`.

Tests in `test_zne.py`:

- The folded statevector matches the original (state fidelity > 1 − 1e-9) for λ = 3, 5.
- CNOT and 1-qubit gate counts scale exactly by λ.
- The Richardson coefficients for (1, 3, 5) equal (1.875, −1.25, 0.375) and sum to 1.
- Richardson recovers E(0) of any quadratic sampled at 1, 3, 5 exactly (within 1e-12).
- The linear extrapolator recovers an exact line.
- The exp extrapolator recovers an exact exponential and returns NaN on a sign change.
- **Statistical test:** under gate-only depolarizing noise (p_ro = 0) at 50,000 shots with a fixed seed, the ZNE abs_error is smaller than the `none` abs_error for one fixed instance with n=4 and L=4.

Acceptance: all tests pass.

### Phase 6: Sweep runner and smoke test

Implement `experiment.py`, `scripts/run_sweep.py`, and `scripts/smoke_test.py`.

Tests in `test_experiment.py`:

- The smoke run produces 3 noise levels × 4 methods = 12 rows with the exact column order from Section 12.
- The required columns are non-NaN for each method.
- Running twice gives identical CSVs, excluding the timing columns.
- At the ideal noise level, `rem` has E_hat equal to `none`'s E_hat within 1e-12, because A = I.

Acceptance: `python scripts/run_sweep.py --smoke` completes and the tests pass.

### Phase 7: Full sweep

Run `python scripts/run_sweep.py --config config/experiment.yaml`.

Acceptance:

- `runs.csv` has 360 rows, and no required column is NaN for its method.
- `environment.txt` exists.
- 90 count files and 30 circuit files exist.
- The log has no errors.

Spot-check, logged and not asserted: the mean abs_error of `none` at moderate noise is greater than at low noise.

### Phase 8: Analysis and figures

Implement `analysis.py` and `plotting.py`, then run both scripts.

Acceptance:

- `summary.csv` has 72 rows.
- All five summary markdown or CSV tables exist.
- Figures F1–F10 exist and are non-empty.
- Every figure is opened and visually checked: labels readable, nothing clipped, legends not covering data.

### Phase 9: Report

Write `report/report.md` following Section 16. Embed the figures with relative paths and include the tables generated in `results/summary/`.

Acceptance:

- Every question in Section 14.2 is answered with a reference to a figure or table.
- Every hypothesis in Section 14.3 is marked supported, partially supported, or not supported, with evidence.

### Phase 10: OPTIONAL extras (only after Phases 1–9 are complete)

Do these in the order listed, each behind a config flag so the core results do not change:

1. `rem_tensored` as a fifth method, with F5 updated.
2. The `E_noisy_exact` diagnostic and F11, if not already done.
3. Bootstrap standard deviations for `rem` and `zne_rem`: 200 multinomial resamples of the counts, seeded with `derive_seed("boot", run_id, method)`.
4. An equal-shot-budget baseline, `none_3072`: an unmitigated run at 3072 shots. It shows that ZNE's gain is reduced bias, not just more shots, because extra shots reduce variance but not bias.
5. The Mitiq cross-check (Section 10.6).
6. The Iris extension (Appendix A).
7. `notebooks/exploration.ipynb` reproducing F4 and F7 interactively.

---

## 16. Report outline (`report/report.md`)

### 16.1 Front matter

- Title: *Effectiveness and Cost of Error Mitigation on Noisy Quantum Circuits*
- Group 20, with member names and USNs
- A 150–200 word abstract

### 16.2 Introduction and background

- **NISQ noise:** gate errors and readout errors.
- **Mitigation vs. correction:** error mitigation reduces bias in expectation values through extra circuits and classical post-processing. It does not correct individual shots and needs no extra qubits.
- **Depolarizing channel:** the formula, and what it means.
- **Readout error model:** the assignment matrix.
- **REM theory:** inverting the assignment matrix, quasi-probabilities, and simplex projection.
- **ZNE theory:** noise scaling by folding, and Richardson extrapolation, including the γ coefficients and the variance amplification factor.

### 16.3 Methodology

- The circuit, with a diagram from `qc.draw("mpl")` for n=4, L=2, saved as `fig_circuit_example.png`.
- The gate-count table.
- The rejection-sampling rule.
- The noise level table.
- The four methods.
- The metric definitions.
- The grid.
- Seeds and pairing.
- The software versions, taken from `environment.txt`.

### 16.4 Results

- Tables from `results/summary/` and figures F1–F10, each with a 2–4 sentence interpretation.
- The noiseless sanity check.
- The overhead results.

### 16.5 Discussion

- Effectiveness vs. cost: the trade-off, shown in F9.
- Scaling with n and L.
- Bias vs. variance, shown in F10.
- **Limitations.** Each must be covered, with evidence where possible:
  - Mitigation is not correction. It improves average expectation values only, and individual shots are still noisy.
  - **ZNE limitations:**
    - It amplifies variance (theoretical factor about 2.3).
    - Its extrapolation model can mismatch the true noise and produce out-of-range estimates.
    - It assumes noise scales faithfully with folding, which holds for our incoherent model but is less clean on hardware.
    - It does not remove readout error.
    - The deepest folded circuit is 5× deeper, which on real hardware can exceed coherence times.
  - **REM limitations:**
    - Full calibration costs 2^n circuits.
    - It assumes readout noise is stable over time and independent of the prepared state.
    - It can produce negative quasi-probabilities.
    - It does nothing about gate noise.
    - The tensored variant assumes uncorrelated readout errors.
  - **Simulation-only:** The depolarizing-plus-readout model is incoherent and Markovian. Real devices also have coherent errors, crosstalk, T1/T2 relaxation, drift, and leakage.
  - **Statistics:** Small qubit counts and only 5 seeds limit statistical power.
  - **Selection bias:** Rejection sampling requires |E_exact| ≥ 0.3, so circuit instances are not uniformly random.
  - **Time:** Simulator wall time is not hardware execution time.

### 16.6 Conclusion and references

- **Conclusion:** a direct answer to the research problem, plus a recommendation on which method to use and when.
- **Further work:** PEC, CDR, dynamical decoupling, real hardware runs, larger n.

References:

- K. Temme, S. Bravyi, J. M. Gambetta, "Error mitigation for short-depth quantum circuits," Phys. Rev. Lett. 119, 180509 (2017).
- A. Kandala et al., "Error mitigation extends the computational reach of a noisy quantum processor," Nature 567, 491–495 (2019).
- T. Giurgica-Tiron et al., "Digital zero noise extrapolation for quantum error mitigation," IEEE QCE (2020).
- R. LaRose et al., "Mitiq: A software package for error mitigation on noisy quantum computers," Quantum 6, 774 (2022).
- S. Bravyi et al., "Mitigating measurement errors in multiqubit experiments," Phys. Rev. A 103, 042605 (2021).
- J. A. Smolin, J. M. Gambetta, G. Smith, "Efficient method for computing the maximum-likelihood quantum state from measurements with additive Gaussian noise," Phys. Rev. Lett. 108, 070502 (2012).
- Z. Cai et al., "Quantum error mitigation," Rev. Mod. Phys. 95, 045005 (2023).
- Qiskit and Qiskit Aer documentation.

### 16.7 Appendix

- Environment versions.
- The QASM of one circuit instance.
- The full summary tables.
- How to reproduce:
  ```
  pip install -r requirements.txt && pip install -e .
  python scripts/run_sweep.py --config config/experiment.yaml
  python scripts/analyze.py --runs results/raw/runs.csv --out results/summary
  python scripts/make_plots.py --runs results/raw/runs.csv --summary results/summary/summary.csv --out results/figures
  ```

---

## 17. Suggested team split (4 members)

| Member | Responsibility | Phases |
|---|---|---|
| A | Circuits, observables, seeding, and their tests | 1, 2 |
| B | Noise models, executor, REM | 3, 4 |
| C | ZNE folding, extrapolators, ZNE section of the report | 5 |
| D | Sweep runner, analysis, plots, report assembly | 6, 7, 8 |

Everyone writes the discussion and limitations (Phase 9). Each member reviews another member's tests.

---

## 18. Deliverables checklist

- [ ] A repository with the layout from Section 4, where `pytest -q` passes.
- [ ] `results/raw/runs.csv` with 360 rows, plus the raw counts, circuits, and calibration matrices.
- [ ] `results/summary/` containing `summary.csv` and all tables.
- [ ] `results/figures/` containing F1–F10.
- [ ] `results/environment.txt`.
- [ ] `report/report.md`, with every Section 14.2 question answered and every Section 14.3 hypothesis evaluated.
- [ ] A README with installation and reproduction steps.
- [ ] `DECISIONS.md` listing any deviations from this plan.

---

## 19. Common pitfalls (check every one)

1. **`optimization_level > 0` on folded circuits.** The folds cancel and ZNE silently does nothing. Always use level 0 with barriers between folds.
2. **Gates outside the noise basis.** Noise is applied only to `ry`, `rz`, and `cx`. Any other gate name runs noise-free with no warning, so keep the executor assertion.
3. **Bit ordering.** Qiskit is little-endian. Qubit q is `key[n-1-q]` and `(index >> q) & 1`.
4. **`measure_all()`.** It adds a second classical register named `meas`, which puts spaces in count keys. Use explicit `measure(range(n), range(n))` into a single register.
5. **Missing count keys.** Count dictionaries omit outcomes with zero counts. Always convert to full 2^n vectors.
6. **Matrix orientation.** A[i, j] is P(measured i | prepared j): columns are prepared states, and each column sums to 1.
7. **Kron ordering** for the tensored matrix: `kron(A_{n-1}, …, A_0)`.
8. **Expectation from the projected distribution.** Projection adds bias. Compute expectations from the unprojected quasi-probabilities and use the projected vector only for distribution metrics.
9. **Correlated samples across scale factors.** Reusing one simulator seed for every λ correlates the samples. Use a distinct derived seed per purpose.
10. **ZNE distributions.** ZNE produces no distribution. Do not report Hellinger fidelity or TVD for `zne` or `zne_rem`.
11. **Improvement % at the ideal level.** It is noise-dominated. Report it as a median, flag it, and exclude it from headline numbers.
12. **`save_density_matrix` and readout error.** The saved density matrix excludes readout error. Apply A_true afterwards for the noisy-exact diagnostic.
13. **Clipping ZNE estimates.** Use raw ZNE estimates for error metrics. Report out-of-range counts separately.
14. **Hand-typed numbers in the report.** Every number must come from the generated tables.
15. **Unpinned optional packages.** If Mitiq downgrades or breaks Qiskit, uninstall it and skip the cross-check.

---

## Appendix A: OPTIONAL Iris binary classification extension

Do this only after Phases 1–9 are complete. It covers the "Iris binary" option on sheet 2 and gives a classification-accuracy view of mitigation.

**Data**

- Load with `sklearn.datasets.load_iris()`.
- Keep classes 1 (versicolor) and 2 (virginica), 100 samples in total. Setosa is excluded because it is linearly separable, which makes the task trivial.
- Labels: y = +1 for versicolor and −1 for virginica.
- Use a stratified 70/30 train/test split with `random_state = derive_seed("iris_split", seed)`.

**Preprocessing**

- Fit `StandardScaler` on the training set.
- Then fit `MinMaxScaler(feature_range=(0, π))` on the training set, and clip the scaled test values to [0, π].

**Feature-to-qubit mapping**

| n | Features |
|---|---|
| 2 | 2 PCA components (PCA fitted on the training set, then min-max scaled as above) |
| 4 | The 4 features |
| 6 | Features [f0, f1, f2, f3, f0, f1] |

**Circuit**

- Encode with `ry(x_q)` on qubit q.
- Follow with the same layered ansatz (Section 7.1), with L ∈ {2, 4} and trainable angles.
- The readout observable is parity, for consistency with the core experiment.
- Prediction: `sign(E)`.

**Training**

- Train on exact statevectors (noiseless).
- Loss: mean squared error between E(x) and y over the training set.
- Optimizer: `scipy.optimize.minimize(method="COBYLA", options={"maxiter": 300})`.
- Initial angles drawn from `default_rng(derive_seed("iris_init", n, L, seed))`.

**Evaluation**

- For each test sample, and for each noise level and method, estimate E at 1024 shots.
- Classification accuracy is the mean of `sign(E_hat) == y`.
- Also report the mean |E_hat − E_exact| and agreement with the noiseless predictions.

**Overhead note:** One REM calibration per (n, noise, seed) is shared across all test samples. This demonstrates amortization: report the per-sample overhead with and without sharing.

**Expected discussion point (verify, do not assume):** Depolarizing noise mostly shrinks E toward 0 without flipping its sign. Classification accuracy may therefore change little with or without mitigation, even when the expectation-value error changes a lot. Accuracy is a coarse metric for judging mitigation, while error and fidelity are more sensitive.

**Outputs**

- `results/iris/runs.csv`
- `results/figures/fig_iris_accuracy.png`: accuracy against noise level, by method, for each n.
- A short "Iris extension" subsection in the report.

---

## Appendix B: Formula quick reference

| Quantity | Formula |
|---|---|
| Parity from probabilities | E = Σ_i (−1)^popcount(i) · p_i |
| ⟨Z_q⟩ from probabilities | Σ_i (−1)^((i>>q)&1) · p_i |
| Readout attenuation of parity | (1 − 2p_ro)^n |
| Depolarizing channel (Qiskit) | E(ρ) = (1−λ)ρ + λ·I/2^k |
| Assignment matrix | A[i, j] = P(measure i \| prepare j) |
| REM | q = A⁻¹·p̃; E = Σ_i (−1)^popcount(i)·q_i |
| Global folding | U_λ = U·(U†U)^((λ−1)/2), λ odd |
| Richardson coefficients | γ_i = Π_{j≠i} λ_j / (λ_j − λ_i); for (1, 3, 5): (1.875, −1.25, 0.375) |
| ZNE estimate | E0 = Σ_i γ_i·E(λ_i) |
| ZNE std estimate | √(Σ_i γ_i²·(1 − E_i²)/N) |
| Shot-noise floor | √((1 − E²)/N) |
| Hellinger fidelity | (Σ_x √(p_x·q_x))² |
| Total variation distance | ½·Σ_x \|p_x − q_x\| |
| Improvement % | 100·(1 − err_method / err_none) |
| 95% CI (5 seeds) | mean ± 2.776·std/√5 |
