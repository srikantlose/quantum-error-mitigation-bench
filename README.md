# Group 20: Quantum Circuit Noise and Error Mitigation

Does error mitigation improve the output of a noisy quantum circuit, what does it cost, and
do the gains carry through to a downstream machine-learning task? This repository has two
tracks. **Track A** simulates a layered parity circuit with Qiskit Aer under depolarizing
and readout noise (course-standard levels: ideal / low / moderate / optional high), and
compares five methods on identical raw data: no mitigation, full readout error mitigation
(REM), scalable tensored REM, zero-noise extrapolation (ZNE), and ZNE combined with REM.
**Track B** applies the same four mitigation methods (no tensored REM) to a variational
quantum classifier (VQC) on the Iris binary task, compared against Logistic Regression and
SVM on identical preprocessed features and the identical train/test split. A third
experiment tests a variance-optimal shot allocation for ZNE against the usual uniform split.

- The findings are in [report/report.md](report/report.md).
- The full specification is in [plan.md](plan.md); the superseded v1 spec is kept at
  [plan_v1.md](plan_v1.md) for reference.
- Deviations from the specification are logged in [DECISIONS.md](DECISIONS.md).
- Literature review notes are in [report/literature_notes.md](report/literature_notes.md).
- Viva prep is in [docs/viva_prep.md](docs/viva_prep.md).

## Install

Python 3.11 is used (3.10–3.12 work). No GPU or IBM Quantum account is needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce everything

```bash
python scripts/run_sweep.py --config config/experiment.yaml         # Track A: 120 conditions, ~20 s
python scripts/run_qml.py --config config/experiment.yaml           # Track B: ~8 min (VQC training)
python scripts/run_improvement.py --config config/experiment.yaml   # shot-allocation study, ~2 min
python scripts/analyze.py --config config/experiment.yaml           # all summary tables + report_numbers.json
python scripts/make_plots.py --config config/experiment.yaml        # figures A1-A13, B1-B6, I1
python scripts/make_design_figures.py --config config/experiment.yaml  # circuit diagrams D1-D5
python scripts/build_report.py                                      # renders report/report.md
```

Each step reads only the outputs of the previous ones. Every random number is derived from
a fixed seed, so a rerun reproduces every CSV exactly, apart from the wall-time columns.

Useful options:

- `run_sweep.py --smoke` / `run_qml.py --smoke` / `run_improvement.py --smoke` run the tiny
  `config/smoke.yaml` configuration into `results/smoke/`.
- `run_sweep.py --only-n 2 4` restricts Track A to the given qubit counts.
- `run_sweep.py --resume` skips Track A conditions already in `runs.csv`.
- `run_qml.py --retrain` retrains the VQC even if a cached model matches the config.
- `--out DIR` (all three sweep scripts) writes somewhere else.

`python scripts/smoke_test.py` runs the Track A smoke sweep and checks its output end to end.

## Test

```bash
pytest -q
```

186 tests at the time of writing (`results/logs/pytest_output.txt` has the saved output).

## Dashboard, field guide and study guide

Three interactive pages read the same result files as the report:

- `dashboard/`: results explorer for Track A, Track B and the shot-allocation experiment.
- `guide/learn-qem.html`: field guide with live demos of shots, noise, REM and ZNE.
- `study/study-guide.html`: study guide covering the whole project in 13 modules, with
  quizzes, flashcards, a 24-question mock viva and a cheat sheet. It is a single file and
  opens directly in a browser.

All three are deployed together at https://quantum-error-mitigation-bench.vercel.app
(dashboard at `/`, study guide at `/study/`, field guide at `/guide/`). Rebuild them after
a resweep, then redeploy from the site folder:

```bash
python scripts/build_dashboard.py     # dashboard + site index
python scripts/build_guide.py         # guide/learn-qem.html + site /guide/
python scripts/build_study_guide.py   # study/study-guide.html + site /study/
cd dashboard/quantum-error-mitigation-bench && vercel deploy --prod
```

`python scripts/package_deliverables.py` bundles every deliverable into
`dist/Group20_QEM_Deliverables.zip` (git-ignored).

## Layout

| Path | Contents |
|---|---|
| `config/` | `experiment.yaml` (full grid) and `smoke.yaml` |
| `src/qem/` | config, seeds, circuits, observables, noise, execution, mitigation (`readout.py`, `zne.py`, `pipeline.py`), metrics, `experiment.py` (Track A), `qml/` (Track B: data, classical, vqc, evaluate, runner), `improvement.py` (shot allocation), analysis, plotting |
| `scripts/` | `run_sweep.py`, `run_qml.py`, `run_improvement.py`, `analyze.py`, `make_plots.py`, `make_design_figures.py`, `smoke_test.py`, `build_report.py`, `build_dashboard.py`, `build_guide.py`, `build_study_guide.py`, `package_deliverables.py` |
| `tests/` | Unit and integration tests (`pytest -q`) |
| `results/raw/` | Track A: `runs.csv` (600 rows), counts, circuit instances, full and tensored calibration matrices |
| `results/qml/` | Track B: `qml_runs.csv` (360 rows), splits, preprocessing params, trained models, per-sample predictions |
| `results/improvement/` | `zne_allocation.csv` (120 rows) |
| `results/summary/` | Both summary CSVs, every markdown table, `stats_tests.csv`, `report_numbers.json` |
| `results/figures/` | Figures A1–A13 (Track A), B1–B6 (Track B), D1–D5 (circuit/pipeline diagrams), I1 (shot allocation) |
| `report/` | `report_template.md` (hand-written, placeholders only), the rendered `report.md`, `literature_notes.md`, `screenshots/` |
| `docs/` | `viva_prep.md` |
| `dashboard/`, `guide/`, `study/` | Interactive pages (templates plus built HTML); `dashboard/quantum-error-mitigation-bench/` is the deployed Vercel site |

The report template contains no result numbers. `scripts/build_report.py` fills every
number and table from `results/summary/`, so the report always matches the data.
