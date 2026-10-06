# Group 20: Quantum Circuit Noise and Error Mitigation

Does error mitigation improve the output of a noisy quantum circuit, and what does it cost?
This repository simulates a layered parity circuit with Qiskit Aer under depolarizing and
readout noise, and compares four methods on identical raw data: no mitigation, readout
error mitigation (REM), zero-noise extrapolation (ZNE), and ZNE combined with REM.

- The findings are in [report/report.md](report/report.md).
- The full specification is in [plan.md](plan.md).
- Deviations from the specification are logged in [DECISIONS.md](DECISIONS.md).

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
python scripts/run_sweep.py --config config/experiment.yaml      # 90 conditions, about 15 s
python scripts/analyze.py --runs results/raw/runs.csv --out results/summary
python scripts/make_plots.py --runs results/raw/runs.csv --summary results/summary/summary.csv --out results/figures
python scripts/build_report.py                                   # renders report/report.md
```

Each step reads only the outputs of the previous ones. Every random number is derived
from a fixed seed, so a rerun reproduces `results/raw/runs.csv` exactly, apart from the
two wall-time columns.

Useful options for `run_sweep.py`:

- `--smoke` runs the tiny configuration (n = 2, L = 2, seed 0) into `results/smoke/`.
- `--only-n 2 4` restricts the qubit counts.
- `--resume` skips conditions already in `runs.csv`.
- `--out DIR` writes somewhere else.

`python scripts/smoke_test.py` runs the smoke sweep and checks its output end to end.

## Dashboard

```bash
python scripts/build_dashboard.py
```

This embeds `results/raw/runs.csv` and `results/summary/report_numbers.json` into
`dashboard/template.html` and writes `dashboard/mitigation-bench.html`, a single
self-contained page. Rebuild it after every sweep and analysis run.

The same build writes a standalone copy to `dashboard/group20-mitigation-bench/index.html`,
which is deployed as a static site to <https://group20-mitigation-bench.vercel.app>.
To publish a new version after rebuilding:

```bash
cd dashboard/group20-mitigation-bench
vercel deploy --prod
```

## Test

```bash
pytest -q
```

## Layout

| Path | Contents |
|---|---|
| `config/` | `experiment.yaml` (full sweep) and `smoke.yaml` |
| `src/qem/` | The library: config, seeds, circuits, observables, noise, execution, mitigation (`readout.py`, `zne.py`), metrics, experiment, analysis, plotting |
| `scripts/` | Command-line entry points listed above |
| `tests/` | Unit and integration tests |
| `results/raw/` | `runs.csv` (360 rows), plus per-condition counts, circuit instances (angles and QASM) and calibration matrices |
| `results/summary/` | `summary.csv`, markdown tables, `stats_tests.csv`, `variance_check.csv`, `report_numbers.json` |
| `results/figures/` | Figures F1–F10 and the circuit diagram |
| `report/` | `report_template.md` (hand-written prose with placeholders) and the rendered `report.md` |

The report template contains no result numbers. `scripts/build_report.py` fills every
number and table from `results/summary/`, so the report always matches the data.
