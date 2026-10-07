# Screenshots checklist (plan.md §20)

These are manual screenshots the team must capture — Claude Code cannot take them. Save each
one into this folder with exactly the filename listed, then reference it from
`report/report.md` §4 ("Implementation") where indicated. The automatically generated text
evidence (test output, sweep logs) is already saved and does not need a screenshot:

- `results/logs/pytest_output.txt` — full `pytest -q` output
- `results/logs/sweep.log` — Track A sweep log
- `results/logs/qml.log` — Track B sweep log
- `results/logs/improvement.log` — shot-allocation sweep log
- `results/environment.txt` — `pip freeze` output
- `results/python_version.txt` — Python version string

## To capture

1. **`01_tests_passing.png`** — a terminal showing `pytest -q` passing (186 tests at the time
   of writing; re-run and confirm the count still matches before capturing). Command:
   `pytest -q`.
2. **`02_track_a_sweep.png`** — `run_sweep.py` in progress or just completed, showing the
   tqdm progress bar and the final "wrote 600 rows to ..." line. Command:
   `python scripts/run_sweep.py --config config/experiment.yaml`.
3. **`03_track_b_run.png`** — `run_qml.py` output, including training progress and the final
   row count. Command: `python scripts/run_qml.py --config config/experiment.yaml`.
4. **`04_improvement_run.png`** — `run_improvement.py` output. Command:
   `python scripts/run_improvement.py --config config/experiment.yaml`.
5. **`05_project_tree.png`** — the repository structure in the IDE's file explorer, showing
   `src/qem/`, `scripts/`, `tests/`, `config/`, `results/`, `report/`, `docs/`.
6. **`06_key_code_folding.png`** — `fold_global` open in the editor
   (`src/qem/circuits.py`).
7. **`07_key_code_rem.png`** — `apply_rem` open in the editor
   (`src/qem/mitigation/readout.py`).
8. **`08_key_code_noise.png`** — `build_noise_model` open in the editor
   (`src/qem/noise.py`).
9. **`09_results_csv.png`** — `results/raw/runs.csv` or a summary table (e.g.
   `results/summary/table_error.md`) open and visible.
10. **`10_circuit_diagram.png`** — `results/figures/fig_D1_trackA_circuit.png` or
    `fig_D4_vqc_circuit.png` open full-size.
11. **`11_config.png`** — `config/experiment.yaml` open, showing the fixed noise levels,
    grid and method lists.

Once captured, the team can either paste these images directly into the submitted report, or
add `![caption](screenshots/NN_name.png)` lines into `report/report_template.md` §4.2 (the
template currently only *references* this checklist in prose, since Claude Code cannot
generate the screenshots itself).
