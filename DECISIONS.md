# Decisions and deviations

Every resolution of an ambiguity or a conflict with `plan.md`, with the date and a one-line reason.

| Date | Topic | Decision | Reason |
|---|---|---|---|
| 2026-10-06 | Python version | The project venv uses CPython 3.11.15, provisioned with `uv venv --python 3.11 --seed .venv`. | The machine's default Python is 3.13, which is outside the 3.10–3.12 range in §3; `--seed` adds pip so `pip freeze` and `pip install -e .` work as written. |
| 2026-10-06 | `pylatexenc` dependency | Added `pylatexenc` to `requirements.txt`. | Qiskit's `qc.draw("mpl")` (§16.3 circuit figure) raises `MissingOptionalLibraryError` without it. |
| 2026-10-06 | Repository root | The repository root is the project folder itself rather than a `qem-group20/` subfolder. | The folder was empty; the layout in §4 is otherwise followed exactly. |
| 2026-10-06 | Library versions | No pin below the §3 minimums; resolved to qiskit 2.5.2 and qiskit-aer 0.17.2. | A noisy density-matrix run, `save_density_matrix`, `qasm2.dumps`, the filtered `depth`, `hellinger_fidelity` and `draw("mpl")` all work on these versions. |
| 2026-10-06 | Extra config validation | `config.py` also rejects `optimization_level != 0`, non-`global` folding, non-`full` calibration and undefined noise levels. | These are FIXED in the spec; failing loudly is safer than running a silently different experiment. |
| 2026-10-06 | Config tests location | Config load and validation tests live in `tests/test_seeds.py` alongside the seed tests. | Keeps the eight test files listed in §4; Phase 1 groups both checks. |
