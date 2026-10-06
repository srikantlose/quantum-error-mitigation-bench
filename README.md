# Group 20: Quantum Circuit Noise and Error Mitigation

Does error mitigation improve the output of a noisy quantum circuit, and what does it cost?
This repository simulates a layered parity circuit with Qiskit Aer under depolarizing and
readout noise, and compares four methods on identical raw data: no mitigation, readout
error mitigation (REM), zero-noise extrapolation (ZNE), and ZNE combined with REM.

The full specification is in [plan.md](plan.md). Deviations from it are logged in
[DECISIONS.md](DECISIONS.md).

## Install

Python 3.11 is used (3.10–3.12 work).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Test

```bash
pytest -q
```
