"""Run the smoke sweep (n=2, L=2, seed 0) and check the output end to end.

    python scripts/smoke_test.py [--out results/smoke]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qem.config import load_config
from qem.experiment import COLUMNS, REQUIRED_COLUMNS, run_sweep

ROOT = Path(__file__).resolve().parents[1]


def check(df: pd.DataFrame, cfg) -> list[str]:
    problems = []
    e = cfg.experiment
    expected_rows = len(e.qubits) * len(e.depths) * len(e.seeds) * len(e.noise_levels) * len(e.methods)
    if len(df) != expected_rows:
        problems.append(f"expected {expected_rows} rows, got {len(df)}")
    if list(df.columns) != COLUMNS:
        problems.append("column order differs from the runs.csv schema")
    for method, cols in REQUIRED_COLUMNS.items():
        sub = df[df["method"] == method]
        bad = [c for c in cols if sub[c].isna().any()]
        if bad:
            problems.append(f"{method}: NaN in required columns {bad}")
    ideal = df[df["noise_level"] == "ideal"].set_index(["run_id", "method"])["E_hat"]
    for rid in ideal.index.get_level_values(0).unique():
        if abs(ideal[(rid, "rem")] - ideal[(rid, "none")]) > 1e-12:
            problems.append(f"{rid}: REM changed the estimate at the ideal level")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "smoke"))
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / "config" / "smoke.yaml")
    run_sweep(cfg, args.out)
    df = pd.read_csv(Path(args.out) / "raw" / "runs.csv")

    table = df.pivot_table(index="noise_level", columns="method", values="abs_error", sort=False)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print("abs_error by noise level and method:")
        print(table[[m for m in cfg.experiment.methods]])
    print("E_exact:", np.unique(df["E_exact"]))

    problems = check(df, cfg)
    for p in problems:
        print("FAIL:", p)
    print("smoke test", "FAILED" if problems else "passed")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
