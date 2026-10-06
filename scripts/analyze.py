"""Aggregate runs.csv into summary.csv, the markdown tables and the statistical tests.

    python scripts/analyze.py --runs results/raw/runs.csv --out results/summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qem import analysis
from qem.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "raw" / "runs.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "summary"))
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    runs_path = Path(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    noise_order = list(cfg.experiment.noise_levels)
    methods = list(cfg.experiment.methods)

    runs = analysis.load_runs(runs_path)
    summary = analysis.summarize(runs, noise_order)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.10g", lineterminator="\n")
    print("wrote", out / "summary.csv", f"({len(summary)} rows)")

    write(out / "table_error.md", analysis.table_error(summary, noise_order, methods))
    write(out / "table_fidelity.md", analysis.table_fidelity(summary, noise_order))
    write(out / "table_overhead.md", analysis.table_overhead(summary, methods))
    write(out / "table_extrapolators.md", analysis.table_extrapolators(runs))
    write(out / "table_improvement.md", analysis.table_improvement(summary, noise_order, methods))
    write(out / "table_bias_sources.md", analysis.table_bias_sources(runs, noise_order))
    write(out / "table_rem_negative_mass.md", analysis.table_rem_negative_mass(runs, noise_order))

    tests = analysis.stats_tests(runs, noise_order)
    tests.to_csv(out / "stats_tests.csv", index=False, float_format="%.10g", lineterminator="\n")
    print("wrote", out / "stats_tests.csv")

    var = analysis.variance_check(runs, cfg, runs_path.parent)
    var.to_csv(out / "variance_check.csv", index=False, float_format="%.10g", lineterminator="\n")
    vs = analysis.variance_summary(var, noise_order)
    write(out / "table_variance.md", analysis.table_variance(vs))

    numbers = analysis.report_numbers(runs, summary, tests, vs, runs_path.parent.parent)
    write(out / "report_numbers.json", json.dumps(numbers, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
