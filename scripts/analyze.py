"""Aggregate runs.csv, qml_runs.csv and zne_allocation.csv into summary tables.

    python scripts/analyze.py --config config/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qem import analysis
from qem.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--results", default=str(ROOT / "results"), help="results directory (default: results/)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    results = Path(args.results)
    runs_path = results / "raw" / "runs.csv"
    out = results / "summary"
    out.mkdir(parents=True, exist_ok=True)

    noise_order_a = list(cfg.track_a.noise_levels)
    headline = list(cfg.noise.headline_levels)
    methods_a = list(cfg.track_a.methods)

    runs = analysis.load_runs(runs_path)
    summary = analysis.summarize(runs, noise_order_a)
    summary.to_csv(out / "summary_track_a.csv", index=False, float_format="%.10g", lineterminator="\n")
    print("wrote", out / "summary_track_a.csv", f"({len(summary)} rows)")

    write(out / "table_error.md", analysis.table_error(summary, noise_order_a, methods_a))
    write(out / "table_fidelity.md", analysis.table_fidelity(summary, noise_order_a))
    write(out / "table_overhead.md", analysis.table_overhead(summary, methods_a))
    write(out / "table_extrapolators.md", analysis.table_extrapolators(runs, headline + ["high"]))
    write(out / "table_improvement.md", analysis.table_improvement(summary, noise_order_a, methods_a))
    write(out / "table_bias_sources.md", analysis.table_bias_sources(runs, headline + ["high"]))
    write(out / "table_rem_negative_mass.md", analysis.table_rem_negative_mass(runs, noise_order_a))
    write(out / "table_rem_full_vs_tensored.md", analysis.table_rem_full_vs_tensored(runs, headline + ["high"]))

    var = analysis.variance_check(runs, cfg, runs_path.parent)
    var.to_csv(out / "variance_check.csv", index=False, float_format="%.10g", lineterminator="\n")
    vs = analysis.variance_summary(var, noise_order_a)
    write(out / "table_variance.md", analysis.table_variance(vs))

    runs_b = summary_b = None
    qml_path = results / "qml" / "qml_runs.csv"
    if qml_path.exists():
        runs_b = pd.read_csv(qml_path)
        noise_order_b = list(cfg.track_b.noise_levels)
        summary_b = analysis.summarize_qml(runs_b)
        summary_b.to_csv(out / "summary_track_b.csv", index=False, float_format="%.10g", lineterminator="\n")
        print("wrote", out / "summary_track_b.csv", f"({len(summary_b)} rows)")
        write(out / "table_qml_classification.md", analysis.table_qml_classification(summary_b, noise_order_b))
        qc = cfg.plots.qml_confusion_condition
        write(out / "table_qml_confusion.md", analysis.table_qml_confusion(runs_b, qc.n, qc.depth))

    write(out / "table_resources.md", analysis.table_resources(summary, runs_b))

    alloc = None
    alloc_path = results / "improvement" / "zne_allocation.csv"
    if alloc_path.exists():
        alloc = pd.read_csv(alloc_path)
        write(out / "table_zne_allocation.md", analysis.table_zne_allocation(alloc))

    tests = analysis.combined_stats_tests(
        runs, noise_order_a, runs_b, (list(cfg.track_b.noise_levels) if runs_b is not None else [])
    )
    tests.to_csv(out / "stats_tests.csv", index=False, float_format="%.10g", lineterminator="\n")
    print("wrote", out / "stats_tests.csv", f"({len(tests)} rows)")

    numbers = analysis.report_numbers(runs, summary, tests, vs, results, headline, runs_b, summary_b, alloc)
    write(out / "report_numbers.json", json.dumps(numbers, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
