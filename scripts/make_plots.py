"""Make every report figure from runs.csv, summary.csv and the raw artifacts.

    python scripts/make_plots.py --runs results/raw/runs.csv --summary results/summary/summary.csv --out results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qem import plotting
from qem.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "raw" / "runs.csv"))
    ap.add_argument("--summary", default=str(ROOT / "results" / "summary" / "summary.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dpi = cfg.plots.dpi
    runs_path = Path(args.runs)
    raw_dir = runs_path.parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(runs_path)
    summary = pd.read_csv(args.summary)

    plotting.apply_style()
    written = [
        plotting.fig_error_vs_noise(summary, runs, out, dpi),
        plotting.fig_fidelity_before_after(summary, runs, out, dpi),
        plotting.fig_error_vs_qubits(summary, out, dpi),
        plotting.fig_zne_extrapolation_example(runs, cfg, out, dpi),
        plotting.fig_overhead_circuits(summary, out, dpi),
        plotting.fig_depth_overhead(plotting.depth_by_scale(cfg, raw_dir), out, dpi),
        *[plotting.fig_distributions(raw_dir, cfg, out, dpi, n) for n in sorted(runs.n_qubits.unique())],
        plotting.fig_improvement_heatmap(summary, out, dpi),
        plotting.fig_cost_benefit(runs, out, dpi),
        plotting.fig_bias_variance(runs, out, dpi),
        plotting.fig_circuit_example(raw_dir, out, dpi),
    ]
    for path in written:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
