"""Make the A-series, B-series and I1 figures from the analysis outputs.

    python scripts/make_plots.py --config config/experiment.yaml
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
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    dpi = cfg.plots.dpi
    results = Path(args.results)
    raw_dir = results / "raw"
    out = results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(raw_dir / "runs.csv")
    summary = pd.read_csv(results / "summary" / "summary_track_a.csv")
    noise_order = list(cfg.track_a.noise_levels)
    headline = [nz for nz in noise_order if nz != "high"]

    plotting.apply_style()
    written = [
        plotting.fig_error_vs_noise(summary, runs, headline, out, dpi),
        plotting.fig_fidelity_before_after(summary, runs, headline, out, dpi),
        plotting.fig_success_probability(summary, runs, headline, out, dpi),
        plotting.fig_error_vs_qubits(summary, out, dpi),
        plotting.fig_zne_extrapolation_example(runs, cfg, out, dpi),
        plotting.fig_overhead_circuits(summary, out, dpi),
        plotting.fig_depth_overhead(plotting.depth_by_scale(cfg, raw_dir), out, dpi),
        *[plotting.fig_distributions(raw_dir, cfg, out, dpi, n) for n in sorted(runs.n_qubits.unique())],
        plotting.fig_improvement_heatmap(summary, out, dpi, noise_levels=[nz for nz in headline if nz != "ideal"]),
        plotting.fig_cost_benefit(runs, out, dpi, noise_levels=[nz for nz in headline if nz != "ideal"]),
        plotting.fig_bias_variance(runs, out, dpi),
        plotting.fig_rem_full_vs_tensored(runs, out, dpi, noise_levels=tuple(nz for nz in noise_order if nz != "ideal")),
        plotting.fig_noisy_exact_vs_sampled(runs, out, dpi),
    ]

    qml_path = results / "qml" / "qml_runs.csv"
    if qml_path.exists():
        runs_b = pd.read_csv(qml_path)
        summary_b_path = results / "summary" / "summary_track_b.csv"
        summary_b = (pd.read_csv(summary_b_path) if summary_b_path.exists()
                    else __import__("qem.analysis", fromlist=["summarize_qml"]).summarize_qml(runs_b))
        qc = cfg.plots.qml_confusion_condition
        written += [
            plotting.fig_qml_accuracy_vs_noise(summary_b, out, dpi, metric="accuracy"),
            plotting.fig_qml_accuracy_vs_noise(summary_b, out, dpi, metric="f1"),
            plotting.fig_qml_confusion_matrices(runs_b, qc.n, qc.depth, out, dpi),
            plotting.fig_qml_expectation_error(summary_b, out, dpi),
            plotting.fig_qml_margin_scatter(results, qc.n, qc.depth, out, dpi),
            plotting.fig_qml_training_curves(results / "qml" / "models", cfg.track_b.qubits,
                                             cfg.track_b.depths, cfg.experiment.seeds, out, dpi),
        ]

    alloc_path = results / "improvement" / "zne_allocation.csv"
    if alloc_path.exists():
        alloc = pd.read_csv(alloc_path)
        written.append(plotting.fig_zne_allocation(alloc, out, dpi))

    for path in written:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
