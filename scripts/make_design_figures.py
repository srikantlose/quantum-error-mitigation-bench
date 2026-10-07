"""Make the circuit-design figures D1-D5 (rubric: algorithm/circuit design).

    python scripts/make_design_figures.py --config config/experiment.yaml --out results/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qem.circuits import (  # noqa: E402
    build_unitary,
    calibration_circuits,
    fold_global,
    load_instance,
    tensored_calibration_circuits,
    with_measurements,
)
from qem.config import load_config  # noqa: E402
from qem.plotting import COLORS  # noqa: E402
from qem.qml.vqc import build_vqc_circuit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def save(fig, path: Path, dpi: int) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


def d1_track_a_circuit(raw_dir: Path, out: Path, dpi: int) -> None:
    path = raw_dir / "circuits" / "n4_L2_s0.json"
    inst = load_instance(path) if path.exists() else None
    if inst is None:
        _, angles = build_unitary(4, 2, np.random.default_rng(0))
        from qem.circuits import build_unitary_from_angles

        unitary = build_unitary_from_angles(angles)
    else:
        unitary = inst.unitary
    qc = with_measurements(unitary)
    fig = qc.draw("mpl", fold=-1, style={"name": "bw"})
    fig.suptitle("D1. Track A ansatz, n = 4, L = 2 (ry, rz per layer, then a CNOT chain)",
                 fontsize=11)
    save(fig, out / "fig_D1_trackA_circuit.png", dpi)


def d2_folded_circuit(out: Path, dpi: int) -> None:
    unitary, _ = build_unitary(2, 1, np.random.default_rng(1))
    folded = fold_global(unitary, 3)
    fig = folded.draw("mpl", fold=-1, style={"name": "bw"})
    fig.suptitle("D2. Global folding at λ = 3: U, then U†, then U again (n = 2, L = 1)",
                 fontsize=11)
    save(fig, out / "fig_D2_folded_circuit.png", dpi)


def d3_calibration_circuits(out: Path, dpi: int) -> None:
    full = calibration_circuits(2)
    tensored = tensored_calibration_circuits(2)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for j, qc in enumerate(full):
        qc.draw("mpl", ax=axes[0, j], style={"name": "bw"})
        axes[0, j].set_title(f"full, j={j} (prepares |{format(j, '02b')}⟩)", fontsize=9)
    labels = ["tensored, all-|00⟩", "tensored, all-|11⟩"]
    for k, qc in enumerate(tensored):
        qc.draw("mpl", ax=axes[1, k], style={"name": "bw"})
        axes[1, k].set_title(labels[k], fontsize=9)
    for k in range(len(tensored), 4):
        axes[1, k].axis("off")
    fig.suptitle("D3. Calibration circuits for n = 2: full (2ⁿ = 4) vs. tensored (2)", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, out / "fig_D3_calibration_circuits.png", dpi)


def d4_vqc_circuit(out: Path, dpi: int) -> None:
    qc, x, w = build_vqc_circuit(4, 2)
    fig = qc.draw("mpl", fold=-1, style={"name": "bw"})
    fig.suptitle("D4. VQC circuit, n = 4, L = 2: angle encoding ry(x[k]), then the ansatz "
                 "with trainable weights w[k]", fontsize=10.5)
    save(fig, out / "fig_D4_vqc_circuit.png", dpi)


def _box(ax, xy, w, h, text, color, fontsize=9):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                   linewidth=1.3, edgecolor=color, facecolor="white")
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#222")


def _arrow(ax, p0, p1, color="#555"):
    ax.annotate("", xy=p1, xytext=p0,
               arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.2, "shrinkA": 2, "shrinkB": 2})


def d5_pipeline(out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.7)
    ax.axis("off")

    _box(ax, (0.2, 4.2), 2.0, 0.9, "Circuit\n(ansatz / VQC)", "#555")
    _box(ax, (0.2, 2.8), 2.0, 0.9, "Noise model\n(depolarizing + readout)", "#555")

    runs = [("Base\n(λ=1)", 4.6), ("Folded\n(λ=3, λ=5)", 3.3), ("Calibration\n(full / tensored)", 2.0)]
    for label, y in runs:
        _box(ax, (3.1, y), 2.0, 0.9, label, "#777")
        _arrow(ax, (2.2, 4.65), (3.1, y + 0.45))

    methods = [("none", COLORS["none"], 4.9), ("rem / rem_tensored", COLORS["rem"], 3.9),
               ("zne", COLORS["zne"], 2.9), ("zne_rem", COLORS["zne_rem"], 1.9)]
    for label, color, y in methods:
        _box(ax, (6.2, y), 1.8, 0.8, label, color)
    for _, y0 in runs:
        for _, color, y1 in methods:
            _arrow(ax, (5.1, y0 + 0.45), (6.2, y1 + 0.4), color="#ccc")

    _box(ax, (8.7, 2.9), 2.0, 1.4, "Metrics\nerror, fidelity,\nsuccess prob.,\noverhead", "#333")
    for _, _, y1 in methods:
        _arrow(ax, (8.0, y1 + 0.4), (8.7, 3.6))

    fig.suptitle("D5. Mitigation pipeline: one noisy circuit feeds every method identically",
                fontsize=11)
    fig.tight_layout()
    save(fig, out / "fig_D5_mitigation_pipeline.png", dpi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw)
    dpi = cfg.plots.dpi

    d1_track_a_circuit(raw_dir, out, dpi)
    d2_folded_circuit(out, dpi)
    d3_calibration_circuits(out, dpi)
    d4_vqc_circuit(out, dpi)
    d5_pipeline(out, dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
