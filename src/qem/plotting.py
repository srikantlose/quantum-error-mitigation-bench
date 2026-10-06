"""All report figures (matplotlib only). Each method keeps one color in every figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from .circuits import circuit_stats, fold_global, load_instance, with_measurements  # noqa: E402
from .config import Config  # noqa: E402
from .execution import Executor  # noqa: E402
from .mitigation.readout import apply_rem  # noqa: E402
from .mitigation.zne import fit_exp, fit_linear, richardson_curve  # noqa: E402
from .observables import counts_to_probvec, exact_reference  # noqa: E402

METHODS = ("none", "rem", "zne", "zne_rem")
# Fixed per method: none = gray, rem = blue, zne = orange, zne_rem = green. Shades chosen so
# that every pair stays distinguishable under color-vision deficiency.
COLORS = {"none": "#8a8a8a", "rem": "#2a78d6", "zne": "#e07b24", "zne_rem": "#226b38"}
MARKERS = {"none": "o", "rem": "s", "zne": "^", "zne_rem": "D"}
LABELS = {"none": "none (no mitigation)", "rem": "rem", "zne": "zne", "zne_rem": "zne_rem"}
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
EXACT_COLOR = "#3b3b3b"
NOISE_LEVELS = ("ideal", "low", "moderate")


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "lines.linewidth": 1.6,
    })


def _save(fig, out: Path, dpi: int) -> Path:
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _top_legend(fig, handles, labels, ncol=None):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol or len(labels), handlelength=1.6, columnspacing=1.4)


def _cell(summary, **kv):
    m = np.ones(len(summary), dtype=bool)
    for k, v in kv.items():
        m &= summary[k].to_numpy() == v
    return summary[m]


# --- F1 / F2: grid of grouped bars ---------------------------------------------------


def _grouped_bar_grid(summary, runs, methods, value_fn, ylabel, title, floor=False):
    qubits = sorted(summary.n_qubits.unique())
    depths = sorted(summary.depth_layers.unique())
    fig, axes = plt.subplots(len(depths), len(qubits), figsize=(10, 5.6), sharey=True,
                             constrained_layout=True)
    x = np.arange(len(NOISE_LEVELS))
    width = 0.8 / len(methods)
    for r, L in enumerate(depths):
        for c, n in enumerate(qubits):
            ax = axes[r, c]
            for k, m in enumerate(methods):
                means, stds = [], []
                for nz in NOISE_LEVELS:
                    row = _cell(summary, n_qubits=n, depth_layers=L, noise_level=nz, method=m).iloc[0]
                    mu, sd = value_fn(row)
                    means.append(mu)
                    stds.append(sd)
                ax.bar(x + (k - (len(methods) - 1) / 2) * width, means, width, yerr=stds,
                       color=COLORS[m], edgecolor="white", linewidth=0.8,
                       error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": INK}, label=LABELS[m])
            if floor:
                fl = runs[(runs.n_qubits == n) & (runs.depth_layers == L)].shot_noise_floor.mean()
                ax.axhline(fl, color=INK, linestyle="--", linewidth=0.9, label="mean shot-noise floor")
            ax.set_xticks(x, NOISE_LEVELS)
            ax.set_title(f"n = {n}, L = {L}")
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(depths) - 1:
                ax.set_xlabel("noise level")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _top_legend(fig, handles, labels)
    fig.suptitle(title, y=1.08, fontsize=11, color=INK)
    return fig


def fig_error_vs_noise(summary, runs, out, dpi):
    fig = _grouped_bar_grid(
        summary, runs, METHODS, lambda r: (r.abs_error_mean, r.abs_error_std),
        "|E_hat − E_exact| (mean ± std, 5 seeds)",
        "F1. Absolute error of the parity estimate by noise level and method", floor=True)
    return _save(fig, out / "fig_error_vs_noise.png", dpi)


def fig_fidelity_before_after(summary, runs, out, dpi):
    fig = _grouped_bar_grid(
        summary, runs, ("none", "rem"),
        lambda r: (1 - r.hellinger_fidelity_mean, r.hellinger_fidelity_std),
        "Hellinger infidelity 1 − F (lower is better)",
        "F2. Distribution fidelity before (none) and after readout mitigation (rem)")
    return _save(fig, out / "fig_fidelity_before_after.png", dpi)


# --- F3 ------------------------------------------------------------------------------


def fig_error_vs_qubits(summary, out, dpi, noise="moderate"):
    depths = sorted(summary.depth_layers.unique())
    qubits = sorted(summary.n_qubits.unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(8.5, 3.4), sharey=True, constrained_layout=True)
    sub = summary[summary.noise_level == noise]
    top = float((sub.abs_error_mean + sub.abs_error_std).max()) * 1.06
    for ax, L in zip(axes, depths):
        for k, m in enumerate(METHODS):
            g = _cell(summary, depth_layers=L, noise_level=noise, method=m).set_index("n_qubits").loc[qubits]
            xs = np.array(qubits) + (k - 1.5) * 0.08
            ax.errorbar(xs, g.abs_error_mean, yerr=g.abs_error_std, color=COLORS[m], marker=MARKERS[m],
                        markersize=5.5, capsize=2, elinewidth=0.8, label=LABELS[m])
        ax.set_xticks(qubits)
        ax.set_xlabel("number of qubits n")
        ax.set_title(f"L = {L} layers")
        ax.set_ylim(0, top)
    axes[0].set_ylabel("|E_hat − E_exact| (mean ± std)")
    handles, labels = axes[0].get_legend_handles_labels()
    _top_legend(fig, handles, labels)
    fig.suptitle(f"F3. Error versus qubit count at {noise} noise", y=1.12, fontsize=11, color=INK)
    return _save(fig, out / "fig_error_vs_qubits.png", dpi)


# --- F4 ------------------------------------------------------------------------------


def fig_zne_extrapolation_example(runs, cfg: Config, out, dpi):
    ex = cfg.plots.example_condition
    rid = f"n{ex.n}_L{ex.depth}_{ex.noise}_s{ex.seed}"
    scales = np.array(cfg.zne.scale_factors, dtype=float)
    grid = np.linspace(0, 5.5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, constrained_layout=True)
    E_exact = runs[runs.run_id == rid].E_exact.iloc[0]
    for ax, method, title in ((axes[0], "zne", "raw counts (zne)"),
                              (axes[1], "zne_rem", "REM-corrected counts (zne_rem)")):
        row = runs[(runs.run_id == rid) & (runs.method == method)].iloc[0]
        vals = np.array([row.E_lambda1, row.E_lambda3, row.E_lambda5])
        color = COLORS[method]
        ax.plot(grid, richardson_curve(scales, vals)(grid), color=color, linestyle="-",
                linewidth=1.4, label="Richardson (quadratic)")
        a, b = fit_linear(scales, vals)
        ax.plot(grid, a + b * grid, color=color, linestyle="--", linewidth=1.2, label="linear fit")
        fit = fit_exp(scales, vals)
        if fit is not None:
            sgn, bb, cc = fit
            ax.plot(grid, sgn * bb * np.exp(-cc * grid), color=color, linestyle=":", linewidth=1.6,
                    label="exponential fit")
        else:
            ax.text(0.03, 0.04, "exponential fit undefined:\nE(λ) changes sign or is ~0",
                    transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
        ax.scatter(scales, vals, s=36, color=color, edgecolor="white", linewidth=1.2, zorder=4,
                   label="measured E(λ)")
        for est, mk, lab in ((row.E_richardson, "*", "Richardson at λ=0"),
                             (row.E_linear, "P", "linear at λ=0"),
                             (row.E_exp, "X", "exp at λ=0")):
            if np.isfinite(est):
                ax.scatter([0], [est], marker=mk, s=80, color=color, edgecolor=INK, linewidth=0.6,
                           zorder=5, label=lab)
        ax.axhline(E_exact, color=INK, linewidth=1.0, linestyle=(0, (6, 3)), label="E_exact")
        ax.set_xlabel("noise scale factor λ")
        ax.set_title(title)
        ax.set_xlim(-0.3, 5.6)
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    axes[0].set_ylabel("parity ⟨Z⊗…⊗Z⟩")
    fig.suptitle(f"F4. Zero-noise extrapolation for n={ex.n}, L={ex.depth}, {ex.noise} noise, "
                 f"seed {ex.seed}", fontsize=11, color=INK)
    return _save(fig, out / "fig_zne_extrapolation_example.png", dpi)


# --- F5 / F6 -------------------------------------------------------------------------


def fig_overhead_circuits(summary, out, dpi):
    qubits = sorted(summary.n_qubits.unique())
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    end_offset = {"none": 0, "rem": -7, "zne": 0, "zne_rem": 7}  # rem and zne_rem end 2 apart
    for m in METHODS:
        g = summary[summary.method == m].groupby("n_qubits").n_circuits.first().loc[qubits]
        ax.plot(qubits, g.values, color=COLORS[m], marker=MARKERS[m], markersize=6, label=LABELS[m])
        ax.annotate(f"{int(g.values[-1])}", (qubits[-1], g.values[-1]), xytext=(7, end_offset[m]),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_yscale("log", base=2)
    ax.set_xticks(qubits)
    ax.set_xlim(qubits[0] - 0.3, qubits[-1] + 0.8)
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel("circuit executions per estimate (log₂)")
    ax.legend(loc="upper left")
    ax.set_title("F5. Circuits per estimate: full REM grows as 1 + 2ⁿ")
    return _save(fig, out / "fig_overhead_circuits.png", dpi)


def depth_by_scale(cfg: Config, raw_dir: Path) -> pd.DataFrame:
    """Transpiled depth (barriers and measurements excluded) at every scale factor."""
    executor = Executor(cfg)
    records = []
    for path in sorted((raw_dir / "circuits").glob("*.json")):
        inst = load_instance(path)
        for s in cfg.zne.scale_factors:
            qc = with_measurements(fold_global(inst.unitary, s))
            (tqc,) = executor.transpile_circuits([qc], transpile_seed=0)
            records.append({"n_qubits": inst.n, "depth_layers": inst.layers, "seed": inst.seed,
                            "scale": s, **circuit_stats(tqc)})
    return pd.DataFrame(records)


def fig_depth_overhead(depths: pd.DataFrame, out, dpi):
    groups = depths.groupby(["n_qubits", "depth_layers", "scale"]).depth.mean().unstack("scale")
    scales = list(groups.columns)
    shades = ["#f6c9a3", "#e9934f", "#a8520f"][: len(scales)]  # one hue, light -> dark with λ
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    x = np.arange(len(groups))
    width = 0.8 / len(scales)
    for k, s in enumerate(scales):
        bars = ax.bar(x + (k - (len(scales) - 1) / 2) * width, groups[s].values, width,
                      color=shades[k], edgecolor="white", linewidth=0.8, label=f"λ = {s}")
        if k == len(scales) - 1:
            ax.bar_label(bars, fmt="%d", fontsize=7, padding=1, color=INK)
    ax.set_xticks(x, [f"n={n}\nL={L}" for n, L in groups.index])
    ax.set_ylabel("transpiled circuit depth")
    ax.set_xlabel("circuit (qubits n, layers L)")
    ax.legend(loc="upper left")
    ax.set_title("F6. Circuit depth at each ZNE scale factor (barriers and measurements excluded)")
    return _save(fig, out / "fig_depth_overhead.png", dpi)


# --- F7 ------------------------------------------------------------------------------


def distributions(raw_dir: Path, n, L, noise, seed, cond_threshold):
    rid = f"n{n}_L{L}_{noise}_s{seed}"
    inst = load_instance(raw_dir / "circuits" / f"n{n}_L{L}_s{seed}.json")
    p_exact = exact_reference(inst.unitary)["probs"]
    record = json.loads((raw_dir / "counts" / f"{rid}.json").read_text(encoding="utf-8"))
    p_noisy = counts_to_probvec(record["scale_counts"]["1"], n)
    A = np.load(raw_dir / "calibration" / f"{rid}.npy")
    p_rem = apply_rem(p_noisy, A, cond_threshold).projected
    return p_exact, p_noisy, p_rem


def fig_distributions(raw_dir: Path, cfg: Config, out, dpi, n, L=4, noise="moderate", seed=0, top=16):
    p_exact, p_noisy, p_rem = distributions(raw_dir, n, L, noise, seed, cfg.rem.cond_threshold)
    idx = np.arange(2**n)
    if len(idx) > top:
        idx = np.sort(np.argsort(-p_exact, kind="stable")[:top])
    labels = [format(i, f"0{n}b") for i in idx]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.48 * len(idx) + 1.5), 3.8), constrained_layout=True)
    x = np.arange(len(idx))
    width = 0.27
    for k, (p, lab, col) in enumerate(((p_exact, "exact (noiseless)", EXACT_COLOR),
                                       (p_noisy, "noisy (none)", COLORS["none"]),
                                       (p_rem, "rem (projected)", COLORS["rem"]))):
        ax.bar(x + (k - 1) * width, p[idx], width, color=col, edgecolor="white", linewidth=0.6, label=lab)
    ax.set_xticks(x, labels, rotation=90 if n > 2 else 0, fontfamily="monospace")
    ax.set_xlabel("basis state (qubit n−1 … qubit 0)" + (f"; {top} most likely of {2**n}" if 2**n > top else ""))
    ax.set_ylabel("probability")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax.set_title(f"F7. Output distribution, n={n}, L={L}, {noise} noise, seed {seed}", pad=24)
    return _save(fig, out / f"fig_distributions_n{n}.png", dpi)


# --- F8 ------------------------------------------------------------------------------


def fig_improvement_heatmap(summary, out, dpi):
    qubits = sorted(summary.n_qubits.unique())
    depths = sorted(summary.depth_layers.unique())
    methods = ("rem", "zne", "zne_rem")
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.4), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    for r, nz in enumerate(("low", "moderate")):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            mat = np.array([[_cell(summary, n_qubits=n, depth_layers=L, noise_level=nz,
                                   method=m).improvement_pct_median.iloc[0] for L in depths] for n in qubits])
            im = ax.imshow(np.clip(mat, -100, 100), cmap="RdBu", norm=norm, aspect="auto")
            for i in range(len(qubits)):
                for j in range(len(depths)):
                    v = mat[i, j]
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > 60 else INK)
            ax.set_xticks(range(len(depths)), [f"L={L}" for L in depths])
            ax.set_yticks(range(len(qubits)), [f"n={n}" for n in qubits])
            ax.grid(False)
            ax.set_title(f"{m}, {nz} noise")
            for s in ax.spines.values():
                s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, label="median improvement over none (%)")
    cb.outline.set_visible(False)
    fig.suptitle("F8. Median error improvement over no mitigation (5 seeds per cell)",
                 fontsize=11, color=INK)
    return _save(fig, out / "fig_improvement_heatmap.png", dpi)


# --- F9 ------------------------------------------------------------------------------


def fig_cost_benefit(runs, out, dpi):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True, constrained_layout=True)
    for ax, nz in zip(axes, ("low", "moderate")):
        sub = runs[runs.noise_level == nz]
        for m in METHODS:
            g = sub[sub.method == m].groupby("n_qubits").agg(
                shots=("total_shots", "first"), erf=("error_reduction_factor", "mean"),
                erf_median=("error_reduction_factor", "median"))
            if m == "none":
                # the baseline, by definition; one point and one label for every n
                ax.scatter([g.shots.iloc[0]], [1.0], marker=MARKERS[m], s=55, color=COLORS[m],
                           edgecolor="white", linewidth=0.8, zorder=3)
                ax.annotate("n=2, 4, 6", (g.shots.iloc[0], 1.0), xytext=(5, -10),
                            textcoords="offset points", fontsize=7, color=MUTED)
                continue
            for shots, lo, hi in zip(g.shots, g.erf_median, g.erf):
                ax.plot([shots, shots], [lo, hi], color=COLORS[m], linewidth=0.8, alpha=0.6, zorder=2)
            ax.scatter(g.shots, g.erf, marker=MARKERS[m], s=55, color=COLORS[m], edgecolor="white",
                       linewidth=0.8, zorder=3)
            ax.scatter(g.shots, g.erf_median, marker=MARKERS[m], s=40, facecolor="white",
                       edgecolor=COLORS[m], linewidth=1.3, zorder=3)
            for n, row in g.iterrows():
                ax.annotate(f"n={n}", (row.shots, row.erf), xytext=(5, 3), textcoords="offset points",
                            fontsize=7, color=MUTED)
        ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_xlabel("total shots per estimate (log scale)")
        ax.set_title(f"{nz} noise")
        ax.grid(True, axis="both")
    axes[0].set_ylabel("error reduction factor err_none / err (log₂)")
    handles = [Line2D([], [], marker=MARKERS[m], linestyle="", markersize=7, color=COLORS[m],
                      markeredgecolor="white") for m in METHODS]
    handles += [Line2D([], [], marker="o", linestyle="", markersize=7, color=INK, markeredgecolor="white"),
                Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="white",
                       markeredgecolor=INK)]
    labels = [LABELS[m] for m in METHODS] + ["filled: mean over L and seeds", "hollow: median"]
    _top_legend(fig, handles, labels, ncol=6)
    fig.suptitle("F9. Cost versus benefit (dashed line: no improvement)", y=1.1, fontsize=11, color=INK)
    return _save(fig, out / "fig_cost_benefit.png", dpi)


# --- F10 -----------------------------------------------------------------------------


def _box_strip(ax, data, rng):
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.4},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED})
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[m])
    for k, (m, d) in enumerate(zip(METHODS, data), start=1):
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(d)), d, s=12, color=COLORS[m], edgecolor="white",
                   linewidth=0.4, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.9, linestyle="--")
    ax.set_xticks(range(1, len(METHODS) + 1), METHODS)
    ax.set_xlabel("method")


def fig_bias_variance(runs, out, dpi, noise="moderate"):
    sub = runs[runs.noise_level == noise]
    raw = [sub[sub.method == m].signed_error.to_numpy() for m in METHODS]
    oriented = [(np.sign(sub[sub.method == m].E_exact) * sub[sub.method == m].signed_error).to_numpy()
                for m in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    rng = np.random.default_rng(0)  # jitter only; does not touch any result
    _box_strip(axes[0], raw, rng)
    axes[0].set_ylabel("signed error E_hat − E_exact")
    axes[0].set_title("signed error")
    _box_strip(axes[1], oriented, rng)
    axes[1].set_ylabel("sign(E_exact) · (E_hat − E_exact)")
    axes[1].set_title("oriented: below 0 = shrunk toward 0, above 0 = overshoot")
    fig.suptitle(f"F10. Bias and spread at {noise} noise (30 runs per method: all n, L, seeds)",
                 fontsize=11, color=INK)
    return _save(fig, out / "fig_bias_variance.png", dpi)


# --- circuit diagram -----------------------------------------------------------------


def fig_circuit_example(raw_dir: Path, out, dpi, n=4, L=2, seed=0):
    inst = load_instance(raw_dir / "circuits" / f"n{n}_L{L}_s{seed}.json")
    fig = with_measurements(inst.unitary).draw("mpl", fold=-1)
    fig.suptitle(f"Ansatz for n={n}, L={L} (seed {seed}): ry and rz on every qubit, then a CNOT chain, per layer",
                 fontsize=11, color=INK)
    return _save(fig, out / "fig_circuit_example.png", dpi)
