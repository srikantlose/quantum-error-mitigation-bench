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
from matplotlib.lines import Line2D  # noqa: E402

from .circuits import circuit_stats, fold_global, load_instance, with_measurements  # noqa: E402
from .config import Config  # noqa: E402
from .execution import Executor  # noqa: E402
from .mitigation.readout import apply_rem  # noqa: E402
from .mitigation.zne import fit_exp, fit_linear, richardson_curve  # noqa: E402
from .observables import counts_to_probvec, exact_reference  # noqa: E402

METHODS = ("none", "rem", "rem_tensored", "zne", "zne_rem")
# Fixed per method: none = gray, rem = blue, rem_tensored = light blue, zne = orange,
# zne_rem = green. Shades chosen so every pair stays distinguishable under colorblindness.
COLORS = {"none": "#8a8a8a", "rem": "#2a78d6", "rem_tensored": "#7fb8e8",
          "zne": "#e07b24", "zne_rem": "#226b38"}
MARKERS = {"none": "o", "rem": "s", "rem_tensored": "P", "zne": "^", "zne_rem": "D"}
LABELS = {"none": "none (no mitigation)", "rem": "rem (full)", "rem_tensored": "rem (tensored)",
          "zne": "zne", "zne_rem": "zne_rem"}
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
EXACT_COLOR = "#3b3b3b"
LR_COLOR = "#111111"
SVM_COLOR = "#444444"


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
               ncol=ncol or len(labels), handlelength=1.6, columnspacing=1.4, fontsize=8)


def _cell(summary, **kv):
    m = np.ones(len(summary), dtype=bool)
    for k, v in kv.items():
        m &= summary[k].to_numpy() == v
    return summary[m]


# ======================================================================================
# A1 / A2 / A3: grid of grouped bars (error, fidelity, success probability)
# ======================================================================================


def _grid_methods_for(summary, methods):
    return [m for m in methods if (summary.method == m).any()]


def _grouped_bar_grid(summary, runs, methods, noise_levels, value_fn, ylabel, title, floor=False):
    qubits = sorted(summary.n_qubits.unique())
    depths = sorted(summary.depth_layers.unique())
    methods = [m for m in methods if (summary.method == m).any()]
    fig, axes = plt.subplots(len(depths), len(qubits), figsize=(11, 5.8), sharey=True,
                             constrained_layout=True, squeeze=False)
    x = np.arange(len(noise_levels))
    width = 0.82 / len(methods)
    for r, L in enumerate(depths):
        for c, n in enumerate(qubits):
            ax = axes[r, c]
            for k, m in enumerate(methods):
                means, stds = [], []
                for nz in noise_levels:
                    sel = _cell(summary, n_qubits=n, depth_layers=L, noise_level=nz, method=m)
                    if not len(sel):
                        means.append(np.nan)
                        stds.append(np.nan)
                        continue
                    mu, sd = value_fn(sel.iloc[0])
                    means.append(mu)
                    stds.append(sd)
                ax.bar(x + (k - (len(methods) - 1) / 2) * width, means, width, yerr=stds,
                       color=COLORS[m], edgecolor="white", linewidth=0.8,
                       error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": INK}, label=LABELS[m])
            if floor:
                fl = runs[(runs.n_qubits == n) & (runs.depth_layers == L)
                         & (runs.noise_level.isin(noise_levels))].shot_noise_floor.mean()
                ax.axhline(fl, color=INK, linestyle="--", linewidth=0.9, label="mean shot-noise floor")
            ax.set_xticks(x, noise_levels, fontsize=7.5)
            ax.set_title(f"n = {n}, L = {L}", fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == len(depths) - 1:
                ax.set_xlabel("noise level")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _top_legend(fig, handles, labels)
    fig.suptitle(title, y=1.1, fontsize=11, color=INK)
    return fig


def fig_error_vs_noise(summary, runs, noise_levels, out, dpi):
    fig = _grouped_bar_grid(
        summary, runs, METHODS, noise_levels, lambda r: (r.abs_error_mean, r.abs_error_std),
        "|E_hat − E_exact| (mean ± std, 5 seeds)",
        "A1. Absolute error of the parity estimate by noise level and method", floor=True)
    return _save(fig, out / "fig_A1_error_vs_noise.png", dpi)


def fig_fidelity_before_after(summary, runs, noise_levels, out, dpi):
    fig = _grouped_bar_grid(
        summary, runs, ("none", "rem", "rem_tensored"), noise_levels,
        lambda r: (1 - r.hellinger_fidelity_mean, r.hellinger_fidelity_std),
        "Hellinger infidelity 1 − F (lower is better)",
        "A2. Distribution fidelity before and after readout mitigation")
    return _save(fig, out / "fig_A2_fidelity_before_after.png", dpi)


def fig_success_probability(summary, runs, noise_levels, out, dpi):
    fig = _grouped_bar_grid(
        summary, runs, METHODS, noise_levels,
        lambda r: (r.P_succ_ratio_mean, r.P_succ_ratio_std),
        "P_succ_hat / P_succ_exact (1 = exact)",
        "A3. Success-probability ratio by noise level and method")
    for ax in fig.axes:
        ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    return _save(fig, out / "fig_A3_success_probability.png", dpi)


# ======================================================================================
# A4: error vs qubits
# ======================================================================================


def fig_error_vs_qubits(summary, out, dpi, noise="moderate"):
    depths = sorted(summary.depth_layers.unique())
    qubits = sorted(summary.n_qubits.unique())
    methods = _grid_methods_for(summary, METHODS)
    fig, axes = plt.subplots(1, len(depths), figsize=(9, 3.6), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, L in zip(axes, depths):
        for k, m in enumerate(methods):
            g = _cell(summary, depth_layers=L, noise_level=noise, method=m)
            if not len(g):
                continue
            g = g.set_index("n_qubits").reindex(qubits)
            xs = np.array(qubits) + (k - (len(methods) - 1) / 2) * 0.08
            ax.errorbar(xs, g.abs_error_mean, yerr=g.abs_error_std, color=COLORS[m], marker=MARKERS[m],
                        markersize=5.5, capsize=2, elinewidth=0.8, label=LABELS[m])
        ax.set_xticks(qubits)
        ax.set_xlabel("number of qubits n")
        ax.set_title(f"L = {L} layers")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("|E_hat − E_exact| (mean ± std)")
    handles, labels = axes[0].get_legend_handles_labels()
    _top_legend(fig, handles, labels)
    fig.suptitle(f"A4. Error versus qubit count at {noise} noise", y=1.14, fontsize=11, color=INK)
    return _save(fig, out / "fig_A4_error_vs_qubits.png", dpi)


# ======================================================================================
# A5: ZNE extrapolation example
# ======================================================================================


def fig_zne_extrapolation_example(runs, cfg: Config, out, dpi):
    ex = cfg.plots.example_condition
    rid = f"n{ex.n}_L{ex.depth}_{ex.noise}_s{ex.seed}"
    scales = np.array(cfg.zne.scale_factors, dtype=float)
    grid = np.linspace(0, 5.5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True, constrained_layout=True)
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
    fig.suptitle(f"A5. Zero-noise extrapolation for n={ex.n}, L={ex.depth}, {ex.noise} noise, "
                 f"seed {ex.seed}", fontsize=11, color=INK)
    return _save(fig, out / "fig_A5_zne_extrapolation_example.png", dpi)


# ======================================================================================
# A6 / A7: overhead and depth
# ======================================================================================


def fig_overhead_circuits(summary, out, dpi):
    qubits = sorted(summary.n_qubits.unique())
    methods = _grid_methods_for(summary, METHODS)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    label_dx = {"none": 0, "rem": -10, "rem_tensored": 10, "zne": 0, "zne_rem": 10}
    for m in methods:
        g = summary[summary.method == m].groupby("n_qubits").n_circuits.first().reindex(qubits)
        ax.plot(qubits, g.values, color=COLORS[m], marker=MARKERS[m], markersize=6, label=LABELS[m])
        ax.annotate(f"{int(g.values[-1])}", (qubits[-1], g.values[-1]),
                    xytext=(label_dx.get(m, 0) + 7, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.set_yscale("log", base=2)
    ax.set_xticks(qubits)
    ax.set_xlim(qubits[0] - 0.3, qubits[-1] + 1.0)
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel("circuit executions per estimate (log₂)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("A6. Circuits per estimate: full REM grows as 1 + 2ⁿ; tensored REM stays at 3")
    return _save(fig, out / "fig_A6_overhead_circuits.png", dpi)


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
    shades = ["#f6c9a3", "#e9934f", "#a8520f"][: len(scales)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    x = np.arange(len(groups))
    width = 0.8 / len(scales)
    for k, s in enumerate(scales):
        bars = ax.bar(x + (k - (len(scales) - 1) / 2) * width, groups[s].values, width,
                      color=shades[k], edgecolor="white", linewidth=0.8, label=f"λ = {s}")
        if k == len(scales) - 1:
            ax.bar_label(bars, fmt="%d", fontsize=6.5, padding=1, color=INK)
    ax.set_xticks(x, [f"n={n}\nL={L}" for n, L in groups.index], fontsize=7.5)
    ax.set_ylabel("transpiled circuit depth")
    ax.set_xlabel("circuit (qubits n, layers L)")
    ax.legend(loc="upper left")
    ax.set_title("A7. Circuit depth at each ZNE scale factor (barriers and measurements excluded)")
    return _save(fig, out / "fig_A7_depth_overhead.png", dpi)


# ======================================================================================
# A8: output distributions
# ======================================================================================


def distributions(raw_dir: Path, n, L, noise, seed, cond_threshold):
    rid = f"n{n}_L{L}_{noise}_s{seed}"
    inst = load_instance(raw_dir / "circuits" / f"n{n}_L{L}_s{seed}.json")
    p_exact = exact_reference(inst.unitary)["probs"]
    record = json.loads((raw_dir / "counts" / f"{rid}.json").read_text(encoding="utf-8"))
    p_noisy = counts_to_probvec(record["scale_counts"]["1"], n)
    A = np.load(raw_dir / "calibration" / f"full_{rid}.npy")
    p_rem = apply_rem(p_noisy, A, cond_threshold).projected
    A_t = np.load(raw_dir / "calibration" / f"tensored_{rid}.npy")
    p_rem_t = apply_rem(p_noisy, A_t, cond_threshold).projected
    return p_exact, p_noisy, p_rem, p_rem_t


def fig_distributions(raw_dir: Path, cfg: Config, out, dpi, n, L=4, noise="moderate", seed=0, top=16):
    p_exact, p_noisy, p_rem, p_rem_t = distributions(raw_dir, n, L, noise, seed, cfg.rem.cond_threshold)
    idx = np.arange(2**n)
    if len(idx) > top:
        idx = np.sort(np.argsort(-p_exact, kind="stable")[:top])
    labels = [format(i, f"0{n}b") for i in idx]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.56 * len(idx) + 1.5), 3.9), constrained_layout=True)
    x = np.arange(len(idx))
    width = 0.2
    series = [(p_exact, "exact (noiseless)", EXACT_COLOR), (p_noisy, "noisy (none)", COLORS["none"]),
              (p_rem, "rem (full)", COLORS["rem"]), (p_rem_t, "rem (tensored)", COLORS["rem_tensored"])]
    for k, (p, lab, col) in enumerate(series):
        ax.bar(x + (k - 1.5) * width, p[idx], width, color=col, edgecolor="white", linewidth=0.5, label=lab)
    ax.set_xticks(x, labels, rotation=90 if n > 2 else 0, fontfamily="monospace", fontsize=7.5)
    ax.set_xlabel("basis state (qubit n−1 … qubit 0)" + (f"; {top} most likely of {2**n}" if 2**n > top else ""))
    ax.set_ylabel("probability")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7.5)
    ax.set_title(f"A8. Output distribution, n={n}, L={L}, {noise} noise, seed {seed}", pad=26)
    return _save(fig, out / f"fig_A8_distributions_n{n}.png", dpi)


# ======================================================================================
# A9: improvement heatmap
# ======================================================================================


def fig_improvement_heatmap(summary, out, dpi, noise_levels=("low", "moderate")):
    qubits = sorted(summary.n_qubits.unique())
    depths = sorted(summary.depth_layers.unique())
    methods = _grid_methods_for(summary, ("rem", "rem_tensored", "zne", "zne_rem"))
    fig, axes = plt.subplots(len(noise_levels), len(methods),
                             figsize=(2.1 * len(methods) + 1.5, 2.5 * len(noise_levels) + 1), constrained_layout=True)
    axes = np.atleast_2d(axes)
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    im = None
    for r, nz in enumerate(noise_levels):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            mat = np.array([[_cell(summary, n_qubits=n, depth_layers=L, noise_level=nz,
                                   method=m).improvement_pct_median.iloc[0]
                             if len(_cell(summary, n_qubits=n, depth_layers=L, noise_level=nz, method=m))
                             else np.nan for L in depths] for n in qubits])
            im = ax.imshow(np.clip(mat, -100, 100), cmap="RdBu", norm=norm, aspect="auto")
            for i in range(len(qubits)):
                for j in range(len(depths)):
                    v = mat[i, j]
                    if np.isnan(v):
                        continue
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                            color="white" if abs(v) > 60 else INK)
            ax.set_xticks(range(len(depths)), [f"L={L}" for L in depths], fontsize=7.5)
            ax.set_yticks(range(len(qubits)), [f"n={n}" for n in qubits], fontsize=7.5)
            ax.grid(False)
            ax.set_title(f"{m}, {nz}", fontsize=8.5)
            for s in ax.spines.values():
                s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, label="median improvement over none (%)")
    cb.outline.set_visible(False)
    fig.suptitle("A9. Median error improvement over no mitigation (5 seeds per cell)",
                 fontsize=11, color=INK)
    return _save(fig, out / "fig_A9_improvement_heatmap.png", dpi)


# ======================================================================================
# A10: cost vs benefit
# ======================================================================================


def fig_cost_benefit(runs, out, dpi, noise_levels=("low", "moderate")):
    methods = tuple(m for m in METHODS if m != "none")
    fig, axes = plt.subplots(1, len(noise_levels), figsize=(5.2 * len(noise_levels), 4.3), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, nz in zip(axes, noise_levels):
        sub = runs[runs.noise_level == nz]
        none_g = sub[sub.method == "none"].groupby("n_qubits").agg(shots=("total_shots", "first"))
        ax.scatter([none_g.shots.iloc[0]], [1.0], marker=MARKERS["none"], s=55, color=COLORS["none"],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate("none, any n", (none_g.shots.iloc[0], 1.0), xytext=(5, -10),
                    textcoords="offset points", fontsize=7, color=MUTED)
        for m in methods:
            if not (sub.method == m).any():
                continue
            g = sub[sub.method == m].groupby("n_qubits").agg(
                shots=("total_shots", "first"), erf=("error_reduction_factor", "mean"))
            left = m == "zne"
            ax.plot(g.shots, g.erf, color=COLORS[m], linewidth=0.8, alpha=0.5, zorder=2)
            ax.scatter(g.shots, g.erf, marker=MARKERS[m], s=55, color=COLORS[m], edgecolor="white",
                       linewidth=0.8, zorder=3)
            for n, row in g.iterrows():
                ax.annotate(f"n={n}", (row.shots, row.erf), xytext=(-10 if left else 7, -7 if left else 3),
                            textcoords="offset points", fontsize=7, color=MUTED,
                            ha="right" if left else "left")
        ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_xlabel("total shots per estimate (log scale)")
        ax.set_title(f"{nz} noise")
        ax.grid(True, axis="both")
    axes[0].set_ylabel("mean error reduction factor err_none / err (log₂)")
    handles = [Line2D([], [], marker=MARKERS[m], linestyle="", markersize=7, color=COLORS[m],
                      markeredgecolor="white") for m in ("none",) + methods]
    labels = [LABELS[m] for m in ("none",) + methods]
    _top_legend(fig, handles, labels, ncol=len(labels))
    fig.suptitle("A10. Cost versus benefit (dashed line: no improvement)", y=1.14, fontsize=11, color=INK)
    return _save(fig, out / "fig_A10_cost_benefit.png", dpi)


# ======================================================================================
# A11: bias / variance box plots
# ======================================================================================


def fig_bias_variance(runs, out, dpi, noise="moderate"):
    methods = _grid_methods_for(pd.DataFrame({"method": runs.method.unique()}), METHODS)
    sub = runs[runs.noise_level == noise]
    raw = [sub[sub.method == m].signed_error.to_numpy() for m in methods]
    oriented = [(np.sign(sub[sub.method == m].E_exact) * sub[sub.method == m].signed_error).to_numpy()
                for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    rng = np.random.default_rng(0)

    def box_strip(ax, data):
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.4},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.35)
            patch.set_edgecolor(COLORS[m])
        for k, (m, d) in enumerate(zip(methods, data), start=1):
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color=COLORS[m], edgecolor="white",
                       linewidth=0.4, zorder=3)
        ax.axhline(0, color=INK, linewidth=0.9, linestyle="--")
        ax.set_xticks(range(1, len(methods) + 1), methods, fontsize=7.5)
        ax.set_xlabel("method")

    box_strip(axes[0], raw)
    axes[0].set_ylabel("signed error E_hat − E_exact")
    axes[0].set_title("signed error")
    box_strip(axes[1], oriented)
    axes[1].set_ylabel("sign(E_exact) · (E_hat − E_exact)")
    axes[1].set_title("oriented: below 0 = shrunk toward 0, above 0 = overshoot")
    fig.suptitle(f"A11. Bias and spread at {noise} noise (pooled over n, L, seeds)",
                 fontsize=11, color=INK)
    return _save(fig, out / "fig_A11_bias_variance.png", dpi)


# ======================================================================================
# A12: full vs. tensored REM
# ======================================================================================


def fig_rem_full_vs_tensored(runs, out, dpi, noise_levels=("low", "moderate", "high")):
    wide = runs[runs.method.isin(["rem", "rem_tensored"])].pivot_table(
        index=["n_qubits", "depth_layers", "noise_level", "seed"], columns="method", values="abs_error")
    fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    qubits = sorted(runs.n_qubits.unique())
    markers = {nz: mk for nz, mk in zip(noise_levels, ("o", "s", "^"))}
    cmap = plt.get_cmap("viridis")
    colors = {n: cmap(i / max(1, len(qubits) - 1)) for i, n in enumerate(qubits)}
    top = 0.0
    for nz in noise_levels:
        g = wide.xs(nz, level="noise_level", drop_level=False).reset_index()
        for n in qubits:
            gn = g[g.n_qubits == n]
            if not len(gn):
                continue
            ax.scatter(gn["rem"], gn["rem_tensored"], marker=markers[nz], color=colors[n], s=36,
                       edgecolor="white", linewidth=0.5, alpha=0.85)
            top = max(top, gn["rem"].max(), gn["rem_tensored"].max())
    top *= 1.08
    ax.plot([0, top], [0, top], color=INK, linewidth=1, linestyle="--")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("full REM |error| (2ⁿ calibration circuits)")
    ax.set_ylabel("tensored REM |error| (2 calibration circuits)")
    shape_handles = [Line2D([], [], marker=markers[nz], linestyle="", color=INK, markersize=7) for nz in noise_levels]
    color_handles = [Line2D([], [], marker="o", linestyle="", color=colors[n], markersize=7) for n in qubits]
    leg1 = ax.legend(shape_handles, list(noise_levels), title="noise", loc="upper left", fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(color_handles, [f"n={n}" for n in qubits], title="qubits", loc="lower right", fontsize=7.5)
    ax.set_title("A12. Full vs. tensored REM: points on the line mean they agree")
    return _save(fig, out / "fig_A12_rem_full_vs_tensored.png", dpi)


# ======================================================================================
# A13 (optional): noisy-exact vs. sampled
# ======================================================================================


def fig_noisy_exact_vs_sampled(runs, out, dpi):
    sub = runs[runs.method == "none"]
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    for nz, marker in zip(("ideal", "low", "moderate", "high"), ("o", "s", "^", "D")):
        g = sub[sub.noise_level == nz]
        if not len(g):
            continue
        ax.scatter(g.E_noisy_exact, g.E_hat, marker=marker, s=24, alpha=0.75, label=nz,
                   edgecolor="white", linewidth=0.3)
    lo = min(sub.E_noisy_exact.min(), sub.E_hat.min()) * 1.05
    hi = max(sub.E_noisy_exact.max(), sub.E_hat.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1, linestyle="--")
    ax.set_xlabel("exact noisy parity (infinite shots)")
    ax.set_ylabel("sampled parity (1024 shots, none)")
    ax.legend(title="noise", fontsize=8)
    ax.set_title("A13. Sampled estimate vs. the exact noisy value: the gap is shot noise")
    return _save(fig, out / "fig_A13_noisy_exact_vs_sampled.png", dpi)


# ======================================================================================
# Track B figures
# ======================================================================================


def fig_qml_accuracy_vs_noise(summary_b, out, dpi, metric="accuracy", fig_id="B1"):
    qubits = sorted(summary_b[summary_b.model == "vqc"].n_qubits.dropna().unique())
    depths = sorted(summary_b[summary_b.model == "vqc"].depth_layers.dropna().unique())
    noise_levels = ["ideal", "low", "moderate", "high"]
    methods = ("none", "rem", "zne", "zne_rem")
    fig, axes = plt.subplots(len(depths), len(qubits), figsize=(10, 6.5), sharey=True,
                             constrained_layout=True, squeeze=False)
    for r, L in enumerate(depths):
        for c, n in enumerate(qubits):
            ax = axes[r, c]
            vqc = summary_b[(summary_b.model == "vqc") & (summary_b.n_qubits == n)
                            & (summary_b.depth_layers == L)]
            for m in methods:
                g = vqc[vqc.method == m].set_index("noise_level").reindex(noise_levels)
                ax.errorbar(range(len(noise_levels)), g[f"{metric}_mean"], yerr=g[f"{metric}_std"],
                           color=COLORS[m], marker=MARKERS[m], markersize=5, capsize=2,
                           elinewidth=0.8, label=LABELS[m])
            lr = summary_b[(summary_b.model == "logreg") & (summary_b.n_qubits == n)]
            svm = summary_b[(summary_b.model == "svm") & (summary_b.n_qubits == n)]
            if len(lr) and metric == "accuracy":
                ax.axhline(lr[f"{metric}_mean"].iloc[0], color=LR_COLOR, linestyle="--", linewidth=1.1,
                          label="Logistic Regression")
            if len(svm) and metric == "accuracy":
                ax.axhline(svm[f"{metric}_mean"].iloc[0], color=SVM_COLOR, linestyle=":", linewidth=1.3,
                          label="SVM")
            ax.set_xticks(range(len(noise_levels)), noise_levels, fontsize=7.5)
            ax.set_title(f"n = {int(n)}, L = {int(L)}", fontsize=9)
            if c == 0:
                ax.set_ylabel(metric)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _top_legend(fig, handles, labels, ncol=3)
    title = {"accuracy": f"{fig_id}. VQC classification accuracy vs. noise, against LR and SVM",
             "f1": f"{fig_id}. VQC F1 score vs. noise level"}[metric]
    fig.suptitle(title, y=1.12, fontsize=11, color=INK)
    fname = {"accuracy": "fig_B1_qml_accuracy_vs_noise.png", "f1": "fig_B2_qml_f1_vs_noise.png"}[metric]
    return _save(fig, out / fname, dpi)


def fig_qml_confusion_matrices(runs_b, n, L, out, dpi):
    conditions = [("LR", {"model": "logreg", "n_qubits": n}),
                  ("SVM", {"model": "svm", "n_qubits": n}),
                  ("VQC\nexact", {"model": "vqc", "n_qubits": n, "depth_layers": L, "noise_level": "exact"}),
                  ("VQC\nmod./none", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                      "noise_level": "moderate", "method": "none"}),
                  ("VQC\nmod./zne_rem", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                         "noise_level": "moderate", "method": "zne_rem"}),
                  ("VQC\nhigh/none", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                      "noise_level": "high", "method": "none"}),
                  ("VQC\nhigh/zne_rem", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                         "noise_level": "high", "method": "zne_rem"})]
    fig, axes = plt.subplots(1, len(conditions), figsize=(2.0 * len(conditions), 2.6), constrained_layout=True)
    for ax, (label, filt) in zip(axes, conditions):
        mask = np.ones(len(runs_b), dtype=bool)
        for k, v in filt.items():
            mask &= (runs_b[k] == v)
        g = runs_b[mask]
        tp, fn, fp, tn = g.tp.sum(), g.fn.sum(), g.fp.sum(), g.tn.sum()
        mat = np.array([[tp, fn], [fp, tn]])
        ax.imshow(mat, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=9,
                       color="white" if mat[i, j] > mat.max() / 2 else INK)
        ax.set_xticks([0, 1], ["+1", "−1"], fontsize=7.5)
        ax.set_yticks([0, 1], ["+1", "−1"], fontsize=7.5)
        ax.set_title(label, fontsize=8)
        ax.grid(False)
    axes[0].set_ylabel("true", fontsize=8)
    fig.text(0.5, -0.04, "predicted", ha="center", fontsize=8)
    fig.suptitle(f"B3. Confusion matrices, pooled over 5 seeds (n={n}, L={L})", fontsize=11, color=INK)
    return _save(fig, out / "fig_B3_qml_confusion_matrices.png", dpi)


def fig_qml_expectation_error(summary_b, out, dpi):
    qubits = sorted(summary_b[summary_b.model == "vqc"].n_qubits.dropna().unique())
    depths = sorted(summary_b[summary_b.model == "vqc"].depth_layers.dropna().unique())
    noise_levels = ["ideal", "low", "moderate", "high"]
    methods = ("none", "rem", "zne", "zne_rem")
    fig, axes = plt.subplots(1, len(qubits), figsize=(5.2 * len(qubits), 4), sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, qubits):
        for m in methods:
            vals = []
            for L in depths:
                g = summary_b[(summary_b.model == "vqc") & (summary_b.n_qubits == n)
                             & (summary_b.depth_layers == L) & (summary_b.method == m)]
                g = g.set_index("noise_level").reindex(noise_levels)
                vals.append(g.mean_abs_E_error_mean.to_numpy())
            mean_over_L = np.nanmean(np.vstack(vals), axis=0)
            ax.plot(range(len(noise_levels)), mean_over_L, color=COLORS[m], marker=MARKERS[m],
                   markersize=5, label=LABELS[m])
        ax.set_xticks(range(len(noise_levels)), noise_levels)
        ax.set_title(f"n = {int(n)} (mean over L)")
        ax.set_xlabel("noise level")
    axes[0].set_ylabel("mean |E_hat − E_exact| over the test set")
    handles, labels = axes[0].get_legend_handles_labels()
    _top_legend(fig, handles, labels)
    fig.suptitle("B4. VQC expectation-value error vs. noise level", y=1.14, fontsize=11, color=INK)
    return _save(fig, out / "fig_B4_qml_expectation_error.png", dpi)


def fig_qml_margin_scatter(root_dir: Path, n, L, out, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8), constrained_layout=True)
    for row, nz in enumerate(("moderate", "high")):
        for col, method in enumerate(("none", "zne_rem")):
            ax = axes[row, col]
            allE, allH = [], []
            for seed in range(5):
                path = root_dir / "qml" / "predictions" / f"n{n}_L{L}_{nz}_s{seed}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                allE.append(df["E_exact"].to_numpy())
                allH.append(df[f"E_hat_{method}"].to_numpy())
            if not allE:
                continue
            E = np.concatenate(allE)
            H = np.concatenate(allH)
            ax.scatter(E, H, s=18, color=COLORS[method], alpha=0.75, edgecolor="white", linewidth=0.3)
            ax.axhline(0, color=MUTED, linewidth=0.7, linestyle=":")
            ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=":")
            ax.plot([-1, 1], [-1, 1], color=INK, linewidth=1, linestyle="--")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(f"{nz} noise, {LABELS[method]}", fontsize=9)
            if col == 0:
                ax.set_ylabel("E_hat")
            if row == 1:
                ax.set_xlabel("E_exact")
    fig.suptitle(f"B5. Margins before and after mitigation (n={n}, L={L}, 100 test predictions pooled)",
                 fontsize=11, color=INK)
    return _save(fig, out / "fig_B5_qml_margin_scatter.png", dpi)


def fig_qml_training_curves(models_dir: Path, qubits, depths, seeds, out, dpi):
    fig, axes = plt.subplots(len(depths), len(qubits), figsize=(9, 5.5), constrained_layout=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for r, L in enumerate(depths):
        for c, n in enumerate(qubits):
            ax = axes[r, c]
            for i, seed in enumerate(seeds):
                path = models_dir / f"n{n}_L{L}_s{seed}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                ax.plot(record["loss_history"], color=cmap(i / max(1, len(seeds) - 1)),
                       linewidth=1.1, label=f"seed {seed}")
            ax.set_title(f"n={n}, L={L}", fontsize=9)
            ax.set_xlabel("objective evaluation")
            if c == 0:
                ax.set_ylabel("MSE loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _top_legend(fig, handles, labels, ncol=len(labels))
    fig.suptitle("B6. VQC training curves (noiseless, exact)", y=1.1, fontsize=11, color=INK)
    return _save(fig, out / "fig_B6_qml_training_curves.png", dpi)


# ======================================================================================
# Improvement: I1
# ======================================================================================


def fig_zne_allocation(alloc: pd.DataFrame, out, dpi):
    groups = alloc.groupby(["n_qubits", "depth_layers", "noise_level"], sort=False)
    labels, uni, opt = [], [], []
    for (n, L, nz), g in groups:
        labels.append(f"n={int(n)}\nL={int(L)}\n{nz}")
        uni.append(g[g.scheme == "uniform"].empirical_std.iloc[0])
        opt.append(g[g.scheme == "optimal"].empirical_std.iloc[0])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels)), 4.3), constrained_layout=True)
    width = 0.35
    ax.bar(x - width / 2, uni, width, color=MUTED, label="uniform (1024 each)")
    ax.bar(x + width / 2, opt, width, color="#2a78d6", label="optimal (∝ |γᵢ|)")
    theory_ratio = (alloc[alloc.scheme == "optimal"].theoretical_std.iloc[0]
                    / alloc[alloc.scheme == "uniform"].theoretical_std.iloc[0])
    ax2 = ax.twinx()
    ax2.axhline(theory_ratio, color=INK, linestyle="--", linewidth=1,
               label=f"theoretical ratio ({theory_ratio:.3f})")
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("—", alpha=0)
    ax2.set_yticks([])
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.set_ylabel("empirical std of the Richardson estimate")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.set_title("I1. Variance-optimal shot allocation vs. uniform, at equal total shots")
    return _save(fig, out / "fig_I1_zne_allocation.png", dpi)
