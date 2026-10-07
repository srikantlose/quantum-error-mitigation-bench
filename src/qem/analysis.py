"""Aggregation over seeds, confidence intervals, statistical tests and markdown tables.

Covers Track A (the mitigation testbed), Track B (the VQC on Iris binary) and the
variance-optimal shot-allocation improvement experiment.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .circuits import fold_global, load_instance
from .config import Config
from .execution import Executor
from .experiment import METHOD_ORDER
from .mitigation.zne import richardson_coeffs
from .observables import parity_from_probs

GROUP = ["n_qubits", "depth_layers", "noise_level", "method"]
SUMMARY_METRICS = [
    "abs_error", "signed_error", "improvement_pct", "error_reduction_factor",
    "hellinger_fidelity", "tvd", "P_succ_ratio", "M_abs_error", "est_std",
]
OVERHEAD = ["n_circuits", "total_shots", "base_depth", "max_depth", "base_cx", "total_cx",
            "total_1q", "total_gates"]
TIMING = ["time_quantum_s", "time_classical_s"]

QML_GROUP = ["model", "n_qubits", "depth_layers", "noise_level", "method"]
QML_SUMMARY_METRICS = ["accuracy", "precision", "recall", "f1", "mean_abs_E_error",
                       "margin_retention", "mean_hellinger", "agreement_with_exact"]
QML_METHOD_ORDER = ("none", "rem", "zne", "zne_rem")
MODEL_ORDER = ("logreg", "svm", "vqc")


def t_crit(k: int) -> float:
    """Two-sided 95% Student t quantile for k samples (2.776 for k = 5)."""
    return float(stats.t.ppf(0.975, k - 1))


def _order(df: pd.DataFrame, noise_order: Sequence[str]) -> pd.DataFrame:
    noise_rank = {nz: i for i, nz in enumerate(noise_order)}
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    key = pd.DataFrame(index=df.index)
    for col in ("n_qubits", "depth_layers"):
        if col in df:
            key[col] = df[col]
    if "noise_level" in df:
        key["noise"] = df["noise_level"].map(noise_rank)
    if "method" in df:
        key["method"] = df["method"].map(method_rank)
    order = key.sort_values(list(key.columns), kind="mergesort").index
    return df.loc[order].reset_index(drop=True)


def load_runs(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# --- shared markdown helpers -----------------------------------------------------------


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str | None = None) -> str:
    align = align or ("l" * len(headers))
    sep = ["---" if a == "l" else "---:" for a in align]

    def esc(c) -> str:
        return str(c).replace("|", "\\|")

    lines = ["| " + " | ".join(esc(h) for h in headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def pm(mean: float, std: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "–"
    if np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _i(x: float) -> str:
    """Whole-number percentage without a "-0"."""
    return f"{round(x) + 0:.0f}"


# ======================================================================================
# Track A
# ======================================================================================


def summarize(runs: pd.DataFrame, noise_order: Sequence[str]) -> pd.DataFrame:
    """One row per (n, L, noise, method): mean, std (ddof=1), median and 95% CI per metric."""
    records = []
    for keys, g in runs.groupby(GROUP, sort=False):
        rec = dict(zip(GROUP, keys))
        rec["n_seeds"] = len(g)
        for m in SUMMARY_METRICS:
            x = g[m].to_numpy(dtype=float)
            x = x[~np.isnan(x) & np.isfinite(x)]
            k = len(x)
            if k == 0:
                mean = std = median = lo = hi = np.nan
            else:
                with np.errstate(invalid="ignore"):
                    mean = float(np.mean(x))
                    median = float(np.median(x))
                    std = float(np.std(x, ddof=1)) if k > 1 else np.nan
                    half = t_crit(k) * std / np.sqrt(k) if k > 1 else np.nan
                lo, hi = mean - half, mean + half
            rec.update({f"{m}_mean": mean, f"{m}_std": std, f"{m}_median": median,
                        f"{m}_ci95_low": lo, f"{m}_ci95_high": hi})
        for c in OVERHEAD:
            rec[c] = g[c].iloc[0]
        for c in TIMING:
            rec[f"{c}_mean"] = float(g[c].mean())
        records.append(rec)
    return _order(pd.DataFrame(records), noise_order)


def table_error(summary: pd.DataFrame, noise_order: Sequence[str], methods: Sequence[str]) -> str:
    """Rows (n, L); columns noise x method, mean ± std abs_error. Headline noise levels
    (everything but 'high') form the main table; 'high' is reported in a separate block,
    since it is a stress test, not a headline result."""
    headline = [nz for nz in noise_order if nz != "high"]
    high = [nz for nz in noise_order if nz == "high"]

    def block(levels):
        headers = ["n", "L"] + [f"{nz} / {m}" for nz in levels for m in methods]
        rows = []
        for (n, L), g in summary.groupby(["n_qubits", "depth_layers"], sort=False):
            row = [str(n), str(L)]
            for nz in levels:
                sub = g[g.noise_level == nz].set_index("method")
                present = [m for m in methods if m in sub.index]
                best = sub.loc[present, "abs_error_mean"].idxmin() if present else None
                for m in methods:
                    if m not in sub.index:
                        row.append("–")
                        continue
                    cell = pm(sub.loc[m, "abs_error_mean"], sub.loc[m, "abs_error_std"])
                    row.append(f"**{cell}**" if m == best else cell)
            rows.append(row)
        return md_table(headers, rows, "rr" + "r" * (len(headers) - 2))

    text = "# Absolute error by method\n\n"
    text += ("Absolute error |E_hat − E_exact| of the parity, mean ± std over 5 seeds. The lowest "
             "mean per (n, L, noise level) is in bold. At the ideal level the error is pure shot "
             "noise.\n\n")
    text += block(headline)
    if high:
        text += ("\n## High noise (stress test, reported separately from the headline results)\n\n"
                 + block(high))
    return text


def table_fidelity(summary: pd.DataFrame, noise_order: Sequence[str],
                   methods: Sequence[str] = ("none", "rem", "rem_tensored")) -> str:
    out = ["# Distribution fidelity and success probability\n",
           "Computed against the exact noiseless distribution, mean ± std over 5 seeds. "
           "ZNE produces no distribution, so `zne` and `zne_rem` have no entries by design.\n"]
    headline = [nz for nz in noise_order if nz != "high"]
    for metric, title in (("hellinger_fidelity", "Hellinger fidelity (higher is better)"),
                          ("tvd", "Total variation distance (lower is better)"),
                          ("P_succ_ratio", "Success-probability ratio P_succ_hat / P_succ_exact")):
        use_methods = methods if metric != "P_succ_ratio" else ("none",) + tuple(
            m for m in ("rem", "rem_tensored", "zne", "zne_rem") if m != "none")
        headers = ["n", "L"] + [f"{nz} / {m}" for nz in headline for m in use_methods]
        rows = []
        for (n, L), g in summary.groupby(["n_qubits", "depth_layers"], sort=False):
            row = [str(n), str(L)]
            for nz in headline:
                sub = g[g.noise_level == nz].set_index("method")
                for m in use_methods:
                    if m not in sub.index or np.isnan(sub.loc[m, f"{metric}_mean"]):
                        row.append("–")
                    else:
                        row.append(pm(sub.loc[m, f"{metric}_mean"], sub.loc[m, f"{metric}_std"]))
            rows.append(row)
        out.append(f"\n## {title}\n\n" + md_table(headers, rows, "rr" + "r" * (len(headers) - 2)))
    return "".join(out)


def table_overhead(summary: pd.DataFrame, methods: Sequence[str]) -> str:
    qubits = sorted(summary.n_qubits.unique())
    headers = ["method"]
    for n in qubits:
        headers += [f"n={n} circuits", f"n={n} shots", f"n={n} max/base depth", f"n={n} total/base CX"]
    rows = []
    for m in methods:
        row = [m]
        for n in qubits:
            g = summary[(summary.method == m) & (summary.n_qubits == n)]
            if g.empty:
                row += ["–", "–", "–", "–"]
                continue
            depth_ratio = g.max_depth / g.base_depth
            cx_ratio = g.total_cx / g.base_cx
            dr = (f"{depth_ratio.min():.2f}" if np.isclose(depth_ratio.min(), depth_ratio.max())
                  else f"{depth_ratio.min():.2f}–{depth_ratio.max():.2f}")
            row += [str(int(g.n_circuits.iloc[0])), str(int(g.total_shots.iloc[0])), dr,
                    f"{cx_ratio.mean():.0f}"]
        rows.append(row)
    text = ("# Overhead per estimate\n\n"
            "Executions and shots needed for one mitigated estimate, the largest executed depth "
            "relative to the base circuit (range over L), and total CNOTs executed relative to the "
            "base circuit. Calibration circuits contain no CNOTs.\n\n")
    text += md_table(headers, rows, "l" + "r" * (len(headers) - 1))

    headers = ["method"] + [f"n={n} quantum (s)" for n in qubits] + [f"n={n} classical (ms)" for n in qubits]
    rows = []
    for m in methods:
        g = summary[summary.method == m]
        row = [m]
        row += [f"{g[g.n_qubits == n].time_quantum_s_mean.mean():.4f}" for n in qubits]
        row += [f"{1000 * g[g.n_qubits == n].time_classical_s_mean.mean():.3f}" for n in qubits]
        rows.append(row)
    text += ("\n## Wall time per estimate\n\n"
             "Simulator time of the circuits a method needs and classical post-processing time, "
             "averaged over L, noise levels and seeds. Simulator time is not hardware time.\n\n")
    text += md_table(headers, rows, "l" + "r" * (len(headers) - 1))
    return text


def table_resources(summary_a: pd.DataFrame, runs_b: pd.DataFrame | None) -> str:
    """Course-standard resource metrics (qubits, depth, gate count, 2-qubit gate count,
    shots) per method and per (n, L), for both tracks."""
    text = "# Resource metrics (qubits, depth, gates, 2-qubit gates, shots)\n\n## Track A\n\n"
    headers = ["method", "n", "L", "qubits", "base depth", "max depth", "total gates",
               "2-qubit gates (base)", "total shots"]
    rows = []
    for (m, n, L), g in summary_a.groupby(["method", "n_qubits", "depth_layers"], sort=False):
        r = g.iloc[0]
        rows.append([m, str(n), str(L), str(n), str(int(r.base_depth)), str(int(r.max_depth)),
                     str(int(r.total_gates)), str(int(r.base_cx)), str(int(r.total_shots))])
    rows.sort(key=lambda r: (int(r[1]), int(r[2]), METHOD_ORDER.index(r[0])))
    text += md_table(headers, rows, "lrrrrrrrr")

    if runs_b is not None and len(runs_b):
        text += "\n## Track B (VQC; base circuit is identical across noise levels and methods)\n\n"
        vqc = runs_b[(runs_b.model == "vqc") & (runs_b.noise_level == "ideal") & (runs_b.method == "none")]
        headers2 = ["n", "L", "qubits", "trainable params", "base depth", "2-qubit gates (base)",
                    "1-qubit gates (base)"]
        rows2 = []
        for (n, L), g in vqc.groupby(["n_qubits", "depth_layers"], sort=False):
            r = g.iloc[0]
            rows2.append([str(int(n)), str(int(L)), str(int(n)), str(int(r.n_params)),
                         str(int(r.base_depth)), str(int(r.base_cx)), str(int(r.base_1q))])
        rows2.sort(key=lambda r: (int(r[0]), int(r[1])))
        text += md_table(headers2, rows2, "rrrrrrr")
    return text


def extrapolator_errors(runs: pd.DataFrame) -> pd.DataFrame:
    z = runs[runs.method.isin(["zne", "zne_rem"])].copy()
    for name in ("richardson", "linear", "exp"):
        z[f"err_{name}"] = (z[f"E_{name}"] - z.E_exact).abs()
    z["exp_nan"] = z.E_exp.isna().astype(int)
    return z


def table_extrapolators(runs: pd.DataFrame, noise_levels: Sequence[str]) -> str:
    z = extrapolator_errors(runs)
    rows = []
    for nz in noise_levels:
        for m in ("zne", "zne_rem"):
            for n in [*sorted(z.n_qubits.unique()), "all"]:
                g = z[(z.noise_level == nz) & (z.method == m)]
                if n != "all":
                    g = g[g.n_qubits == n]
                rows.append([nz, m, str(n), str(len(g)),
                             f"{g.err_richardson.mean():.3f}", f"{g.err_linear.mean():.3f}",
                             f"{g.err_exp.mean():.3f}" if g.err_exp.notna().any() else "–",
                             str(int(g.extrapolation_out_of_range.sum())), str(int(g.exp_nan.sum()))])
    headers = ["noise", "method", "n", "runs", "Richardson |err|", "linear |err|",
               "exp |err| (valid fits)", "Richardson out of [−1, 1]", "exp fit failed (NaN)"]
    return ("# Extrapolator comparison\n\n"
            "Mean absolute error of each extrapolator, pooled over L and seeds (`all` pools n too). "
            "The exp error averages only the fits that succeeded. `high` is the optional course "
            "stress level, where the folded signal is expected to decay fastest.\n\n"
            + md_table(headers, rows, "lllrrrrrr"))


def table_rem_full_vs_tensored(runs: pd.DataFrame, noise_levels: Sequence[str]) -> str:
    """Improvement 1: does the 2-circuit tensored calibration match the 2^n-circuit full
    calibration? Paired by run_id."""
    wide = runs[runs.method.isin(["rem", "rem_tensored"])].pivot_table(
        index=["n_qubits", "depth_layers", "noise_level", "seed"], columns="method",
        values=["abs_error", "hellinger_fidelity", "P_succ_ratio"],
    )
    rows = []
    for (n, L, nz), g in wide.groupby(level=[0, 1, 2]):
        if nz not in noise_levels:
            continue
        err_full = g[("abs_error", "rem")]
        err_tens = g[("abs_error", "rem_tensored")]
        diff = (err_tens - err_full).to_numpy()
        rows.append([
            str(n), str(L), nz,
            pm(float(err_full.mean()), float(err_full.std(ddof=1))),
            pm(float(err_tens.mean()), float(err_tens.std(ddof=1))),
            f"{float(np.mean(diff)):+.4f}",
            pm(float(g[('hellinger_fidelity', 'rem')].mean()), float(g[('hellinger_fidelity', 'rem')].std(ddof=1))),
            pm(float(g[('hellinger_fidelity', 'rem_tensored')].mean()),
               float(g[('hellinger_fidelity', 'rem_tensored')].std(ddof=1))),
        ])
    rank = {nz: i for i, nz in enumerate(noise_levels)}
    rows.sort(key=lambda r: (int(r[0]), int(r[1]), rank[r[2]]))
    headers = ["n", "L", "noise", "full REM |err|", "tensored REM |err|", "mean diff (tensored − full)",
               "full REM fidelity", "tensored REM fidelity"]
    return ("# Full vs. tensored REM (2 circuits instead of 2ⁿ)\n\n"
            "Full calibration uses 2ⁿ circuits; tensored calibration uses 2, assuming uncorrelated "
            "readout errors (true by construction in this noise model). Paired per circuit instance "
            "and noise level, mean ± std over 5 seeds.\n\n"
            + md_table(headers, rows, "rrlrrrrr"))


def bias_decomposition(runs: pd.DataFrame) -> pd.DataFrame:
    """Split the infinite-shot bias of `none` into gate and readout parts.

    Symmetric readout error scales the parity by exactly a = (1 - 2 p_ro)^n, so the exact
    gate-noise-only parity is E_gate = E_noisy_exact / a.
    """
    d = runs[runs.method == "none"].copy()
    a = (1 - 2 * d.p_ro) ** d.n_qubits
    d["E_gate_exact"] = d.E_noisy_exact / a
    d["gate_bias"] = (d.E_gate_exact - d.E_exact).abs()
    d["readout_bias"] = (d.E_noisy_exact - d.E_gate_exact).abs()
    d["total_bias"] = (d.E_noisy_exact - d.E_exact).abs()
    d["readout_share"] = d.readout_bias / (d.gate_bias + d.readout_bias)
    return d


def table_bias_sources(runs: pd.DataFrame, noise_levels: Sequence[str]) -> str:
    d = bias_decomposition(runs)
    d = d[d.noise_level.isin(noise_levels)]
    rows = []
    for (n, L, nz), g in d.groupby(["n_qubits", "depth_layers", "noise_level"], sort=False):
        rows.append([str(n), str(L), nz, f"{g.total_bias.mean():.3f}", f"{g.gate_bias.mean():.3f}",
                     f"{g.readout_bias.mean():.3f}", f"{100 * g.readout_share.mean():.0f}%"])
    headers = ["n", "L", "noise", "total |bias|", "gate part", "readout part", "readout share"]
    return ("# Sources of the unmitigated bias\n\n"
            "Infinite-shot bias of the unmitigated parity, from the exact noisy density matrix "
            "(`E_noisy_exact`), split exactly into gate-noise and readout parts. Mean over 5 seeds.\n\n"
            + md_table(headers, _sort_rows(rows, noise_levels), "rrlrrrr"))


def _sort_rows(rows, noise_order):
    rank = {nz: i for i, nz in enumerate(noise_order)}
    return sorted(rows, key=lambda r: (int(r[0]), int(r[1]), rank[r[2]]))


def table_rem_negative_mass(runs: pd.DataFrame, noise_order: Sequence[str]) -> str:
    out = ["# REM quasi-probabilities\n",
           "Total negative mass Σ|q_i| over q_i < 0 of the REM quasi-probability vector, over "
           "both depths and 5 seeds (10 runs per row).\n"]
    for method, title in (("rem", "Full calibration"), ("rem_tensored", "Tensored calibration")):
        r = runs[runs.method == method]
        rows = []
        for (n, nz), g in r.groupby(["n_qubits", "noise_level"], sort=False):
            rows.append([str(n), nz, f"{g.rem_negative_mass.mean():.4f}", f"{g.rem_negative_mass.max():.4f}",
                         f"{(g.rem_negative_mass > 0).mean() * 100:.0f}%",
                         f"{g.rem_condition_number.mean():.3f}"])
        rank = {nz: i for i, nz in enumerate(noise_order)}
        rows.sort(key=lambda x: (int(x[0]), rank[x[1]]))
        headers = ["n", "noise", "mean negative mass", "max negative mass", "runs with negative mass",
                   "mean cond(A)"]
        out.append(f"\n## {title}\n\n" + md_table(headers, rows, "llrrrr"))
    return "".join(out)


def table_improvement(summary: pd.DataFrame, noise_order: Sequence[str], methods: Sequence[str]) -> str:
    ms = [m for m in methods if m != "none"]
    headline = [nz for nz in noise_order if nz != "high"]
    headers = ["n", "L"] + [f"{nz} / {m}" for nz in headline for m in ms]
    rows = []
    for (n, L), g in summary.groupby(["n_qubits", "depth_layers"], sort=False):
        row = [str(n), str(L)]
        for nz in headline:
            sub = g[g.noise_level == nz].set_index("method")
            for m in ms:
                if m not in sub.index:
                    row.append("–")
                    continue
                v = sub.loc[m, "improvement_pct_median"]
                row.append(f"{_i(v)}{' †' if nz == 'ideal' else ''}")
        rows.append(row)
    return ("# Median improvement over no mitigation\n\n"
            "Median over 5 seeds of 100·(1 − |err| / |err_none|), in percent; negative values mean "
            "the method made the estimate worse. † At the ideal level the baseline error is pure "
            "shot noise, so these values are unstable and excluded from headline numbers. The "
            "optional `high` stress level is reported separately (`table_error.md`).\n\n"
            + md_table(headers, rows, "rr" + "r" * (len(headers) - 2)))


# --- statistical tests, Track A -------------------------------------------------------


def stats_tests_track_a(runs: pd.DataFrame, noise_order: Sequence[str]) -> pd.DataFrame:
    """Paired tests of each method's abs_error against `none`."""
    wide = runs.pivot_table(index=["n_qubits", "depth_layers", "noise_level", "seed"],
                            columns="method", values="abs_error")
    records = []
    methods = [m for m in METHOD_ORDER if m in wide.columns and m != "none"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (n, L, nz), g in wide.groupby(level=[0, 1, 2], sort=False):
            for m in methods:
                diff = (g[m] - g["none"]).to_numpy()
                if np.allclose(diff, 0):
                    t, p = np.nan, np.nan
                else:
                    res = stats.ttest_rel(g[m], g["none"])
                    t, p = float(res.statistic), float(res.pvalue)
                records.append({"track": "A", "scope": "per_condition", "test": "ttest_rel",
                                "n_qubits": n, "depth_layers": L, "noise_level": nz, "metric": "abs_error",
                                "comparison": m, "n_pairs": len(diff), "statistic": t, "p_value": p,
                                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff))})
        for nz in noise_order:
            g = wide.xs(nz, level="noise_level")
            for m in methods:
                diff = (g[m] - g["none"]).to_numpy()
                try:
                    res = stats.wilcoxon(g[m], g["none"])
                    s, p = float(res.statistic), float(res.pvalue)
                except ValueError:
                    s, p = np.nan, np.nan
                records.append({"track": "A", "scope": "pooled", "test": "wilcoxon", "n_qubits": np.nan,
                                "depth_layers": np.nan, "noise_level": nz, "metric": "abs_error",
                                "comparison": m, "n_pairs": len(diff), "statistic": s, "p_value": p,
                                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff))})
    df = pd.DataFrame(records)
    per = _order(df[df.scope == "per_condition"], noise_order)
    pooled = df[df.scope == "pooled"].reset_index(drop=True)
    return pd.concat([per, pooled], ignore_index=True)


# --- empirical variance of ZNE ---------------------------------------------------------


def variance_check(runs: pd.DataFrame, cfg: Config, raw_dir: Path) -> pd.DataFrame:
    """Shot-noise residuals of `none` and `zne` against their exact infinite-shot values.

    The infinite-shot value of `none` is E_noisy_exact; that of `zne` is the Richardson
    combination of the exact noisy parities of the folded circuits, computed here from the
    density matrix (no sampling). The residuals then contain shot noise only.
    """
    executor = Executor(cfg)
    scales = tuple(cfg.zne.scale_factors)
    gammas = richardson_coeffs(scales)
    instances = {}
    records = []
    zne = runs[runs.method == "zne"].set_index("run_id")
    for _, row in runs[runs.method == "none"].iterrows():
        key = (row.n_qubits, row.depth_layers, row.seed)
        if key not in instances:
            instances[key] = load_instance(raw_dir / "circuits" / f"n{key[0]}_L{key[1]}_s{key[2]}.json")
        u = instances[key].unitary
        exact_by_scale = [row.E_noisy_exact] + [
            parity_from_probs(executor.exact_noisy_probs(fold_global(u, s), row.noise_level))
            for s in scales[1:]
        ]
        z = zne.loc[row.run_id]
        zne_inf = float(gammas @ np.array(exact_by_scale))
        a = (1 - 2 * row.p_ro) ** row.n_qubits
        records.append({
            "run_id": row.run_id, "n_qubits": row.n_qubits, "depth_layers": row.depth_layers,
            "noise_level": row.noise_level, "seed": row.seed,
            "none_residual": row.E_hat - row.E_noisy_exact,
            "none_pred_std": row.est_std,
            "zne_inf_shot": zne_inf,
            "zne_residual": z.E_hat - zne_inf,
            "zne_pred_std": z.est_std,
            "zne_inf_shot_abs_error": abs(zne_inf - row.E_exact),
            "zne_gate_only_inf_shot_abs_error": abs(zne_inf / a - row.E_exact),
        })
    return pd.DataFrame(records)


def variance_summary(var: pd.DataFrame, noise_order: Sequence[str]) -> pd.DataFrame:
    recs = []
    for nz in noise_order:
        g = var[var.noise_level == nz]
        if not len(g):
            continue
        rms_none = float(np.sqrt(np.mean(g.none_residual**2)))
        rms_zne = float(np.sqrt(np.mean(g.zne_residual**2)))
        recs.append({
            "noise_level": nz, "runs": len(g),
            "rms_residual_none": rms_none, "rms_residual_zne": rms_zne,
            "empirical_std_ratio": rms_zne / rms_none,
            "predicted_std_ratio_mean": float(np.mean(g.zne_pred_std / g.none_pred_std)),
            "zne_inf_shot_abs_error_mean": float(g.zne_inf_shot_abs_error.mean()),
            "zne_gate_only_inf_shot_abs_error_mean": float(g.zne_gate_only_inf_shot_abs_error.mean()),
        })
    return pd.DataFrame(recs)


def table_variance(vs: pd.DataFrame) -> str:
    rows = [[r.noise_level, str(r.runs), f"{r.rms_residual_none:.4f}", f"{r.rms_residual_zne:.4f}",
             f"{r.empirical_std_ratio:.2f}", f"{r.predicted_std_ratio_mean:.2f}",
             f"{r.zne_inf_shot_abs_error_mean:.4f}", f"{r.zne_gate_only_inf_shot_abs_error_mean:.4f}"]
            for r in vs.itertuples()]
    headers = ["noise", "runs", "RMS shot residual none", "RMS shot residual zne",
               "empirical std ratio", "predicted std ratio", "zne |err|, infinite shots",
               "zne |err|, infinite shots, readout removed"]
    return ("# Variance amplification and residual bias of ZNE\n\n"
            "Residuals of each estimate against its own exact infinite-shot value (density-matrix "
            "simulation of the base and folded circuits), so they contain shot noise only. "
            "The predicted ratio is the mean of zne_std / est_std(none); the theoretical value at "
            "E = 0 is √5.22 ≈ 2.28. The last two columns are the error ZNE would leave with unlimited "
            "shots: as run, and with the exact readout attenuation (1 − 2p_ro)ⁿ divided out, which "
            "isolates the extrapolation error on gate noise alone.\n\n"
            + md_table(headers, rows, "lrrrrrrr"))


# ======================================================================================
# Track B
# ======================================================================================


def summarize_qml(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, n, L, noise, method): mean, std, median, CI95 per metric.

    Classical rows have no L/noise/method (NaN), so groupby keeps NaN as its own group.
    """
    records = []
    for keys, g in runs.groupby(QML_GROUP, sort=False, dropna=False):
        rec = dict(zip(QML_GROUP, keys))
        rec["n_seeds"] = len(g)
        for m in QML_SUMMARY_METRICS:
            x = g[m].to_numpy(dtype=float)
            x = x[~np.isnan(x) & np.isfinite(x)]
            k = len(x)
            if k == 0:
                mean = std = median = lo = hi = np.nan
            else:
                mean = float(np.mean(x))
                median = float(np.median(x))
                std = float(np.std(x, ddof=1)) if k > 1 else np.nan
                half = t_crit(k) * std / np.sqrt(k) if k > 1 else np.nan
                lo, hi = mean - half, mean + half
            rec.update({f"{m}_mean": mean, f"{m}_std": std, f"{m}_median": median,
                        f"{m}_ci95_low": lo, f"{m}_ci95_high": hi})
        for c in ("n_train", "n_test", "n_params"):
            rec[c] = g[c].iloc[0]
        records.append(rec)
    df = pd.DataFrame(records)
    model_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    method_rank = {m: i for i, m in enumerate(QML_METHOD_ORDER)}
    key = pd.DataFrame({
        "model": df.model.map(model_rank), "n": df.n_qubits,
        "L": df.depth_layers.fillna(-1), "noise": df.noise_level.fillna("exact"),
        "method": df.method.map(method_rank).fillna(-1),
    })
    order = key.sort_values(["model", "n", "L", "noise", "method"], kind="mergesort").index
    return df.loc[order].reset_index(drop=True)


def table_qml_classification(summary: pd.DataFrame, noise_order: Sequence[str]) -> str:
    text = ("# Classification metrics: VQC vs. Logistic Regression and SVM\n\n"
            "Mean ± std over 5 seeds. LR and SVM do not depend on noise level, depth or method; "
            "the same row is shown in every (n, L) block for comparison. `exact` is the noiseless "
            "VQC. No quantum advantage is claimed; see the fairness statement in the report.\n\n")
    for n in sorted(summary.n_qubits.unique()):
        lr = summary[(summary.model == "logreg") & (summary.n_qubits == n)]
        svm = summary[(summary.model == "svm") & (summary.n_qubits == n)]
        for L in sorted(summary[(summary.model == "vqc") & (summary.n_qubits == n)].depth_layers.dropna().unique()):
            text += f"\n## n = {int(n)}, L = {int(L)}\n\n"
            headers = ["model / condition", "accuracy", "precision", "recall", "f1"]
            rows = []
            for label, g in (("Logistic Regression", lr), ("SVM", svm)):
                if len(g):
                    r = g.iloc[0]
                    rows.append([label, pm(r.accuracy_mean, r.accuracy_std), pm(r.precision_mean, r.precision_std),
                                pm(r.recall_mean, r.recall_std), pm(r.f1_mean, r.f1_std)])
            vqc = summary[(summary.model == "vqc") & (summary.n_qubits == n) & (summary.depth_layers == L)]
            ex = vqc[vqc.noise_level == "exact"]
            if len(ex):
                r = ex.iloc[0]
                rows.append(["VQC, exact (noiseless)", pm(r.accuracy_mean, r.accuracy_std),
                            pm(r.precision_mean, r.precision_std), pm(r.recall_mean, r.recall_std),
                            pm(r.f1_mean, r.f1_std)])
            for nz in noise_order:
                for m in QML_METHOD_ORDER:
                    g = vqc[(vqc.noise_level == nz) & (vqc.method == m)]
                    if not len(g):
                        continue
                    r = g.iloc[0]
                    rows.append([f"VQC, {nz} / {m}", pm(r.accuracy_mean, r.accuracy_std),
                                pm(r.precision_mean, r.precision_std), pm(r.recall_mean, r.recall_std),
                                pm(r.f1_mean, r.f1_std)])
            text += md_table(headers, rows, "lrrrr")
    return text


def table_qml_confusion(runs: pd.DataFrame, n: int, L: int) -> str:
    """Confusion matrices summed over the 5 seeds, for the configured (n, L) condition."""
    conditions = [("Logistic Regression", {"model": "logreg", "n_qubits": n}),
                  ("SVM", {"model": "svm", "n_qubits": n}),
                  ("VQC, exact", {"model": "vqc", "n_qubits": n, "depth_layers": L, "noise_level": "exact"}),
                  ("VQC, moderate / none", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                            "noise_level": "moderate", "method": "none"}),
                  ("VQC, moderate / zne_rem", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                               "noise_level": "moderate", "method": "zne_rem"}),
                  ("VQC, high / none", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                        "noise_level": "high", "method": "none"}),
                  ("VQC, high / zne_rem", {"model": "vqc", "n_qubits": n, "depth_layers": L,
                                           "noise_level": "high", "method": "zne_rem"})]
    rows = []
    for label, filt in conditions:
        mask = np.ones(len(runs), dtype=bool)
        for k, v in filt.items():
            mask &= (runs[k] == v)
        g = runs[mask]
        if not len(g):
            continue
        tp, fn, fp, tn = int(g.tp.sum()), int(g.fn.sum()), int(g.fp.sum()), int(g.tn.sum())
        acc = (tp + tn) / (tp + fn + fp + tn)
        rows.append([label, tp, fn, fp, tn, f"{acc:.3f}"])
    headers = ["condition", "TP", "FN", "FP", "TN", "accuracy (pooled)"]
    return (f"# Confusion matrices, summed over 5 seeds (n = {n}, L = {L})\n\n"
            "Rows: TP/FN/FP/TN with versicolor (+1) as the positive class, pooled over the 5 "
            "seeds' 20-sample test sets (100 predictions per condition).\n\n"
            + md_table(headers, rows, "lrrrrr"))


def stats_tests_track_b(runs: pd.DataFrame, noise_order: Sequence[str]) -> pd.DataFrame:
    vqc = runs[runs.model == "vqc"]
    methods = [m for m in QML_METHOD_ORDER if m != "none"]
    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for metric in ("accuracy", "mean_abs_E_error"):
            wide = vqc.pivot_table(index=["n_qubits", "depth_layers", "noise_level", "seed"],
                                   columns="method", values=metric)
            for (n, L, nz), g in wide.groupby(level=[0, 1, 2], sort=False):
                if nz not in noise_order:
                    continue
                for m in methods:
                    diff = (g[m] - g["none"]).to_numpy()
                    if np.allclose(diff, 0):
                        t, p = np.nan, np.nan
                    else:
                        res = stats.ttest_rel(g[m], g["none"])
                        t, p = float(res.statistic), float(res.pvalue)
                    records.append({"track": "B", "scope": "per_condition", "test": "ttest_rel",
                                    "n_qubits": n, "depth_layers": L, "noise_level": nz,
                                    "metric": metric, "comparison": m, "n_pairs": len(diff),
                                    "statistic": t, "p_value": p, "mean_diff": float(diff.mean()),
                                    "median_diff": float(np.median(diff))})

        # VQC (exact, noiseless) vs. LR and vs. SVM, paired by seed, per (n, L)
        exact = vqc[vqc.noise_level == "exact"]
        for n in sorted(exact.n_qubits.unique()):
            for comparator in ("logreg", "svm"):
                comp = runs[(runs.model == comparator) & (runs.n_qubits == n)].set_index("seed")["accuracy"]
                for L in sorted(exact[exact.n_qubits == n].depth_layers.unique()):
                    e = exact[(exact.n_qubits == n) & (exact.depth_layers == L)].set_index("seed")["accuracy"]
                    seeds = sorted(set(e.index) & set(comp.index))
                    diff = (e.loc[seeds] - comp.loc[seeds]).to_numpy()
                    if np.allclose(diff, 0):
                        t, p = np.nan, np.nan
                    else:
                        res = stats.ttest_rel(e.loc[seeds], comp.loc[seeds])
                        t, p = float(res.statistic), float(res.pvalue)
                    records.append({"track": "B", "scope": "vs_classical", "test": "ttest_rel",
                                    "n_qubits": n, "depth_layers": L, "noise_level": "exact",
                                    "metric": "accuracy", "comparison": f"vqc_vs_{comparator}",
                                    "n_pairs": len(seeds), "statistic": t, "p_value": p,
                                    "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff))})
    return pd.DataFrame(records)


# ======================================================================================
# Improvement: variance-optimal shot allocation
# ======================================================================================


def table_zne_allocation(alloc: pd.DataFrame) -> str:
    headers = ["n", "L", "noise", "scheme", "shots (λ1,λ3,λ5)", "bias", "empirical std",
               "rmse", "theoretical std"]
    rows = []
    for (n, L, nz), g in alloc.groupby(["n_qubits", "depth_layers", "noise_level"], sort=False):
        for scheme in ("uniform", "optimal"):
            r = g[g.scheme == scheme]
            if not len(r):
                continue
            r = r.iloc[0]
            rows.append([str(int(n)), str(int(L)), nz, scheme,
                         f"{int(r.shots_lambda1)},{int(r.shots_lambda3)},{int(r.shots_lambda5)}",
                         f"{r.bias:+.4f}", f"{r.empirical_std:.4f}", f"{r.rmse:.4f}",
                         f"{r.theoretical_std:.4f}"])
    rows.sort(key=lambda r: (int(r[0]), int(r[1]), r[2], r[3]))
    text = ("# Variance-optimal ZNE shot allocation vs. uniform\n\n"
            f"Same total budget (3072 shots) split uniformly (1024 each) or proportional to "
            "|γ_i| (the variance-minimizing allocation). Each row aggregates "
            f"{int(alloc.repetitions.iloc[0])} repetitions at a fixed (n, L, noise). "
            "Theoretical std assumes σ_i ≈ 1 at every scale.\n\n")
    text += md_table(headers, rows, "rrllrrrrr")

    summary_rows = []
    for scheme in ("uniform", "optimal"):
        g = alloc[alloc.scheme == scheme]
        summary_rows.append([scheme, f"{g.empirical_std.mean():.4f}", f"{g.rmse.mean():.4f}",
                             f"{g.theoretical_std.iloc[0]:.4f}"])
    ratio_emp = alloc[alloc.scheme == "optimal"].empirical_std.mean() / alloc[alloc.scheme == "uniform"].empirical_std.mean()
    ratio_theory = alloc[alloc.scheme == "optimal"].theoretical_std.iloc[0] / alloc[alloc.scheme == "uniform"].theoretical_std.iloc[0]
    text += ("\n## Pooled over every condition\n\n"
            + md_table(["scheme", "mean empirical std", "mean rmse", "theoretical std"], summary_rows, "lrrr")
            + f"\nEmpirical std ratio (optimal / uniform): **{ratio_emp:.3f}**. "
              f"Theoretical ratio: **{ratio_theory:.3f}**.\n")
    return text


# ======================================================================================
# Combined statistical tests
# ======================================================================================

STATS_COLUMNS = ["track", "scope", "test", "n_qubits", "depth_layers", "noise_level", "metric",
                 "comparison", "n_pairs", "statistic", "p_value", "mean_diff", "median_diff"]


def combined_stats_tests(runs_a: pd.DataFrame, noise_order_a: Sequence[str],
                         runs_b: pd.DataFrame | None, noise_order_b: Sequence[str]) -> pd.DataFrame:
    parts = [stats_tests_track_a(runs_a, noise_order_a)]
    if runs_b is not None and len(runs_b):
        parts.append(stats_tests_track_b(runs_b, noise_order_b))
    return pd.concat(parts, ignore_index=True).reindex(columns=STATS_COLUMNS)


# ======================================================================================
# Numbers quoted in the report
# ======================================================================================


def _f(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _pval(p: float) -> str:
    return "n/a" if not np.isfinite(p) else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def _env_versions(env_path: Path) -> dict:
    out = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "==" in line:
                name, ver = line.split("==", 1)
                out[name.strip().lower()] = ver.strip()
    return out


def report_numbers(
    runs: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame, vs: pd.DataFrame,
    results_dir: Path, headline_levels: Sequence[str],
    runs_b: pd.DataFrame | None = None, summary_b: pd.DataFrame | None = None,
    alloc: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Every number the report text quotes, formatted, keyed by a stable name."""
    num: dict[str, str] = {}
    mitigators = ("rem", "rem_tensored", "zne", "zne_rem")
    headline = list(headline_levels)

    env = _env_versions(results_dir / "environment.txt")
    for pkg in ("qiskit", "qiskit-aer", "numpy", "scipy", "pandas", "matplotlib", "scikit-learn"):
        num[f"env.{pkg}"] = env.get(pkg, "unknown")
    pyv = results_dir / "python_version.txt"
    num["env.python"] = pyv.read_text(encoding="utf-8").splitlines()[0] if pyv.exists() else "unknown"
    num["grid.rows"] = str(len(runs))
    num["grid.conditions"] = str(runs.run_id.nunique())
    inst = runs.drop_duplicates(["n_qubits", "depth_layers", "seed"])
    num["grid.instances"] = str(len(inst))
    num["attempts.max"] = str(int(inst.circuit_attempts.max()))
    num["attempts.median"] = f"{inst.circuit_attempts.median():.0f}"
    num["E_exact.absmin"] = _f(inst.E_exact.abs().min())
    num["E_exact.absmax"] = _f(inst.E_exact.abs().max())

    for nz in ("ideal",) + tuple(headline) + ("high",):
        lvl = runs[runs.noise_level == nz]
        if not len(lvl):
            continue
        for m in METHOD_ORDER:
            sub = lvl[lvl.method == m]
            if len(sub):
                num[f"err.mean.{m}.{nz}"] = _f(sub.abs_error.mean())
        for m in mitigators:
            g = lvl[lvl.method == m]
            if not len(g):
                continue
            num[f"imp.median.{m}.{nz}"] = _i(g.improvement_pct.median())
            num[f"imp.better.{m}.{nz}"] = f"{int((g.improvement_pct > 0).sum())}/{len(g)}"

    per = tests[(tests.track == "A") & (tests.scope == "per_condition")]
    pooled = tests[(tests.track == "A") & (tests.scope == "pooled")]
    for nz in headline:
        for m in mitigators:
            rowp = pooled[(pooled.noise_level == nz) & (pooled.comparison == m)]
            if len(rowp):
                row = rowp.iloc[0]
                num[f"wilcoxon.p.{m}.{nz}"] = _pval(row.p_value)
                num[f"wilcoxon.meandiff.{m}.{nz}"] = _f(row.mean_diff)
            pc = per[(per.noise_level == nz) & (per.comparison == m)]
            if len(pc):
                num[f"ttest.sig.{m}.{nz}"] = f"{int(((pc.p_value < 0.05) & (pc.mean_diff < 0)).sum())}/{len(pc)}"
        cells = summary[summary.noise_level == nz]
        if len(cells):
            best = cells.loc[cells.groupby(["n_qubits", "depth_layers"]).abs_error_mean.idxmin()]
            for m in METHOD_ORDER:
                num[f"best.count.{m}.{nz}"] = f"{int((best.method == m).sum())}/{len(best)}"

    bias = bias_decomposition(runs)
    for nz in headline:
        b = bias[bias.noise_level == nz]
        if not len(b):
            continue
        num[f"bias.total.{nz}"] = _f(b.total_bias.mean())
        num[f"bias.gate.{nz}"] = _f(b.gate_bias.mean())
        num[f"bias.readout.{nz}"] = _f(b.readout_bias.mean())
        num[f"bias.readout_share.{nz}"] = f"{100 * b.readout_share.mean():.0f}"

    for m in METHOD_ORDER:
        for n in sorted(runs.n_qubits.unique()):
            g = runs[(runs.method == m) & (runs.n_qubits == n)]
            if not len(g):
                continue
            num[f"circuits.{m}.n{n}"] = str(int(g.n_circuits.iloc[0]))
            num[f"shots.{m}.n{n}"] = f"{int(g.total_shots.iloc[0]):,}"
    z = runs[runs.method == "zne"]
    if len(z):
        num["zne.depth_ratio.max"] = _f((z.max_depth / z.base_depth).max(), 2)
        num["zne.cx_ratio"] = f"{(z.total_cx / z.base_cx).mean():.0f}"
        num["zne.depth.max"] = str(int(z.max_depth.max()))

    ideal = runs[runs.noise_level == "ideal"].pivot_table(index="run_id", columns="method", values="E_hat")
    if "rem" in ideal.columns and "none" in ideal.columns:
        maxdiff = (ideal["rem"] - ideal["none"]).abs().max()
        num["ideal.rem_none_maxdiff"] = "0" if maxdiff == 0 else f"{maxdiff:.1e}"
    ie = runs[runs.noise_level == "ideal"].pivot_table(index="run_id", columns="method", values="abs_error")
    if "zne" in ie.columns and "none" in ie.columns:
        num["ideal.zne_worse"] = f"{int((ie['zne'] > ie['none']).sum())}/{len(ie)}"

    if vs is not None and len(vs):
        for r in vs.itertuples():
            num[f"var.ratio.emp.{r.noise_level}"] = _f(r.empirical_std_ratio, 2)
            num[f"var.ratio.pred.{r.noise_level}"] = _f(r.predicted_std_ratio_mean, 2)
        num["var.ratio.theory"] = _f(float(np.sqrt(np.sum(richardson_coeffs((1, 3, 5)) ** 2))), 2)

    zz = runs[runs.method.isin(["zne", "zne_rem"])]
    for m in ("zne", "zne_rem"):
        g = zz[zz.method == m]
        num[f"oor.{m}"] = f"{int(g.extrapolation_out_of_range.sum())}/{len(g)}"
        num[f"expnan.{m}"] = f"{int(g.E_exp.isna().sum())}/{len(g)}"
    num["oor.zne.high"] = f"{int(zz[(zz.method=='zne')&(zz.noise_level=='high')].extrapolation_out_of_range.sum())}"

    r = runs[runs.method == "rem"]
    rt = runs[runs.method == "rem_tensored"]
    if len(r):
        num["negmass.max"] = f"{r.rem_negative_mass.max():.4f}"
        num["lstsq.count"] = str(int((r.rem_condition_number >= 1e8).sum()))
    if len(rt):
        num["negmass.max.tensored"] = f"{rt.rem_negative_mass.max():.4f}"

    wide_rt = runs[runs.method.isin(["rem", "rem_tensored"])].pivot_table(
        index=["n_qubits", "depth_layers", "noise_level", "seed"], columns="method", values="abs_error")
    if "rem" in wide_rt.columns and "rem_tensored" in wide_rt.columns:
        diff = (wide_rt["rem_tensored"] - wide_rt["rem"]).abs()
        num["rem_tensored.max_abs_diff"] = _f(diff.max(), 4)
        num["rem_tensored.mean_abs_diff"] = _f(diff.mean(), 4)

    # Track B
    if runs_b is not None and len(runs_b):
        vqc = runs_b[runs_b.model == "vqc"]
        for n in sorted(runs_b.n_qubits.dropna().unique()):
            lr = runs_b[(runs_b.model == "logreg") & (runs_b.n_qubits == n)]
            svm = runs_b[(runs_b.model == "svm") & (runs_b.n_qubits == n)]
            if len(lr):
                num[f"qml.logreg.acc.n{int(n)}"] = _f(lr.accuracy.mean())
            if len(svm):
                num[f"qml.svm.acc.n{int(n)}"] = _f(svm.accuracy.mean())
        ex = vqc[vqc.noise_level == "exact"]
        num["qml.vqc_exact.acc.mean"] = _f(ex.accuracy.mean())
        num["qml.vqc_exact.acc.min"] = _f(ex.accuracy.min())
        num["qml.vqc_exact.acc.max"] = _f(ex.accuracy.max())
        for nz in ("ideal",) + tuple(headline) + ("high",):
            for m in QML_METHOD_ORDER:
                g = vqc[(vqc.noise_level == nz) & (vqc.method == m)]
                if len(g):
                    num[f"qml.acc.{m}.{nz}"] = _f(g.accuracy.mean())
                    num[f"qml.meanE.{m}.{nz}"] = _f(g.mean_abs_E_error.mean())
                    num[f"qml.margin.{m}.{nz}"] = _f(g.margin_retention.mean())
        ideal_vqc = vqc[vqc.noise_level == "ideal"].pivot_table(index=["n_qubits", "depth_layers", "seed"],
                                                                columns="method", values="accuracy")
        if "rem" in ideal_vqc.columns:
            num["qml.ideal.rem_none_maxdiff"] = _f((ideal_vqc["rem"] - ideal_vqc["none"]).abs().max(), 6)

    if alloc is not None and len(alloc):
        ratio = alloc[alloc.scheme == "optimal"].empirical_std.mean() / alloc[alloc.scheme == "uniform"].empirical_std.mean()
        theory_ratio = (alloc[alloc.scheme == "optimal"].theoretical_std.iloc[0]
                        / alloc[alloc.scheme == "uniform"].theoretical_std.iloc[0])
        num["alloc.ratio.empirical"] = _f(ratio, 3)
        num["alloc.ratio.theory"] = _f(theory_ratio, 3)
        num["alloc.shots.optimal"] = ",".join(
            str(int(alloc[alloc.scheme == "optimal"].iloc[0][f"shots_lambda{s}"])) for s in (1, 3, 5))
        num["alloc.repetitions"] = str(int(alloc.repetitions.iloc[0]))

    return num
