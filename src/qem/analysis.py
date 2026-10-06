"""Aggregation over seeds, confidence intervals, statistical tests and markdown tables."""

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
    "hellinger_fidelity", "tvd", "M_abs_error", "est_std",
]
OVERHEAD = ["n_circuits", "total_shots", "base_depth", "max_depth", "base_cx", "total_cx", "total_1q"]
TIMING = ["time_quantum_s", "time_classical_s"]
NOISY_LEVELS = ("low", "moderate")


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


# --- summary.csv ---------------------------------------------------------------------


def summarize(runs: pd.DataFrame, noise_order: Sequence[str]) -> pd.DataFrame:
    """One row per (n, L, noise, method): mean, std (ddof=1), median and 95% CI per metric."""
    records = []
    for keys, g in runs.groupby(GROUP, sort=False):
        rec = dict(zip(GROUP, keys))
        rec["n_seeds"] = len(g)
        for m in SUMMARY_METRICS:
            x = g[m].to_numpy(dtype=float)
            x = x[~np.isnan(x)]
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


# --- markdown helpers ----------------------------------------------------------------


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str | None = None) -> str:
    align = align or ("l" * len(headers))
    sep = ["---" if a == "l" else "---:" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def pm(mean: float, std: float, digits: int = 3) -> str:
    if np.isnan(mean):
        return "–"
    if np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _cell(summary, n, L, noise, method, col):
    row = summary[(summary.n_qubits == n) & (summary.depth_layers == L)
                  & (summary.noise_level == noise) & (summary.method == method)]
    return row.iloc[0] if len(row) else None


# --- tables --------------------------------------------------------------------------


def table_error(summary: pd.DataFrame, noise_order, methods) -> str:
    headers = ["n", "L"] + [f"{nz} / {m}" for nz in noise_order for m in methods]
    rows = []
    for (n, L), g in summary.groupby(["n_qubits", "depth_layers"], sort=False):
        row = [str(n), str(L)]
        for nz in noise_order:
            sub = g[g.noise_level == nz].set_index("method")
            best = sub.loc[list(methods), "abs_error_mean"].idxmin()
            for m in methods:
                cell = pm(sub.loc[m, "abs_error_mean"], sub.loc[m, "abs_error_std"])
                row.append(f"**{cell}**" if m == best else cell)
        rows.append(row)
    intro = (
        "Absolute error |E_hat − E_exact| of the parity, mean ± std over 5 seeds. "
        "The lowest mean per (n, L, noise level) is in bold. At the ideal level the error "
        "is pure shot noise.\n\n"
    )
    return "# Absolute error by method\n\n" + intro + md_table(headers, rows, "rr" + "r" * (len(headers) - 2))


def table_fidelity(summary: pd.DataFrame, noise_order) -> str:
    out = ["# Distribution fidelity: none vs rem\n",
           "Computed against the exact noiseless distribution, mean ± std over 5 seeds. "
           "ZNE produces no distribution, so `zne` and `zne_rem` have no entries by design.\n"]
    for metric, title in (("hellinger_fidelity", "Hellinger fidelity (higher is better)"),
                          ("tvd", "Total variation distance (lower is better)")):
        headers = ["n", "L"] + [f"{nz} / {m}" for nz in noise_order for m in ("none", "rem")]
        rows = []
        for (n, L), g in summary.groupby(["n_qubits", "depth_layers"], sort=False):
            row = [str(n), str(L)]
            for nz in noise_order:
                sub = g[g.noise_level == nz].set_index("method")
                for m in ("none", "rem"):
                    row.append(pm(sub.loc[m, f"{metric}_mean"], sub.loc[m, f"{metric}_std"]))
            rows.append(row)
        out.append(f"\n## {title}\n\n" + md_table(headers, rows, "rr" + "r" * (len(headers) - 2)))
    return "".join(out)


def table_overhead(summary: pd.DataFrame, methods) -> str:
    qubits = sorted(summary.n_qubits.unique())
    headers = ["method"]
    for n in qubits:
        headers += [f"n={n} circuits", f"n={n} shots", f"n={n} max/base depth", f"n={n} total/base CX"]
    rows = []
    for m in methods:
        row = [m]
        for n in qubits:
            g = summary[(summary.method == m) & (summary.n_qubits == n)]
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


def extrapolator_errors(runs: pd.DataFrame) -> pd.DataFrame:
    z = runs[runs.method.isin(["zne", "zne_rem"]) & runs.noise_level.isin(NOISY_LEVELS)].copy()
    for name in ("richardson", "linear", "exp"):
        z[f"err_{name}"] = (z[f"E_{name}"] - z.E_exact).abs()
    z["exp_nan"] = z.E_exp.isna().astype(int)
    return z


def table_extrapolators(runs: pd.DataFrame) -> str:
    z = extrapolator_errors(runs)
    rows = []
    for nz in NOISY_LEVELS:
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
            "Mean absolute error of each extrapolator at the noisy levels, pooled over L and seeds "
            "(`all` pools n too). The exp error averages only the fits that succeeded.\n\n"
            + md_table(headers, rows, "lllrrrrrr"))


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


def table_bias_sources(runs: pd.DataFrame, noise_order) -> str:
    d = bias_decomposition(runs)
    d = d[d.noise_level.isin(NOISY_LEVELS)]
    rows = []
    for (n, L, nz), g in d.groupby(["n_qubits", "depth_layers", "noise_level"], sort=False):
        rows.append([str(n), str(L), nz, f"{g.total_bias.mean():.3f}", f"{g.gate_bias.mean():.3f}",
                     f"{g.readout_bias.mean():.3f}", f"{100 * g.readout_share.mean():.0f}%"])
    headers = ["n", "L", "noise", "total |bias|", "gate part", "readout part", "readout share"]
    return ("# Sources of the unmitigated bias\n\n"
            "Infinite-shot bias of the unmitigated parity, from the exact noisy density matrix "
            "(`E_noisy_exact`), split exactly into gate-noise and readout parts. Mean over 5 seeds.\n\n"
            + md_table(headers, _sort_rows(rows, noise_order), "rrlrrrr"))


def _sort_rows(rows, noise_order):
    rank = {nz: i for i, nz in enumerate(noise_order)}
    return sorted(rows, key=lambda r: (int(r[0]), int(r[1]), rank[r[2]]))


def table_rem_negative_mass(runs: pd.DataFrame, noise_order) -> str:
    r = runs[runs.method == "rem"]
    rows = []
    for (n, nz), g in r.groupby(["n_qubits", "noise_level"], sort=False):
        rows.append([str(n), nz, f"{g.rem_negative_mass.mean():.4f}", f"{g.rem_negative_mass.max():.4f}",
                     f"{(g.rem_negative_mass > 0).mean() * 100:.0f}%",
                     f"{g.rem_condition_number.mean():.3f}"])
    rank = {nz: i for i, nz in enumerate(noise_order)}
    rows.sort(key=lambda x: (int(x[0]), rank[x[1]]))
    headers = ["n", "noise", "mean negative mass", "max negative mass", "runs with negative mass", "mean cond(A)"]
    return ("# REM quasi-probabilities\n\n"
            "Total negative mass Σ|q_i| over q_i < 0 of the REM quasi-probability vector, over "
            "both depths and 5 seeds (10 runs per row).\n\n"
            + md_table(headers, rows, "llrrrr"))


def table_improvement(summary: pd.DataFrame, noise_order, methods) -> str:
    ms = [m for m in methods if m != "none"]
    headers = ["n", "L"] + [f"{nz} / {m}" for nz in noise_order for m in ms]
    rows = []
    for (n, L), g in summary.groupby(["n_qubits", "depth_layers"], sort=False):
        row = [str(n), str(L)]
        for nz in noise_order:
            sub = g[g.noise_level == nz].set_index("method")
            for m in ms:
                v = sub.loc[m, "improvement_pct_median"]
                row.append(f"{v:.0f}{' †' if nz == 'ideal' else ''}")
        rows.append(row)
    return ("# Median improvement over no mitigation\n\n"
            "Median over 5 seeds of 100·(1 − |err| / |err_none|), in percent; negative values mean "
            "the method made the estimate worse. † At the ideal level the baseline error is pure "
            "shot noise, so these values are unstable and excluded from headline numbers.\n\n"
            + md_table(headers, rows, "rr" + "r" * (len(headers) - 2)))


# --- statistical tests ---------------------------------------------------------------


def stats_tests(runs: pd.DataFrame, noise_order) -> pd.DataFrame:
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
                records.append({"scope": "per_condition", "test": "ttest_rel", "n_qubits": n,
                                "depth_layers": L, "noise_level": nz, "method": m,
                                "n_pairs": len(diff), "statistic": t, "p_value": p,
                                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff))})
        for nz in NOISY_LEVELS:
            g = wide.xs(nz, level="noise_level")
            for m in methods:
                diff = (g[m] - g["none"]).to_numpy()
                try:
                    res = stats.wilcoxon(g[m], g["none"])
                    s, p = float(res.statistic), float(res.pvalue)
                except ValueError:
                    s, p = np.nan, np.nan
                records.append({"scope": "pooled", "test": "wilcoxon", "n_qubits": np.nan,
                                "depth_layers": np.nan, "noise_level": nz, "method": m,
                                "n_pairs": len(diff), "statistic": s, "p_value": p,
                                "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff))})
    df = pd.DataFrame(records)
    per = _order(df[df.scope == "per_condition"], noise_order)
    pooled = df[df.scope == "pooled"].reset_index(drop=True)
    return pd.concat([per, pooled], ignore_index=True)


# --- empirical variance of ZNE -------------------------------------------------------


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
        records.append({
            "run_id": row.run_id, "n_qubits": row.n_qubits, "depth_layers": row.depth_layers,
            "noise_level": row.noise_level, "seed": row.seed,
            "none_residual": row.E_hat - row.E_noisy_exact,
            "none_pred_std": row.est_std,
            "zne_inf_shot": float(gammas @ np.array(exact_by_scale)),
            "zne_residual": z.E_hat - float(gammas @ np.array(exact_by_scale)),
            "zne_pred_std": z.est_std,
            "zne_inf_shot_abs_error": abs(float(gammas @ np.array(exact_by_scale)) - row.E_exact),
        })
    return pd.DataFrame(records)


def variance_summary(var: pd.DataFrame, noise_order) -> pd.DataFrame:
    recs = []
    for nz in noise_order:
        g = var[var.noise_level == nz]
        rms_none = float(np.sqrt(np.mean(g.none_residual**2)))
        rms_zne = float(np.sqrt(np.mean(g.zne_residual**2)))
        recs.append({
            "noise_level": nz, "runs": len(g),
            "rms_residual_none": rms_none, "rms_residual_zne": rms_zne,
            "empirical_std_ratio": rms_zne / rms_none,
            "predicted_std_ratio_mean": float(np.mean(g.zne_pred_std / g.none_pred_std)),
            "zne_inf_shot_abs_error_mean": float(g.zne_inf_shot_abs_error.mean()),
        })
    return pd.DataFrame(recs)


def table_variance(vs: pd.DataFrame) -> str:
    rows = [[r.noise_level, str(r.runs), f"{r.rms_residual_none:.4f}", f"{r.rms_residual_zne:.4f}",
             f"{r.empirical_std_ratio:.2f}", f"{r.predicted_std_ratio_mean:.2f}",
             f"{r.zne_inf_shot_abs_error_mean:.4f}"] for r in vs.itertuples()]
    headers = ["noise", "runs", "RMS shot residual none", "RMS shot residual zne",
               "empirical std ratio", "predicted std ratio", "zne |err| at infinite shots"]
    return ("# Variance amplification of ZNE\n\n"
            "Residuals of each estimate against its own exact infinite-shot value (density-matrix "
            "simulation of the base and folded circuits), so they contain shot noise only. "
            "The predicted ratio is the mean of zne_std / est_std(none); the theoretical value at "
            "E = 0 is √5.22 ≈ 2.28. The last column is the bias ZNE would leave with unlimited shots.\n\n"
            + md_table(headers, rows, "lrrrrrr"))
