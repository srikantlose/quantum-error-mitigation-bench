"""Track A: run one condition (n, L, noise, seed) for every method, and run the full sweep."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import metrics
from .circuits import (
    CircuitInstance,
    calibration_circuits,
    circuit_stats,
    fold_global,
    save_instance,
    select_instance,
    tensored_calibration_circuits,
    with_measurements,
)
from .config import Config
from .execution import Executor
from .mitigation.pipeline import apply_methods
from .mitigation.readout import build_assignment_matrix, build_tensored_assignment_matrix
from .observables import counts_to_probvec, exact_reference, parity_from_probs
from .seeds import derive_seed

log = logging.getLogger(__name__)

METHOD_ORDER = ("none", "rem", "rem_tensored", "zne", "zne_rem")
SCHEMA_SCALES = (1, 3, 5)  # the E_lambda1/3/5 columns of the runs.csv schema

COLUMNS = [
    "run_id", "n_qubits", "depth_layers", "noise_level", "p1", "p2", "p_ro", "seed", "method",
    "circuit_attempts",
    "E_exact", "E_ideal_shots", "E_noisy_exact", "E_hat", "E_hat_clipped", "abs_error",
    "signed_error", "rel_error",
    "abs_error_none", "improvement_pct", "error_reduction_factor",
    "M_exact", "M_hat", "M_abs_error",
    "P_succ_exact", "P_succ_hat", "P_succ_ratio",
    "E_lambda1", "E_lambda3", "E_lambda5", "E_richardson", "E_linear", "E_exp",
    "extrapolation_out_of_range",
    "hellinger_fidelity", "tvd",
    "rem_negative_mass", "rem_condition_number",
    "est_std", "shot_noise_floor",
    "n_circuits", "total_shots", "base_depth", "max_depth", "base_cx", "total_cx", "total_1q",
    "total_gates", "time_quantum_s", "time_classical_s",
]
TIMING_COLUMNS = ["time_quantum_s", "time_classical_s"]
INT_COLUMNS = [
    "n_qubits", "depth_layers", "seed", "circuit_attempts", "n_circuits", "total_shots",
    "base_depth", "max_depth", "base_cx", "total_cx", "total_1q", "total_gates",
]

_COMMON_REQUIRED = [
    "run_id", "n_qubits", "depth_layers", "noise_level", "p1", "p2", "p_ro", "seed", "method",
    "circuit_attempts", "E_exact", "E_ideal_shots", "E_noisy_exact", "E_hat", "E_hat_clipped",
    "abs_error", "signed_error", "rel_error", "abs_error_none", "M_exact", "M_hat",
    "M_abs_error", "P_succ_exact", "P_succ_hat", "P_succ_ratio", "shot_noise_floor",
    "n_circuits", "total_shots", "base_depth", "max_depth", "base_cx", "total_cx", "total_1q",
    "total_gates", "time_quantum_s", "time_classical_s",
]
_VS_NONE = ["improvement_pct", "error_reduction_factor"]
_ZNE = ["E_lambda1", "E_lambda3", "E_lambda5", "E_richardson", "E_linear",
        "extrapolation_out_of_range"]
_REM_DIAG = ["rem_negative_mass", "rem_condition_number"]
_DIST = ["hellinger_fidelity", "tvd"]
# Columns that must be non-NaN for each method. E_exp may be NaN by design (failed fit).
REQUIRED_COLUMNS = {
    "none": _COMMON_REQUIRED + _DIST + ["est_std"],
    "rem": _COMMON_REQUIRED + _VS_NONE + _DIST + _REM_DIAG,
    "rem_tensored": _COMMON_REQUIRED + _VS_NONE + _DIST + _REM_DIAG,
    "zne": _COMMON_REQUIRED + _VS_NONE + _ZNE + ["est_std"],
    "zne_rem": _COMMON_REQUIRED + _VS_NONE + _ZNE + _REM_DIAG,
}


def run_id_for(n: int, layers: int, noise: str, seed: int) -> str:
    return f"n{n}_L{layers}_{noise}_s{seed}"


@dataclass
class Reference:
    E_exact: float
    M_exact: float
    p_exact: np.ndarray
    target_index: int
    P_succ_exact: float
    ideal_counts: dict[str, int]
    E_ideal_shots: float


def _timed(fn, *args):
    t0 = perf_counter()
    out = fn(*args)
    return out, perf_counter() - t0


# --- overhead ------------------------------------------------------------------------


def _overhead(method: str, stats_by_scale: dict[int, dict], cal_full_stats: list[dict],
              cal_tensored_stats: list[dict], shots: int, cal_shots: int) -> dict:
    base = stats_by_scale[1]
    if method in ("none", "rem", "rem_tensored"):
        used = [base]
    else:  # zne, zne_rem
        used = list(stats_by_scale.values())
    total_shots = shots * len(used)
    cal_stats = {"rem": cal_full_stats, "rem_tensored": cal_tensored_stats,
                 "zne_rem": cal_full_stats}.get(method)
    if cal_stats is not None:
        used = used + cal_stats
        total_shots += cal_shots * len(cal_stats)
    return {
        "n_circuits": len(used),
        "total_shots": total_shots,
        "base_depth": base["depth"],
        "max_depth": max(s["depth"] for s in used),
        "base_cx": base["cx"],
        "total_cx": sum(s["cx"] for s in used),
        "total_1q": sum(s["n1q"] for s in used),
        "total_gates": sum(s["total_gates"] for s in used),
    }


# --- one condition -------------------------------------------------------------------


def run_condition(
    executor: Executor,
    cfg: Config,
    inst: CircuitInstance,
    ref: Reference,
    noise: str,
    raw_dir: Path,
) -> list[dict]:
    """Execute the base, folded and both calibration circuits for one (n, L, noise, seed)
    and return one row per configured Track A method. Raw counts and both assignment
    matrices are saved."""
    n, L, seed = inst.n, inst.layers, inst.seed
    shots = cfg.experiment.shots
    cal_shots = cfg.rem.calibration_shots
    scales = tuple(cfg.zne.scale_factors)
    lvl = cfg.noise.level(noise)
    run_id = run_id_for(n, L, noise, seed)

    def sim_seed(purpose):
        return derive_seed("sim", n, L, noise, seed, purpose)

    def transpile_seed(purpose):
        return derive_seed("transpile", n, L, noise, seed, purpose)

    counts_by_scale, stats_by_scale, time_by_scale = {}, {}, {}
    for s in scales:
        purpose = "base" if s == 1 else f"fold{s}"
        circuit = with_measurements(inst.unitary if s == 1 else fold_global(inst.unitary, s))
        (counts,), elapsed = executor.run(
            [circuit], noise, shots, sim_seed(purpose), transpile_seed(purpose)
        )
        counts_by_scale[s] = counts
        stats_by_scale[s] = circuit_stats(executor.last_transpiled[0])
        time_by_scale[s] = elapsed

    cal_counts, t_cal = executor.run(
        calibration_circuits(n), noise, cal_shots, sim_seed("cal"), transpile_seed("cal")
    )
    cal_stats = [circuit_stats(tc) for tc in executor.last_transpiled]
    A = build_assignment_matrix(cal_counts, n, cal_shots)
    np.save(raw_dir / "calibration" / f"full_{run_id}.npy", A)

    cal_t_counts, t_cal_t = executor.run(
        tensored_calibration_circuits(n), noise, cal_shots, sim_seed("cal_tensored"),
        transpile_seed("cal_tensored"),
    )
    cal_t_stats = [circuit_stats(tc) for tc in executor.last_transpiled]
    A_t = build_tensored_assignment_matrix(cal_t_counts[0], cal_t_counts[1], n)
    np.save(raw_dir / "calibration" / f"tensored_{run_id}.npy", A_t)

    E_noisy_exact = parity_from_probs(executor.exact_noisy_probs(inst.unitary, noise))

    p_by_scale = {s: counts_to_probvec(counts_by_scale[s], n) for s in scales}
    t_quantum_by_method = {
        "none": time_by_scale[1],
        "rem": time_by_scale[1] + t_cal,
        "rem_tensored": time_by_scale[1] + t_cal_t,
        "zne": sum(time_by_scale.values()),
        "zne_rem": sum(time_by_scale.values()) + t_cal,
    }
    (estimates, t_classical) = _timed(
        apply_methods, p_by_scale, A, A_t, cfg.track_a.methods, n, ref.target_index, shots, scales,
        cfg.rem.cond_threshold,
    )

    err_none = metrics.abs_error(estimates["none"].E_hat, ref.E_exact)

    common = {
        "run_id": run_id, "n_qubits": n, "depth_layers": L, "noise_level": noise,
        "p1": lvl.p1, "p2": lvl.p2, "p_ro": lvl.p_ro, "seed": seed,
        "circuit_attempts": inst.attempts, "E_exact": ref.E_exact,
        "E_ideal_shots": ref.E_ideal_shots, "E_noisy_exact": E_noisy_exact,
        "M_exact": ref.M_exact, "P_succ_exact": ref.P_succ_exact, "abs_error_none": err_none,
        "shot_noise_floor": metrics.shot_noise_floor(ref.E_exact, shots),
    }

    rows = []
    for method in cfg.track_a.methods:
        est = estimates[method]
        E_hat = est.E_hat
        err = metrics.abs_error(E_hat, ref.E_exact)
        row = dict(common)
        row.update({
            "method": method,
            "E_hat": E_hat,
            "E_hat_clipped": est.E_hat_clipped,
            "abs_error": err,
            "signed_error": metrics.signed_error(E_hat, ref.E_exact),
            "rel_error": metrics.rel_error(E_hat, ref.E_exact),
            "M_hat": est.M_hat,
            "M_abs_error": metrics.abs_error(est.M_hat, ref.M_exact),
            "P_succ_hat": est.P_succ_hat,
            "P_succ_ratio": metrics.success_ratio(est.P_succ_hat, ref.P_succ_exact),
            "time_quantum_s": t_quantum_by_method[method],
            "time_classical_s": t_classical / len(cfg.track_a.methods),
        })
        if method != "none":
            row["improvement_pct"] = metrics.improvement_pct(err, err_none)
            row["error_reduction_factor"] = metrics.error_reduction_factor(err, err_none)
        if est.distribution is not None:
            row["hellinger_fidelity"] = metrics.hellinger_fidelity(est.distribution, ref.p_exact)
            row["tvd"] = metrics.tvd(est.distribution, ref.p_exact)
        if est.est_std is not None:
            row["est_std"] = est.est_std
        if est.rem_negative_mass is not None:
            row["rem_negative_mass"] = est.rem_negative_mass
            row["rem_condition_number"] = est.rem_condition_number
        if est.E_by_scale is not None:
            for s in SCHEMA_SCALES:
                row[f"E_lambda{s}"] = est.E_by_scale[s]
            row["E_richardson"] = est.E_richardson
            row["E_linear"] = est.E_linear
            row["E_exp"] = est.E_exp
            row["extrapolation_out_of_range"] = int(est.extrapolation_out_of_range)
            if est.extrapolation_out_of_range:
                log.warning("%s %s: Richardson estimate %.4f is outside [-1, 1]", run_id, method, E_hat)
        row.update(_overhead(method, stats_by_scale, cal_stats, cal_t_stats, shots, cal_shots))
        rows.append(row)

    record = {
        "run_id": run_id,
        "shots": shots,
        "calibration_shots": cal_shots,
        "ideal_shots": ref.ideal_counts,
        "scale_counts": {str(s): counts_by_scale[s] for s in scales},
        "calibration_full": cal_counts,
        "calibration_tensored": {"all_zero": cal_t_counts[0], "all_one": cal_t_counts[1]},
    }
    (raw_dir / "counts" / f"{run_id}.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return rows


# --- runs.csv ------------------------------------------------------------------------


def sort_runs(df: pd.DataFrame, noise_order: Iterable[str]) -> pd.DataFrame:
    noise_rank = {name: i for i, name in enumerate(noise_order)}
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    key = pd.DataFrame({
        "n": df["n_qubits"], "L": df["depth_layers"],
        "noise": df["noise_level"].map(noise_rank), "seed": df["seed"],
        "method": df["method"].map(method_rank),
    })
    order = key.sort_values(["n", "L", "noise", "seed", "method"], kind="mergesort").index
    return df.loc[order].reset_index(drop=True)


def write_runs_csv(df: pd.DataFrame, path: Path, noise_order: Iterable[str]) -> pd.DataFrame:
    df = sort_runs(df.reindex(columns=COLUMNS), noise_order)
    for col in INT_COLUMNS:
        df[col] = df[col].astype("int64")
    df.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return df


# --- sweep ---------------------------------------------------------------------------


def write_environment(out_dir: Path) -> None:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    ).stdout
    (out_dir / "environment.txt").write_text(freeze.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    (out_dir / "python_version.txt").write_text(
        f"{platform.python_version()}\n{sys.version}\n", encoding="utf-8", newline="\n"
    )


def _attach_log_file(log_path: Path) -> logging.Handler:
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    pkg = logging.getLogger("qem")
    pkg.setLevel(logging.INFO)
    pkg.addHandler(handler)
    return handler


def run_sweep(
    cfg: Config,
    out_dir: str | Path,
    only_n: Iterable[int] | None = None,
    resume: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Run every condition of the Track A grid and write runs.csv plus all raw artifacts."""
    if tuple(cfg.zne.scale_factors) != SCHEMA_SCALES:
        raise ValueError(f"runs.csv has E_lambda columns for scales {SCHEMA_SCALES} only")
    out = Path(out_dir)
    raw = out / "raw"
    for sub in ("counts", "circuits", "calibration"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    handler = _attach_log_file(out / "logs" / "sweep.log")
    try:
        return _run_sweep(cfg, out, raw, only_n, resume, progress)
    except Exception:
        log.exception("sweep failed")
        raise
    finally:
        logging.getLogger("qem").removeHandler(handler)
        handler.close()


def _run_sweep(cfg, out, raw, only_n, resume, progress):
    a = cfg.track_a
    write_environment(out)
    runs_path = raw / "runs.csv"
    qubits = [n for n in a.qubits if only_n is None or n in set(only_n)]
    log.info("sweep start: config=%s qubits=%s depths=%s noise=%s seeds=%s resume=%s",
             cfg.experiment.name, qubits, a.depths, a.noise_levels, cfg.experiment.seeds, resume)

    frames = []
    done: set[str] = set()
    if resume and runs_path.exists():
        existing = pd.read_csv(runs_path)
        frames.append(existing)
        done = set(existing["run_id"])
        log.info("resume: %d run ids already present", len(done))

    executor = Executor(cfg)
    new_rows: list[dict] = []

    def flush() -> pd.DataFrame:
        parts = frames + ([pd.DataFrame(new_rows)] if new_rows else [])
        if not parts:
            raise RuntimeError("the sweep produced no rows")
        return write_runs_csv(pd.concat(parts, ignore_index=True), runs_path, a.noise_levels)

    total = len(qubits) * len(a.depths) * len(cfg.experiment.seeds) * len(a.noise_levels)
    t_sweep = perf_counter()
    with tqdm(total=total, disable=not progress, desc="conditions") as bar:
        for n in qubits:
            for L in a.depths:
                for seed in cfg.experiment.seeds:
                    pending = [nz for nz in a.noise_levels if run_id_for(n, L, nz, seed) not in done]
                    if not pending:
                        bar.update(len(a.noise_levels))
                        continue
                    inst = select_instance(
                        n, L, seed, a.circuit.min_abs_exact_expectation,
                        a.circuit.max_resample_attempts, a.circuit.angle_low,
                        a.circuit.angle_high,
                    )
                    exact = exact_reference(inst.unitary)
                    save_instance(inst, raw / "circuits")
                    (ideal_counts,), _ = executor.run(
                        [with_measurements(inst.unitary)], "ideal", cfg.experiment.shots,
                        derive_seed("sim", n, L, "ideal_shots", seed),
                        derive_seed("transpile", n, L, "ideal", seed, "ideal_shots"),
                    )
                    ref = Reference(
                        exact["E_exact"], exact["M_exact"], exact["probs"], exact["target_index"],
                        exact["P_succ_exact"], ideal_counts,
                        parity_from_probs(counts_to_probvec(ideal_counts, n)),
                    )
                    for noise in a.noise_levels:
                        rid = run_id_for(n, L, noise, seed)
                        if rid in done:
                            bar.update(1)
                            continue
                        t0 = perf_counter()
                        new_rows.extend(run_condition(executor, cfg, inst, ref, noise, raw))
                        log.info("%s done in %.2fs (attempts=%d, E_exact=%.4f)",
                                 rid, perf_counter() - t0, inst.attempts, exact["E_exact"])
                        flush()  # keep runs.csv current so --resume can pick up after a crash
                        bar.update(1)

    df = flush()
    log.info("sweep finished: %d rows in %.1fs", len(df), perf_counter() - t_sweep)

    headline = df[df["noise_level"].isin(cfg.noise.headline_levels)]
    none_by_level = headline[headline["method"] == "none"].groupby("noise_level")["abs_error"].mean()
    if {"low", "moderate"} <= set(none_by_level.index):
        log.info("spot-check: mean abs_error of none at moderate (%.4f) %s low (%.4f)",
                 none_by_level["moderate"], ">" if none_by_level["moderate"] > none_by_level["low"] else "<=",
                 none_by_level["low"])
    return df
