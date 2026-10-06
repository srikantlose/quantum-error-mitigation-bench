"""Improvement 2 (plan.md §14.2): variance-optimal shot allocation for ZNE.

Compares uniform shot allocation (1024 per scale factor) against the variance-minimizing
allocation N_i proportional to |gamma_i| at the same total shot budget, by repeating the
three-scale measurement many times and looking at the spread of the Richardson estimate.

Uses the *same* circuit instances as Track A (`select_instance` with identical seeding),
so bias is directly comparable between this experiment and the main grid.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from .circuits import fold_global, select_instance, with_measurements
from .config import Config
from .execution import Executor
from .experiment import write_environment
from .mitigation.zne import extrapolate_richardson, optimal_allocation, richardson_coeffs
from .observables import counts_to_probvec, exact_reference, parity_from_probs
from .seeds import derive_seed

log = logging.getLogger(__name__)

SCHEMES = ("uniform", "optimal")

ALLOCATION_COLUMNS = [
    "n_qubits", "depth_layers", "noise_level", "seed", "scheme",
    "repetitions", "shots_lambda1", "shots_lambda3", "shots_lambda5", "total_shots",
    "E_exact", "mean_E0", "bias", "empirical_std", "rmse", "theoretical_std",
]


def _attach_log_file(log_path: Path) -> logging.Handler:
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    pkg = logging.getLogger("qem")
    pkg.setLevel(logging.INFO)
    pkg.addHandler(handler)
    return handler


def _shots_for_scheme(scheme: str, gammas: np.ndarray, total: int) -> list[int]:
    if scheme == "uniform":
        base = total // len(gammas)
        shots = [base] * len(gammas)
        shots[0] += total - sum(shots)  # keep the sum exact even if not divisible
        return shots
    if scheme == "optimal":
        return optimal_allocation(gammas, np.ones(len(gammas)), total)
    raise ValueError(f"unknown scheme {scheme!r}")


def run_condition(executor: Executor, cfg: Config, n: int, L: int, noise: str, seed: int) -> list[dict]:
    a = cfg.track_a
    scales = tuple(cfg.zne.scale_factors)
    gammas = richardson_coeffs(scales)
    sa = cfg.improvement.shot_allocation
    total = sa.total_shots
    reps = sa.repetitions

    inst = select_instance(n, L, seed, a.circuit.min_abs_exact_expectation, a.circuit.max_resample_attempts,
                           a.circuit.angle_low, a.circuit.angle_high)
    E_exact = exact_reference(inst.unitary)["E_exact"]
    folded = {s: with_measurements(inst.unitary if s == 1 else fold_global(inst.unitary, s)) for s in scales}

    rows = []
    for scheme in SCHEMES:
        shots_list = _shots_for_scheme(scheme, gammas, total)
        # One batched run per scale: `reps` copies of the same circuit in one Aer call.
        # Aer assigns each copy in the batch its own deterministic, independent random
        # draw from the base seed (verified empirically), so this is equivalent to `reps`
        # separate calls but far faster.
        E_by_rep = np.zeros((len(scales), reps))
        for i, (s, shots_i) in enumerate(zip(scales, shots_list)):
            batch_seed = derive_seed("alloc", n, L, noise, seed, scheme, s)
            counts, _ = executor.run([folded[s]] * reps, noise, shots_i, batch_seed)
            for r in range(reps):
                E_by_rep[i, r] = parity_from_probs(counts_to_probvec(counts[r], n))
        E0 = gammas @ E_by_rep  # Richardson estimate for every repetition, vectorized
        mean_E0 = float(np.mean(E0))
        bias = mean_E0 - E_exact
        empirical_std = float(np.std(E0, ddof=1)) if reps > 1 else float("nan")
        rmse = float(np.sqrt(np.mean((E0 - E_exact) ** 2)))
        theoretical_std = float(np.sqrt(np.sum(gammas**2 / np.asarray(shots_list, dtype=float))))

        row = {
            "n_qubits": n, "depth_layers": L, "noise_level": noise, "seed": seed, "scheme": scheme,
            "repetitions": reps, "total_shots": total, "E_exact": E_exact, "mean_E0": mean_E0,
            "bias": bias, "empirical_std": empirical_std, "rmse": rmse, "theoretical_std": theoretical_std,
        }
        for s, shots_i in zip(scales, shots_list):
            row[f"shots_lambda{s}"] = shots_i
        rows.append(row)
    return rows


def run_improvement(cfg: Config, out_dir: str | Path, progress: bool = True) -> pd.DataFrame:
    sa = cfg.improvement.shot_allocation
    if not sa.enabled:
        raise ValueError("improvement.shot_allocation.enabled is false")
    out = Path(out_dir)
    imp_dir = out / "improvement"
    imp_dir.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    write_environment(out)
    handler = _attach_log_file(out / "logs" / "improvement.log")
    a = cfg.track_a
    log.info("improvement sweep start: qubits=%s depths=%s noise=%s seeds=%s reps=%d total_shots=%d",
             a.qubits, a.depths, sa.noise_levels, cfg.experiment.seeds, sa.repetitions, sa.total_shots)
    try:
        executor = Executor(cfg)
        rows: list[dict] = []
        total = len(a.qubits) * len(a.depths) * len(sa.noise_levels) * len(cfg.experiment.seeds)
        t0 = perf_counter()
        with tqdm(total=total, disable=not progress, desc="allocation") as bar:
            for n in a.qubits:
                for L in a.depths:
                    for noise in sa.noise_levels:
                        for seed in cfg.experiment.seeds:
                            rows.extend(run_condition(executor, cfg, n, L, noise, seed))
                            bar.update(1)
        df = pd.DataFrame(rows).reindex(columns=ALLOCATION_COLUMNS)
        df = df.sort_values(["n_qubits", "depth_layers", "noise_level", "seed", "scheme"],
                            key=lambda c: c.map({"uniform": 0, "optimal": 1}) if c.name == "scheme" else c,
                            kind="mergesort").reset_index(drop=True)
        path = imp_dir / "zne_allocation.csv"
        df.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        log.info("improvement sweep finished: %d rows in %.1fs", len(df), perf_counter() - t0)
        ratio = (df[df.scheme == "optimal"].empirical_std.mean() / df[df.scheme == "uniform"].empirical_std.mean())
        log.info("spot-check: mean empirical std ratio (optimal/uniform) = %.3f (theory ~0.885)", ratio)
        return df
    except Exception:
        log.exception("improvement sweep failed")
        raise
    finally:
        logging.getLogger("qem").removeHandler(handler)
        handler.close()
