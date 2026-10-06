"""Noisy inference, mitigation and classification metrics for the trained VQC.

Every method at a given (n, L, noise, seed) shares the same base/folded counts for all 20
test samples, and REM shares one calibration matrix across the whole test set -- the
amortization the course standard asks us to demonstrate (§13.10).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ..circuits import calibration_circuits, circuit_stats, fold_global, with_measurements
from ..config import Config
from ..execution import Executor
from ..mitigation.pipeline import apply_methods
from ..mitigation.readout import build_assignment_matrix
from ..observables import counts_to_probvec, parity_from_probs
from ..seeds import derive_seed
from .data import PreprocessedData
from .vqc import VqcModel, bind_sample, build_vqc_circuit

log = logging.getLogger(__name__)

QML_COLUMNS = [
    "model", "n_qubits", "depth_layers", "noise_level", "seed", "method", "n_train", "n_test", "n_params",
    "accuracy", "precision", "recall", "f1", "tp", "fn", "fp", "tn",
    "mean_abs_E_error", "margin_retention", "mean_hellinger", "agreement_with_exact",
    "train_accuracy_exact", "train_loss_final", "train_iterations", "train_time_s",
    "n_circuits_per_sample", "n_circuits_per_sample_amortized", "total_shots_eval",
    "base_depth", "base_cx", "base_1q", "time_eval_s",
]


@dataclass
class ExactReference:
    E: np.ndarray  # (n_test,) exact <Z^n> per sample
    y_pred: np.ndarray  # exact prediction per sample
    probs: list[np.ndarray]  # exact distribution per sample
    target_index: list[int]  # argmax(probs) per sample


def exact_vqc_reference(prep: PreprocessedData, model: VqcModel) -> ExactReference:
    qc, x, w = build_vqc_circuit(prep.n, model.layers)
    obs = SparsePauliOp("Z" * prep.n)
    E, probs, target = [], [], []
    for xv in prep.X_test:
        bound = bind_sample(qc, x, w, xv, model.weights)
        sv = Statevector(bound)
        p = sv.probabilities()
        E.append(sv.expectation_value(obs).real)
        probs.append(p)
        target.append(int(np.argmax(p)))
    E = np.asarray(E, dtype=float)
    return ExactReference(E=E, y_pred=np.where(E >= 0, 1, -1), probs=probs, target_index=target)


def _classification_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(np.mean(y_true == y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
    tp, fn, fp, tn = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def classical_row(model_name: str, prep: PreprocessedData, y_pred: np.ndarray) -> dict:
    row = {c: np.nan for c in QML_COLUMNS}
    row.update({"model": model_name, "n_qubits": prep.n, "seed": prep.seed,
                "n_train": len(prep.y_train), "n_test": len(prep.y_test)})
    row.update(_classification_row(prep.y_test, y_pred))
    return row


def exact_vqc_row(prep: PreprocessedData, model: VqcModel, ref: ExactReference) -> dict:
    row = {c: np.nan for c in QML_COLUMNS}
    n_par = len(model.weights)
    row.update({
        "model": "vqc", "n_qubits": prep.n, "depth_layers": model.layers, "noise_level": "exact",
        "seed": prep.seed, "method": "none", "n_train": len(prep.y_train), "n_test": len(prep.y_test),
        "n_params": n_par,
        "mean_abs_E_error": 0.0, "margin_retention": 1.0, "mean_hellinger": 1.0, "agreement_with_exact": 1.0,
        "train_accuracy_exact": model.train_accuracy_exact, "train_loss_final": model.loss_history[-1],
        "train_iterations": model.n_iterations, "train_time_s": model.train_time_s,
        "n_circuits_per_sample": 1, "n_circuits_per_sample_amortized": 1,
        "total_shots_eval": 0, "base_depth": np.nan, "base_cx": np.nan, "base_1q": np.nan,
        "time_eval_s": 0.0,
    })
    row.update(_classification_row(prep.y_test, ref.y_pred))
    return row


def _resource_counts(method: str, n: int, n_test: int, shots: int, cal_shots: int) -> dict:
    per_sample = 3 if method in ("zne", "zne_rem") else 1
    uses_cal = method in ("rem", "zne_rem")
    n_cal = 2**n if uses_cal else 0
    actual_circuits = per_sample * n_test + n_cal
    actual_shots = per_sample * n_test * shots + n_cal * cal_shots
    return {
        "n_circuits_per_sample": per_sample + n_cal,  # unamortized worst case, as in Track A
        "n_circuits_per_sample_amortized": actual_circuits / n_test,
        "total_shots_eval": actual_shots,
    }


def evaluate_noisy(
    executor: Executor,
    cfg: Config,
    prep: PreprocessedData,
    model: VqcModel,
    exact: ExactReference,
    noise: str,
    predictions_dir: Path,
) -> list[dict]:
    """Noisy inference and mitigation at one noise level, for every configured Track B
    method, amortizing one REM calibration across the whole test set."""
    n, layers, seed = prep.n, model.layers, prep.seed
    shots = cfg.experiment.shots
    cal_shots = cfg.rem.calibration_shots
    scales = tuple(cfg.zne.scale_factors)
    n_test = len(prep.y_test)
    qc, x, w = build_vqc_circuit(n, layers)
    unitaries = [bind_sample(qc, x, w, xv, model.weights) for xv in prep.X_test]

    def sim_seed(purpose):
        return derive_seed("qml_sim", n, layers, noise, seed, purpose)

    def transpile_seed(purpose):
        return derive_seed("qml_transpile", n, layers, noise, seed, purpose)

    t0 = perf_counter()
    counts_by_scale, stats_by_scale = {}, {}
    for s in scales:
        purpose = "base" if s == 1 else f"fold{s}"
        circuits = [with_measurements(u if s == 1 else fold_global(u, s)) for u in unitaries]
        counts, _ = executor.run(circuits, noise, shots, sim_seed(purpose), transpile_seed(purpose))
        counts_by_scale[s] = counts
        stats_by_scale[s] = circuit_stats(executor.last_transpiled[0])

    A = None
    if any(m in cfg.track_b.methods for m in ("rem", "zne_rem")):
        cal_counts, _ = executor.run(
            calibration_circuits(n), noise, cal_shots, sim_seed("cal"), transpile_seed("cal")
        )
        A = build_assignment_matrix(cal_counts, n, cal_shots)

    per_method = {m: {"E_hat": [], "y_pred": [], "dist": [], "hellinger": []} for m in cfg.track_b.methods}
    for i in range(n_test):
        p_by_scale = {s: counts_to_probvec(counts_by_scale[s][i], n) for s in scales}
        est = apply_methods(p_by_scale, A, None, cfg.track_b.methods, n, exact.target_index[i], shots, scales,
                            cfg.rem.cond_threshold)
        for m, e in est.items():
            per_method[m]["E_hat"].append(e.E_hat)
            per_method[m]["y_pred"].append(1 if e.E_hat >= 0 else -1)
            if e.distribution is not None:
                from .. import metrics as qem_metrics

                per_method[m]["hellinger"].append(qem_metrics.hellinger_fidelity(e.distribution, exact.probs[i]))
    time_eval = perf_counter() - t0

    rows = []
    for method in cfg.track_b.methods:
        d = per_method[method]
        E_hat = np.asarray(d["E_hat"], dtype=float)
        y_pred = np.asarray(d["y_pred"], dtype=int)
        mean_abs_err = float(np.mean(np.abs(E_hat - exact.E)))
        denom = float(np.mean(np.abs(exact.E)))
        margin_retention = float(np.mean(E_hat * np.sign(exact.E)) / denom) if denom > 1e-12 else float("nan")
        mean_hell = float(np.mean(d["hellinger"])) if d["hellinger"] else float("nan")
        agreement = float(np.mean(y_pred == exact.y_pred))

        row = {c: np.nan for c in QML_COLUMNS}
        row.update({
            "model": "vqc", "n_qubits": n, "depth_layers": layers, "noise_level": noise, "seed": seed,
            "method": method, "n_train": len(prep.y_train), "n_test": n_test, "n_params": len(model.weights),
            "mean_abs_E_error": mean_abs_err, "margin_retention": margin_retention,
            "mean_hellinger": mean_hell, "agreement_with_exact": agreement,
            "train_accuracy_exact": model.train_accuracy_exact, "train_loss_final": model.loss_history[-1],
            "train_iterations": model.n_iterations, "train_time_s": model.train_time_s,
            "base_depth": stats_by_scale[1]["depth"], "base_cx": stats_by_scale[1]["cx"],
            "base_1q": stats_by_scale[1]["n1q"], "time_eval_s": time_eval,
        })
        row.update(_classification_row(prep.y_test, y_pred))
        row.update(_resource_counts(method, n, n_test, shots, cal_shots))
        rows.append(row)

    _save_predictions(predictions_dir / f"n{n}_L{layers}_{noise}_s{seed}.csv",
                      prep, exact, per_method, cfg.track_b.methods)
    return rows


def _save_predictions(path: Path, prep: PreprocessedData, exact: ExactReference,
                      per_method: dict, methods) -> None:
    fieldnames = ["sample_idx", "y_true", "E_exact"] + \
        [f"E_hat_{m}" for m in methods] + [f"yhat_{m}" for m in methods]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(prep.y_test)):
            row = {"sample_idx": i, "y_true": int(prep.y_test[i]), "E_exact": f"{exact.E[i]:.10g}"}
            for m in methods:
                row[f"E_hat_{m}"] = f"{per_method[m]['E_hat'][i]:.10g}"
                row[f"yhat_{m}"] = per_method[m]["y_pred"][i]
            writer.writerow(row)
