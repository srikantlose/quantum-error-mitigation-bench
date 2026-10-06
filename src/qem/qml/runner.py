"""Track B sweep: data -> classical baselines -> VQC training (cached) -> exact and noisy
evaluation, for every (n, L, noise, seed, method) combination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..execution import Executor
from ..experiment import write_environment
from .classical import fit_predict_logreg, fit_predict_svm
from .data import load_iris_binary, make_split, preprocess, save_preprocessing, save_split
from .evaluate import QML_COLUMNS, classical_row, evaluate_noisy, exact_vqc_reference, exact_vqc_row
from .vqc import train_or_load

log = logging.getLogger(__name__)

MODEL_ORDER = ("logreg", "svm", "vqc")
METHOD_ORDER = ("none", "rem", "zne", "zne_rem")


def _attach_log_file(log_path: Path) -> logging.Handler:
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    pkg = logging.getLogger("qem")
    pkg.setLevel(logging.INFO)
    pkg.addHandler(handler)
    return handler


def sort_qml_runs(df: pd.DataFrame, noise_order) -> pd.DataFrame:
    noise_rank = {name: i for i, name in enumerate(("exact",) + tuple(noise_order))}
    key = pd.DataFrame({
        "model": df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}),
        "n": df["n_qubits"],
        "L": df["depth_layers"].fillna(-1),
        "noise": df["noise_level"].fillna("exact").map(noise_rank),
        "seed": df["seed"],
        "method": df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(-1),
    })
    order = key.sort_values(["model", "n", "L", "noise", "seed", "method"], kind="mergesort").index
    return df.loc[order].reset_index(drop=True)


def run_qml(cfg: Config, out_dir: str | Path, retrain: bool = False, progress: bool = True) -> pd.DataFrame:
    b = cfg.track_b
    out = Path(out_dir)
    qml_dir = out / "qml"
    splits_dir, prep_dir, models_dir, pred_dir = (
        qml_dir / "splits", qml_dir / "preprocessing", qml_dir / "models", qml_dir / "predictions"
    )
    for d in (splits_dir, prep_dir, models_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    write_environment(out)
    handler = _attach_log_file(out / "logs" / "qml.log")
    log.info("track B sweep start: qubits=%s depths=%s noise=%s seeds=%s retrain=%s",
             b.qubits, b.depths, b.noise_levels, cfg.experiment.seeds, retrain)
    try:
        return _run_qml(cfg, out, splits_dir, prep_dir, models_dir, pred_dir, retrain, progress)
    except Exception:
        log.exception("track B sweep failed")
        raise
    finally:
        logging.getLogger("qem").removeHandler(handler)
        handler.close()


def _run_qml(cfg, out, splits_dir, prep_dir, models_dir, pred_dir, retrain, progress):
    b = cfg.track_b
    X, y = load_iris_binary(b)
    executor = Executor(cfg)
    rows: list[dict] = []
    total = len(cfg.experiment.seeds) * len(b.qubits) * (1 + len(b.depths) * (1 + len(b.noise_levels)))
    t_sweep = perf_counter()
    with tqdm(total=total, disable=not progress, desc="track B") as bar:
        for seed in cfg.experiment.seeds:
            split = make_split(X, y, seed, b.test_size)
            save_split(split, splits_dir)
            for n in b.qubits:
                prep = preprocess(split, n, b.feature_range)
                save_preprocessing(prep, prep_dir)

                lr_pred = fit_predict_logreg(prep, b.classical)
                svm_pred = fit_predict_svm(prep, b.classical)
                rows.append(classical_row("logreg", prep, lr_pred.y_pred))
                rows.append(classical_row("svm", prep, svm_pred.y_pred))
                log.info("n=%d seed=%d: logreg acc=%.3f svm acc=%.3f", n, seed,
                         float((lr_pred.y_pred == prep.y_test).mean()),
                         float((svm_pred.y_pred == prep.y_test).mean()))
                bar.update(1)

                for L in b.depths:
                    model = train_or_load(prep, L, b.training, models_dir, retrain)
                    exact = exact_vqc_reference(prep, model)
                    rows.append(exact_vqc_row(prep, model, exact))
                    log.info("n=%d L=%d seed=%d: trained in %.2fs, %d iters, train_acc=%.3f, test_acc(exact)=%.3f",
                             n, L, seed, model.train_time_s, model.n_iterations, model.train_accuracy_exact,
                             float((exact.y_pred == prep.y_test).mean()))
                    bar.update(1)

                    for noise in b.noise_levels:
                        rows.extend(evaluate_noisy(executor, cfg, prep, model, exact, noise, pred_dir))
                        bar.update(1)

    df = sort_qml_runs(pd.DataFrame(rows).reindex(columns=QML_COLUMNS), b.noise_levels)
    path = out / "qml" / "qml_runs.csv"
    df.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    log.info("track B sweep finished: %d rows in %.1fs", len(df), perf_counter() - t_sweep)
    return df
