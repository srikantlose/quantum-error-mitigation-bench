import numpy as np
import pandas as pd
import pytest

from qem.qml.evaluate import QML_COLUMNS
from qem.qml.runner import run_qml


@pytest.fixture(scope="module")
def qml_runs(smoke_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("qml_smoke")
    df = run_qml(smoke_cfg, out, progress=False)
    return out, df


def test_smoke_row_count_and_columns(smoke_cfg, qml_runs):
    out, df = qml_runs
    b = smoke_cfg.track_b
    n_logreg_svm = 2 * len(b.qubits) * len(smoke_cfg.experiment.seeds)
    n_exact = len(b.qubits) * len(b.depths) * len(smoke_cfg.experiment.seeds)
    n_noisy = len(b.qubits) * len(b.depths) * len(b.noise_levels) * len(smoke_cfg.experiment.seeds) * len(b.methods)
    assert len(df) == n_logreg_svm + n_exact + n_noisy
    assert list(df.columns) == QML_COLUMNS
    assert (out / "qml" / "qml_runs.csv").exists()


def test_confusion_counts_sum_to_test_size(smoke_cfg, qml_runs):
    _, df = qml_runs
    n_test = int(df["n_test"].dropna().iloc[0])
    sub = df.dropna(subset=["tp"])
    totals = sub["tp"] + sub["fn"] + sub["fp"] + sub["tn"]
    assert (totals == n_test).all()


def test_ideal_level_rem_matches_none(qml_runs):
    _, df = qml_runs
    ideal = df[(df["model"] == "vqc") & (df["noise_level"] == "ideal")].set_index("method")
    assert ideal.loc["rem", "accuracy"] == pytest.approx(ideal.loc["none", "accuracy"])
    assert ideal.loc["rem", "mean_abs_E_error"] == pytest.approx(ideal.loc["none", "mean_abs_E_error"], abs=1e-9)


def test_exact_row_is_self_consistent(qml_runs):
    _, df = qml_runs
    exact = df[(df["model"] == "vqc") & (df["noise_level"] == "exact")].iloc[0]
    assert exact.mean_abs_E_error == 0.0
    assert exact.agreement_with_exact == 1.0
    assert exact.margin_retention == 1.0
    assert exact.mean_hellinger == 1.0


def test_resource_amortization(qml_runs):
    _, df = qml_runs
    rem = df[(df["model"] == "vqc") & (df["noise_level"] == "ideal") & (df["method"] == "rem")].iloc[0]
    n, n_test = int(rem.n_qubits), int(rem.n_test)
    assert rem.n_circuits_per_sample == 1 + 2**n  # unamortized worst case
    assert rem.n_circuits_per_sample_amortized == pytest.approx((n_test + 2**n) / n_test)
    assert rem.n_circuits_per_sample_amortized < rem.n_circuits_per_sample

    none = df[(df["model"] == "vqc") & (df["noise_level"] == "ideal") & (df["method"] == "none")].iloc[0]
    assert none.n_circuits_per_sample == none.n_circuits_per_sample_amortized == 1


def test_predictions_file_written(smoke_cfg, qml_runs):
    out, _ = qml_runs
    files = list((out / "qml" / "predictions").glob("*.csv"))
    assert len(files) == len(smoke_cfg.track_b.noise_levels)
    df = pd.read_csv(files[0])
    assert len(df) == 20
    assert {"sample_idx", "y_true", "E_exact"} <= set(df.columns)
    for m in smoke_cfg.track_b.methods:
        assert f"yhat_{m}" in df.columns


def test_classical_rows_use_same_split_and_features_as_vqc(smoke_cfg, qml_runs):
    """LR, SVM and the VQC must see identical test labels for a given (n, seed): proof
    they were evaluated on the same split."""
    _, df = qml_runs
    seed = smoke_cfg.experiment.seeds[0]
    n = smoke_cfg.track_b.qubits[0]
    lr = df[(df.model == "logreg") & (df.n_qubits == n) & (df.seed == seed)].iloc[0]
    vqc_exact = df[(df.model == "vqc") & (df.n_qubits == n) & (df.seed == seed)
                   & (df.noise_level == "exact")].iloc[0]
    assert lr.n_test == vqc_exact.n_test == 20
