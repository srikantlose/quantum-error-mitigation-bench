import numpy as np
import pandas as pd
import pytest

from qem import analysis
from qem.experiment import run_sweep
from qem.improvement import run_improvement
from qem.qml.runner import run_qml


@pytest.fixture(scope="module")
def runs_a(smoke_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis_a")
    run_sweep(smoke_cfg, out, progress=False)
    return pd.read_csv(out / "raw" / "runs.csv")


@pytest.fixture(scope="module")
def runs_b(smoke_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis_b")
    df = run_qml(smoke_cfg, out, progress=False)
    return df


@pytest.fixture(scope="module")
def alloc(smoke_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis_alloc")
    return run_improvement(smoke_cfg, out, progress=False)


@pytest.fixture(scope="module")
def noise_order(smoke_cfg):
    return list(smoke_cfg.track_a.noise_levels)


@pytest.fixture(scope="module")
def summary_a(runs_a, noise_order):
    return analysis.summarize(runs_a, noise_order)


def test_summarize_track_a_row_count_and_values(runs_a, summary_a):
    # smoke.yaml: 1 n x 1 L x 4 noise x 5 methods = 20 groups
    assert len(summary_a) == 20
    assert list(summary_a.method.unique()) == ["none", "rem", "rem_tensored", "zne", "zne_rem"]
    row = summary_a[(summary_a.noise_level == "low") & (summary_a.method == "rem")].iloc[0]
    expected = runs_a[(runs_a.noise_level == "low") & (runs_a.method == "rem")].abs_error.mean()
    assert row.abs_error_mean == pytest.approx(expected)
    assert row.n_circuits == 5  # 1 n = 2, so 1 + 2^2
    assert analysis.t_crit(5) == pytest.approx(2.776, abs=1e-3)


def test_bias_decomposition_exact_split(runs_a):
    bias = analysis.bias_decomposition(runs_a)
    assert (bias[bias.noise_level == "ideal"].readout_bias == 0).all()
    assert np.allclose(bias.E_gate_exact * (1 - 2 * bias.p_ro) ** bias.n_qubits, bias.E_noisy_exact)


def test_track_a_tables_render(summary_a, runs_a, noise_order):
    methods = ["none", "rem", "rem_tensored", "zne", "zne_rem"]
    for text in (
        analysis.table_error(summary_a, noise_order, methods),
        analysis.table_fidelity(summary_a, noise_order),
        analysis.table_overhead(summary_a, methods),
        analysis.table_extrapolators(runs_a, ["low", "moderate", "high"]),
        analysis.table_improvement(summary_a, noise_order, methods),
        analysis.table_bias_sources(runs_a, ["low", "moderate", "high"]),
        analysis.table_rem_negative_mass(runs_a, noise_order),
        analysis.table_rem_full_vs_tensored(runs_a, ["low", "moderate", "high"]),
    ):
        assert text.startswith("# ")
        assert "| " in text
    # high noise gets its own block in table_error, separate from the headline levels
    assert "High noise" in analysis.table_error(summary_a, noise_order, methods)


def test_stats_tests_track_a_scopes(runs_a, noise_order):
    tests = analysis.stats_tests_track_a(runs_a, noise_order)
    assert set(tests.scope) == {"per_condition", "pooled"}
    assert set(tests.comparison) == {"rem", "rem_tensored", "zne", "zne_rem"}


def test_variance_check_and_table(runs_a, smoke_cfg, root):
    var = analysis.variance_check(runs_a, smoke_cfg, root / "results" / "raw")
    assert "zne_residual" in var.columns
    assert len(var) == len(runs_a[runs_a.method == "none"])
    vs = analysis.variance_summary(var, list(smoke_cfg.track_a.noise_levels))
    text = analysis.table_variance(vs)
    assert text.startswith("# ")


# --- Track B ---------------------------------------------------------------------------


def test_summarize_track_b_row_count(runs_b):
    summary = analysis.summarize_qml(runs_b)
    # smoke.yaml: 1 n -> 2 classical groups + 1 vqc-exact group + (4 noise x 4 methods) vqc groups
    assert len(summary) == 2 + 1 + 16
    assert set(summary.model.unique()) == {"logreg", "svm", "vqc"}


def test_table_qml_classification_and_confusion(runs_b, smoke_cfg):
    summary = analysis.summarize_qml(runs_b)
    text = analysis.table_qml_classification(summary, list(smoke_cfg.track_b.noise_levels))
    assert "Logistic Regression" in text and "VQC, exact" in text
    n, L = smoke_cfg.track_b.qubits[0], smoke_cfg.track_b.depths[0]
    conf = analysis.table_qml_confusion(runs_b, n, L)
    assert "TP" in conf and "TN" in conf


def test_stats_tests_track_b(runs_b, smoke_cfg):
    tests = analysis.stats_tests_track_b(runs_b, list(smoke_cfg.track_b.noise_levels))
    assert set(tests.scope) == {"per_condition", "vs_classical"}
    assert {"vqc_vs_logreg", "vqc_vs_svm"} <= set(tests[tests.scope == "vs_classical"].comparison)


def test_combined_stats_tests(runs_a, runs_b, noise_order, smoke_cfg):
    tests = analysis.combined_stats_tests(runs_a, noise_order, runs_b, list(smoke_cfg.track_b.noise_levels))
    assert set(tests.track) == {"A", "B"}
    assert list(tests.columns) == analysis.STATS_COLUMNS


# --- improvement -------------------------------------------------------------------------


def test_table_zne_allocation(alloc):
    text = analysis.table_zne_allocation(alloc)
    assert "optimal" in text and "uniform" in text
    assert "Empirical std ratio" in text


# --- report numbers ----------------------------------------------------------------------


def test_report_numbers_covers_both_tracks(runs_a, summary_a, runs_b, alloc, smoke_cfg, root):
    tests = analysis.combined_stats_tests(runs_a, noise_order_of(smoke_cfg), runs_b,
                                          list(smoke_cfg.track_b.noise_levels))
    var = analysis.variance_check(runs_a, smoke_cfg, root / "results" / "raw")
    vs = analysis.variance_summary(var, noise_order_of(smoke_cfg))
    summary_b = analysis.summarize_qml(runs_b)
    numbers = analysis.report_numbers(runs_a, summary_a, tests, vs, root / "results",
                                      list(smoke_cfg.noise.headline_levels), runs_b, summary_b, alloc)
    for key in ("grid.rows", "env.qiskit", "alloc.ratio.empirical", "qml.vqc_exact.acc.mean",
                "bias.readout_share.low", "var.ratio.theory"):
        assert key in numbers, key


def noise_order_of(cfg):
    return list(cfg.track_a.noise_levels)


def test_resources_table(summary_a, runs_b):
    text = analysis.table_resources(summary_a, runs_b)
    assert "Track A" in text and "Track B" in text
