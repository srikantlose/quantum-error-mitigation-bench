import json

import numpy as np
import pandas as pd
import pytest

from qem.experiment import COLUMNS, METHOD_ORDER, REQUIRED_COLUMNS, TIMING_COLUMNS, run_sweep


@pytest.fixture(scope="module")
def smoke_runs(smoke_cfg, tmp_path_factory):
    out_a = tmp_path_factory.mktemp("smoke_a")
    out_b = tmp_path_factory.mktemp("smoke_b")
    run_sweep(smoke_cfg, out_a, progress=False)
    run_sweep(smoke_cfg, out_b, progress=False)
    return out_a, out_b


@pytest.fixture(scope="module")
def df(smoke_runs):
    return pd.read_csv(smoke_runs[0] / "raw" / "runs.csv")


def test_row_count_and_exact_column_order(smoke_runs, df):
    # smoke.yaml: 1 n x 1 L x 1 seed x 4 noise levels x 5 methods = 20 rows
    assert len(df) == 4 * 5
    header = (smoke_runs[0] / "raw" / "runs.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == COLUMNS
    assert list(df["method"]) == list(METHOD_ORDER) * 4
    assert list(df["noise_level"]) == ["ideal"] * 5 + ["low"] * 5 + ["moderate"] * 5 + ["high"] * 5


def test_required_columns_are_present(df):
    for method, cols in REQUIRED_COLUMNS.items():
        sub = df[df["method"] == method]
        assert len(sub) == 4
        missing = [c for c in cols if sub[c].isna().any()]
        assert not missing, f"{method}: {missing}"


def test_not_applicable_columns_are_empty(df):
    zne = df[df["method"].isin(["zne", "zne_rem"])]
    assert zne[["hellinger_fidelity", "tvd"]].isna().all().all()
    assert df[df["method"] == "none"][["improvement_pct", "E_richardson"]].isna().all().all()
    assert df[df["method"].isin(["rem", "rem_tensored", "zne_rem"])]["est_std"].isna().all()
    assert df[df["method"].isin(["none", "zne"])][["rem_negative_mass", "rem_condition_number"]].isna().all().all()


def test_runs_are_reproducible(smoke_runs):
    a, b = smoke_runs
    da = pd.read_csv(a / "raw" / "runs.csv", dtype=str, keep_default_na=False)
    db = pd.read_csv(b / "raw" / "runs.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(da.drop(columns=TIMING_COLUMNS), db.drop(columns=TIMING_COLUMNS))
    for sub in ("counts", "circuits", "calibration"):
        files_a = sorted(p.name for p in (a / "raw" / sub).iterdir())
        assert files_a == sorted(p.name for p in (b / "raw" / sub).iterdir())
        for name in files_a:
            assert (a / "raw" / sub / name).read_bytes() == (b / "raw" / sub / name).read_bytes()


def test_rem_is_identity_at_ideal_level(df):
    ideal = df[df["noise_level"] == "ideal"].set_index("method")
    for m in ("rem", "rem_tensored"):
        assert abs(ideal.loc[m, "E_hat"] - ideal.loc["none", "E_hat"]) <= 1e-12
        assert ideal.loc[m, "rem_condition_number"] == pytest.approx(1.0)
        assert ideal.loc[m, "rem_negative_mass"] == 0.0


def test_overhead_matches_formulas(df):
    n, L = 2, 2
    c, n1q = (n - 1) * L, 2 * n * L
    cal_1q_full = n * 2 ** (n - 1)  # one ry(pi) per set bit over all 2^n preparations
    cal_1q_tensored = n  # only the all-|1> circuit carries ry(pi) gates
    expected = {
        "none": (1, 1024, c, n1q),
        "rem": (1 + 2**n, 1024 * (1 + 2**n), c, n1q + cal_1q_full),
        "rem_tensored": (3, 3072, c, n1q + cal_1q_tensored),
        "zne": (3, 3072, 9 * c, 9 * n1q),
        "zne_rem": (3 + 2**n, 1024 * (3 + 2**n), 9 * c, 9 * n1q + cal_1q_full),
    }
    for _, row in df.iterrows():
        assert (row.n_circuits, row.total_shots, row.total_cx, row.total_1q) == expected[row.method]
        assert row.total_gates == row.total_1q + row.total_cx
        assert row.base_cx == c
        if row.method in ("zne", "zne_rem"):
            assert row.base_depth < row.max_depth <= 5 * row.base_depth
        else:
            assert row.max_depth == row.base_depth


def test_pairing_and_derived_columns(df):
    for _, g in df.groupby("run_id"):
        assert g["abs_error_none"].nunique() == 1
        none = g[g["method"] == "none"].iloc[0]
        assert none["abs_error"] == pytest.approx(none["abs_error_none"])
        zne = g[g["method"] == "zne"].iloc[0]
        # paired by construction: zne's lambda=1 point is the none estimate
        assert zne["E_lambda1"] == pytest.approx(none["E_hat"], abs=1e-12)
        g15 = np.array([1.875, -1.25, 0.375]) @ zne[["E_lambda1", "E_lambda3", "E_lambda5"]].to_numpy(float)
        assert zne["E_richardson"] == pytest.approx(g15, abs=1e-9)
        rem = g[g["method"] == "rem"].iloc[0]
        assert rem["improvement_pct"] == pytest.approx(100 * (1 - rem["abs_error"] / none["abs_error"]))
        # success-probability ratio is consistent with the stored exact/estimated values
        assert none["P_succ_ratio"] == pytest.approx(none["P_succ_hat"] / none["P_succ_exact"])
    assert np.allclose(df["abs_error"], (df["E_hat"] - df["E_exact"]).abs(), atol=1e-9)


def test_raw_artifacts(smoke_runs):
    out = smoke_runs[0]
    assert (out / "environment.txt").read_text().strip()
    assert (out / "python_version.txt").read_text().startswith("3.")
    assert (out / "logs" / "sweep.log").exists()
    assert len(list((out / "raw" / "counts").glob("*.json"))) == 4
    assert len(list((out / "raw" / "circuits").glob("*.json"))) == 1
    assert len(list((out / "raw" / "calibration").glob("full_*.npy"))) == 4
    assert len(list((out / "raw" / "calibration").glob("tensored_*.npy"))) == 4
    A = np.load(out / "raw" / "calibration" / "full_n2_L2_moderate_s0.npy")
    assert A.shape == (4, 4) and np.allclose(A.sum(axis=0), 1.0)
    A_t = np.load(out / "raw" / "calibration" / "tensored_n2_L2_moderate_s0.npy")
    assert A_t.shape == (4, 4) and np.allclose(A_t.sum(axis=0), 1.0)
    record = json.loads((out / "raw" / "counts" / "n2_L2_moderate_s0.json").read_text())
    assert set(record["scale_counts"]) == {"1", "3", "5"}
    assert len(record["calibration_full"]) == 4
    assert set(record["calibration_tensored"]) == {"all_zero", "all_one"}
    assert sum(record["ideal_shots"].values()) == 1024


def test_resume_skips_existing_runs(smoke_cfg, smoke_runs, tmp_path):
    import shutil

    shutil.copytree(smoke_runs[0], tmp_path / "copy")
    before = (tmp_path / "copy" / "raw" / "runs.csv").read_bytes()
    df = run_sweep(smoke_cfg, tmp_path / "copy", resume=True, progress=False)
    assert len(df) == 20
    assert (tmp_path / "copy" / "raw" / "runs.csv").read_bytes() == before


def test_md_table_escapes_pipes():
    from qem.analysis import md_table

    text = md_table(["a |x|", "b"], [["|1|", "2"]], "lr")
    assert text.splitlines()[0] == r"| a \|x\| | b |"
    assert text.splitlines()[2] == r"| \|1\| | 2 |"


def test_build_report_renders_placeholders(root, tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_report", root / "scripts" / "build_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    (tmp_path / "t.md").write_text("# Title\n\n## Sub\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", encoding="utf-8")
    (tmp_path / "n1_L1_s0.json").write_text(json.dumps({"qasm": "OPENQASM 2.0;\n"}), encoding="utf-8")
    out = mod.render("x={{num:k}} {{table:t}} {{qasm:n1_L1_s0}}", {"k": "0.5"}, tmp_path, tmp_path)
    assert out.startswith("x=0.5 **Sub**")
    assert "| 1 | 2 |" in out and "OPENQASM 2.0;" in out and "# Title" not in out
    with pytest.raises(KeyError):
        mod.render("{{num:missing}}", {}, tmp_path, tmp_path)
