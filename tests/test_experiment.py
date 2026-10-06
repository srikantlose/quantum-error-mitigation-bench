import json

import numpy as np
import pandas as pd
import pytest

from qem.experiment import COLUMNS, REQUIRED_COLUMNS, TIMING_COLUMNS, run_sweep


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
    assert len(df) == 3 * 4
    header = (smoke_runs[0] / "raw" / "runs.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == COLUMNS
    assert list(df["method"]) == ["none", "rem", "zne", "zne_rem"] * 3
    assert list(df["noise_level"]) == ["ideal"] * 4 + ["low"] * 4 + ["moderate"] * 4


def test_required_columns_are_present(df):
    for method, cols in REQUIRED_COLUMNS.items():
        sub = df[df["method"] == method]
        assert len(sub) == 3
        missing = [c for c in cols if sub[c].isna().any()]
        assert not missing, f"{method}: {missing}"


def test_not_applicable_columns_are_empty(df):
    zne = df[df["method"].isin(["zne", "zne_rem"])]
    assert zne[["hellinger_fidelity", "tvd"]].isna().all().all()
    assert df[df["method"] == "none"][["improvement_pct", "E_richardson"]].isna().all().all()
    assert df[df["method"].isin(["rem", "zne_rem"])]["est_std"].isna().all()


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
    assert abs(ideal.loc["rem", "E_hat"] - ideal.loc["none", "E_hat"]) <= 1e-12
    assert ideal.loc["rem", "rem_condition_number"] == 1.0
    assert ideal.loc["rem", "rem_negative_mass"] == 0.0


def test_overhead_matches_formulas(df):
    n, L = 2, 2
    c, n1q = (n - 1) * L, 2 * n * L
    cal_1q = n * 2 ** (n - 1)  # one ry(pi) per set bit over all 2^n preparations
    expected = {
        "none": (1, 1024, c, n1q),
        "rem": (1 + 2**n, 1024 * (1 + 2**n), c, n1q + cal_1q),
        "zne": (3, 3072, 9 * c, 9 * n1q),
        "zne_rem": (3 + 2**n, 1024 * (3 + 2**n), 9 * c, 9 * n1q + cal_1q),
    }
    for _, row in df.iterrows():
        assert (row.n_circuits, row.total_shots, row.total_cx, row.total_1q) == expected[row.method]
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
    assert np.allclose(df["abs_error"], (df["E_hat"] - df["E_exact"]).abs(), atol=1e-9)


def test_raw_artifacts(smoke_runs):
    out = smoke_runs[0]
    assert (out / "environment.txt").read_text().strip()
    assert (out / "python_version.txt").read_text().startswith("3.")
    assert (out / "logs" / "sweep.log").exists()
    assert len(list((out / "raw" / "counts").glob("*.json"))) == 3
    assert len(list((out / "raw" / "circuits").glob("*.json"))) == 1
    A = np.load(out / "raw" / "calibration" / "n2_L2_moderate_s0.npy")
    assert A.shape == (4, 4) and np.allclose(A.sum(axis=0), 1.0)
    record = json.loads((out / "raw" / "counts" / "n2_L2_moderate_s0.json").read_text())
    assert set(record["scale_counts"]) == {"1", "3", "5"}
    assert len(record["calibration"]) == 4
    assert sum(record["ideal_shots"].values()) == 1024


def test_resume_skips_existing_runs(smoke_cfg, smoke_runs, tmp_path):
    import shutil

    shutil.copytree(smoke_runs[0], tmp_path / "copy")
    before = (tmp_path / "copy" / "raw" / "runs.csv").read_bytes()
    df = run_sweep(smoke_cfg, tmp_path / "copy", resume=True, progress=False)
    assert len(df) == 12
    assert (tmp_path / "copy" / "raw" / "runs.csv").read_bytes() == before
