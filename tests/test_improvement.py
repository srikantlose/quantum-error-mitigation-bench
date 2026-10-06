import numpy as np
import pandas as pd
import pytest

from qem.improvement import ALLOCATION_COLUMNS, run_improvement
from qem.mitigation.zne import richardson_coeffs


@pytest.fixture(scope="module")
def smoke_allocation(smoke_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("alloc_smoke")
    df = run_improvement(smoke_cfg, out, progress=False)
    return out, df


def test_smoke_schema_and_row_count(smoke_cfg, smoke_allocation):
    out, df = smoke_allocation
    a = smoke_cfg.track_a
    sa = smoke_cfg.improvement.shot_allocation
    expected = len(a.qubits) * len(a.depths) * len(sa.noise_levels) * len(smoke_cfg.experiment.seeds) * 2
    assert len(df) == expected
    assert list(df.columns) == ALLOCATION_COLUMNS
    assert (out / "improvement" / "zne_allocation.csv").exists()
    assert (out / "logs" / "improvement.log").exists()


def test_shots_sum_to_total_and_match_scales(smoke_allocation):
    _, df = smoke_allocation
    for _, row in df.iterrows():
        total = row.shots_lambda1 + row.shots_lambda3 + row.shots_lambda5
        assert total == row.total_shots == 3072


def test_uniform_is_equal_split():
    from qem.improvement import _shots_for_scheme

    g = richardson_coeffs((1, 3, 5))
    shots = _shots_for_scheme("uniform", g, 3072)
    assert shots == [1024, 1024, 1024]


def test_optimal_matches_worked_example(smoke_allocation):
    _, df = smoke_allocation
    opt = df[df.scheme == "optimal"].iloc[0]
    assert [opt.shots_lambda1, opt.shots_lambda3, opt.shots_lambda5] == pytest.approx([1646, 1097, 329], abs=1)


def test_theoretical_std_is_scheme_constant_and_optimal_is_lower(smoke_allocation):
    _, df = smoke_allocation
    uni_std = df[df.scheme == "uniform"].theoretical_std
    opt_std = df[df.scheme == "optimal"].theoretical_std
    assert uni_std.nunique() == 1
    assert opt_std.nunique() == 1
    assert uni_std.iloc[0] == pytest.approx(np.sqrt(5.21875 / 1024), abs=1e-6)
    ratio = opt_std.iloc[0] / uni_std.iloc[0]
    assert ratio == pytest.approx(0.885, abs=0.005)


def test_same_circuit_instance_as_track_a(smoke_cfg, smoke_allocation):
    """The improvement experiment must reuse Track A's circuit instance (same seeding),
    so E_exact here matches the one Track A would compute for the same (n, L, seed)."""
    from qem.circuits import select_instance
    from qem.observables import exact_reference

    _, df = smoke_allocation
    row = df.iloc[0]
    a = smoke_cfg.track_a
    inst = select_instance(int(row.n_qubits), int(row.depth_layers), int(row.seed),
                           a.circuit.min_abs_exact_expectation, a.circuit.max_resample_attempts)
    assert row.E_exact == pytest.approx(exact_reference(inst.unitary)["E_exact"], abs=1e-12)


def test_full_run_std_ratio_approaches_theory(cfg, tmp_path):
    """With more repetitions than the smoke config, the empirical std ratio should move
    toward the ~0.885 theoretical ratio. Uses a tiny single-condition slice of the full
    config rather than the whole grid, to keep this test fast."""
    import dataclasses

    from qem.config import ShotAllocationConfig

    trimmed = dataclasses.replace(
        cfg,
        track_a=dataclasses.replace(cfg.track_a, qubits=(2,), depths=(2,)),
        experiment=dataclasses.replace(cfg.experiment, seeds=(0,)),
        improvement=dataclasses.replace(cfg.improvement, shot_allocation=dataclasses.replace(
            cfg.improvement.shot_allocation, noise_levels=("moderate",), repetitions=200,
        )),
    )
    df = run_improvement(trimmed, tmp_path, progress=False)
    uni = df[df.scheme == "uniform"].iloc[0]
    opt = df[df.scheme == "optimal"].iloc[0]
    ratio = opt.empirical_std / uni.empirical_std
    assert 0.5 < ratio < 1.3  # loose bound: a genuine reduction, without demanding exact theory at N=200
