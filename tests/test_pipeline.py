import numpy as np
import pytest

from qem.circuits import (
    calibration_circuits,
    fold_global,
    select_instance,
    tensored_calibration_circuits,
    with_measurements,
)
from qem.mitigation.pipeline import METHODS_ALL, apply_methods
from qem.mitigation.readout import build_assignment_matrix, build_tensored_assignment_matrix
from qem.observables import counts_to_probvec, exact_reference
from qem.seeds import derive_seed


def _run_all_scales(ex, unitary, level, shots, tag):
    p_by_scale = {}
    for s in (1, 3, 5):
        qc = with_measurements(unitary if s == 1 else fold_global(unitary, s))
        (counts,), _ = ex.run([qc], level, shots, derive_seed("pipeline-test", tag, level, s))
        p_by_scale[s] = counts_to_probvec(counts, unitary.num_qubits)
    return p_by_scale


@pytest.fixture(scope="module")
def ex(cfg):
    from qem.execution import Executor

    return Executor(cfg)


def test_apply_methods_all_methods_present(ex):
    inst = select_instance(2, 2, 0, 0.3, 20000)
    n, shots = inst.n, 1024
    ref = exact_reference(inst.unitary)
    p_by_scale = _run_all_scales(ex, inst.unitary, "moderate", shots, "all")
    cal, _ = ex.run(calibration_circuits(n), "moderate", shots, derive_seed("pipeline-test", "cal"))
    A = build_assignment_matrix(cal, n, shots)
    (c0, c1), _ = ex.run(tensored_calibration_circuits(n), "moderate", shots, derive_seed("pipeline-test", "calt"))
    A_t = build_tensored_assignment_matrix(c0, c1, n)

    est = apply_methods(p_by_scale, A, A_t, METHODS_ALL, n, ref["target_index"], shots)
    assert set(est) == set(METHODS_ALL)

    # none and rem variants carry a distribution; zne variants carry none
    for m in ("none", "rem", "rem_tensored"):
        assert est[m].distribution is not None
        assert est[m].E_by_scale is None
    for m in ("zne", "zne_rem"):
        assert est[m].distribution is None
        assert est[m].E_by_scale == {1: pytest.approx(est[m].E_by_scale[1]), 3: pytest.approx(est[m].E_by_scale[3]),
                                     5: pytest.approx(est[m].E_by_scale[5])}
        assert set(est[m].E_by_scale) == {1, 3, 5}

    # est_std is populated for none and zne only
    assert est["none"].est_std is not None
    assert est["zne"].est_std is not None
    assert est["rem"].est_std is None
    assert est["rem_tensored"].est_std is None
    assert est["zne_rem"].est_std is None

    # rem diagnostics only on rem-family methods
    for m in ("rem", "rem_tensored", "zne_rem"):
        assert est[m].rem_negative_mass is not None
        assert est[m].rem_condition_number is not None
    assert est["none"].rem_negative_mass is None
    assert est["zne"].rem_negative_mass is None


def test_apply_methods_ideal_rem_matches_none(ex):
    inst = select_instance(2, 2, 1, 0.3, 20000)
    n, shots = inst.n, 1024
    ref = exact_reference(inst.unitary)
    p_by_scale = _run_all_scales(ex, inst.unitary, "ideal", shots, "ideal")
    cal, _ = ex.run(calibration_circuits(n), "ideal", shots, derive_seed("pipeline-test", "ideal-cal"))
    A = build_assignment_matrix(cal, n, shots)
    (c0, c1), _ = ex.run(tensored_calibration_circuits(n), "ideal", shots, derive_seed("pipeline-test", "ideal-calt"))
    A_t = build_tensored_assignment_matrix(c0, c1, n)

    est = apply_methods(p_by_scale, A, A_t, ("none", "rem", "rem_tensored"), n, ref["target_index"], shots)
    assert est["rem"].E_hat == pytest.approx(est["none"].E_hat, abs=1e-12)
    assert est["rem_tensored"].E_hat == pytest.approx(est["none"].E_hat, abs=1e-12)


def test_apply_methods_only_requested_methods_computed(ex):
    inst = select_instance(2, 2, 2, 0.3, 20000)
    n, shots = inst.n, 1024
    ref = exact_reference(inst.unitary)
    p_by_scale = _run_all_scales(ex, inst.unitary, "low", shots, "subset")
    # None of these methods need calibration matrices, so passing None for both must work.
    est = apply_methods(p_by_scale, None, None, ("none", "zne"), n, ref["target_index"], shots)
    assert set(est) == {"none", "zne"}
