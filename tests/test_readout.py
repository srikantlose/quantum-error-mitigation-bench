import logging

import numpy as np
import pytest

from conftest import cfg_with_levels
from qem.circuits import calibration_circuits, circuit_stats, select_instance, with_measurements
from qem.execution import Executor
from qem.mitigation.readout import apply_rem, build_assignment_matrix, project_to_simplex
from qem.noise import true_assignment_matrix
from qem.observables import counts_to_probvec, parity_from_probs


@pytest.fixture(scope="module")
def ex(cfg):
    return Executor(cfg_with_levels(cfg, ro_only=(0.0, 0.0, 0.03)))


def calibrate(ex, n, level, shots, seed):
    cal, _ = ex.run(calibration_circuits(n), level, shots, seed)
    return build_assignment_matrix(cal, n, shots)


def test_calibration_circuit_structure():
    circs = calibration_circuits(3)
    assert len(circs) == 8
    for j, qc in enumerate(circs):
        assert set(qc.count_ops()) <= {"ry", "barrier", "measure"}
        assert circuit_stats(qc)["n1q"] == bin(j).count("1")
        assert circuit_stats(qc)["cx"] == 0


@pytest.mark.parametrize("n", [2, 3])
def test_calibration_circuit_j_gives_key_j(ex, n):
    cal, _ = ex.run(calibration_circuits(n), "ideal", 512, seed=4)
    for j, counts in enumerate(cal):
        assert counts == {format(j, f"0{n}b"): 512}


@pytest.mark.parametrize("n", [2, 4])
def test_ideal_assignment_matrix_is_identity(ex, n):
    A = calibrate(ex, n, "ideal", 1024, seed=8)
    assert np.array_equal(A, np.eye(2**n))


@pytest.mark.parametrize("n", [2, 3])
def test_readout_only_matrix_matches_tensor_product(ex, n):
    A = calibrate(ex, n, "ro_only", 50_000, seed=9)
    assert np.max(np.abs(A - true_assignment_matrix(0.03, n))) < 0.01


def test_columns_sum_to_one_under_moderate_noise(ex):
    A = calibrate(ex, 4, "moderate", 1024, seed=10)
    assert np.allclose(A.sum(axis=0), 1.0, atol=1e-12, rtol=0)
    assert np.all(A >= 0)


def test_rem_recovers_parity_with_readout_only_noise(ex):
    n, shots = 4, 50_000
    inst = select_instance(n, 4, 0, 0.3, 20000)
    sigma = np.sqrt((1 - inst.E_exact**2) / shots)
    (counts,), _ = ex.run([with_measurements(inst.unitary)], "ro_only", shots, seed=31)
    p_noisy = counts_to_probvec(counts, n)
    A = calibrate(ex, n, "ro_only", shots, seed=32)
    rem = apply_rem(p_noisy, A)
    E_none = parity_from_probs(p_noisy)
    E_rem = parity_from_probs(rem.quasi)
    assert abs(E_rem - inst.E_exact) < 4 * sigma
    assert abs(E_none - inst.E_exact) > 4 * sigma


def test_apply_rem_inverts_exactly():
    rng = np.random.default_rng(1)
    A = true_assignment_matrix(0.05, 3)
    p_true = rng.dirichlet(np.ones(8))
    res = apply_rem(A @ p_true, A)
    assert np.allclose(res.quasi, p_true, atol=1e-12)
    assert res.quasi.sum() == pytest.approx(1.0)
    assert res.negative_mass == pytest.approx(0.0, abs=1e-12)
    assert res.cond == pytest.approx(np.linalg.cond(A))
    assert not res.used_lstsq


def test_apply_rem_identity_is_exact():
    p = np.array([0.25, 0.5, 0.125, 0.125])
    res = apply_rem(p, np.eye(4))
    assert np.array_equal(res.quasi, p)
    assert res.cond == 1.0


def test_apply_rem_negative_mass_and_lstsq_fallback(caplog):
    A = true_assignment_matrix(0.1, 2)
    p = np.array([0.95, 0.02, 0.02, 0.01])  # sharper than A can produce -> negative quasi-probs
    res = apply_rem(p, A)
    assert res.negative_mass > 0
    assert res.quasi.sum() == pytest.approx(1.0)
    assert res.negative_mass == pytest.approx(-res.quasi[res.quasi < 0].sum())
    with caplog.at_level(logging.WARNING):
        res2 = apply_rem(p, A, cond_threshold=1.0)
    assert res2.used_lstsq
    assert "lstsq" in caplog.text
    assert np.allclose(res2.quasi, res.quasi, atol=1e-12)


def test_project_to_simplex_outputs_valid_distribution():
    rng = np.random.default_rng(2)
    for _ in range(200):
        v = rng.normal(size=8) * rng.uniform(0.01, 3)
        x = project_to_simplex(v)
        assert np.all(x >= 0)
        assert x.sum() == pytest.approx(1.0, abs=1e-12)
        # it is the closest point: no random simplex point is closer
        y = rng.dirichlet(np.ones(8))
        assert np.linalg.norm(x - v) <= np.linalg.norm(y - v) + 1e-12


def test_project_to_simplex_leaves_valid_input_unchanged():
    rng = np.random.default_rng(3)
    for p in [np.array([1.0, 0, 0, 0]), np.array([0.25] * 4), rng.dirichlet(np.ones(16)),
              np.array([0.5, 0.0, 0.5, 0.0])]:
        assert np.max(np.abs(project_to_simplex(p) - p)) < 1e-12


def test_project_to_simplex_known_case():
    # quasi-probabilities summing to 1 with a negative entry
    x = project_to_simplex(np.array([0.7, -0.1, 0.25, 0.15]))
    assert np.allclose(x, [0.7 - 1 / 30, 0.0, 0.25 - 1 / 30, 0.15 - 1 / 30])
