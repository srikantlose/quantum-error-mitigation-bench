import numpy as np
import pytest
from qiskit import QuantumCircuit

from conftest import cfg_with_levels
from qem.circuits import select_instance, with_measurements
from qem.execution import Executor, assert_noise_basis
from qem.noise import build_noise_model, readout_attenuation, true_assignment_matrix
from qem.observables import counts_to_probvec, exact_reference, parity_from_probs


@pytest.fixture(scope="module")
def ex(cfg):
    return Executor(cfg_with_levels(cfg, ro_only=(0.0, 0.0, 0.03), gate_only=(0.005, 0.03, 0.0)))


def test_ideal_level_has_no_noise_model(cfg):
    assert build_noise_model(cfg.noise.level("ideal"), ["ry", "rz"], ["cx"]) is None


def test_noise_model_covers_basis(cfg):
    nm = build_noise_model(cfg.noise.level("moderate"), ["ry", "rz"], ["cx"])
    assert {"ry", "rz", "cx", "measure"} <= set(nm.noise_instructions)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_ideal_run_of_zero_state_gives_one_key(ex, n):
    (counts,), _ = ex.run([with_measurements(QuantumCircuit(n))], "ideal", 1024, seed=5)
    assert counts == {"0" * n: 1024}


def test_readout_attenuation_of_zero_state(ex):
    n, shots = 4, 50_000
    (counts,), _ = ex.run([with_measurements(QuantumCircuit(n))], "ro_only", shots, seed=17)
    E = parity_from_probs(counts_to_probvec(counts, n))
    expected = readout_attenuation(0.03, n)
    assert expected == pytest.approx(0.94**4)
    sigma = np.sqrt((1 - expected**2) / shots)
    assert abs(E - expected) < 4 * sigma


def test_true_assignment_matrix_scales_parity_exactly():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5):
        p = rng.dirichlet(np.ones(2**n))
        A = true_assignment_matrix(0.03, n)
        assert np.allclose(A.sum(axis=0), 1.0)
        assert parity_from_probs(A @ p) == pytest.approx(parity_from_probs(p) * readout_attenuation(0.03, n))


def test_moderate_gate_noise_shrinks_parity(ex):
    inst = select_instance(4, 4, 0, 0.3, 20000)
    qc = with_measurements(inst.unitary)
    (ideal,), _ = ex.run([qc], "ideal", 20_000, seed=3)
    (noisy,), _ = ex.run([qc], "moderate", 20_000, seed=3)
    E_ideal = parity_from_probs(counts_to_probvec(ideal, 4))
    E_noisy = parity_from_probs(counts_to_probvec(noisy, 4))
    assert abs(E_noisy) < abs(E_ideal)


def test_guard_rejects_gate_outside_noise_basis(ex):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))
    with pytest.raises(AssertionError, match="h"):
        ex.run([qc], "moderate", 100, seed=1)
    x = QuantumCircuit(1)
    x.x(0)
    with pytest.raises(AssertionError, match="x"):
        assert_noise_basis([x], frozenset({"ry", "rz", "cx", "measure", "barrier"}))


def test_same_seed_same_counts_different_seed_different(ex):
    inst = select_instance(4, 2, 1, 0.3, 20000)
    qc = with_measurements(inst.unitary)
    a, _ = ex.run([qc], "moderate", 1024, seed=99)
    b, _ = ex.run([qc], "moderate", 1024, seed=99)
    c, _ = ex.run([qc], "moderate", 1024, seed=100)
    assert a == b
    assert a != c


def test_counts_keys_are_normalized(ex):
    inst = select_instance(4, 2, 1, 0.3, 20000)
    (counts,), elapsed = ex.run([with_measurements(inst.unitary)], "moderate", 1024, seed=7)
    assert all(len(k) == 4 and " " not in k for k in counts)
    assert list(counts) == sorted(counts)
    assert sum(counts.values()) == 1024
    assert elapsed > 0


def test_exact_noisy_probs(ex):
    inst = select_instance(4, 4, 0, 0.3, 20000)
    ref = exact_reference(inst.unitary)
    E_exact, p_exact = ref["E_exact"], ref["probs"]
    # ideal: identical to the statevector distribution
    assert np.allclose(ex.exact_noisy_probs(inst.unitary, "ideal"), p_exact, atol=1e-12)
    # readout only: parity attenuated by exactly (1 - 2p)^n
    p_ro = ex.exact_noisy_probs(inst.unitary, "ro_only")
    assert parity_from_probs(p_ro) == pytest.approx(E_exact * readout_attenuation(0.03, 4), abs=1e-12)
    # gate + readout noise: sampled estimate is consistent with the exact noisy value
    p_mod = ex.exact_noisy_probs(inst.unitary, "moderate")
    E_noisy_exact = parity_from_probs(p_mod)
    assert p_mod.sum() == pytest.approx(1.0)
    assert abs(E_noisy_exact) < abs(E_exact)
    shots = 50_000
    (counts,), _ = ex.run([with_measurements(inst.unitary)], "moderate", shots, seed=21)
    E_hat = parity_from_probs(counts_to_probvec(counts, 4))
    assert abs(E_hat - E_noisy_exact) < 4 * np.sqrt((1 - E_noisy_exact**2) / shots)
