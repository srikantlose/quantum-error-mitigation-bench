import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from qem.circuits import select_instance, with_measurements
from qem.observables import (
    counts_to_probvec,
    exact_reference,
    success_probability,
    magnetization_from_probs,
    parity_from_probs,
    parity_signs,
    z_expectations,
    z_signs,
)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_bit_order_ry_pi_on_qubit0(n):
    u = QuantumCircuit(n)
    u.ry(np.pi, 0)
    counts = (
        AerSimulator(method="density_matrix")
        .run(with_measurements(u), shots=1024, seed_simulator=11)
        .result()
        .get_counts()
    )
    key = "0" * (n - 1) + "1"
    assert counts == {key: 1024}
    p = counts_to_probvec(counts, n)
    assert p[1] == 1.0
    z = z_expectations(p, n)
    assert z[0] == -1.0
    assert np.all(z[1:] == 1.0)
    # Statevector uses the same index convention
    assert Statevector(u).probabilities()[1] == pytest.approx(1.0)


def test_counts_to_probvec_missing_spaces_and_short_keys():
    p = counts_to_probvec({"01": 3, "11": 1}, 2)
    assert np.allclose(p, [0, 0.75, 0, 0.25])
    p = counts_to_probvec({"0 1": 2, "1 0": 2}, 2)
    assert np.allclose(p, [0, 0.5, 0.5, 0])
    p = counts_to_probvec({"1": 1, "10": 1, "0": 2}, 3)
    assert np.allclose(p, [0.5, 0.25, 0.25, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        counts_to_probvec({"0101": 1}, 3)
    with pytest.raises(ValueError):
        counts_to_probvec({}, 2)


@pytest.mark.parametrize("key, expected", [("00", 1.0), ("01", -1.0), ("10", -1.0), ("11", 1.0)])
def test_parity_of_basis_states(key, expected):
    assert parity_from_probs(counts_to_probvec({key: 100}, 2)) == expected


def test_signs():
    assert list(parity_signs(3)) == [1, -1, -1, 1, -1, 1, 1, -1]
    assert list(z_signs(3, 0)) == [1, -1, 1, -1, 1, -1, 1, -1]
    assert list(z_signs(3, 2)) == [1, 1, 1, 1, -1, -1, -1, -1]
    with pytest.raises(ValueError):
        parity_signs(2)[0] = 5  # cached arrays are read-only


def test_parity_is_linear_for_quasi_probs():
    q = np.array([0.7, -0.1, 0.25, 0.15])
    assert parity_from_probs(q) == pytest.approx(0.7 + 0.1 - 0.25 + 0.15)


def test_magnetization():
    # |q1 q0> = |01>: <Z0> = -1, <Z1> = +1 -> M = 0
    assert magnetization_from_probs(counts_to_probvec({"01": 1}, 2), 2) == 0.0
    assert magnetization_from_probs(counts_to_probvec({"000": 1}, 3), 3) == 1.0
    assert magnetization_from_probs(counts_to_probvec({"111": 1}, 3), 3) == -1.0


@pytest.mark.parametrize("n, L, seed", [(2, 2, 0), (4, 4, 1), (6, 4, 2)])
def test_exact_reference_matches_pauli_expectation(n, L, seed):
    inst = select_instance(n, L, seed, 0.3, 20000)
    ref = exact_reference(inst.unitary)
    E, M, probs = ref["E_exact"], ref["M_exact"], ref["probs"]
    sv = Statevector(inst.unitary)
    assert abs(E - sv.expectation_value(SparsePauliOp("Z" * n)).real) < 1e-9
    m_ref = np.mean([
        sv.expectation_value(SparsePauliOp("I" * (n - 1 - q) + "Z" + "I" * q)).real for q in range(n)
    ])
    assert abs(M - m_ref) < 1e-9
    assert probs.sum() == pytest.approx(1.0)
    assert ref["target_index"] == int(np.argmax(probs))
    assert ref["P_succ_exact"] == pytest.approx(probs.max())
    assert success_probability(probs, ref["target_index"]) == pytest.approx(ref["P_succ_exact"])


def test_success_probability():
    p = counts_to_probvec({"00": 1, "01": 3}, 2)
    assert success_probability(p, 0) == pytest.approx(0.25)
    assert success_probability(p, 1) == pytest.approx(0.75)
    # linear in quasi-probabilities too
    q = np.array([0.6, -0.1, 0.3, 0.2])
    assert success_probability(q, 1) == -0.1
