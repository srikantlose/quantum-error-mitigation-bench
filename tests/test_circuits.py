import numpy as np
import pytest

from qem.circuits import (
    build_unitary,
    build_unitary_from_angles,
    circuit_stats,
    load_instance,
    save_instance,
    select_instance,
    with_measurements,
)
from qem.observables import exact_reference
from qem.seeds import derive_seed

COMBOS = [(2, 2), (2, 4), (4, 2), (4, 4), (6, 2), (6, 4)]
GATE_TABLE = {  # §7.1: (n, L) -> (1-qubit gates, CNOTs)
    (2, 2): (8, 2),
    (2, 4): (16, 4),
    (4, 2): (16, 6),
    (4, 4): (32, 12),
    (6, 2): (24, 10),
    (6, 4): (48, 20),
}


@pytest.mark.parametrize("n, L", COMBOS)
def test_gate_counts_match_table(n, L):
    u, angles = build_unitary(n, L, np.random.default_rng(0))
    stats = circuit_stats(u)
    assert (stats["n1q"], stats["cx"]) == GATE_TABLE[(n, L)]
    assert stats["total_gates"] == sum(GATE_TABLE[(n, L)])
    assert angles.shape == (L, 2, n)
    assert set(u.count_ops()) == {"ry", "rz", "cx"}
    # measurements and barriers are excluded from the stats
    assert circuit_stats(with_measurements(u)) == stats


def test_angles_drawn_in_fixed_order_and_range():
    n, L = 4, 3
    _, angles = build_unitary(n, L, np.random.default_rng(123))
    flat = np.random.default_rng(123).uniform(0.0, 2 * np.pi, size=2 * n * L)
    # theta(layer 0), phi(layer 0), theta(layer 1), ...
    assert np.array_equal(angles.ravel(), flat)
    assert np.all(angles >= 0.0) and np.all(angles < 2 * np.pi)


def test_layer_structure():
    angles = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3) / 10
    u = build_unitary_from_angles(angles)
    names = [(inst.operation.name, [u.find_bit(q).index for q in inst.qubits]) for inst in u.data]
    layer = [("ry", [0]), ("ry", [1]), ("ry", [2]), ("rz", [0]), ("rz", [1]), ("rz", [2]),
             ("cx", [0, 1]), ("cx", [1, 2])]
    assert names == layer + layer
    assert float(u.data[3].operation.params[0]) == pytest.approx(angles[0, 1, 0])


def test_with_measurements_single_register():
    u, _ = build_unitary(3, 2, np.random.default_rng(1))
    qc = with_measurements(u)
    assert len(qc.cregs) == 1 and qc.num_clbits == 3
    assert qc.count_ops()["measure"] == 3
    assert u.num_clbits == 0  # the unitary itself is untouched


def test_same_seed_same_angles_different_seed_different():
    a = select_instance(4, 2, 0, 0.3, 20000)
    b = select_instance(4, 2, 0, 0.3, 20000)
    c = select_instance(4, 2, 1, 0.3, 20000)
    assert np.array_equal(a.angles, b.angles)
    assert a.attempts == b.attempts and a.E_exact == b.E_exact
    assert not np.array_equal(a.angles, c.angles)


@pytest.mark.parametrize("n, L", COMBOS)
def test_select_instance_meets_threshold(n, L):
    inst = select_instance(n, L, 0, 0.3, 20000)
    assert inst.attempts >= 1
    assert abs(inst.E_exact) >= 0.3
    ref = exact_reference(inst.unitary)
    E_exact, probs = ref["E_exact"], ref["probs"]
    assert E_exact == pytest.approx(inst.E_exact, abs=1e-12)
    assert probs.shape == (2**n,)


def test_select_instance_uses_circuit_seed():
    inst = select_instance(2, 2, 3, 0.0, 10)  # threshold 0 accepts the first draw
    assert inst.attempts == 1
    _, angles = build_unitary(2, 2, np.random.default_rng(derive_seed("circuit", 2, 2, 3)))
    assert np.array_equal(inst.angles, angles)


def test_select_instance_raises_when_exhausted():
    with pytest.raises(RuntimeError):
        select_instance(2, 2, 0, 1.01, 5)


def test_instance_roundtrip(tmp_path):
    inst = select_instance(4, 4, 2, 0.3, 20000)
    path = save_instance(inst, tmp_path)
    assert path.name == "n4_L4_s2.json"
    back = load_instance(path)
    assert np.array_equal(back.angles, inst.angles)
    assert back.attempts == inst.attempts and back.E_exact == inst.E_exact
    assert circuit_stats(back.unitary) == circuit_stats(inst.unitary)
    assert exact_reference(back.unitary)["E_exact"] == pytest.approx(inst.E_exact, abs=1e-12)
    assert "OPENQASM 2.0" in path.read_text()
