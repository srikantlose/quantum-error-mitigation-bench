import numpy as np
import pytest
from qiskit.quantum_info import Statevector, state_fidelity

from conftest import cfg_with_levels
from qem.circuits import circuit_stats, fold_global, select_instance, with_measurements
from qem.execution import Executor
from qem.mitigation.zne import (
    extrapolate_exp,
    extrapolate_linear,
    extrapolate_richardson,
    fit_exp,
    richardson_coeffs,
    richardson_curve,
    zne_std,
)
from qem.observables import counts_to_probvec, exact_reference, parity_from_probs
from qem.seeds import derive_seed

SCALES = (1, 3, 5)


@pytest.mark.parametrize("n, L, seed", [(2, 2, 0), (4, 4, 1), (6, 4, 2)])
@pytest.mark.parametrize("scale", [3, 5])
def test_folded_statevector_matches_original(n, L, seed, scale):
    u = select_instance(n, L, seed, 0.3, 20000).unitary
    f = fold_global(u, scale)
    assert state_fidelity(Statevector(u), Statevector(f)) > 1 - 1e-9


@pytest.mark.parametrize("n, L", [(2, 2), (4, 4), (6, 2)])
@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_folded_gate_counts_scale_exactly(n, L, scale):
    u = select_instance(n, L, 0, 0.3, 20000).unitary
    base = circuit_stats(u)
    folded = fold_global(u, scale)
    st = circuit_stats(folded)
    assert st["cx"] == scale * base["cx"]
    assert st["n1q"] == scale * base["n1q"]
    assert set(folded.count_ops()) <= {"ry", "rz", "cx", "barrier"}
    assert st["depth"] <= scale * base["depth"]


def test_fold_inverse_negates_angles():
    u = select_instance(2, 2, 0, 0.3, 20000).unitary
    f = fold_global(u, 3)
    gates = [inst for inst in f.data if inst.operation.name != "barrier"]
    m = len(u.data)
    inverse_part = gates[m:2 * m]
    assert inverse_part[-1].operation.name == "ry"
    assert float(inverse_part[-1].operation.params[0]) == pytest.approx(-float(u.data[0].operation.params[0]))


def test_fold_rejects_even_or_nonpositive_scale():
    u = select_instance(2, 2, 0, 0.3, 20000).unitary
    for bad in (0, 2, 4, -1):
        with pytest.raises(ValueError):
            fold_global(u, bad)


def test_richardson_coeffs_135():
    g = richardson_coeffs(SCALES)
    assert np.allclose(g, [1.875, -1.25, 0.375], atol=1e-15)
    assert g.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.sqrt(np.sum(g**2)) == pytest.approx(2.2845, abs=1e-4)


def test_richardson_coeffs_general():
    assert np.allclose(richardson_coeffs([1, 2, 3]), [3, -3, 1])
    assert np.allclose(richardson_coeffs([1, 3]), [1.5, -0.5])
    assert richardson_coeffs([1, 3, 5, 7]).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        richardson_coeffs([1, 3, 3])


def test_richardson_recovers_any_quadratic():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b, c = rng.normal(size=3)
        values = [a + b * s + c * s**2 for s in SCALES]
        assert abs(extrapolate_richardson(SCALES, values) - a) < 1e-12
        curve = richardson_curve(SCALES, values)
        assert np.allclose(curve(np.array([0.0, 1.0, 2.0, 5.0])), [a, a + b + c, a + 2 * b + 4 * c, a + 5 * b + 25 * c])


def test_linear_recovers_exact_line():
    values = [0.8 - 0.07 * s for s in SCALES]
    assert extrapolate_linear(SCALES, values) == pytest.approx(0.8, abs=1e-12)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_exp_recovers_exact_exponential(sign):
    b, c = 0.73, 0.21
    values = [sign * b * np.exp(-c * s) for s in SCALES]
    assert extrapolate_exp(SCALES, values) == pytest.approx(sign * b, abs=1e-12)
    s, bb, cc = fit_exp(SCALES, values)
    assert (s, bb, cc) == (sign, pytest.approx(b), pytest.approx(c))


def test_exp_returns_nan_when_invalid():
    assert np.isnan(extrapolate_exp(SCALES, [0.5, 0.2, -0.05]))
    assert np.isnan(extrapolate_exp(SCALES, [0.5, 0.2, 1e-7]))
    assert np.isnan(extrapolate_exp(SCALES, [0.5, 0.0, 0.1]))


def test_zne_std_formula():
    values = [0.6, 0.3, 0.1]
    g = np.array([1.875, -1.25, 0.375])
    expected = np.sqrt(np.sum(g**2 * (1 - np.array(values) ** 2) / 1024))
    assert zne_std(SCALES, values, 1024) == pytest.approx(expected)
    # at E = 0 the amplification over a single run is sqrt(sum gamma^2)
    assert zne_std(SCALES, [0, 0, 0], 1024) / np.sqrt(1 / 1024) == pytest.approx(np.sqrt(5.21875))


def test_zne_beats_none_under_gate_only_noise(cfg):
    n, L, shots = 4, 4, 50_000
    ex = Executor(cfg_with_levels(cfg, gate_only=(0.005, 0.03, 0.0)))
    inst = select_instance(n, L, 0, 0.3, 20000)
    E_exact = exact_reference(inst.unitary)[0]
    values = []
    for scale in SCALES:
        qc = with_measurements(fold_global(inst.unitary, scale))
        (counts,), _ = ex.run([qc], "gate_only", shots, derive_seed("test-zne", scale))
        values.append(parity_from_probs(counts_to_probvec(counts, n)))
    err_none = abs(values[0] - E_exact)
    err_zne = abs(extrapolate_richardson(SCALES, values) - E_exact)
    assert err_zne < err_none
