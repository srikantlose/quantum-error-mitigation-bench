import numpy as np
import pytest
from qiskit.quantum_info import hellinger_fidelity as qiskit_hellinger

from qem import metrics
from qem.observables import counts_to_probvec


@pytest.mark.parametrize(
    "a, b",
    [
        ({"00": 500, "11": 524}, {"00": 480, "01": 20, "11": 524}),
        ({"000": 10, "101": 30, "111": 60}, {"000": 25, "011": 5, "101": 25, "110": 45}),
        ({"01": 7}, {"10": 3}),
    ],
)
def test_hellinger_matches_qiskit(a, b):
    n = len(next(iter(a)))
    ours = metrics.hellinger_fidelity(counts_to_probvec(a, n), counts_to_probvec(b, n))
    assert ours == pytest.approx(qiskit_hellinger(a, b), abs=1e-12)


def test_hellinger_and_tvd_extremes():
    p = np.array([0.5, 0.5, 0, 0])
    q = np.array([0, 0, 0.5, 0.5])
    assert metrics.hellinger_fidelity(p, p) == pytest.approx(1.0)
    assert metrics.hellinger_fidelity(p, q) == 0.0
    assert metrics.tvd(p, p) == 0.0
    assert metrics.tvd(p, q) == pytest.approx(1.0)
    assert metrics.tvd(p, np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        metrics.hellinger_fidelity(np.array([1.1, -0.1]), np.array([0.5, 0.5]))


def test_error_metrics():
    assert metrics.abs_error(0.4, 0.5) == pytest.approx(0.1)
    assert metrics.signed_error(0.4, 0.5) == pytest.approx(-0.1)
    assert metrics.rel_error(-0.4, -0.5) == pytest.approx(0.2)
    assert metrics.rel_error(0.4, -0.5) == pytest.approx(1.8)


def test_improvement_and_reduction_factor():
    assert metrics.improvement_pct(0.025, 0.1) == pytest.approx(75.0)
    assert metrics.improvement_pct(0.2, 0.1) == pytest.approx(-100.0)
    assert np.isnan(metrics.improvement_pct(0.1, 0.0))
    assert metrics.error_reduction_factor(0.025, 0.1) == pytest.approx(4.0)
    assert metrics.error_reduction_factor(0.0, 0.1) == float("inf")


def test_noise_floor_and_std():
    assert metrics.shot_noise_floor(0.0, 1024) == pytest.approx(1 / 32)
    assert metrics.shot_noise_floor(0.6, 100) == pytest.approx(0.08)
    assert metrics.binomial_std(1.2, 100) == 0.0
