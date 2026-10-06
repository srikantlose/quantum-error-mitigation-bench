"""Aer noise models: depolarizing gate noise plus symmetric readout error.

Qiskit's ``depolarizing_error(lam, k)`` is E(rho) = (1 - lam) rho + lam Tr(rho) I / 2^k; a
non-identity Pauli is applied with probability lam (4^k - 1) / 4^k.

Noise is attached by gate name, so any gate outside ``one_q_gates`` / ``two_q_gates`` runs
noise-free. The executor guards against that.
"""

from __future__ import annotations

from functools import reduce
from typing import Sequence

import numpy as np
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .config import NoiseLevelConfig


def build_noise_model(
    level: NoiseLevelConfig, one_q_gates: Sequence[str], two_q_gates: Sequence[str]
) -> NoiseModel | None:
    if level.p1 == 0 and level.p2 == 0 and level.p_ro == 0:
        return None
    nm = NoiseModel()
    if level.p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(level.p1, 1), list(one_q_gates))
    if level.p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(level.p2, 2), list(two_q_gates))
    if level.p_ro > 0:
        # Row k gives P(measured | prepared k).
        nm.add_all_qubit_readout_error(ReadoutError([[1 - level.p_ro, level.p_ro], [level.p_ro, 1 - level.p_ro]]))
    return nm


def readout_attenuation(p_ro: float, n: int) -> float:
    """Factor by which symmetric readout error multiplies the n-qubit parity."""
    return (1 - 2 * p_ro) ** n


def single_qubit_assignment(p_ro: float) -> np.ndarray:
    """A_q[i, j] = P(measure i | prepare j) for symmetric flip probability p_ro."""
    return np.array([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])


def true_assignment_matrix(p_ro: float, n: int) -> np.ndarray:
    """A_true = A_{n-1} ⊗ ... ⊗ A_0 (qubit n-1 is the most significant bit)."""
    a = single_qubit_assignment(p_ro)
    return reduce(np.kron, [a] * n)
