"""Readout error mitigation with a full 2^n x 2^n assignment matrix."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..observables import counts_to_probvec

log = logging.getLogger(__name__)


@dataclass
class RemResult:
    quasi: np.ndarray  # A^-1 p: sums to 1, may have negative entries
    projected: np.ndarray  # quasi projected onto the probability simplex
    negative_mass: float
    cond: float
    used_lstsq: bool = False


def build_assignment_matrix(cal_counts: list[dict[str, int]], n: int, shots: int) -> np.ndarray:
    """A[i, j] = P(measured i | prepared j); column j comes from calibration circuit j."""
    if len(cal_counts) != 2**n:
        raise ValueError(f"expected {2**n} calibration results, got {len(cal_counts)}")
    A = np.zeros((2**n, 2**n))
    for j, counts in enumerate(cal_counts):
        total = sum(counts.values())
        if total != shots:
            raise ValueError(f"calibration circuit {j} has {total} shots, expected {shots}")
        A[:, j] = counts_to_probvec(counts, n)
    col_sums = A.sum(axis=0)
    if not np.all(np.abs(col_sums - 1.0) <= 1e-12):
        raise AssertionError(f"assignment matrix columns do not sum to 1: {col_sums}")
    return A


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x : x >= 0, sum(x) = 1}."""
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, len(v) + 1)
    rho = j[u - (css - 1) / j > 0][-1]
    tau = (css[rho - 1] - 1) / rho
    return np.maximum(v - tau, 0.0)


def apply_rem(p_noisy: np.ndarray, A: np.ndarray, cond_threshold: float = 1e8) -> RemResult:
    """Invert the assignment matrix. Expectations should use ``quasi``, distributions ``projected``."""
    p_noisy = np.asarray(p_noisy, dtype=float)
    cond = float(np.linalg.cond(A))
    used_lstsq = cond >= cond_threshold
    if used_lstsq:
        log.warning("assignment matrix cond=%.3g >= %.3g; using lstsq", cond, cond_threshold)
        q = np.linalg.lstsq(A, p_noisy, rcond=None)[0]
    else:
        q = np.linalg.solve(A, p_noisy)
    negative_mass = float(np.abs(q[q < 0]).sum())
    return RemResult(quasi=q, projected=project_to_simplex(q), negative_mass=negative_mass,
                     cond=cond, used_lstsq=used_lstsq)
