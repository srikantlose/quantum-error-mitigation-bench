"""Per-row error, fidelity and noise-floor metrics."""

from __future__ import annotations

import numpy as np

ZERO_ERROR = 1e-12


def abs_error(E_hat: float, E_exact: float) -> float:
    return abs(E_hat - E_exact)


def signed_error(E_hat: float, E_exact: float) -> float:
    return E_hat - E_exact


def rel_error(E_hat: float, E_exact: float) -> float:
    return abs(E_hat - E_exact) / abs(E_exact)


def improvement_pct(err: float, err_none: float) -> float:
    """100 (1 - err / err_none); NaN when the baseline error is zero."""
    if err_none < ZERO_ERROR:
        return float("nan")
    return 100.0 * (1.0 - err / err_none)


def error_reduction_factor(err: float, err_none: float) -> float:
    """err_none / err; inf when the method's error is (numerically) zero."""
    if err < ZERO_ERROR:
        return float("inf")
    return err_none / err


def success_ratio(p_succ_hat: float, p_succ_exact: float) -> float:
    """P_succ_hat / P_succ_exact; NaN when the exact success probability is (numerically) zero."""
    if abs(p_succ_exact) < ZERO_ERROR:
        return float("nan")
    return p_succ_hat / p_succ_exact


def hellinger_fidelity(p: np.ndarray, q: np.ndarray) -> float:
    """(sum_x sqrt(p_x q_x))^2 on full 2^n vectors; matches qiskit's hellinger_fidelity."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any(p < -1e-12) or np.any(q < -1e-12):
        raise ValueError("hellinger fidelity needs non-negative distributions")
    return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))) ** 2)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance 1/2 sum_x |p_x - q_x|."""
    return float(0.5 * np.sum(np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float))))


def shot_noise_floor(E_exact: float, shots: int) -> float:
    """sqrt((1 - E_exact^2) / shots): the standard deviation of an ideal shot estimate."""
    return float(np.sqrt(max(0.0, 1.0 - E_exact**2) / shots))


def binomial_std(E_hat: float, shots: int) -> float:
    """sqrt((1 - E_hat^2) / shots): estimated std of a single-run +-1 average."""
    return float(np.sqrt(max(0.0, 1.0 - E_hat**2) / shots))
