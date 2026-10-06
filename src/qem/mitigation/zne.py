"""Zero-noise extrapolators and the ZNE variance estimate.

Each extrapolator takes the noise scale factors and the expectation values measured at
them, and returns the estimate at scale 0.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

EXP_MIN_ABS = 1e-6


def richardson_coeffs(scales: Sequence[float]) -> np.ndarray:
    """gamma_i = prod_{j != i} s_j / (s_j - s_i), so that E0 = sum_i gamma_i E(s_i)."""
    s = np.asarray(scales, dtype=float)
    if len(np.unique(s)) != len(s):
        raise ValueError(f"scale factors must be distinct, got {list(scales)}")
    gammas = np.ones(len(s))
    for i in range(len(s)):
        for j in range(len(s)):
            if j != i:
                gammas[i] *= s[j] / (s[j] - s[i])
    return gammas


def extrapolate_richardson(scales: Sequence[float], values: Sequence[float]) -> float:
    """Exact degree m-1 polynomial through all m points, evaluated at 0."""
    return float(richardson_coeffs(scales) @ np.asarray(values, dtype=float))


def richardson_curve(scales: Sequence[float], values: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    """The interpolating polynomial itself (Lagrange form), for plotting."""
    s = np.asarray(scales, dtype=float)
    v = np.asarray(values, dtype=float)

    def curve(x):
        x = np.asarray(x, dtype=float)
        total = np.zeros_like(x)
        for i in range(len(s)):
            term = np.full_like(x, v[i])
            for j in range(len(s)):
                if j != i:
                    term = term * (x - s[j]) / (s[i] - s[j])
            total = total + term
        return total

    return curve


def fit_linear(scales: Sequence[float], values: Sequence[float]) -> tuple[float, float]:
    """Least-squares E(lambda) = a + b lambda; returns (a, b)."""
    s = np.asarray(scales, dtype=float)
    X = np.column_stack([np.ones_like(s), s])
    (a, b), *_ = np.linalg.lstsq(X, np.asarray(values, dtype=float), rcond=None)
    return float(a), float(b)


def extrapolate_linear(scales: Sequence[float], values: Sequence[float]) -> float:
    return fit_linear(scales, values)[0]


def fit_exp(scales: Sequence[float], values: Sequence[float]) -> tuple[float, float, float] | None:
    """Fit E(lambda) = sign * b * exp(-c lambda) by regressing ln|E| on lambda.

    Returns (sign, b, c), or None when the values change sign or any |E| <= 1e-6.
    """
    v = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(v)) or np.any(np.abs(v) <= EXP_MIN_ABS):
        return None
    signs = np.sign(v)
    if not np.all(signs == signs[0]):
        return None
    ln_b, minus_c = fit_linear(scales, np.log(np.abs(v)))
    return float(signs[0]), float(np.exp(ln_b)), float(-minus_c)


def extrapolate_exp(scales: Sequence[float], values: Sequence[float]) -> float:
    fit = fit_exp(scales, values)
    if fit is None:
        return float("nan")
    sign, b, _ = fit
    return sign * b


def zne_std(scales: Sequence[float], values: Sequence[float], shots: int) -> float:
    """sqrt(sum_i gamma_i^2 (1 - E_i^2) / shots), treating each E_i as an independent
    average of +-1 outcomes."""
    g = richardson_coeffs(scales)
    v = np.asarray(values, dtype=float)
    var_terms = np.clip(1 - v**2, 0.0, None) / shots
    return float(np.sqrt(np.sum(g**2 * var_terms)))


EXTRAPOLATORS: dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
    "richardson": extrapolate_richardson,
    "linear": extrapolate_linear,
    "exp": extrapolate_exp,
}
