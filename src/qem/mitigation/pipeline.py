"""Shared mitigation pipeline: turns per-scale probability vectors into per-method estimates.

Both Track A (the core testbed, `experiment.py`) and Track B (the VQC, `qml/evaluate.py`)
call ``apply_methods`` so that none, rem, rem_tensored, zne and zne_rem are computed
identically in both places, and the paired-design guarantee (every method reads the same
counts) holds across the whole project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..observables import magnetization_from_probs, parity_from_probs, success_probability
from .readout import apply_rem
from .zne import extrapolate_exp, extrapolate_linear, extrapolate_richardson, zne_std

METHODS_ALL = ("none", "rem", "rem_tensored", "zne", "zne_rem")


@dataclass
class MethodEstimate:
    method: str
    E_hat: float
    E_hat_clipped: float
    M_hat: float
    P_succ_hat: float
    distribution: np.ndarray | None = None  # projected distribution; None for zne/zne_rem
    E_by_scale: dict[int, float] | None = None  # only set for zne/zne_rem
    E_richardson: float | None = None
    E_linear: float | None = None
    E_exp: float | None = None
    extrapolation_out_of_range: bool | None = None
    est_std: float | None = None
    rem_negative_mass: float | None = None
    rem_condition_number: float | None = None


def _none_estimate(p1: np.ndarray, n: int, target_index: int, shots: int) -> MethodEstimate:
    E = parity_from_probs(p1)
    return MethodEstimate(
        method="none", E_hat=E, E_hat_clipped=float(np.clip(E, -1.0, 1.0)),
        M_hat=magnetization_from_probs(p1, n), P_succ_hat=success_probability(p1, target_index),
        distribution=p1, est_std=float(np.sqrt(max(0.0, 1.0 - E**2) / shots)),
    )


def _rem_estimate(method: str, p1: np.ndarray, A: np.ndarray, n: int, target_index: int,
                  cond_threshold: float) -> MethodEstimate:
    res = apply_rem(p1, A, cond_threshold)
    E = parity_from_probs(res.quasi)
    return MethodEstimate(
        method=method, E_hat=E, E_hat_clipped=float(np.clip(E, -1.0, 1.0)),
        M_hat=magnetization_from_probs(res.quasi, n),
        P_succ_hat=success_probability(res.quasi, target_index),
        distribution=res.projected,
        rem_negative_mass=res.negative_mass, rem_condition_number=res.cond,
    )


def _zne_estimate(method: str, p_by_scale: dict[int, np.ndarray], scales: Sequence[int], n: int,
                  target_index: int, shots: int, A: np.ndarray | None,
                  cond_threshold: float) -> MethodEstimate:
    """REM-corrected when A is not None (zne_rem); raw counts otherwise (zne)."""
    E_vals, M_vals, psucc_vals, neg_masses = [], [], [], []
    cond = float("nan")
    for s in scales:
        p = p_by_scale[s]
        if A is not None:
            res = apply_rem(p, A, cond_threshold)
            p = res.quasi
            neg_masses.append(res.negative_mass)
            cond = res.cond
        E_vals.append(parity_from_probs(p))
        M_vals.append(magnetization_from_probs(p, n))
        psucc_vals.append(success_probability(p, target_index))
    E_rich = extrapolate_richardson(scales, E_vals)
    out = MethodEstimate(
        method=method, E_hat=E_rich, E_hat_clipped=float(np.clip(E_rich, -1.0, 1.0)),
        M_hat=extrapolate_richardson(scales, M_vals),
        P_succ_hat=extrapolate_richardson(scales, psucc_vals),
        E_by_scale=dict(zip(scales, E_vals)),
        E_richardson=E_rich, E_linear=extrapolate_linear(scales, E_vals),
        E_exp=extrapolate_exp(scales, E_vals),
        extrapolation_out_of_range=abs(E_rich) > 1.0,
        est_std=None if A is not None else zne_std(scales, E_vals, shots),
    )
    if A is not None:
        out.rem_negative_mass = float(np.mean(neg_masses))
        out.rem_condition_number = cond
    return out


def apply_methods(
    p_by_scale: dict[int, np.ndarray],
    A_full: np.ndarray | None,
    A_tensored: np.ndarray | None,
    methods: Sequence[str],
    n: int,
    target_index: int,
    shots: int,
    scales: Sequence[int] = (1, 3, 5),
    cond_threshold: float = 1e8,
) -> dict[str, MethodEstimate]:
    """Compute one MethodEstimate per requested method from the same per-scale distributions.

    ``p_by_scale`` maps each scale factor to its probability vector (already normalized from
    counts). ``p_by_scale[1]`` is the unfolded, base-noise distribution. ``A_full`` is required
    for "rem" and "zne_rem"; ``A_tensored`` for "rem_tensored". A method not present in
    ``methods`` needs no matrix for it.
    """
    out: dict[str, MethodEstimate] = {}
    for m in methods:
        if m == "none":
            out[m] = _none_estimate(p_by_scale[1], n, target_index, shots)
        elif m == "rem":
            out[m] = _rem_estimate("rem", p_by_scale[1], A_full, n, target_index, cond_threshold)
        elif m == "rem_tensored":
            out[m] = _rem_estimate("rem_tensored", p_by_scale[1], A_tensored, n, target_index, cond_threshold)
        elif m == "zne":
            out[m] = _zne_estimate("zne", p_by_scale, scales, n, target_index, shots, None, cond_threshold)
        elif m == "zne_rem":
            out[m] = _zne_estimate("zne_rem", p_by_scale, scales, n, target_index, shots, A_full, cond_threshold)
        else:
            raise ValueError(f"unknown method {m!r}")
    return out
