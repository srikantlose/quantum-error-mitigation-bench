"""Counts -> probability vectors, parity, magnetization and success probability, and exact
noiseless references.

Bit ordering follows Qiskit (little-endian): in a count key ``s`` of length n, qubit q is
``s[n-1-q]``; in the integer index ``i = int(s, 2)`` qubit q is bit ``(i >> q) & 1``.
``Statevector.probabilities()`` uses the same index convention, so every function here
works on length-2^n vectors indexed this way.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


def normalize_key(key: str, n: int) -> str:
    k = key.replace(" ", "")
    if len(k) > n:
        raise ValueError(f"count key {key!r} has more than {n} bits")
    return k.zfill(n)


def counts_to_probvec(counts: dict[str, int], n: int) -> np.ndarray:
    """Full 2^n probability vector from a counts dict (missing outcomes are zero)."""
    p = np.zeros(2**n, dtype=float)
    total = 0
    for key, c in counts.items():
        p[int(normalize_key(key, n), 2)] += c
        total += c
    if total <= 0:
        raise ValueError("counts are empty")
    return p / total


def num_qubits_of(p: np.ndarray) -> int:
    n = int(round(np.log2(len(p))))
    if 2**n != len(p):
        raise ValueError(f"vector length {len(p)} is not a power of two")
    return n


@lru_cache(maxsize=None)
def parity_signs(n: int) -> np.ndarray:
    """s[i] = (-1)^popcount(i)."""
    idx = np.arange(2**n)
    popcount = np.array([bin(i).count("1") for i in idx])
    s = (1 - 2 * (popcount % 2)).astype(float)
    s.setflags(write=False)
    return s


@lru_cache(maxsize=None)
def z_signs(n: int, q: int) -> np.ndarray:
    """(-1)^((i >> q) & 1): the eigenvalue of Z_q on basis state i."""
    if not 0 <= q < n:
        raise ValueError(f"qubit {q} out of range for n={n}")
    idx = np.arange(2**n)
    s = (1 - 2 * ((idx >> q) & 1)).astype(float)
    s.setflags(write=False)
    return s


def parity_from_probs(p: np.ndarray) -> float:
    """<Z⊗...⊗Z> from a (quasi-)probability vector. Linear, so valid for quasi-probs."""
    p = np.asarray(p, dtype=float)
    return float(parity_signs(num_qubits_of(p)) @ p)


def z_expectations(p: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.array([z_signs(n, q) @ p for q in range(n)])


def magnetization_from_probs(p: np.ndarray, n: int) -> float:
    """(1/n) * sum_q <Z_q>."""
    return float(np.mean(z_expectations(p, n)))


def success_probability(p: np.ndarray, target_index: int) -> float:
    """p[target_index]: the (quasi-)probability of the most likely ideal bitstring."""
    return float(np.asarray(p, dtype=float)[target_index])


def exact_parity(unitary: QuantumCircuit) -> float:
    return parity_from_probs(Statevector(unitary).probabilities())


def exact_reference(unitary: QuantumCircuit) -> dict:
    """Exact (infinite-shot) parity, magnetization, distribution and success target.

    Returns a dict with keys: E_exact, M_exact, probs, target_index, P_succ_exact.
    target_index = argmax(probs) is the most likely ideal bitstring; P_succ_exact is its
    exact probability.
    """
    n = unitary.num_qubits
    sv = Statevector(unitary)
    probs = sv.probabilities()
    E_exact = parity_from_probs(probs)
    E_pauli = float(sv.expectation_value(SparsePauliOp("Z" * n)).real)
    if abs(E_exact - E_pauli) > 1e-9:
        raise AssertionError(f"parity cross-check failed: {E_exact} vs {E_pauli}")
    M_exact = magnetization_from_probs(probs, n)
    target_index = int(np.argmax(probs))
    return {
        "E_exact": E_exact,
        "M_exact": M_exact,
        "probs": probs,
        "target_index": target_index,
        "P_succ_exact": float(probs[target_index]),
    }
