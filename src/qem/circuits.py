"""Ansatz construction, measurement, instance selection and circuit statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm2

from .observables import exact_parity
from .seeds import derive_seed

TWO_PI = 2 * np.pi
_NOT_GATES = ("barrier", "measure")


def _is_gate(inst) -> bool:
    return inst.operation.name not in _NOT_GATES


def build_unitary_from_angles(angles: np.ndarray) -> QuantumCircuit:
    """Layered ry-rz-cx ansatz from an angles array of shape (L, 2, n).

    angles[l, 0, q] is the ry angle and angles[l, 1, q] the rz angle on qubit q in layer l.
    """
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 3 or angles.shape[1] != 2:
        raise ValueError(f"angles must have shape (L, 2, n), got {angles.shape}")
    layers, _, n = angles.shape
    qc = QuantumCircuit(n)
    for layer in range(layers):
        for q in range(n):
            qc.ry(float(angles[layer, 0, q]), q)
        for q in range(n):
            qc.rz(float(angles[layer, 1, q]), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def build_unitary(
    n: int,
    layers: int,
    rng: np.random.Generator,
    angle_low: float = 0.0,
    angle_high: float = TWO_PI,
) -> tuple[QuantumCircuit, np.ndarray]:
    """Draw angles and build the ansatz. No classical bits, no measurements.

    Drawing a C-ordered (L, 2, n) block reproduces the fixed draw order of §7.1: all theta
    for layer 0, then all phi for layer 0, then layer 1, and so on.
    """
    angles = rng.uniform(angle_low, angle_high, size=(layers, 2, n))
    return build_unitary_from_angles(angles), angles


def with_measurements(unitary: QuantumCircuit) -> QuantumCircuit:
    """Measure every qubit q into classical bit q of a single register (no measure_all)."""
    n = unitary.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(unitary, inplace=True)
    qc.barrier()
    qc.measure(range(n), range(n))
    return qc


def fold_global(u: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Global unitary folding U_lambda = U (U^dagger U)^k for odd scale lambda = 2k + 1.

    Barriers separate the folds; the result must never be transpiled at optimization
    level > 0, which would cancel the U^dagger U pairs. Returns a unitary-only circuit.
    """
    if not (isinstance(scale, int) and scale >= 1 and scale % 2 == 1):
        raise ValueError(f"scale must be an odd positive integer, got {scale!r}")
    k = (scale - 1) // 2
    folded = u.copy()
    for _ in range(k):
        folded.barrier()
        folded.compose(u.inverse(), inplace=True)
        folded.barrier()
        folded.compose(u, inplace=True)
    return folded


def calibration_circuits(n: int) -> list[QuantumCircuit]:
    """2^n readout-calibration circuits; circuit j prepares the basis state |j>.

    ry(pi) is used instead of x so the preparation stays inside the noisy gate basis.
    """
    circuits = []
    for j in range(2**n):
        qc = QuantumCircuit(n, n, name=f"cal_{j}")
        for q in range(n):
            if (j >> q) & 1:
                qc.ry(np.pi, q)
        qc.barrier()
        qc.measure(range(n), range(n))
        circuits.append(qc)
    return circuits


def circuit_stats(qc: QuantumCircuit) -> dict:
    """Depth and gate counts, excluding barriers and measurements."""
    gates = [inst for inst in qc.data if _is_gate(inst)]
    return {
        "depth": qc.depth(filter_function=_is_gate),
        "cx": sum(1 for inst in gates if inst.operation.name == "cx"),
        "n1q": sum(1 for inst in gates if inst.operation.num_qubits == 1),
        "total_gates": len(gates),
    }


@dataclass
class CircuitInstance:
    n: int
    layers: int
    seed: int
    unitary: QuantumCircuit
    angles: np.ndarray
    attempts: int
    E_exact: float  # exact parity expectation


def select_instance(
    n: int,
    layers: int,
    seed: int,
    threshold: float,
    max_attempts: int,
    angle_low: float = 0.0,
    angle_high: float = TWO_PI,
) -> CircuitInstance:
    """Rejection-sample angles until the exact parity satisfies |E_exact| >= threshold."""
    rng = np.random.default_rng(derive_seed("circuit", n, layers, seed))
    for attempt in range(1, max_attempts + 1):
        unitary, angles = build_unitary(n, layers, rng, angle_low, angle_high)
        E = exact_parity(unitary)
        if abs(E) >= threshold:
            return CircuitInstance(n, layers, seed, unitary, angles, attempt, E)
    raise RuntimeError(
        f"no instance with |E_exact| >= {threshold} found for n={n}, L={layers}, "
        f"seed={seed} within {max_attempts} attempts"
    )


def instance_filename(n: int, layers: int, seed: int) -> str:
    return f"n{n}_L{layers}_s{seed}.json"


def save_instance(inst: CircuitInstance, directory: str | Path) -> Path:
    path = Path(directory) / instance_filename(inst.n, inst.layers, inst.seed)
    record = {
        "n": inst.n,
        "layers": inst.layers,
        "seed": inst.seed,
        "attempts": inst.attempts,
        "E_exact": inst.E_exact,
        "angles": inst.angles.tolist(),
        "qasm": qasm2.dumps(with_measurements(inst.unitary)),
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def load_instance(path: str | Path) -> CircuitInstance:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    angles = np.asarray(record["angles"], dtype=float)
    return CircuitInstance(
        n=record["n"],
        layers=record["layers"],
        seed=record["seed"],
        unitary=build_unitary_from_angles(angles),
        angles=angles,
        attempts=record["attempts"],
        E_exact=record["E_exact"],
    )
