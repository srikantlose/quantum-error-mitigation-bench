"""Execution layer: transpile at optimization level 0, guard the gate set, run on Aer, time it."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import numpy as np
import qiskit_aer  # noqa: F401  (registers save_density_matrix on QuantumCircuit)
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator

from .config import Config
from .noise import build_noise_model, true_assignment_matrix
from .observables import normalize_key
from .seeds import derive_seed

NON_GATE_OPS = frozenset({"measure", "barrier"})


def assert_noise_basis(circuits: Iterable[QuantumCircuit], allowed: frozenset[str]) -> None:
    """Fail if any circuit contains an op the noise model does not cover.

    Noise is attached by gate name, so an op outside the basis would run noise-free.
    """
    for i, qc in enumerate(circuits):
        bad = {inst.operation.name for inst in qc.data} - allowed
        if bad:
            raise AssertionError(
                f"circuit {i} ({qc.name}) contains ops outside the noise basis: {sorted(bad)}"
            )


def normalize_counts(counts: dict[str, int], n: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, c in counts.items():
        k = normalize_key(key, n)
        out[k] = out.get(k, 0) + int(c)
    return dict(sorted(out.items()))


class Executor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.basis = list(cfg.simulator.transpile_basis)
        self.allowed_ops = frozenset(self.basis) | NON_GATE_OPS
        self._levels = {lvl.name: lvl for lvl in cfg.noise.levels}
        self._sims: dict[str, AerSimulator] = {}
        for name, lvl in self._levels.items():
            nm = build_noise_model(lvl, cfg.noise.one_qubit_gates, cfg.noise.two_qubit_gates)
            if nm is None:
                self._sims[name] = AerSimulator(method=cfg.simulator.method)
            else:
                self._sims[name] = AerSimulator(method=cfg.simulator.method, noise_model=nm)

    def simulator(self, noise_level: str) -> AerSimulator:
        if noise_level not in self._sims:
            raise KeyError(f"unknown noise level {noise_level!r}")
        return self._sims[noise_level]

    def transpile_circuits(
        self, circuits: list[QuantumCircuit], transpile_seed: int
    ) -> list[QuantumCircuit]:
        # Checked before transpiling as well: every circuit we build is already in the basis,
        # and a rewrite by the transpiler would silently change the gate counts we report.
        assert_noise_basis(circuits, self.allowed_ops)
        tcircs = transpile(
            list(circuits),
            basis_gates=self.basis,
            optimization_level=self.cfg.simulator.optimization_level,
            seed_transpiler=transpile_seed,
        )
        assert_noise_basis(tcircs, self.allowed_ops)
        return tcircs

    def run(
        self,
        circuits: list[QuantumCircuit],
        noise_level: str,
        shots: int,
        seed: int,
        transpile_seed: int | None = None,
    ) -> tuple[list[dict[str, int]], float]:
        if transpile_seed is None:
            transpile_seed = derive_seed("transpile", seed)
        tcircs = self.transpile_circuits(circuits, transpile_seed)
        sim = self.simulator(noise_level)
        t0 = perf_counter()
        result = sim.run(tcircs, shots=shots, seed_simulator=seed).result()
        counts = [normalize_counts(result.get_counts(i), tc.num_clbits) for i, tc in enumerate(tcircs)]
        return counts, perf_counter() - t0

    def exact_noisy_probs(self, unitary: QuantumCircuit, noise_level: str) -> np.ndarray:
        """Infinite-shot outcome distribution under the noise model, readout included.

        save_density_matrix captures gate noise only, so the true assignment matrix is
        applied afterwards.
        """
        assert_noise_basis([unitary], self.allowed_ops)
        qc = unitary.copy()
        qc.save_density_matrix()
        result = self.simulator(noise_level).run(qc).result()
        p_gate = DensityMatrix(result.data(0)["density_matrix"]).probabilities()
        p_ro = self._levels[noise_level].p_ro
        if p_ro > 0:
            return true_assignment_matrix(p_ro, unitary.num_qubits) @ p_gate
        return p_gate
