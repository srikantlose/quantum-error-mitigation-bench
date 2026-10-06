"""The variational quantum classifier: angle encoding + the Track A ansatz, trained
noiseless and exact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from ..config import TrackBTrainingConfig
from ..seeds import derive_seed
from .data import PreprocessedData


def build_vqc_circuit(n: int, layers: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Angle encoding ry(x_k) on qubit k, then the Track A ry-rz-cx ansatz with L layers
    of trainable weights. No measurements: this is the unitary, ready for binding and
    then either a statevector evaluation or ``with_measurements`` for noisy execution.
    """
    x = ParameterVector("x", n)
    w = ParameterVector("w", 2 * n * layers)
    qc = QuantumCircuit(n)
    for k in range(n):
        qc.ry(x[k], k)
    idx = 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(w[idx], q)
            idx += 1
        for q in range(n):
            qc.rz(w[idx], q)
            idx += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc, x, w


def n_params(n: int, layers: int) -> int:
    return 2 * n * layers


def _param_order_indices(qc: QuantumCircuit, x: ParameterVector, w: ParameterVector) -> tuple[list[int], list[int]]:
    """Positions of the x and w parameters within ``qc.parameters`` (name-sorted, so this
    must never be assumed to be [w..., x...] without checking)."""
    order = list(qc.parameters)
    x_pos = [order.index(p) for p in x]
    w_pos = [order.index(p) for p in w]
    assert sorted(x_pos + w_pos) == list(range(len(order))), "parameter positions must partition qc.parameters"
    return x_pos, w_pos


def bind_sample(qc: QuantumCircuit, x: ParameterVector, w: ParameterVector,
                x_values: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
    """Bind one sample's features and the shared weights, by name (order-independent)."""
    mapping = {p: float(v) for p, v in zip(x, x_values)}
    mapping.update({p: float(v) for p, v in zip(w, weights)})
    return qc.assign_parameters(mapping)


def batch_expectations(qc: QuantumCircuit, x: ParameterVector, w: ParameterVector,
                       X: np.ndarray, weights: np.ndarray,
                       estimator: StatevectorEstimator | None = None) -> np.ndarray:
    """Exact <Z^n> for every row of X (shape (N, n)) with the shared weights, via one
    batched StatevectorEstimator PUB call. Falls back to a per-sample Statevector loop if
    the estimator primitive is unavailable.
    """
    n = qc.num_qubits
    obs = SparsePauliOp("Z" * n)
    X = np.asarray(X, dtype=float)
    if estimator is not None:
        x_pos, w_pos = _param_order_indices(qc, x, w)
        values = np.empty((len(X), len(qc.parameters)))
        values[:, w_pos] = np.asarray(weights, dtype=float)
        values[:, x_pos] = X
        result = estimator.run([(qc, obs, values)]).result()
        return np.asarray(result[0].data.evs, dtype=float)
    from qiskit.quantum_info import Statevector  # fallback path

    out = np.empty(len(X))
    for i, xv in enumerate(X):
        bound = bind_sample(qc, x, w, xv, weights)
        out[i] = Statevector(bound).expectation_value(obs).real
    return out


def predict(E: np.ndarray) -> np.ndarray:
    """sign(E) with ties (E == 0) going to +1."""
    return np.where(E >= 0, 1, -1)


@dataclass
class VqcModel:
    n: int
    layers: int
    seed: int
    weights: np.ndarray
    loss_history: list[float]
    n_iterations: int
    train_accuracy_exact: float
    train_time_s: float
    config_hash: str


def _config_hash(n: int, layers: int, seed: int, training: TrackBTrainingConfig, n_train: int) -> str:
    key = f"{n}|{layers}|{seed}|{training.optimizer}|{training.min_maxiter}|{training.maxiter_per_param}|{n_train}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def train_vqc(prep: PreprocessedData, layers: int, training: TrackBTrainingConfig) -> VqcModel:
    """Noiseless, exact training: COBYLA on the MSE of <Z^n> against y in {+1, -1}.

    Training is deliberately noiseless, to isolate the effect of noise and mitigation at
    inference time; noise-aware training is listed as future work.
    """
    n = prep.n
    qc, x, w = build_vqc_circuit(n, layers)
    estimator = StatevectorEstimator()
    rng = np.random.default_rng(derive_seed("qml_init", n, layers, prep.seed))
    w0 = rng.uniform(0.0, 2 * np.pi, size=n_params(n, layers))

    history: list[float] = []

    def objective(weights):
        E = batch_expectations(qc, x, w, prep.X_train, weights, estimator)
        loss = float(np.mean((E - prep.y_train) ** 2))
        history.append(loss)
        return loss

    maxiter = max(training.min_maxiter, training.maxiter_per_param * len(w0))
    t0 = perf_counter()
    result = minimize(objective, w0, method=training.optimizer, options={"maxiter": maxiter})
    train_time = perf_counter() - t0

    E_train = batch_expectations(qc, x, w, prep.X_train, result.x, estimator)
    train_acc = float(np.mean(predict(E_train) == prep.y_train))

    return VqcModel(
        n=n, layers=layers, seed=prep.seed, weights=np.asarray(result.x, dtype=float),
        loss_history=history, n_iterations=len(history), train_accuracy_exact=train_acc,
        train_time_s=train_time, config_hash=_config_hash(n, layers, prep.seed, training, len(prep.y_train)),
    )


def model_path(directory: str | Path, n: int, layers: int, seed: int) -> Path:
    return Path(directory) / f"n{n}_L{layers}_s{seed}.json"


def save_model(model: VqcModel, directory: str | Path) -> Path:
    path = model_path(directory, model.n, model.layers, model.seed)
    record = {
        "n": model.n, "layers": model.layers, "seed": model.seed,
        "weights": model.weights.tolist(), "loss_history": model.loss_history,
        "n_iterations": model.n_iterations, "train_accuracy_exact": model.train_accuracy_exact,
        "train_time_s": model.train_time_s, "config_hash": model.config_hash,
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def load_model(path: str | Path) -> VqcModel:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return VqcModel(
        n=record["n"], layers=record["layers"], seed=record["seed"],
        weights=np.asarray(record["weights"], dtype=float), loss_history=record["loss_history"],
        n_iterations=record["n_iterations"], train_accuracy_exact=record["train_accuracy_exact"],
        train_time_s=record["train_time_s"], config_hash=record["config_hash"],
    )


def train_or_load(prep: PreprocessedData, layers: int, training: TrackBTrainingConfig,
                  directory: str | Path, retrain: bool = False) -> VqcModel:
    """Train, or reuse a cached model if one exists and its config hash still matches."""
    path = model_path(directory, prep.n, layers, prep.seed)
    expected_hash = _config_hash(prep.n, layers, prep.seed, training, len(prep.y_train))
    if not retrain and path.exists():
        cached = load_model(path)
        if cached.config_hash == expected_hash:
            return cached
    model = train_vqc(prep, layers, training)
    save_model(model, directory)
    return model
