import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from qem.qml.data import load_iris_binary, make_split, preprocess
from qem.qml.vqc import (
    bind_sample,
    build_vqc_circuit,
    n_params,
    predict,
    train_or_load,
    train_vqc,
)

PARAM_TABLE = {  # (n, L) -> trainable params
    (2, 2): 8, (2, 4): 16, (4, 2): 16, (4, 4): 32,
}


@pytest.mark.parametrize("n, L", list(PARAM_TABLE))
def test_param_counts_match_table(n, L):
    qc, x, w = build_vqc_circuit(n, L)
    assert len(w) == PARAM_TABLE[(n, L)]
    assert n_params(n, L) == PARAM_TABLE[(n, L)]
    assert len(x) == n
    assert len(qc.parameters) == n + PARAM_TABLE[(n, L)]
    assert set(qc.count_ops()) <= {"ry", "rz", "cx"}


def test_bound_circuit_matches_statevector_estimator(cfg):
    from qiskit.primitives import StatevectorEstimator

    from qem.qml.vqc import batch_expectations

    n, L = 4, 2
    qc, x, w = build_vqc_circuit(n, L)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(5, n))
    weights = rng.uniform(0, 2 * np.pi, size=n_params(n, L))

    via_estimator = batch_expectations(qc, x, w, X, weights, StatevectorEstimator())
    obs = SparsePauliOp("Z" * n)
    via_statevector = np.array([
        Statevector(bind_sample(qc, x, w, xv, weights)).expectation_value(obs).real for xv in X
    ])
    assert np.allclose(via_estimator, via_statevector, atol=1e-9)

    # the fallback path (no estimator) must agree too
    via_fallback = batch_expectations(qc, x, w, X, weights, estimator=None)
    assert np.allclose(via_fallback, via_statevector, atol=1e-9)


def test_predict_tie_goes_to_plus_one():
    assert list(predict(np.array([0.5, -0.1, 0.0, -1e-9]))) == [1, -1, 1, -1]


def test_training_lowers_loss_and_same_seed_reproducible(cfg):
    X, y = load_iris_binary(cfg.track_b)
    split = make_split(X, y, seed=0, test_size=cfg.track_b.test_size)
    prep = preprocess(split, 2, cfg.track_b.feature_range)

    model_a = train_vqc(prep, 2, cfg.track_b.training)
    assert model_a.loss_history[-1] <= model_a.loss_history[0]
    assert 0.0 <= model_a.train_accuracy_exact <= 1.0

    model_b = train_vqc(prep, 2, cfg.track_b.training)
    assert np.allclose(model_a.weights, model_b.weights)
    assert model_a.loss_history == pytest.approx(model_b.loss_history)


def test_model_cache_reuses_matching_config(cfg, tmp_path):
    X, y = load_iris_binary(cfg.track_b)
    split = make_split(X, y, seed=0, test_size=cfg.track_b.test_size)
    prep = preprocess(split, 2, cfg.track_b.feature_range)

    m1 = train_or_load(prep, 2, cfg.track_b.training, tmp_path)
    calls = []
    import qem.qml.vqc as vqc_mod

    orig = vqc_mod.train_vqc

    def spy(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    vqc_mod.train_vqc = spy
    try:
        m2 = train_or_load(prep, 2, cfg.track_b.training, tmp_path)
        assert not calls  # cache hit: train_vqc was not called again
        assert np.allclose(m1.weights, m2.weights)
        m3 = train_or_load(prep, 2, cfg.track_b.training, tmp_path, retrain=True)
        assert calls  # --retrain forces a fresh call
    finally:
        vqc_mod.train_vqc = orig
