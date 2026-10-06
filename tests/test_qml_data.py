import numpy as np
import pytest

from qem.qml.data import load_iris_binary, make_split, preprocess, save_preprocessing, save_split


def test_load_iris_binary_shape_and_labels(cfg):
    X, y = load_iris_binary(cfg.track_b)
    assert X.shape == (100, 4)
    assert set(np.unique(y)) == {1, -1}
    assert (y == 1).sum() == 50
    assert (y == -1).sum() == 50


def test_split_is_stratified_80_20(cfg):
    X, y = load_iris_binary(cfg.track_b)
    split = make_split(X, y, seed=0, test_size=cfg.track_b.test_size)
    assert len(split.y_train) == 80
    assert len(split.y_test) == 20
    assert (split.y_train == 1).sum() == 40
    assert (split.y_train == -1).sum() == 40
    assert (split.y_test == 1).sum() == 10
    assert (split.y_test == -1).sum() == 10
    assert set(split.train_idx) & set(split.test_idx) == set()
    assert len(set(split.train_idx) | set(split.test_idx)) == 100


def test_split_is_seed_deterministic(cfg):
    X, y = load_iris_binary(cfg.track_b)
    a = make_split(X, y, seed=1, test_size=cfg.track_b.test_size)
    b = make_split(X, y, seed=1, test_size=cfg.track_b.test_size)
    c = make_split(X, y, seed=2, test_size=cfg.track_b.test_size)
    assert np.array_equal(a.train_idx, b.train_idx)
    assert not np.array_equal(a.train_idx, c.train_idx)


@pytest.mark.parametrize("n", [2, 4])
def test_preprocessing_output_shape_and_range(cfg, n):
    X, y = load_iris_binary(cfg.track_b)
    split = make_split(X, y, seed=0, test_size=cfg.track_b.test_size)
    prep = preprocess(split, n, cfg.track_b.feature_range)
    lo, hi = cfg.track_b.feature_range
    assert prep.X_train.shape == (80, n)
    assert prep.X_test.shape == (20, n)
    assert prep.X_train.min() >= lo - 1e-9
    assert prep.X_train.max() <= hi + 1e-9
    # the test set is clipped, so it can never leave the range even though it is not fit on it
    assert prep.X_test.min() >= lo - 1e-9
    assert prep.X_test.max() <= hi + 1e-9
    if n == 2:
        assert prep.pca_components is not None
        assert prep.pca_components.shape == (2, 4)
    else:
        assert prep.pca_components is None


def test_preprocessing_fit_on_train_only_no_leakage(cfg):
    """Fitting the same preprocessing on train+test must give different parameters,
    which proves the train-only fit does not see the test rows."""
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris_binary(cfg.track_b)
    split = make_split(X, y, seed=0, test_size=cfg.track_b.test_size)
    prep = preprocess(split, 4, cfg.track_b.feature_range)

    leaked_scaler = StandardScaler().fit(np.concatenate([split.X_train, split.X_test]))
    assert not np.allclose(prep.scaler_mean, leaked_scaler.mean_)


def test_preprocessing_scaler_sees_only_training_rows(cfg):
    X, y = load_iris_binary(cfg.track_b)
    split = make_split(X, y, seed=0, test_size=cfg.track_b.test_size)
    prep = preprocess(split, 4, cfg.track_b.feature_range)
    assert np.allclose(prep.scaler_mean, split.X_train.mean(axis=0))


def test_save_split_and_preprocessing(cfg, tmp_path):
    import json

    X, y = load_iris_binary(cfg.track_b)
    split = make_split(X, y, seed=3, test_size=cfg.track_b.test_size)
    path = save_split(split, tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["seed"] == 3
    assert len(record["train_idx"]) == 80 and len(record["test_idx"]) == 20

    prep = preprocess(split, 2, cfg.track_b.feature_range)
    ppath = save_preprocessing(prep, tmp_path)
    precord = json.loads(ppath.read_text(encoding="utf-8"))
    assert len(precord["scaler_mean"]) == 4
    assert len(precord["pca_components"]) == 2
    assert len(precord["minmax_data_min"]) == 2
