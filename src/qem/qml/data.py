"""Load, split and preprocess the Iris binary task (versicolor vs. virginica).

All scalers and PCA are fitted on the training split only; the test split is transformed
with those fitted parameters and clipped, never refit. This is checked by
``tests/test_qml_data.py`` by refitting on train+test and asserting the parameters differ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..config import TrackBConfig
from ..seeds import derive_seed


def load_iris_binary(cfg: TrackBConfig) -> tuple[np.ndarray, np.ndarray]:
    """The 100-sample versicolor/virginica subset of Iris, with y in {+1, -1}.

    versicolor (sklearn target 1) -> +1; virginica (sklearn target 2) -> -1. setosa is
    excluded: it is linearly separable from the other two, which makes the task trivial.
    """
    data = load_iris()
    mask = (data.target == cfg.classes["positive"]) | (data.target == cfg.classes["negative"])
    X = data.data[mask]
    y = np.where(data.target[mask] == cfg.classes["positive"], 1, -1)
    return X, y


@dataclass
class DataSplit:
    seed: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_split(X: np.ndarray, y: np.ndarray, seed: int, test_size: float) -> DataSplit:
    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, stratify=y, random_state=derive_seed("qml_split", seed)
    )
    return DataSplit(seed, X[idx_train], X[idx_test], y_train, y_test, idx_train, idx_test)


def save_split(split: DataSplit, directory: str | Path) -> Path:
    path = Path(directory) / f"s{split.seed}.json"
    record = {
        "seed": split.seed,
        "train_idx": split.train_idx.tolist(),
        "test_idx": split.test_idx.tolist(),
        "y_train": split.y_train.tolist(),
        "y_test": split.y_test.tolist(),
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


@dataclass
class PreprocessedData:
    n: int
    seed: int
    X_train: np.ndarray  # (n_train, n), in feature_range
    X_test: np.ndarray  # (n_test, n), in feature_range, clipped
    y_train: np.ndarray
    y_test: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_components: np.ndarray | None
    pca_explained_variance_ratio: np.ndarray | None
    minmax_data_min: np.ndarray
    minmax_data_max: np.ndarray


def preprocess(split: DataSplit, n: int, feature_range: tuple[float, float]) -> PreprocessedData:
    """n == 4: StandardScaler then MinMaxScaler. n == 2: StandardScaler, PCA(2), MinMaxScaler.

    Every step is fitted on ``split.X_train`` only; ``split.X_test`` is only ever
    transformed, and its output is clipped to ``feature_range`` since PCA/scaling can push
    a test point slightly outside the training range.
    """
    scaler = StandardScaler().fit(split.X_train)
    train = scaler.transform(split.X_train)
    test = scaler.transform(split.X_test)

    pca = None
    if n == 2:
        pca = PCA(n_components=2, random_state=derive_seed("qml_pca", split.seed)).fit(train)
        train = pca.transform(train)
        test = pca.transform(test)
    elif n != split.X_train.shape[1]:
        raise ValueError(f"n={n} must be 2 (PCA) or {split.X_train.shape[1]} (no PCA)")

    mm = MinMaxScaler(feature_range=feature_range).fit(train)
    train = mm.transform(train)
    test = np.clip(mm.transform(test), feature_range[0], feature_range[1])

    return PreprocessedData(
        n=n, seed=split.seed, X_train=train, X_test=test, y_train=split.y_train, y_test=split.y_test,
        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
        pca_components=None if pca is None else pca.components_,
        pca_explained_variance_ratio=None if pca is None else pca.explained_variance_ratio_,
        minmax_data_min=mm.data_min_, minmax_data_max=mm.data_max_,
    )


def save_preprocessing(prep: PreprocessedData, directory: str | Path) -> Path:
    path = Path(directory) / f"n{prep.n}_s{prep.seed}.json"
    record = {
        "n": prep.n, "seed": prep.seed,
        "scaler_mean": prep.scaler_mean.tolist(), "scaler_scale": prep.scaler_scale.tolist(),
        "pca_components": None if prep.pca_components is None else prep.pca_components.tolist(),
        "pca_explained_variance_ratio": (
            None if prep.pca_explained_variance_ratio is None else prep.pca_explained_variance_ratio.tolist()
        ),
        "minmax_data_min": prep.minmax_data_min.tolist(), "minmax_data_max": prep.minmax_data_max.tolist(),
    }
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path
