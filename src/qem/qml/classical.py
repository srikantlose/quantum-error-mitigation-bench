"""Classical baselines: Logistic Regression and SVM, on the exact same preprocessed
features and split as the VQC. No hyperparameter tuning is done on the test set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ..config import TrackBClassicalConfig
from ..seeds import derive_seed
from .data import PreprocessedData


@dataclass
class ClassicalResult:
    model: str
    n_qubits: int
    seed: int
    y_pred: np.ndarray


def fit_predict_logreg(prep: PreprocessedData, cfg: TrackBClassicalConfig) -> ClassicalResult:
    lr = cfg.logistic_regression
    model = LogisticRegression(
        C=lr.C, max_iter=lr.max_iter, random_state=derive_seed("qml_lr", prep.n, prep.seed)
    )
    model.fit(prep.X_train, prep.y_train)
    return ClassicalResult("logreg", prep.n, prep.seed, model.predict(prep.X_test))


def fit_predict_svm(prep: PreprocessedData, cfg: TrackBClassicalConfig) -> ClassicalResult:
    svm = cfg.svm
    model = SVC(
        kernel=svm.kernel, C=svm.C, gamma=svm.gamma, random_state=derive_seed("qml_svm", prep.n, prep.seed)
    )
    model.fit(prep.X_train, prep.y_train)
    return ClassicalResult("svm", prep.n, prep.seed, model.predict(prep.X_test))
