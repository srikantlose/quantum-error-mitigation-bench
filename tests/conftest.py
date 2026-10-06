from pathlib import Path

import pytest

from qem.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config" / "experiment.yaml")


@pytest.fixture(scope="session")
def smoke_cfg():
    return load_config(ROOT / "config" / "smoke.yaml")
