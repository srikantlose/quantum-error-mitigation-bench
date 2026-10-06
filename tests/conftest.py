from dataclasses import replace
from pathlib import Path

import pytest

from qem.config import NoiseLevelConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def cfg_with_levels(cfg, **levels):
    """Copy of cfg with extra noise levels, e.g. ro_only=(0, 0, 0.03) as (p1, p2, p_ro)."""
    extra = tuple(NoiseLevelConfig(name, *ps) for name, ps in levels.items())
    return replace(cfg, noise=replace(cfg.noise, levels=cfg.noise.levels + extra))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "config" / "experiment.yaml")


@pytest.fixture(scope="session")
def smoke_cfg():
    return load_config(ROOT / "config" / "smoke.yaml")
