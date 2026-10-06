import copy

import pytest
import yaml

from qem.config import config_from_dict, load_config
from qem.seeds import derive_seed


def test_derive_seed_is_deterministic():
    assert derive_seed("circuit", 4, 2, 0) == derive_seed("circuit", 4, 2, 0)
    assert derive_seed("sim", 2, 4, "low", 3, "base") == derive_seed("sim", 2, 4, "low", 3, "base")


def test_different_parts_give_different_seeds():
    seeds = {
        derive_seed("circuit", 4, 2, 0),
        derive_seed("circuit", 4, 2, 1),
        derive_seed("circuit", 2, 4, 0),
        derive_seed("sim", 4, 2, "low", 0, "base"),
        derive_seed("sim", 4, 2, "low", 0, "fold3"),
        derive_seed("sim", 4, 2, "low", 0, "fold5"),
        derive_seed("sim", 4, 2, "low", 0, "cal"),
    }
    assert len(seeds) == 7


def test_seed_range():
    for i in range(2000):
        s = derive_seed("range-check", i)
        assert isinstance(s, int)
        assert 0 <= s < 2**31 - 1


# --- config loading and validation -------------------------------------------------


@pytest.fixture()
def raw(root):
    with open(root / "config" / "experiment.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_experiment_config_loads(cfg):
    assert cfg.experiment.qubits == (2, 4, 6)
    assert cfg.experiment.depths == (2, 4)
    assert cfg.experiment.noise_levels == ("ideal", "low", "moderate")
    assert cfg.experiment.seeds == (0, 1, 2, 3, 4)
    assert cfg.experiment.shots == 1024
    assert cfg.experiment.methods == ("none", "rem", "zne", "zne_rem")
    assert cfg.zne.scale_factors == (1, 3, 5)
    assert cfg.noise.level("ideal").is_ideal
    mod = cfg.noise.level("moderate")
    assert (mod.p1, mod.p2, mod.p_ro) == (0.005, 0.03, 0.03)


def test_smoke_config_loads(smoke_cfg, cfg):
    assert smoke_cfg.experiment.qubits == (2,)
    assert smoke_cfg.experiment.depths == (2,)
    assert smoke_cfg.experiment.seeds == (0,)
    # identical to the full config everywhere else
    assert smoke_cfg.noise == cfg.noise
    assert smoke_cfg.zne == cfg.zne
    assert smoke_cfg.rem == cfg.rem
    assert smoke_cfg.circuit == cfg.circuit
    assert smoke_cfg.experiment.shots == cfg.experiment.shots


def test_config_is_frozen(cfg):
    with pytest.raises(Exception):
        cfg.experiment.shots = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["experiment"].update(qubits=[1, 2]), "qubits"),
        (lambda r: r["experiment"].update(depths=[0]), "depth"),
        (lambda r: r["noise"]["levels"]["low"].update(p2=0.5), "p2"),
        (lambda r: r["noise"]["levels"]["low"].update(p_ro=-0.1), "p_ro"),
        (lambda r: r["zne"].update(scale_factors=[1, 2, 5]), "odd"),
        (lambda r: r["zne"].update(scale_factors=[3, 5]), r"scale_factors\[0\]"),
        (lambda r: r["experiment"].update(shots=0), "shots"),
        (lambda r: r["experiment"].update(methods=["none", "pec"]), "method"),
        (lambda r: r["experiment"].update(noise_levels=["ideal", "high"]), "high"),
        (lambda r: r["simulator"].update(optimization_level=1), "optimization_level"),
        (lambda r: r.pop("zne"), "zne"),
    ],
)
def test_invalid_configs_raise(raw, mutate, match):
    bad = copy.deepcopy(raw)
    mutate(bad)
    with pytest.raises(ValueError, match=match):
        config_from_dict(bad)


def test_load_config_from_path(root):
    cfg = load_config(root / "config" / "experiment.yaml")
    assert cfg.output_dir == "results"
