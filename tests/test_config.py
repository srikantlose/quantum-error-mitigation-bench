import copy

import pytest
import yaml

from qem.config import config_from_dict, load_config


@pytest.fixture()
def raw(root):
    with open(root / "config" / "experiment.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_experiment_config_loads(cfg):
    assert cfg.experiment.seeds == (0, 1, 2, 3, 4)
    assert cfg.experiment.shots == 1024

    assert cfg.track_a.qubits == (2, 4, 6)
    assert cfg.track_a.depths == (2, 4)
    assert cfg.track_a.noise_levels == ("ideal", "low", "moderate", "high")
    assert cfg.track_a.methods == ("none", "rem", "rem_tensored", "zne", "zne_rem")

    assert cfg.track_b.qubits == (2, 4)
    assert cfg.track_b.depths == (2, 4)
    assert cfg.track_b.methods == ("none", "rem", "zne", "zne_rem")
    assert cfg.track_b.test_size == pytest.approx(0.2)
    assert cfg.track_b.classes == {"positive": 1, "negative": 2}

    assert cfg.zne.scale_factors == (1, 3, 5)
    assert cfg.noise.level("ideal").is_ideal
    low = cfg.noise.level("low")
    assert (low.p1, low.p2, low.p_ro) == (0.001, 0.005, 0.01)
    mod = cfg.noise.level("moderate")
    assert (mod.p1, mod.p2, mod.p_ro) == (0.01, 0.02, 0.03)
    high = cfg.noise.level("high")
    assert (high.p1, high.p2, high.p_ro) == (0.03, 0.05, 0.05)
    assert cfg.noise.headline_levels == ("low", "moderate")

    assert cfg.improvement.shot_allocation.total_shots == 3072
    assert cfg.improvement.shot_allocation.repetitions == 50


def test_smoke_config_loads(smoke_cfg, cfg):
    assert smoke_cfg.track_a.qubits == (2,)
    assert smoke_cfg.track_a.depths == (2,)
    assert smoke_cfg.experiment.seeds == (0,)
    assert smoke_cfg.improvement.shot_allocation.repetitions == 3
    # identical to the full config everywhere else
    assert smoke_cfg.noise == cfg.noise
    assert smoke_cfg.zne == cfg.zne
    assert smoke_cfg.rem == cfg.rem
    assert smoke_cfg.track_a.circuit == cfg.track_a.circuit
    assert smoke_cfg.track_b.training.optimizer == cfg.track_b.training.optimizer
    assert smoke_cfg.experiment.shots == cfg.experiment.shots


def test_config_is_frozen(cfg):
    with pytest.raises(Exception):
        cfg.experiment.shots = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["track_a"].update(qubits=[1, 2]), "qubits"),
        (lambda r: r["track_a"].update(depths=[0]), "depth"),
        (lambda r: r["noise"]["levels"]["low"].update(p2=0.5), "p2"),
        (lambda r: r["noise"]["levels"]["low"].update(p_ro=-0.1), "p_ro"),
        (lambda r: r["zne"].update(scale_factors=[1, 2, 5]), "odd"),
        (lambda r: r["zne"].update(scale_factors=[3, 5]), r"scale_factors\[0\]"),
        (lambda r: r["experiment"].update(shots=0), "shots"),
        (lambda r: r["track_a"].update(methods=["none", "pec"]), "method"),
        (lambda r: r["track_b"].update(methods=["none", "rem_tensored"]), "method"),
        (lambda r: r["track_a"].update(noise_levels=["ideal", "extreme"]), "extreme"),
        (lambda r: r["track_b"].update(test_size=1.2), "test_size"),
        (lambda r: r["simulator"].update(optimization_level=1), "optimization_level"),
        (lambda r: r.pop("zne"), "zne"),
        (lambda r: r.pop("track_b"), "track_b"),
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
