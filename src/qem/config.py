"""Load the experiment YAML into frozen dataclasses and validate it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_METHODS = ("none", "rem", "zne", "zne_rem")
VALID_EXTRAPOLATORS = ("richardson", "linear", "exp")


@dataclass(frozen=True)
class NoiseLevelConfig:
    name: str
    p1: float
    p2: float
    p_ro: float

    @property
    def is_ideal(self) -> bool:
        return self.p1 == 0 and self.p2 == 0 and self.p_ro == 0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    qubits: tuple[int, ...]
    depths: tuple[int, ...]
    noise_levels: tuple[str, ...]
    seeds: tuple[int, ...]
    shots: int
    methods: tuple[str, ...]


@dataclass(frozen=True)
class CircuitConfig:
    type: str
    entanglement: str
    angle_low: float
    angle_high: float
    observable: str
    min_abs_exact_expectation: float
    max_resample_attempts: int


@dataclass(frozen=True)
class NoiseConfig:
    one_qubit_gates: tuple[str, ...]
    two_qubit_gates: tuple[str, ...]
    levels: tuple[NoiseLevelConfig, ...]

    def level(self, name: str) -> NoiseLevelConfig:
        for lvl in self.levels:
            if lvl.name == name:
                return lvl
        raise KeyError(f"unknown noise level {name!r}")


@dataclass(frozen=True)
class RemConfig:
    calibration: str
    calibration_shots: int
    cond_threshold: float


@dataclass(frozen=True)
class ZneConfig:
    folding: str
    scale_factors: tuple[int, ...]
    extrapolators: tuple[str, ...]
    primary_extrapolator: str


@dataclass(frozen=True)
class SimulatorConfig:
    method: str
    transpile_basis: tuple[str, ...]
    optimization_level: int


@dataclass(frozen=True)
class ExampleCondition:
    n: int
    depth: int
    noise: str
    seed: int


@dataclass(frozen=True)
class PlotsConfig:
    example_condition: ExampleCondition
    dpi: int


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    circuit: CircuitConfig
    noise: NoiseConfig
    rem: RemConfig
    zne: ZneConfig
    simulator: SimulatorConfig
    plots: PlotsConfig
    output_dir: str


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _section(raw: dict, key: str) -> dict:
    if key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"config is missing the '{key}' section")
    return raw[key]


def config_from_dict(raw: dict) -> Config:
    """Build a validated Config from a parsed YAML mapping. Raises ValueError on bad input."""
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    try:
        e = _section(raw, "experiment")
        experiment = ExperimentConfig(
            name=str(e["name"]),
            qubits=tuple(e["qubits"]),
            depths=tuple(e["depths"]),
            noise_levels=tuple(e["noise_levels"]),
            seeds=tuple(e["seeds"]),
            shots=e["shots"],
            methods=tuple(e["methods"]),
        )
        c = _section(raw, "circuit")
        circuit = CircuitConfig(
            type=str(c["type"]),
            entanglement=str(c["entanglement"]),
            angle_low=float(c["angle_low"]),
            angle_high=float(c["angle_high"]),
            observable=str(c["observable"]),
            min_abs_exact_expectation=float(c["min_abs_exact_expectation"]),
            max_resample_attempts=c["max_resample_attempts"],
        )
        nz = _section(raw, "noise")
        levels = tuple(
            NoiseLevelConfig(name=str(k), p1=float(v["p1"]), p2=float(v["p2"]), p_ro=float(v["p_ro"]))
            for k, v in nz["levels"].items()
        )
        noise = NoiseConfig(
            one_qubit_gates=tuple(nz["one_qubit_gates"]),
            two_qubit_gates=tuple(nz["two_qubit_gates"]),
            levels=levels,
        )
        r = _section(raw, "rem")
        rem = RemConfig(
            calibration=str(r["calibration"]),
            calibration_shots=r["calibration_shots"],
            cond_threshold=float(r["cond_threshold"]),
        )
        z = _section(raw, "zne")
        zne = ZneConfig(
            folding=str(z["folding"]),
            scale_factors=tuple(z["scale_factors"]),
            extrapolators=tuple(z["extrapolators"]),
            primary_extrapolator=str(z["primary_extrapolator"]),
        )
        s = _section(raw, "simulator")
        simulator = SimulatorConfig(
            method=str(s["method"]),
            transpile_basis=tuple(s["transpile_basis"]),
            optimization_level=s["optimization_level"],
        )
        p = _section(raw, "plots")
        ex = p["example_condition"]
        plots = PlotsConfig(
            example_condition=ExampleCondition(
                n=ex["n"], depth=ex["depth"], noise=str(ex["noise"]), seed=ex["seed"]
            ),
            dpi=p["dpi"],
        )
        output_dir = str(_section(raw, "output")["dir"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed config: {exc!r}") from exc

    cfg = Config(experiment, circuit, noise, rem, zne, simulator, plots, output_dir)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    e = cfg.experiment
    if not e.qubits:
        raise ValueError("experiment.qubits must not be empty")
    for n in e.qubits:
        if not _is_int(n) or n < 2:
            raise ValueError(f"every qubits value must be an integer >= 2, got {n!r}")
    if not e.depths:
        raise ValueError("experiment.depths must not be empty")
    for d in e.depths:
        if not _is_int(d) or d < 1:
            raise ValueError(f"every depth must be an integer >= 1, got {d!r}")
    if not e.seeds:
        raise ValueError("experiment.seeds must not be empty")
    for sd in e.seeds:
        if not _is_int(sd) or sd < 0:
            raise ValueError(f"every seed must be a non-negative integer, got {sd!r}")
    if not _is_int(e.shots) or e.shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {e.shots!r}")
    if not e.methods:
        raise ValueError("experiment.methods must not be empty")
    for m in e.methods:
        if m not in VALID_METHODS:
            raise ValueError(f"unknown method {m!r}; valid methods are {VALID_METHODS}")

    level_names = {lvl.name for lvl in cfg.noise.levels}
    for name in e.noise_levels:
        if name not in level_names:
            raise ValueError(f"noise level {name!r} is not defined under noise.levels")
    for lvl in cfg.noise.levels:
        for label, p in (("p1", lvl.p1), ("p2", lvl.p2), ("p_ro", lvl.p_ro)):
            if not 0.0 <= p < 0.5:
                raise ValueError(f"noise.levels.{lvl.name}.{label} must lie in [0, 0.5), got {p}")

    sf = cfg.zne.scale_factors
    if not sf:
        raise ValueError("zne.scale_factors must not be empty")
    for s in sf:
        if not _is_int(s) or s < 1 or s % 2 != 1:
            raise ValueError(f"every scale factor must be an odd positive integer, got {s!r}")
    if sf[0] != 1:
        raise ValueError(f"zne.scale_factors[0] must be 1, got {sf[0]!r}")
    if len(set(sf)) != len(sf):
        raise ValueError("zne.scale_factors must be distinct")
    for x in cfg.zne.extrapolators:
        if x not in VALID_EXTRAPOLATORS:
            raise ValueError(f"unknown extrapolator {x!r}; valid: {VALID_EXTRAPOLATORS}")
    if cfg.zne.primary_extrapolator not in cfg.zne.extrapolators:
        raise ValueError("zne.primary_extrapolator must be one of zne.extrapolators")
    if cfg.zne.folding != "global":
        raise ValueError(f"only global folding is implemented, got {cfg.zne.folding!r}")

    if cfg.rem.calibration != "full":
        raise ValueError(f"only full REM calibration is implemented, got {cfg.rem.calibration!r}")
    if not _is_int(cfg.rem.calibration_shots) or cfg.rem.calibration_shots <= 0:
        raise ValueError("rem.calibration_shots must be a positive integer")
    if cfg.rem.cond_threshold <= 0:
        raise ValueError("rem.cond_threshold must be positive")

    c = cfg.circuit
    if not c.angle_low < c.angle_high:
        raise ValueError("circuit.angle_low must be smaller than circuit.angle_high")
    if not 0.0 <= c.min_abs_exact_expectation <= 1.0:
        raise ValueError("circuit.min_abs_exact_expectation must lie in [0, 1]")
    if not _is_int(c.max_resample_attempts) or c.max_resample_attempts < 1:
        raise ValueError("circuit.max_resample_attempts must be a positive integer")

    # Folded circuits must never be optimized: level > 0 cancels the U†U pairs.
    if cfg.simulator.optimization_level != 0:
        raise ValueError("simulator.optimization_level must be 0 (higher levels cancel ZNE folds)")
    if cfg.simulator.method != "density_matrix":
        raise ValueError("simulator.method must be 'density_matrix'")

    if not _is_int(cfg.plots.dpi) or cfg.plots.dpi <= 0:
        raise ValueError("plots.dpi must be a positive integer")


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return config_from_dict(raw)
