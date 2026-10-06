"""Load the experiment YAML into frozen dataclasses and validate it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

VALID_METHODS_A = ("none", "rem", "rem_tensored", "zne", "zne_rem")
VALID_METHODS_B = ("none", "rem", "zne", "zne_rem")
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
class NoiseConfig:
    one_qubit_gates: tuple[str, ...]
    two_qubit_gates: tuple[str, ...]
    levels: tuple[NoiseLevelConfig, ...]
    headline_levels: tuple[str, ...]

    def level(self, name: str) -> NoiseLevelConfig:
        for lvl in self.levels:
            if lvl.name == name:
                return lvl
        raise KeyError(f"unknown noise level {name!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    shots: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class TrackACircuitConfig:
    type: str
    entanglement: str
    angle_low: float
    angle_high: float
    observable: str
    min_abs_exact_expectation: float
    max_resample_attempts: int


@dataclass(frozen=True)
class TrackAConfig:
    enabled: bool
    qubits: tuple[int, ...]
    depths: tuple[int, ...]
    noise_levels: tuple[str, ...]
    methods: tuple[str, ...]
    circuit: TrackACircuitConfig


@dataclass(frozen=True)
class LogRegConfig:
    C: float
    max_iter: int


@dataclass(frozen=True)
class SvmConfig:
    kernel: str
    C: float
    gamma: str


@dataclass(frozen=True)
class TrackBTrainingConfig:
    optimizer: str
    min_maxiter: int
    maxiter_per_param: int


@dataclass(frozen=True)
class TrackBClassicalConfig:
    logistic_regression: LogRegConfig
    svm: SvmConfig


@dataclass(frozen=True)
class TrackBConfig:
    enabled: bool
    dataset: str
    classes: Mapping[str, int]
    test_size: float
    stratified: bool
    qubits: tuple[int, ...]
    depths: tuple[int, ...]
    noise_levels: tuple[str, ...]
    methods: tuple[str, ...]
    feature_range: tuple[float, float]
    observable: str
    training: TrackBTrainingConfig
    classical: TrackBClassicalConfig


@dataclass(frozen=True)
class RemConfig:
    calibration_shots: int
    cond_threshold: float


@dataclass(frozen=True)
class ZneConfig:
    folding: str
    scale_factors: tuple[int, ...]
    extrapolators: tuple[str, ...]
    primary_extrapolator: str


@dataclass(frozen=True)
class ShotAllocationConfig:
    enabled: bool
    noise_levels: tuple[str, ...]
    total_shots: int
    repetitions: int


@dataclass(frozen=True)
class ImprovementConfig:
    shot_allocation: ShotAllocationConfig


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
class QmlConfusionCondition:
    n: int
    depth: int


@dataclass(frozen=True)
class PlotsConfig:
    example_condition: ExampleCondition
    qml_confusion_condition: QmlConfusionCondition
    dpi: int


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    noise: NoiseConfig
    track_a: TrackAConfig
    track_b: TrackBConfig
    rem: RemConfig
    zne: ZneConfig
    improvement: ImprovementConfig
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
        experiment = ExperimentConfig(name=str(e["name"]), shots=e["shots"], seeds=tuple(e["seeds"]))

        nz = _section(raw, "noise")
        levels = tuple(
            NoiseLevelConfig(name=str(k), p1=float(v["p1"]), p2=float(v["p2"]), p_ro=float(v["p_ro"]))
            for k, v in nz["levels"].items()
        )
        noise = NoiseConfig(
            one_qubit_gates=tuple(nz["one_qubit_gates"]),
            two_qubit_gates=tuple(nz["two_qubit_gates"]),
            levels=levels,
            headline_levels=tuple(nz.get("headline_levels", ["low", "moderate"])),
        )

        ta = _section(raw, "track_a")
        tac = _section(ta, "circuit")
        track_a = TrackAConfig(
            enabled=bool(ta["enabled"]),
            qubits=tuple(ta["qubits"]),
            depths=tuple(ta["depths"]),
            noise_levels=tuple(ta["noise_levels"]),
            methods=tuple(ta["methods"]),
            circuit=TrackACircuitConfig(
                type=str(tac["type"]), entanglement=str(tac["entanglement"]),
                angle_low=float(tac["angle_low"]), angle_high=float(tac["angle_high"]),
                observable=str(tac["observable"]),
                min_abs_exact_expectation=float(tac["min_abs_exact_expectation"]),
                max_resample_attempts=tac["max_resample_attempts"],
            ),
        )

        tb = _section(raw, "track_b")
        tbt = _section(tb, "training")
        tbc = _section(tb, "classical")
        lr = _section(tbc, "logistic_regression")
        svm = _section(tbc, "svm")
        track_b = TrackBConfig(
            enabled=bool(tb["enabled"]),
            dataset=str(tb["dataset"]),
            classes=dict(tb["classes"]),
            test_size=float(tb["test_size"]),
            stratified=bool(tb["stratified"]),
            qubits=tuple(tb["qubits"]),
            depths=tuple(tb["depths"]),
            noise_levels=tuple(tb["noise_levels"]),
            methods=tuple(tb["methods"]),
            feature_range=tuple(float(x) for x in tb["feature_range"]),
            observable=str(tb["observable"]),
            training=TrackBTrainingConfig(
                optimizer=str(tbt["optimizer"]), min_maxiter=tbt["min_maxiter"],
                maxiter_per_param=tbt["maxiter_per_param"],
            ),
            classical=TrackBClassicalConfig(
                logistic_regression=LogRegConfig(C=float(lr["C"]), max_iter=lr["max_iter"]),
                svm=SvmConfig(kernel=str(svm["kernel"]), C=float(svm["C"]), gamma=str(svm["gamma"])),
            ),
        )

        r = _section(raw, "rem")
        rem = RemConfig(calibration_shots=r["calibration_shots"], cond_threshold=float(r["cond_threshold"]))

        z = _section(raw, "zne")
        zne = ZneConfig(
            folding=str(z["folding"]), scale_factors=tuple(z["scale_factors"]),
            extrapolators=tuple(z["extrapolators"]), primary_extrapolator=str(z["primary_extrapolator"]),
        )

        imp = _section(raw, "improvement")
        sa = _section(imp, "shot_allocation")
        improvement = ImprovementConfig(shot_allocation=ShotAllocationConfig(
            enabled=bool(sa["enabled"]), noise_levels=tuple(sa["noise_levels"]),
            total_shots=sa["total_shots"], repetitions=sa["repetitions"],
        ))

        s = _section(raw, "simulator")
        simulator = SimulatorConfig(
            method=str(s["method"]), transpile_basis=tuple(s["transpile_basis"]),
            optimization_level=s["optimization_level"],
        )

        p = _section(raw, "plots")
        ex = p["example_condition"]
        qc = p["qml_confusion_condition"]
        plots = PlotsConfig(
            example_condition=ExampleCondition(n=ex["n"], depth=ex["depth"], noise=str(ex["noise"]), seed=ex["seed"]),
            qml_confusion_condition=QmlConfusionCondition(n=qc["n"], depth=qc["depth"]),
            dpi=p["dpi"],
        )
        output_dir = str(_section(raw, "output")["dir"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed config: {exc!r}") from exc

    cfg = Config(experiment, noise, track_a, track_b, rem, zne, improvement, simulator, plots, output_dir)
    validate(cfg)
    return cfg


def _validate_qubits_depths(qubits, depths, prefix: str) -> None:
    if not qubits:
        raise ValueError(f"{prefix}.qubits must not be empty")
    for n in qubits:
        if not _is_int(n) or n < 2:
            raise ValueError(f"every {prefix} qubits value must be an integer >= 2, got {n!r}")
    if not depths:
        raise ValueError(f"{prefix}.depths must not be empty")
    for d in depths:
        if not _is_int(d) or d < 1:
            raise ValueError(f"every {prefix} depth must be an integer >= 1, got {d!r}")


def validate(cfg: Config) -> None:
    e = cfg.experiment
    if not e.seeds:
        raise ValueError("experiment.seeds must not be empty")
    for sd in e.seeds:
        if not _is_int(sd) or sd < 0:
            raise ValueError(f"every seed must be a non-negative integer, got {sd!r}")
    if not _is_int(e.shots) or e.shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {e.shots!r}")

    level_names = {lvl.name for lvl in cfg.noise.levels}
    for lvl in cfg.noise.levels:
        for label, p in (("p1", lvl.p1), ("p2", lvl.p2), ("p_ro", lvl.p_ro)):
            if not 0.0 <= p < 0.5:
                raise ValueError(f"noise.levels.{lvl.name}.{label} must lie in [0, 0.5), got {p}")
    for name in cfg.noise.headline_levels:
        if name not in level_names:
            raise ValueError(f"headline level {name!r} is not defined under noise.levels")

    _validate_qubits_depths(cfg.track_a.qubits, cfg.track_a.depths, "track_a")
    for name in cfg.track_a.noise_levels:
        if name not in level_names:
            raise ValueError(f"track_a noise level {name!r} is not defined under noise.levels")
    if not cfg.track_a.methods:
        raise ValueError("track_a.methods must not be empty")
    for m in cfg.track_a.methods:
        if m not in VALID_METHODS_A:
            raise ValueError(f"unknown track_a method {m!r}; valid methods are {VALID_METHODS_A}")
    c = cfg.track_a.circuit
    if not c.angle_low < c.angle_high:
        raise ValueError("track_a.circuit.angle_low must be smaller than angle_high")
    if not 0.0 <= c.min_abs_exact_expectation <= 1.0:
        raise ValueError("track_a.circuit.min_abs_exact_expectation must lie in [0, 1]")
    if not _is_int(c.max_resample_attempts) or c.max_resample_attempts < 1:
        raise ValueError("track_a.circuit.max_resample_attempts must be a positive integer")

    _validate_qubits_depths(cfg.track_b.qubits, cfg.track_b.depths, "track_b")
    for name in cfg.track_b.noise_levels:
        if name not in level_names:
            raise ValueError(f"track_b noise level {name!r} is not defined under noise.levels")
    if not cfg.track_b.methods:
        raise ValueError("track_b.methods must not be empty")
    for m in cfg.track_b.methods:
        if m not in VALID_METHODS_B:
            raise ValueError(f"unknown track_b method {m!r}; valid methods are {VALID_METHODS_B}")
    if not 0.0 < cfg.track_b.test_size < 1.0:
        raise ValueError(f"track_b.test_size must lie in (0, 1), got {cfg.track_b.test_size}")
    lo, hi = cfg.track_b.feature_range
    if not lo < hi:
        raise ValueError("track_b.feature_range must have low < high")

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

    if not _is_int(cfg.rem.calibration_shots) or cfg.rem.calibration_shots <= 0:
        raise ValueError("rem.calibration_shots must be a positive integer")
    if cfg.rem.cond_threshold <= 0:
        raise ValueError("rem.cond_threshold must be positive")

    sa = cfg.improvement.shot_allocation
    for name in sa.noise_levels:
        if name not in level_names:
            raise ValueError(f"improvement.shot_allocation noise level {name!r} is not defined")
    if not _is_int(sa.total_shots) or sa.total_shots <= 0:
        raise ValueError("improvement.shot_allocation.total_shots must be a positive integer")
    if not _is_int(sa.repetitions) or sa.repetitions <= 0:
        raise ValueError("improvement.shot_allocation.repetitions must be a positive integer")

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
