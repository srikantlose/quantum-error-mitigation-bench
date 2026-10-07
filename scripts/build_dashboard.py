"""Build the interactive results dashboard from the sweep and analysis outputs.

    python scripts/build_dashboard.py

Reads results/raw/runs.csv (Track A), results/qml/qml_runs.csv (Track B),
results/improvement/zne_allocation.csv and results/summary/report_numbers.json, embeds them
as JSON into dashboard/template.html, and writes dashboard/mitigation-bench.html (the
artifact page) plus dashboard/quantum-error-mitigation-bench/index.html (a standalone page
for Vercel).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from qem.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = '"__BENCH_DATA__"'

TRACK_A_COLUMNS = {
    "n_qubits": "n", "depth_layers": "L", "noise_level": "noise", "seed": "seed", "method": "method",
    "E_exact": "E_exact", "E_noisy_exact": "E_noisy", "E_ideal_shots": "E_ideal_shots",
    "E_hat": "E_hat", "abs_error": "err", "signed_error": "serr",
    "improvement_pct": "imp", "error_reduction_factor": "erf",
    "hellinger_fidelity": "hf", "tvd": "tvd",
    "P_succ_exact": "psucc_exact", "P_succ_hat": "psucc_hat", "P_succ_ratio": "psucc_ratio",
    "E_lambda1": "E1", "E_lambda3": "E3", "E_lambda5": "E5",
    "E_richardson": "E_rich", "E_linear": "E_lin", "E_exp": "E_exp",
    "extrapolation_out_of_range": "oor", "rem_negative_mass": "negmass",
    "rem_condition_number": "cond", "est_std": "est_std", "shot_noise_floor": "floor",
    "n_circuits": "circuits", "total_shots": "shots", "base_depth": "depth", "max_depth": "max_depth",
    "base_cx": "cx", "total_cx": "total_cx", "total_gates": "total_gates", "p_ro": "p_ro",
}

TRACK_B_COLUMNS = {
    "model": "model", "n_qubits": "n", "depth_layers": "L", "noise_level": "noise", "seed": "seed",
    "method": "method", "accuracy": "acc", "precision": "prec", "recall": "rec", "f1": "f1",
    "tp": "tp", "fn": "fn", "fp": "fp", "tn": "tn",
    "mean_abs_E_error": "meanE", "margin_retention": "margin", "mean_hellinger": "hell",
    "agreement_with_exact": "agree", "n_test": "n_test",
}

ALLOC_COLUMNS = {
    "n_qubits": "n", "depth_layers": "L", "noise_level": "noise", "seed": "seed", "scheme": "scheme",
    "shots_lambda1": "s1", "shots_lambda3": "s3", "shots_lambda5": "s5", "total_shots": "total",
    "E_exact": "E_exact", "mean_E0": "mean_E0", "bias": "bias", "empirical_std": "std",
    "rmse": "rmse", "theoretical_std": "theory_std",
}


def _clean(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(f"{v:.6g}")
    if isinstance(v, str) and v != v:  # defensive; shouldn't hit for strings
        return None
    return v


def _cols(df: pd.DataFrame, mapping: dict) -> dict:
    return {short: [_clean(x) for x in df[col].tolist()] for col, short in mapping.items()}


def build_payload(runs_a: pd.DataFrame, runs_b: pd.DataFrame | None, alloc: pd.DataFrame | None,
                  numbers: dict, cfg) -> dict:
    levels = []
    for name in cfg.track_a.noise_levels:
        lvl = cfg.noise.level(name)
        levels.append({"name": name, "p1": lvl.p1, "p2": lvl.p2, "p_ro": lvl.p_ro})
    ex = cfg.plots.example_condition
    qc = cfg.plots.qml_confusion_condition
    payload = {
        "a": {"cols": _cols(runs_a, TRACK_A_COLUMNS)},
        "numbers": numbers,
        "meta": {
            "noise_levels": levels,
            "qubits": list(cfg.track_a.qubits),
            "depths": list(cfg.track_a.depths),
            "seeds": list(cfg.experiment.seeds),
            "methods": list(cfg.track_a.methods),
            "shots": cfg.experiment.shots,
            "scales": list(cfg.zne.scale_factors),
            "threshold": cfg.track_a.circuit.min_abs_exact_expectation,
            "example": {"n": ex.n, "L": ex.depth, "noise": ex.noise, "seed": ex.seed},
            "qml_qubits": list(cfg.track_b.qubits),
            "qml_depths": list(cfg.track_b.depths),
            "qml_methods": list(cfg.track_b.methods),
            "qml_noise_levels": list(cfg.track_b.noise_levels),
            "qml_confusion": {"n": qc.n, "L": qc.depth},
        },
    }
    if runs_b is not None:
        payload["b"] = {"cols": _cols(runs_b, TRACK_B_COLUMNS)}
    if alloc is not None:
        payload["alloc"] = {"cols": _cols(alloc, ALLOC_COLUMNS)}
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "raw" / "runs.csv"))
    ap.add_argument("--qml-runs", default=str(ROOT / "results" / "qml" / "qml_runs.csv"))
    ap.add_argument("--alloc-runs", default=str(ROOT / "results" / "improvement" / "zne_allocation.csv"))
    ap.add_argument("--numbers", default=str(ROOT / "results" / "summary" / "report_numbers.json"))
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--template", default=str(ROOT / "dashboard" / "template.html"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "mitigation-bench.html"))
    ap.add_argument("--site", default=str(ROOT / "dashboard" / "quantum-error-mitigation-bench"),
                    help="folder for the standalone index.html that is deployed to Vercel")
    args = ap.parse_args(argv)

    runs_a = pd.read_csv(args.runs)
    qml_path = Path(args.qml_runs)
    runs_b = pd.read_csv(qml_path) if qml_path.exists() else None
    alloc_path = Path(args.alloc_runs)
    alloc = pd.read_csv(alloc_path) if alloc_path.exists() else None
    numbers = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    payload = build_payload(runs_a, runs_b, alloc, numbers, load_config(args.config))
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    template = Path(args.template).read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise SystemExit(f"template must contain {PLACEHOLDER} exactly once")
    page = template.replace(PLACEHOLDER, blob)
    Path(args.out).write_text(page, encoding="utf-8", newline="\n")
    print("wrote", args.out, f"({len(blob) // 1024} KB of data)")

    site = Path(args.site)
    site.mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text(standalone(page), encoding="utf-8", newline="\n")
    print("wrote", site / "index.html")
    return 0


FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='6' fill='%230B1426'/%3E"
    "%3Ccircle cx='16' cy='16' r='9' fill='none' stroke='%23E8A93A' stroke-width='3'/%3E"
    "%3Ccircle cx='16' cy='16' r='3' fill='%2343D3EA'/%3E%3C/svg%3E"
)
DESCRIPTION = (
    "Results explorer for Group 20's study of quantum error mitigation: Track A error, "
    "cost and bias sources, Track B VQC vs. LogReg/SVM, and the two proposed improvements."
)


def standalone(page: str) -> str:
    """The artifact page is a fragment; wrap it in a full HTML document for static hosting."""
    head, sep, body = page.partition("</style>")
    if not sep:
        raise SystemExit("template has no </style> to split the head from the body")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f'<meta name="description" content="{DESCRIPTION}">\n'
        f'<link rel="icon" href="{FAVICON}">\n'
        + head.strip() + "\n" + sep + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
