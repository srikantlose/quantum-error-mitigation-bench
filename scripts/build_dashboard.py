"""Build the interactive results dashboard from the sweep and analysis outputs.

    python scripts/build_dashboard.py

Reads results/raw/runs.csv and results/summary/report_numbers.json, embeds them as JSON
into dashboard/template.html, and writes dashboard/mitigation-bench.html (the artifact
page) plus dashboard/group20-mitigation-bench/index.html (a standalone page for Vercel).
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

COLUMNS = {
    "n_qubits": "n", "depth_layers": "L", "noise_level": "noise", "seed": "seed", "method": "method",
    "E_exact": "E_exact", "E_noisy_exact": "E_noisy", "E_ideal_shots": "E_ideal_shots",
    "E_hat": "E_hat", "abs_error": "err", "signed_error": "serr",
    "improvement_pct": "imp", "error_reduction_factor": "erf",
    "hellinger_fidelity": "hf", "tvd": "tvd",
    "E_lambda1": "E1", "E_lambda3": "E3", "E_lambda5": "E5",
    "E_richardson": "E_rich", "E_linear": "E_lin", "E_exp": "E_exp",
    "extrapolation_out_of_range": "oor", "rem_negative_mass": "negmass",
    "rem_condition_number": "cond", "est_std": "est_std", "shot_noise_floor": "floor",
    "n_circuits": "circuits", "total_shots": "shots", "base_depth": "depth", "max_depth": "max_depth",
    "base_cx": "cx", "total_cx": "total_cx", "p_ro": "p_ro",
}


def _clean(v):
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            return None
        return float(f"{v:.6g}")
    return v


def build_payload(runs: pd.DataFrame, numbers: dict, cfg) -> dict:
    cols = {short: [_clean(x) for x in runs[col].tolist()] for col, short in COLUMNS.items()}
    levels = []
    for name in cfg.experiment.noise_levels:
        lvl = cfg.noise.level(name)
        levels.append({"name": name, "p1": lvl.p1, "p2": lvl.p2, "p_ro": lvl.p_ro})
    ex = cfg.plots.example_condition
    return {
        "cols": cols,
        "numbers": numbers,
        "meta": {
            "noise_levels": levels,
            "qubits": list(cfg.experiment.qubits),
            "depths": list(cfg.experiment.depths),
            "seeds": list(cfg.experiment.seeds),
            "methods": list(cfg.experiment.methods),
            "shots": cfg.experiment.shots,
            "scales": list(cfg.zne.scale_factors),
            "threshold": cfg.circuit.min_abs_exact_expectation,
            "example": {"n": ex.n, "L": ex.depth, "noise": ex.noise, "seed": ex.seed},
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "raw" / "runs.csv"))
    ap.add_argument("--numbers", default=str(ROOT / "results" / "summary" / "report_numbers.json"))
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--template", default=str(ROOT / "dashboard" / "template.html"))
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "mitigation-bench.html"))
    ap.add_argument("--site", default=str(ROOT / "dashboard" / "group20-mitigation-bench"),
                    help="folder for the standalone index.html that is deployed to Vercel")
    args = ap.parse_args(argv)

    runs = pd.read_csv(args.runs)
    numbers = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    payload = build_payload(runs, numbers, load_config(args.config))
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
    "Results explorer for Group 20's study of quantum error mitigation: error, improvement, "
    "bias sources, extrapolation and cost across 360 simulated runs."
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
