"""Build the interactive learning guide from the analysis outputs.

    python scripts/build_guide.py

Embeds results/summary/report_numbers.json, one example circuit's ZNE points, the
per-size bias split and two real calibration matrices into guide/template.html, and
writes guide/learn-qem.html plus a standalone copy for the Vercel site
(dashboard/quantum-error-mitigation-bench/guide/index.html).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qem.analysis import bias_decomposition
from qem.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = '"__GUIDE_DATA__"'
CALIBRATION_RUN = "n2_L2_moderate_s0"
CALIBRATION_FULL = "full_" + CALIBRATION_RUN
CALIBRATION_TENSORED = "tensored_" + CALIBRATION_RUN


def r6(x) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else float(f"{x:.6g}")


def build_payload(runs: pd.DataFrame, numbers: dict, cfg, raw_dir: Path) -> dict:
    ex = cfg.plots.example_condition
    example = {}
    for _, row in runs[(runs.n_qubits == ex.n) & (runs.depth_layers == ex.depth)
                       & (runs.noise_level == ex.noise) & (runs.seed == ex.seed)].iterrows():
        example[row.method] = {k: r6(row[c]) for k, c in (
            ("E_exact", "E_exact"), ("E_noisy", "E_noisy_exact"), ("E_hat", "E_hat"), ("err", "abs_error"),
            ("E1", "E_lambda1"), ("E3", "E_lambda3"), ("E5", "E_lambda5"),
            ("E_rich", "E_richardson"), ("E_lin", "E_linear"))}

    bias = bias_decomposition(runs)
    bias = bias[bias.noise_level != "ideal"].groupby(["n_qubits", "depth_layers", "noise_level"]).agg(
        gate=("gate_bias", "mean"), readout=("readout_bias", "mean"), E=("E_exact", lambda s: s.abs().mean()))
    bias_rows = [{"n": int(n), "L": int(L), "noise": nz, "gate": r6(r.gate), "readout": r6(r.readout), "absE": r6(r.E)}
                 for (n, L, nz), r in bias.iterrows()]

    A_full = np.load(raw_dir / "calibration" / f"{CALIBRATION_FULL}.npy")
    A_tensored = np.load(raw_dir / "calibration" / f"{CALIBRATION_TENSORED}.npy")
    levels = [{"name": nm, "p1": cfg.noise.level(nm).p1, "p2": cfg.noise.level(nm).p2, "p_ro": cfg.noise.level(nm).p_ro}
              for nm in cfg.track_a.noise_levels]
    return {
        "numbers": numbers,
        "example": {"n": ex.n, "L": ex.depth, "noise": ex.noise, "seed": ex.seed, "methods": example},
        "bias": bias_rows,
        "calibration": {
            "run": CALIBRATION_RUN,
            "A_full": [[round(float(v), 4) for v in row] for row in A_full],
            "A_tensored": [[round(float(v), 4) for v in row] for row in A_tensored],
        },
        "levels": levels,
        "shots": cfg.experiment.shots,
        "methods": list(cfg.track_a.methods),
        "qml_methods": list(cfg.track_b.methods),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "results" / "raw" / "runs.csv"))
    ap.add_argument("--numbers", default=str(ROOT / "results" / "summary" / "report_numbers.json"))
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--template", default=str(ROOT / "guide" / "template.html"))
    ap.add_argument("--out", default=str(ROOT / "guide" / "learn-qem.html"))
    ap.add_argument("--site", default=str(ROOT / "dashboard" / "quantum-error-mitigation-bench" / "guide" / "index.html"),
                    help="standalone copy for the Vercel site")
    args = ap.parse_args(argv)

    runs_path = Path(args.runs)
    payload = build_payload(pd.read_csv(runs_path),
                            json.loads(Path(args.numbers).read_text(encoding="utf-8")),
                            load_config(args.config), runs_path.parent)
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = Path(args.template).read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise SystemExit(f"template must contain {PLACEHOLDER} exactly once")
    page = template.replace(PLACEHOLDER, blob)
    Path(args.out).write_text(page, encoding="utf-8", newline="\n")
    print("wrote", args.out)
    site = Path(args.site)
    site.parent.mkdir(parents=True, exist_ok=True)
    site.write_text(standalone(page), encoding="utf-8", newline="\n")
    print("wrote", site)
    return 0


def standalone(page: str) -> str:
    """Wrap the page fragment in a full HTML document for static hosting."""
    head, sep, body = page.partition("</style>")
    if not sep:
        raise SystemExit("template has no </style> to split the head from the body")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head.strip() + "\n" + sep + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
