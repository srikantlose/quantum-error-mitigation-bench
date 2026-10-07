"""Build the interactive study guide from the analysis outputs.

    python scripts/build_study_guide.py

Embeds results/summary/report_numbers.json, the row counts of the three sweeps, the
worked ZNE example condition, the noise levels and a set of downscaled figures into
study/template.html. Writes study/study-guide.html (a complete HTML document), a copy
with a link back to the results dashboard into the Vercel site folder
(dashboard/quantum-error-mitigation-bench/study/index.html) and, with --fragment, the page
without the document skeleton.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from qem.config import load_config

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = '"__STUDY_DATA__"'
SITE_NAV = "<!--SITE_NAV-->"
SITE_LINK = '<a class="sitenav" href="../">Results dashboard</a>'
FIG_DIR = ROOT / "results" / "figures"
FIG_MAX_WIDTH = 1100

FIGURES = {
    "D1": ("fig_D1_trackA_circuit.png", "D1. The Track A ansatz for n = 4, L = 2: ry and rz on every qubit, then a CNOT chain."),
    "D2": ("fig_D2_folded_circuit.png", "D2. Global folding at λ = 3 (n = 2, L = 1): U, then U†, then U again."),
    "D3": ("fig_D3_calibration_circuits.png", "D3. Full calibration (2ⁿ circuits, n = 2 shown) vs. tensored calibration (2 circuits at any n)."),
    "D4": ("fig_D4_vqc_circuit.png", "D4. The VQC for n = 4, L = 2: angle encoding, then the trainable ansatz."),
    "D5": ("fig_D5_mitigation_pipeline.png", "D5. One circuit's base, folded and calibration runs feed every method from the same shots."),
    "A1": ("fig_A1_error_vs_noise.png", "A1. Mean absolute error by noise level and method."),
    "A2": ("fig_A2_fidelity_before_after.png", "A2. Hellinger infidelity before and after REM (full and tensored)."),
    "A5": ("fig_A5_zne_extrapolation_example.png", "A5. Extrapolation for the example condition: raw counts (zne) vs. REM-corrected counts (zne_rem)."),
    "A6": ("fig_A6_overhead_circuits.png", "A6. Circuits per estimate: full REM grows as 1 + 2ⁿ; tensored REM stays at 3."),
    "A8": ("fig_A8_distributions_n4.png", "A8. Output distribution at n = 4, L = 4, moderate noise: noise spreads probability away from the ideal outcomes."),
    "A9": ("fig_A9_improvement_heatmap.png", "A9. Median improvement over no mitigation for every (n, L)."),
    "A10": ("fig_A10_cost_benefit.png", "A10. Cost (shots) against benefit (error reduction factor)."),
    "A12": ("fig_A12_rem_full_vs_tensored.png", "A12. Full vs. tensored REM error: points on the diagonal mean the two agree."),
    "B1": ("fig_B1_qml_accuracy_vs_noise.png", "B1. VQC accuracy vs. noise level, with LR and SVM as horizontal lines."),
    "B5": ("fig_B5_qml_margin_scatter.png", "B5. E_hat against E_exact on the test set: noise shrinks points toward 0 without flipping their sign."),
    "B6": ("fig_B6_qml_training_curves.png", "B6. VQC training curves (noiseless, exact)."),
    "I1": ("fig_I1_zne_allocation.png", "I1. Empirical std of the Richardson estimate, uniform vs. optimal allocation, at equal total shots."),
}


def figure_data_uri(path: Path) -> str:
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if im.width > FIG_MAX_WIDTH:
        h = round(im.height * FIG_MAX_WIDTH / im.width)
        im = im.resize((FIG_MAX_WIDTH, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _f(x):
    x = float(x)
    return None if not math.isfinite(x) else float(f"{x:.6g}")


def build_payload(cfg, numbers: dict) -> dict:
    runs = pd.read_csv(ROOT / "results" / "raw" / "runs.csv")
    qml = pd.read_csv(ROOT / "results" / "qml" / "qml_runs.csv")
    alloc = pd.read_csv(ROOT / "results" / "improvement" / "zne_allocation.csv")

    ex = cfg.plots.example_condition
    sel = runs[(runs.n_qubits == ex.n) & (runs.depth_layers == ex.depth)
               & (runs.noise_level == ex.noise) & (runs.seed == ex.seed)]
    example = {"n": ex.n, "L": ex.depth, "noise": ex.noise, "seed": ex.seed, "methods": {}}
    for _, row in sel.iterrows():
        example["methods"][row.method] = {
            "E_exact": _f(row.E_exact), "E_hat": _f(row.E_hat),
            "E1": _f(row.E_lambda1), "E3": _f(row.E_lambda3), "E5": _f(row.E_lambda5),
            "E_rich": _f(row.E_richardson), "E_lin": _f(row.E_linear), "E_exp": _f(row.E_exp),
        }

    log = (ROOT / "results" / "logs" / "pytest_output.txt").read_text(encoding="utf-8")
    m = re.search(r"(\d+) passed", log)

    levels = [{"name": nm, "p1": cfg.noise.level(nm).p1, "p2": cfg.noise.level(nm).p2,
               "p_ro": cfg.noise.level(nm).p_ro} for nm in cfg.track_a.noise_levels]
    figures = {k: {"src": figure_data_uri(FIG_DIR / fn), "caption": cap} for k, (fn, cap) in FIGURES.items()}
    return {
        "numbers": numbers,
        "levels": levels,
        "counts": {"track_a_rows": len(runs), "track_b_rows": len(qml), "alloc_rows": len(alloc),
                   "tests_passed": int(m.group(1)) if m else None},
        "shots": cfg.experiment.shots,
        "scales": list(cfg.zne.scale_factors),
        "alloc_total": cfg.improvement.shot_allocation.total_shots,
        "example": example,
        "figures": figures,
    }


FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='6' fill='%230B1426'/%3E"
    "%3Ccircle cx='16' cy='16' r='9' fill='none' stroke='%23E8A93A' stroke-width='3'/%3E"
    "%3Ccircle cx='16' cy='16' r='3' fill='%2343D3EA'/%3E%3C/svg%3E"
)


def standalone(page: str) -> str:
    """Wrap the page fragment in a full HTML document for opening from disk."""
    head, sep, body = page.partition("</style>")
    if not sep:
        raise SystemExit("template has no </style> to split the head from the body")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        f'<link rel="icon" href="{FAVICON}">\n'
        + head.strip() + "\n" + sep + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--numbers", default=str(ROOT / "results" / "summary" / "report_numbers.json"))
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--template", default=str(ROOT / "study" / "template.html"))
    ap.add_argument("--out", default=str(ROOT / "study" / "study-guide.html"))
    ap.add_argument("--site", default=str(ROOT / "dashboard" / "quantum-error-mitigation-bench" / "study" / "index.html"),
                    help="copy for the Vercel site, with a link back to the dashboard")
    ap.add_argument("--fragment", default=None, help="also write the page without the document skeleton here")
    args = ap.parse_args(argv)

    numbers = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    payload = build_payload(load_config(args.config), numbers)
    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")

    template = Path(args.template).read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise SystemExit(f"template must contain {PLACEHOLDER} exactly once")
    page = template.replace(PLACEHOLDER, blob)
    Path(args.out).write_text(standalone(page), encoding="utf-8", newline="\n")
    print("wrote", args.out, f"({len(page) // 1024} KB)")
    site = Path(args.site)
    site.parent.mkdir(parents=True, exist_ok=True)
    site.write_text(standalone(page.replace(SITE_NAV, SITE_LINK)), encoding="utf-8", newline="\n")
    print("wrote", site)
    if args.fragment:
        Path(args.fragment).write_text(page, encoding="utf-8", newline="\n")
        print("wrote", args.fragment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
