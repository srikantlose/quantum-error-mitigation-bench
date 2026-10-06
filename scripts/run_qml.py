"""Run the Track B (Iris binary VQC) sweep and write results/qml/qml_runs.csv.

    python scripts/run_qml.py --config config/experiment.yaml [--smoke] [--retrain]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qem.config import load_config
from qem.qml.runner import run_qml

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--smoke", action="store_true", help="load config/smoke.yaml and write to <output.dir>/smoke")
    ap.add_argument("--retrain", action="store_true", help="retrain the VQC even if a cached model matches")
    ap.add_argument("--out", help="output directory (default: output.dir from the config)")
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / "config" / "smoke.yaml" if args.smoke else args.config)
    if args.out:
        out = Path(args.out)
    else:
        out = ROOT / cfg.output_dir
        if args.smoke:
            out = out / "smoke"
    df = run_qml(cfg, out, retrain=args.retrain)
    print(f"wrote {len(df)} rows to {out / 'qml' / 'qml_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
