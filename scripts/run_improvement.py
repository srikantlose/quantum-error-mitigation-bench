"""Run the variance-optimal shot allocation experiment (plan.md §14.2) and write
results/improvement/zne_allocation.csv.

    python scripts/run_improvement.py --config config/experiment.yaml [--smoke]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qem.config import load_config
from qem.improvement import run_improvement

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--smoke", action="store_true", help="load config/smoke.yaml and write to <output.dir>/smoke")
    ap.add_argument("--out", help="output directory (default: output.dir from the config)")
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / "config" / "smoke.yaml" if args.smoke else args.config)
    if args.out:
        out = Path(args.out)
    else:
        out = ROOT / cfg.output_dir
        if args.smoke:
            out = out / "smoke"
    df = run_improvement(cfg, out)
    print(f"wrote {len(df)} rows to {out / 'improvement' / 'zne_allocation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
