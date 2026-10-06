"""Run the experiment sweep and write results/raw/runs.csv plus all raw artifacts.

    python scripts/run_sweep.py --config config/experiment.yaml [--smoke] [--only-n 2 4] [--resume]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qem.config import load_config
from qem.experiment import run_sweep

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="load config/smoke.yaml and write to <output.dir>/smoke")
    ap.add_argument("--only-n", type=int, nargs="+", metavar="N", help="run only these qubit counts")
    ap.add_argument("--resume", action="store_true", help="skip run ids already in runs.csv")
    ap.add_argument("--out", help="output directory (default: output.dir from the config)")
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / "config" / "smoke.yaml" if args.smoke else args.config)
    if args.out:
        out = Path(args.out)
    else:
        out = ROOT / cfg.output_dir
        if args.smoke:
            out = out / "smoke"  # never overwrite the full sweep's results
    df = run_sweep(cfg, out, only_n=args.only_n, resume=args.resume)
    print(f"wrote {len(df)} rows to {out / 'raw' / 'runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
