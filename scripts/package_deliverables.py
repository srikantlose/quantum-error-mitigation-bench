"""Bundle every deliverable into one zip for submission or download.

    python scripts/package_deliverables.py

Writes dist/Group20_QEM_Deliverables.zip with the code, configs, tests, all results and
figures, the report and literature notes, viva prep, the dashboard, the field guide and the
study guide, under a single top-level folder. Build artefacts (virtualenv, caches, egg-info,
Vercel link files, the smoke-test output and dist/ itself) are left out. Run the build_*
scripts first so the HTML pages match the current results.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_FOLDER = "Group20_QEM"
FILES = [
    "README.md", "DECISIONS.md", "plan.md", "plan_v1.md", "pyproject.toml",
    "requirements.txt", "requirements-optional.txt", ".gitignore", ".gitattributes",
]
DIRS = ["config", "src", "scripts", "tests", "results", "report", "docs", "dashboard", "guide", "study"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".vercel", ".venv", ".git", "dist"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SKIP_PREFIXES = ["results/smoke/"]


def wanted(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in SKIP_PARTS or p.endswith(".egg-info") for p in parts):
        return False
    if any(rel.startswith(pre) for pre in SKIP_PREFIXES):
        return False
    return Path(rel).suffix not in SKIP_SUFFIXES


def collect() -> list[str]:
    out = [f for f in FILES if (ROOT / f).is_file()]
    for d in DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file():
                rel = p.relative_to(ROOT).as_posix()
                if wanted(rel):
                    out.append(rel)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "dist" / "Group20_QEM_Deliverables.zip"))
    args = ap.parse_args(argv)

    files = collect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            zf.write(ROOT / rel, f"{TOP_FOLDER}/{rel}")
    print(f"wrote {out} ({len(files)} files, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
