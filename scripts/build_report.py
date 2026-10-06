"""Render report/report.md from report/report_template.md.

Placeholders, all filled from generated files so no result number is typed by hand:

    {{num:key}}        a value from results/summary/report_numbers.json
    {{table:name}}     results/summary/<name>.md, with its title line removed
    {{qasm:instance}}  the QASM of results/raw/circuits/<instance>.json

    python scripts/build_report.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{\{(num|table|qasm):([A-Za-z0-9_.\-]+)\}\}")


def render(template: str, numbers: dict, summary_dir: Path, circuits_dir: Path) -> str:
    missing = []

    def table(name: str) -> str:
        lines = (summary_dir / f"{name}.md").read_text(encoding="utf-8").strip().splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        # sub-headings inside a table file become bold captions
        lines = [f"**{ln[3:]}**" if ln.startswith("## ") else ln for ln in lines]
        return "\n".join(lines).strip()

    def qasm(name: str) -> str:
        record = json.loads((circuits_dir / f"{name}.json").read_text(encoding="utf-8"))
        return "```\n" + record["qasm"].strip() + "\n```"

    def sub(match: re.Match) -> str:
        kind, key = match.groups()
        if kind == "num":
            if key not in numbers:
                missing.append(key)
                return match.group(0)
            return numbers[key]
        if kind == "table":
            return table(key)
        return qasm(key)

    out = PLACEHOLDER.sub(sub, template)
    if missing:
        raise KeyError(f"unknown report numbers: {sorted(set(missing))}")
    if "{{" in out:
        raise ValueError("unrendered placeholder left in the report")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=str(ROOT / "report" / "report_template.md"))
    ap.add_argument("--summary", default=str(ROOT / "results" / "summary"))
    ap.add_argument("--circuits", default=str(ROOT / "results" / "raw" / "circuits"))
    ap.add_argument("--out", default=str(ROOT / "report" / "report.md"))
    args = ap.parse_args(argv)

    summary_dir = Path(args.summary)
    numbers = json.loads((summary_dir / "report_numbers.json").read_text(encoding="utf-8"))
    template = Path(args.template).read_text(encoding="utf-8")
    try:
        text = render(template, numbers, summary_dir, Path(args.circuits))
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
