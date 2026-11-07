"""Score a parse against the batch's ground truth and write the report.

    uv run python -m statementscreen.score data/parsed/ data/synthetic/

Reads the parse output (rows/statements/issues.parquet, parse_run.json) and the batch's
transactions.parquet and manifest.csv. Writes reports/week2_parse_report.json (sorted
keys), carrying reports/week2_first_contact.json alongside if it exists.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.scoring import build_report, load_parsed, load_truth

DEFAULT_OUT = Path("reports/week2_parse_report.json")
DEFAULT_FIRST_CONTACT = Path("reports/week2_first_contact.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.score",
        description="Score parsed statements against the truth.",
    )
    p.add_argument("parsed", type=Path, help="folder written by statementscreen.parse")
    p.add_argument("batch", type=Path, help="generated batch folder (truth + manifest)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report path")
    p.add_argument("--first-contact", type=Path, default=DEFAULT_FIRST_CONTACT)
    p.add_argument("--target", type=float, default=0.995, help="exact-row rate to meet")
    return p


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.4%}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for folder, needed in ((args.parsed, ("rows.parquet", "statements.parquet",
                                          "issues.parquet", "parse_run.json")),
                           (args.batch, ("transactions.parquet", "manifest.csv"))):  # fmt: skip
        missing = [name for name in needed if not (folder / name).exists()]
        if missing:
            print(f"error: {folder} is missing {', '.join(missing)}", file=sys.stderr)
            return 2

    first_contact = None
    if args.first_contact.exists():
        first_contact = json.loads(args.first_contact.read_text(encoding="utf-8"))
    else:
        print(f"note: {args.first_contact} not found; first_contact will be null")

    truth, manifest = load_truth(args.batch)
    report = build_report(load_parsed(args.parsed), truth, manifest,
                          first_contact=first_contact, target=args.target)  # fmt: skip
    report["batch"] = args.batch.name
    report["parse_run"] = json.loads((args.parsed / "parse_run.json").read_text("utf-8"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    o = report["overall"]
    print(f"Wrote {args.out}")
    print(f"  statements {o['statements']}, truth rows {o['truth_rows']}, "
          f"parsed rows {o['parsed_rows']}")  # fmt: skip
    for layout, s in report["by_layout"].items():
        verdict = "meets" if s["meets_target"] else "BELOW"
        print(f"  {layout:8} exact rows {_pct(s['exact_row_rate'])} ({verdict} "
              f"{args.target:.1%}), fully clean {_pct(s['fully_clean_rate'])}")  # fmt: skip
    if first_contact:
        print(f"  first_contact (drcr, columns rules forced): "
              f"{_pct(first_contact.get('exact_row_rate'))}")  # fmt: skip
    print(f"  self-check flags {report['self_check_flags']['count']}, "
          f"parse issues {report['parse_issues']['count']}, "
          f"field failures {report['failures']['count']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
