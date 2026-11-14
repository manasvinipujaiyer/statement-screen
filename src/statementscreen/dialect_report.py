"""How much of a narration dialect do the 29 known styles (patterns/clean.py) recognise?

    uv run python -m statementscreen.dialect_report data/week4_shift_n300/

Reads a batch generated with --narration-dialect: transactions.parquet (the narrations,
as generated, before any PDF layout cuts them) and narration_formats.parquet (which
dialect line wrote each one). This is evaluation, like `score`: pattern detection never
reads these files. Writes reports/week4_dialect_<name>.json.

Two measures, per narration and per dialect line, with no pass/fail threshold:

- opening match:     the narration starts with the opening of one of the known styles
                     (e.g. "NEFT CR-"), so clean.py tries that style's field layout;
- fully recognised:  clean.py accepts every field (style is not "<name>?" or "unknown").

Fully recognised means clean.py's field checks passed, not that it picked the right
counterparty: a different field order can still pass the checks.

Each line is also put in a group from its template alone (known opening / new opening),
by filling the placeholders with sample values. Week 4's comparison breaks results down
by this group.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.patterns.clean import _OPENINGS, clean_narration
from statementscreen.synth.dialect import Dialect, Format, load_dialect

SAMPLE = {
    "employer": "SAMPLE NAME", "lender": "SAMPLE NAME", "payee": "SAMPLE NAME",
    "merchant": "SAMPLE NAME", "biller": "SAMPLE NAME", "sender": "SAMPLE NAME",
    "city": "PUNE", "ref": "123456789012", "utr": "VSTKN25001123456",
    "umrn": "VSTK7000000000000001", "ifsc": "VSTK0A1B2C3", "vpa": "sample12@vstk",
    "month": "JAN25",
}  # fmt: skip
KNOWN, NEW = "known_opening", "new_opening"


def opening_style(narration: str) -> str | None:
    """Name of the first known style whose opening starts the narration, else None."""
    text = narration.strip()
    for style, opening in _OPENINGS:
        if opening.match(text):
            return style.name
    return None


def fully_recognised(narration: str) -> bool:
    style = clean_narration(narration).style
    return style != "unknown" and not style.endswith("?")


def format_group(fmt: Format) -> str:
    """known_opening or new_opening, from the template filled with sample values."""
    return KNOWN if opening_style(fmt.render(SAMPLE)) else NEW


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def build_report(batch: Path, dialect: Dialect) -> dict:
    columns = ["statement_id", "row", "description"]
    texts = pl.read_parquet(batch / "transactions.parquet", columns=columns)
    formats = pl.read_parquet(batch / "narration_formats.parquet")
    names = set(formats["dialect"].unique())
    if names != {dialect.name}:
        raise ValueError(f"{batch} was generated with dialect {sorted(names)}, not {dialect.name}")
    rows = texts.join(formats, on=["statement_id", "row"], how="left", validate="1:1")

    per_line: dict[int, dict] = {}
    for fmt in dialect.formats:
        per_line[fmt.line] = {"line": fmt.line, "kind": fmt.kind, "direction": fmt.direction,
                              "format": fmt.template, "group": format_group(fmt),
                              "narrations": 0, "opening_match": 0, "fully_recognised": 0,
                              "styles_matched": Counter(), "example": None}  # fmt: skip
    kept_default = 0
    for r in rows.iter_rows(named=True):
        if r["format_line"] is None:
            kept_default += 1  # advance tax / GST payment: today's narration, not measured
            continue
        entry = per_line[r["format_line"]]
        style = opening_style(r["description"])
        entry["narrations"] += 1
        entry["opening_match"] += style is not None
        entry["fully_recognised"] += fully_recognised(r["description"])
        entry["styles_matched"][clean_narration(r["description"]).style] += 1
        entry["example"] = entry["example"] or r["description"]

    lines = []
    for e in per_line.values():
        e["opening_match_share"] = _share(e["opening_match"], e["narrations"])
        e["fully_recognised_share"] = _share(e["fully_recognised"], e["narrations"])
        e["styles_matched"] = dict(e["styles_matched"].most_common())
        lines.append(e)
    total = sum(e["narrations"] for e in lines)
    opening = sum(e["opening_match"] for e in lines)
    recognised = sum(e["fully_recognised"] for e in lines)
    groups = {g: [e["line"] for e in lines if e["group"] == g] for g in (KNOWN, NEW)}
    return {
        "dialect": dialect.name,
        "batch": batch.name,
        "dialect_narrations": total,
        "rows_keeping_default_narration": kept_default,
        "opening_match_share": _share(opening, total),
        "fully_recognised_share": _share(recognised, total),
        "lines_by_group": groups,
        "narrations_by_group": {
            g: sum(e["narrations"] for e in lines if e["group"] == g) for g in (KNOWN, NEW)
        },
        "lines": lines,
        "lines_never_generated": [e["line"] for e in lines if e["narrations"] == 0],
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.dialect_report",
        description="Measure how much of a dialect batch the known styles recognise.",
    )
    p.add_argument("batch", type=Path, help="a batch generated with --narration-dialect")
    p.add_argument("--out", type=Path, help="default: reports/week4_dialect_<name>.json")
    args = p.parse_args(argv)
    try:
        formats = pl.read_parquet(args.batch / "narration_formats.parquet", columns=["dialect"])
        dialect = load_dialect(formats["dialect"][0])
        report = build_report(args.batch, dialect)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = args.out or Path(f"reports/week4_dialect_{dialect.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    kept = report["rows_keeping_default_narration"]
    print(f"Wrote {out}: {report['dialect_narrations']} {dialect.name} narrations "
          f"({kept} tax rows kept today's narration)")  # fmt: skip
    print(f"  opening matches a known style: {report['opening_match_share']:.2%}")
    print(f"  fully recognised by clean.py:  {report['fully_recognised_share']:.2%}")
    print(f"  {'line':>4}  {'kind':17} {'group':13} {'n':>6} {'opening':>8} {'recog.':>8}  format")
    for e in report["lines"]:
        o, f = e["opening_match_share"], e["fully_recognised_share"]
        o = "-" if o is None else f"{o:.0%}"
        f = "-" if f is None else f"{f:.0%}"
        print(f"  {e['line']:>4}  {e['kind']:17} {e['group']:13} {e['narrations']:>6} "
              f"{o:>8} {f:>8}  {e['format']}")  # fmt: skip
    if report["lines_never_generated"]:
        print(f"  lines never generated: {report['lines_never_generated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
