"""Parse a folder of statement PDFs into tables.

    uv run python -m statementscreen.parse data/synthetic/pdfs/ --out data/parsed/

The parser gets only each PDF's path: it never reads the batch's manifest, labels or
transactions.parquet. PDFs are taken from an explicitly sorted file list and parsed one
at a time (no multiprocessing). Each one's tables go to temporary Parquet parts, which
Polars then merges in sorted order, streaming, as in Week 1. Outputs in ``--out``:

    rows.parquet        one row per parsed transaction
    statements.parquet  one row per PDF tried, including unreadable ones (with reason)
    issues.parquet      every parse issue and self-check flag
    parse_run.json      counts, seconds per PDF and peak memory

A PDF that can't be read, or that crashes the parser, is recorded as an error with its
reason. It is never skipped silently.
"""

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.generate import OutputFolderError, prepare_output_folder
from statementscreen.parsing.models import ParsedStatement
from statementscreen.parsing.pdf import parse_pdf
from statementscreen.parsing.tables import (
    ISSUE_SCHEMA,
    ROW_SCHEMA,
    STATEMENT_SCHEMA,
    issues_frame,
    rows_frame,
    statement_frame,
)
from statementscreen.summary import peak_memory_bytes

TABLES = {"rows": ROW_SCHEMA, "statements": STATEMENT_SCHEMA, "issues": ISSUE_SCHEMA}
OUTPUT_NAMES = ("_parts", "rows.parquet", "statements.parquet", "issues.parquet",
                "parse_run.json")  # fmt: skip


def parse_one(pdf: Path) -> tuple[ParsedStatement, float]:
    """Parse one PDF. A parser crash becomes an error result, never a lost statement."""
    started = time.perf_counter()
    try:
        result = parse_pdf(pdf)
    except Exception as e:  # a bug in the parser: record it and keep going
        result = ParsedStatement(pdf.stem, None, 0, "error", f"parser crashed: {e!r}")
        result.issue("PARSER_CRASH", None, None, traceback.format_exc(limit=3))
    return result, time.perf_counter() - started


def merge_parts(out: Path) -> None:
    """Stream each table's parts, in sorted order, into one Parquet file."""
    for name, schema in TABLES.items():
        parts = sorted((out / "_parts" / name).glob("stmt_*.parquet"))  # explicit, sorted
        pl.scan_parquet(parts, schema=schema).sink_parquet(out / f"{name}.parquet")
    shutil.rmtree(out / "_parts")


def run_summary(pdfs: Sequence[Path], seconds: list[float], statuses: Counter,
                layouts: Counter, issue_kinds: Counter, wall: float) -> dict:  # fmt: skip
    ordered = sorted(seconds)
    return {
        "pdfs": len(pdfs),
        "status": dict(sorted(statuses.items())),
        "detected_layouts": dict(sorted(layouts.items())),
        "issues": dict(sorted(issue_kinds.items())),
        "seconds_total": round(wall, 1),
        "seconds_per_pdf": {
            "mean": round(statistics.fmean(ordered), 4),
            "median": round(statistics.median(ordered), 4),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
            "max": round(ordered[-1], 4),
        },
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
        "python": platform.python_version(),
        "platform": sys.platform,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m statementscreen.parse",
                                description="Parse statement PDFs into tables.")  # fmt: skip
    p.add_argument("pdfs", type=Path, help="folder of stmt_*.pdf files")
    p.add_argument("--out", type=Path, default=Path("data/parsed"), help="output folder")
    p.add_argument("--limit", type=int, help="parse only the first N PDFs (sorted by name)")
    p.add_argument("--overwrite", action="store_true", help="replace a previous parse in --out")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.pdfs.is_dir():
        print(f"error: {args.pdfs} is not a folder", file=sys.stderr)
        return 2
    pdfs = sorted(args.pdfs.glob("*.pdf"))[: args.limit]  # explicit, sorted file list
    if not pdfs:
        print(f"error: no .pdf files in {args.pdfs}", file=sys.stderr)
        return 2
    try:
        prepare_output_folder(args.out, args.overwrite, allowed=OUTPUT_NAMES)
    except OutputFolderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for name in TABLES:
        (args.out / "_parts" / name).mkdir(parents=True)

    seconds: list[float] = []
    statuses: Counter = Counter()
    layouts: Counter = Counter()
    issue_kinds: Counter = Counter()
    errors: list[tuple[str, str]] = []
    started = time.perf_counter()
    for i, pdf in enumerate(pdfs, start=1):
        result, took = parse_one(pdf)
        frames = {"rows": rows_frame(result), "statements": statement_frame(result, took),
                  "issues": issues_frame(result)}  # fmt: skip
        for name, frame in frames.items():
            frame.write_parquet(args.out / "_parts" / name / f"{pdf.stem}.parquet")
        seconds.append(took)
        statuses[result.status] += 1
        layouts[result.layout or "none"] += 1
        issue_kinds.update(frames["issues"]["kind"].to_list())
        if result.status != "ok":
            errors.append((pdf.name, result.error or "unknown error"))
        if i % 100 == 0 or i == len(pdfs):
            print(f"  parsed {i}/{len(pdfs)}", flush=True)
    wall = time.perf_counter() - started

    merge_parts(args.out)
    summary = run_summary(pdfs, seconds, statuses, layouts, issue_kinds, wall)
    (args.out / "parse_run.json").write_text(json.dumps(summary, indent=2, sort_keys=True)
                                             + "\n", encoding="utf-8")  # fmt: skip

    print(f"Parsed {len(pdfs)} PDFs into {args.out} in {summary['seconds_total']} s "
          f"({summary['seconds_per_pdf']['mean']} s per PDF, "
          f"peak memory {summary['peak_memory_mb']} MB)")  # fmt: skip
    print(f"  status: {summary['status']}  issues: {summary['issues'] or 'none'}")
    for name, reason in errors:  # every failed PDF, by name
        print(f"  ERROR {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
