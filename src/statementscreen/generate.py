"""Command line: plan and write a batch of synthetic statements, then verify it.

    uv run python -m statementscreen.generate --n 50 --seed 42 --out data/synthetic/
    uv run python -m statementscreen.generate --n 2000 --dry-run      # plan only, no files

Options:
    --n               how many statements (1-9999)
    --seed            main seed; each statement's seed is derived from it
    --mix             customer-type shares, e.g. salaried=0.6,gig=0.2,self_employed=0.2
    --hard-case-rate  chance of each applicable hard case per statement (default 0.2)
    --out             output folder (refused if not empty, unless --overwrite)
    --overwrite       replace a previous batch in --out (only this tool's files are removed)
    --dry-run         print the plan summary and stop; writes nothing
"""

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from statementscreen.batch import (
    DEFAULT_HARD_CASE_RATE,
    BatchConfig,
    StatementPlan,
    parse_mix,
    plan_batch,
)
from statementscreen.synth.hard_cases import HARD_CASES

# Everything a batch writes. --overwrite removes these and nothing else.
OUTPUT_NAMES = ("pdfs", "labels", "_parts", "transactions.parquet", "manifest.csv",
                "run_summary.json")  # fmt: skip


class OutputFolderError(Exception):
    pass


def prepare_output_folder(
    out: Path, overwrite: bool, allowed: Sequence[str] = OUTPUT_NAMES
) -> None:
    """Make sure ``out`` is ready for a fresh run, without ever deleting foreign files.

    ``allowed`` names the files and folders this tool writes (the batch outputs by
    default); anything else in ``out`` means it isn't ours, so nothing is deleted.
    """
    if out.exists() and not out.is_dir():
        raise OutputFolderError(f"{out} exists and is not a folder")
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in out.iterdir())
    if not existing:
        return
    if not overwrite:
        raise OutputFolderError(
            f"{out} is not empty ({len(existing)} entries); pass --overwrite to replace a "
            "previous batch, or choose another --out"
        )
    foreign = [name for name in existing if name not in allowed]
    if foreign:
        raise OutputFolderError(
            f"{out} contains files this tool didn't write ({', '.join(foreign[:5])}); "
            "refusing to overwrite. Choose an empty folder."
        )
    for name in existing:
        path = out / name
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def plan_summary(plans: Sequence[StatementPlan]) -> str:
    types = Counter(p.customer_type for p in plans)
    layouts = Counter(p.layout for p in plans)
    cases = Counter(c for p in plans for c in p.hard_cases)
    months = [p.months for p in plans]
    lines = [f"Planned {len(plans)} statements"]
    lines.append("  customer types: " + ", ".join(f"{k}={v}" for k, v in sorted(types.items())))
    lines.append("  layouts:        " + ", ".join(f"{k}={v}" for k, v in sorted(layouts.items())))
    lines.append(f"  months:         {min(months)}-{max(months)}")
    lines.append("  hard cases:     " + ", ".join(f"{c}={cases[c]}" for c in HARD_CASES))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.generate",
        description="Generate a batch of synthetic bank statements.",
    )
    p.add_argument("--n", type=int, required=True, help="number of statements (1-9999)")
    p.add_argument("--seed", type=int, default=42, help="main seed (default 42)")
    # Parsed in main(), not by argparse, so a bad mix shows our explanation of what's wrong.
    p.add_argument(
        "--mix",
        default=None,
        help="customer-type shares, default salaried=0.6,gig=0.2,self_employed=0.2",
    )
    p.add_argument("--hard-case-rate", type=float, default=DEFAULT_HARD_CASE_RATE,
                   help="chance of each applicable hard case (default 0.2)")  # fmt: skip
    p.add_argument("--out", type=Path, default=Path("data/synthetic"), help="output folder")
    p.add_argument("--overwrite", action="store_true", help="replace a previous batch in --out")
    p.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kwargs = {"mix": parse_mix(args.mix)} if args.mix is not None else {}
        config = BatchConfig(seed=args.seed, hard_case_rate=args.hard_case_rate, **kwargs)
        plans = plan_batch(config, args.n)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(plan_summary(plans))
    if args.dry_run:
        return 0
    try:
        prepare_output_folder(args.out, args.overwrite)
    except OutputFolderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    from statementscreen.writer import write_batch  # imported here: --dry-run needs no Polars

    def progress(done: int, total: int) -> None:
        if done == total or done % max(total // 10, 1) == 0:
            print(f"  wrote {done}/{total}", file=sys.stderr)

    from statementscreen.summary import build_run_summary
    from statementscreen.verify import verify_batch

    started = time.perf_counter()
    rows = write_batch(plans, args.out, progress)
    write_seconds = time.perf_counter() - started

    # Check the batch from its files alone, then record what the run produced and cost.
    started = time.perf_counter()
    report = verify_batch(args.out)
    verify_seconds = time.perf_counter() - started
    summary = build_run_summary(
        config, plans, rows, report, write_seconds, verify_seconds, args.out
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (args.out / "run_summary.json").write_text(text, encoding="utf-8")

    print(_describe(summary))
    if not report.ok:
        print(f"error: verify found {len(report.failures)} problem(s):", file=sys.stderr)
        for failure in report.failures[:20]:
            print(f"    {failure}", file=sys.stderr)
        return 1
    return 0


def _describe(summary: dict) -> str:
    c, v, t, d = summary["counts"], summary["verify"], summary["timing"], summary["disk"]
    return "\n".join([
        f"Wrote {summary['n']} statements ({c['transactions']:,} transactions)",
        "  customer types: " + ", ".join(f"{k}={n}" for k, n in c["customer_type"].items()),
        "  layouts:        " + ", ".join(f"{k}={n}" for k, n in c["layout"].items()),
        "  hard cases:     " + ", ".join(f"{k}={n}" for k, n in c["hard_cases"].items()),
        f"  reconciliation: {v['reconciled']}/{v['statements_checked']} "
        f"({v['reconciliation_pass_rate']:.0%})",
        f"  time:           {t['write_seconds']} s writing + {t['verify_seconds']} s verifying "
        f"({t['seconds_per_statement']} s per statement)",
        f"  disk:           {d['total_bytes'] / 1024**2:.1f} MB "
        f"(PDFs {d.get('pdfs_bytes', 0) / 1024**2:.1f} MB, "
        f"Parquet {d.get('transactions.parquet_bytes', 0) / 1024**2:.1f} MB)",
        f"  peak memory:    {summary['peak_memory_mb']} MB",
    ])  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
