"""Write a planned batch to disk, one statement at a time.

Output layout (all paths relative to --out):

    pdfs/stmt_0001.pdf          the rendered statement
    labels/stmt_0001.json       per-statement ground truth (see synth/labels.py)
    transactions.parquet        every transaction of every statement, with statement_id
    manifest.csv                one row per statement
    narration_formats.parquet   only with a narration dialect: per transaction, the dialect
                                kind and line that wrote its narration (null where the
                                event has no kind and kept today's narration)

Memory stays flat: each statement is generated, rendered and written, then dropped.
Its transactions go to a temporary Parquet *part* (one file per statement, one fixed
schema). At the end Polars merges the parts, from an explicitly sorted file list, into
transactions.parquet by streaming (not by loading them all), and the parts are deleted.

Reproducibility: no timestamps, no absolute paths and no machine-specific values are
written anywhere. JSON keys are sorted. PDFs use ReportLab's invariant mode (see
synth/render.py for why that matters to the Week 6 tamper detector).
"""

import csv
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

from statementscreen.batch import GENERATOR_VERSION, StatementPlan
from statementscreen.synth.dialect import Dialect
from statementscreen.synth.generator import generate_statement
from statementscreen.synth.labels import statement_truth, transaction_rows
from statementscreen.synth.render import render_statement

TRANSACTION_SCHEMA = {
    "statement_id": pl.String,
    "row": pl.Int32,
    "posted": pl.Date,
    "description": pl.String,
    "amount_paise": pl.Int64,
    "balance_paise": pl.Int64,
    "category": pl.String,
    "canonical_counterparty": pl.String,
    "is_salary": pl.Boolean,
    "is_emi": pl.Boolean,
    "is_bounce": pl.Boolean,
    "hard_case_tags": pl.String,
}
MANIFEST_COLUMNS = ("statement_id", "customer_type", "layout", "months", "transaction_count",
                    "hard_case_tags", "seed", "generator_version")  # fmt: skip
FORMATS_SCHEMA = {"statement_id": pl.String, "row": pl.Int32, "dialect": pl.String,
                  "kind": pl.String, "format_line": pl.Int32}  # fmt: skip
FORMATS_PARTS = "_formats"


def _transactions_frame(rows: list[dict]) -> pl.DataFrame:
    """Rows -> a DataFrame with exactly TRANSACTION_SCHEMA (names, order and types)."""
    frame = pl.DataFrame(rows, schema={**TRANSACTION_SCHEMA, "posted": pl.String})
    return frame.with_columns(pl.col("posted").str.to_date("%Y-%m-%d")).select(
        [pl.col(name).cast(dtype) for name, dtype in TRANSACTION_SCHEMA.items()]
    )


def _formats_frame(s, sid: str, dialect: Dialect) -> pl.DataFrame:
    rows = []
    for i, t in enumerate(s.transactions):
        kind, line = s.narration_formats.get(t.description, (None, None))
        rows.append({"statement_id": sid, "row": i, "dialect": dialect.name, "kind": kind,
                     "format_line": line})  # fmt: skip
    return pl.DataFrame(rows, schema=FORMATS_SCHEMA)


def write_statement(plan: StatementPlan, out: Path, dialect: Dialect | None = None) -> dict:
    """Generate, render and write one statement; return its manifest row."""
    s = generate_statement(
        plan.seed, plan.customer_type, plan.months, plan.start, hard_cases=plan.hard_cases,
        dialect=dialect,
    )  # fmt: skip
    sid = plan.statement_id
    render_statement(s, out / "pdfs" / f"{sid}.pdf", plan.layout)

    truth = statement_truth(s, sid)
    truth.update(
        seed=plan.seed,
        layout=plan.layout,
        months=plan.months,
        generator_version=GENERATOR_VERSION,
        pdf=f"pdfs/{sid}.pdf",  # relative, so the batch can be moved or zipped
    )
    label = json.dumps(truth, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (out / "labels" / f"{sid}.json").write_text(label, encoding="utf-8")

    _transactions_frame(transaction_rows(s, sid)).write_parquet(out / "_parts" / f"{sid}.parquet")
    if dialect is not None:
        _formats_frame(s, sid, dialect).write_parquet(out / FORMATS_PARTS / f"{sid}.parquet")

    return {
        "statement_id": sid,
        "customer_type": plan.customer_type,
        "layout": plan.layout,
        "months": plan.months,
        "transaction_count": len(s.transactions),
        "hard_case_tags": "|".join(plan.hard_cases),
        "seed": plan.seed,
        "generator_version": GENERATOR_VERSION,
    }


def merge_parts(out: Path) -> None:
    """Stream the per-statement parts, in sorted order, into transactions.parquet."""
    parts = sorted((out / "_parts").glob("stmt_*.parquet"))  # explicit, sorted file list
    if not parts:
        raise RuntimeError("no Parquet parts to merge")
    pl.scan_parquet(parts, schema=TRANSACTION_SCHEMA).sink_parquet(
        out / "transactions.parquet", statistics=False
    )
    shutil.rmtree(out / "_parts")


def merge_format_parts(out: Path) -> None:
    parts = sorted((out / FORMATS_PARTS).glob("stmt_*.parquet"))
    pl.scan_parquet(parts, schema=FORMATS_SCHEMA).sink_parquet(
        out / "narration_formats.parquet", statistics=False
    )
    shutil.rmtree(out / FORMATS_PARTS)


def write_batch(
    plans: Sequence[StatementPlan],
    out: Path,
    progress: Callable[[int, int], None] | None = None,
    dialect: Dialect | None = None,
) -> list[dict]:
    """Write every planned statement, then merge the Parquet parts. Returns manifest rows.

    dialect: a narration dialect (synth/dialect.py), or None for today's formats.
    """
    subs = ("pdfs", "labels", "_parts") + ((FORMATS_PARTS,) if dialect is not None else ())
    for sub in subs:
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for k, plan in enumerate(plans, start=1):
            row = write_statement(plan, out, dialect)
            writer.writerow(row)
            rows.append(row)  # a few small strings per statement, not the statement itself
            if progress:
                progress(k, len(plans))
    merge_parts(out)
    if dialect is not None:
        merge_format_parts(out)
    return rows
