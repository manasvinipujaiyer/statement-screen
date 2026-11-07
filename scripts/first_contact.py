"""Week 2, step 5: how far does the 'columns' parser get on 'drcr' PDFs, untouched?

Runs BEFORE any drcr-specific parsing code exists. The columns rules are forced
(layout detection is bypassed), so this measures how general the columns parser is.
It also records what normal detection says about these PDFs.

Usage:
    uv run python scripts/first_contact.py data/week1_n2000 --limit 200

Writes reports/week2_first_contact.json. Only this scoring side reads the manifest and
truth; the parser only ever gets each PDF's path.
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import polars as pl

from statementscreen.parsing.pdf import parse_pdf
from statementscreen.scoring import compare_rows, per_statement


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("batch", type=Path)
    p.add_argument("--limit", type=int, default=200, help="use statements 1..limit")
    p.add_argument("--out", type=Path, default=Path("reports/week2_first_contact.json"))
    args = p.parse_args()

    with (args.batch / "manifest.csv").open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))[: args.limit]
    drcr_ids = [r["statement_id"] for r in manifest if r["layout"] == "drcr"]

    started = time.perf_counter()
    rows, detection, issue_codes = [], Counter(), Counter()
    for sid in drcr_ids:
        pdf = args.batch / "pdfs" / f"{sid}.pdf"
        normal = parse_pdf(pdf)  # what detection says, with no forcing
        detection[normal.layout or normal.error or "unknown"] += 1
        forced = parse_pdf(pdf, force_layout="columns")
        issue_codes.update(i.code for i in forced.issues)
        rows += [
            {"statement_id": r.statement_id, "row": r.row, "posted": r.posted,
             "narration": r.narration, "amount_paise": r.amount_paise,
             "balance_paise": r.balance_paise}
            for r in forced.rows
        ]  # fmt: skip
    seconds = time.perf_counter() - started

    truth = pl.read_parquet(args.batch / "transactions.parquet").filter(
        pl.col("statement_id").is_in(drcr_ids)
    )
    schema = {
        "statement_id": pl.String,
        "row": pl.Int32,
        "posted": pl.Date,
        "narration": pl.String,
        "amount_paise": pl.Int64,
        "balance_paise": pl.Int64,
    }
    parsed = pl.DataFrame(rows, schema=schema)
    compared = compare_rows(parsed, truth)
    stats = per_statement(compared)
    truth_rows = int(compared["in_truth"].sum())

    report = {
        "what": "columns parser rules forced onto drcr PDFs, before any drcr code existed",
        "batch": args.batch.name,
        "statements": len(drcr_ids),
        "sample": f"drcr statements among stmt_0001..stmt_{args.limit:04d}",
        "normal_detection": dict(detection),
        "truth_rows": truth_rows,
        "parsed_rows": int(compared["in_parsed"].sum()),
        "exact_rows": int(compared["exact_row"].sum()),
        "exact_row_rate": float(compared["exact_row"].sum() / truth_rows) if truth_rows else 0.0,
        "statements_with_any_row": int((stats["parsed_rows"] > 0).sum()),
        "fully_clean_statements": int(
            (stats["row_count_match"] & (stats["exact_rows"] == stats["truth_rows"])).sum()
        ),
        "issue_codes": dict(sorted(issue_codes.items())),
        "seconds": round(seconds, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
