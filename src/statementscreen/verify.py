"""Command line: check a written batch using only the files on disk.

    uv run python -m statementscreen.verify data/synthetic/

This never calls the generator: it reads manifest.csv, labels/*.json,
transactions.parquet and pdfs/*.pdf back and checks that they agree with each other.
That makes it an independent check: it catches a writer bug, a file damaged later, or a
single edited amount.

Checks:
- the same statement IDs in the manifest, the labels, the Parquet file and the PDFs
- the same transaction count in the manifest, the label and the Parquet rows
- Parquet sorted by statement_id, with rows numbered 0, 1, 2, ... within each statement
- every running balance recomputed exactly (integer paise) from the label's opening
  balance, ending at the label's closing balance
- every PDF opens with pypdf and has at least one page

Every failure names its statement. Any failure fails the run (exit code 1); a failing
statement is reported, never skipped.
"""

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from pypdf import PdfReader


@dataclass
class VerifyReport:
    statements: int = 0
    reconciled: int = 0  # statements whose every balance recomputed exactly
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def reconciliation_rate(self) -> float:
        return self.reconciled / self.statements if self.statements else 0.0

    def fail(self, statement_id: str, problem: str) -> None:
        self.failures.append(f"{statement_id}: {problem}")


def _read_manifest(out: Path) -> list[dict]:
    with (out / "manifest.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify_batch(out: Path) -> VerifyReport:
    report = VerifyReport()
    for name in ("manifest.csv", "transactions.parquet", "labels", "pdfs"):
        if not (out / name).exists():
            report.fail("batch", f"missing {name}")
    if not report.ok:
        return report

    manifest = _read_manifest(out)
    manifest_ids = [row["statement_id"] for row in manifest]
    report.statements = len(manifest_ids)

    # 1. IDs agree everywhere, no duplicates.
    if len(set(manifest_ids)) != len(manifest_ids):
        report.fail("manifest", "duplicate statement IDs")
    label_ids = sorted(p.stem for p in (out / "labels").glob("*.json"))
    pdf_ids = sorted(p.stem for p in (out / "pdfs").glob("*.pdf"))
    # Only the columns we need, so a 2000-statement file stays small in memory.
    df = pl.read_parquet(
        out / "transactions.parquet",
        columns=["statement_id", "row", "amount_paise", "balance_paise"],
    )
    parquet_ids = df["statement_id"].unique(maintain_order=True).to_list()
    expected = set(manifest_ids)
    for source, ids in (("labels", label_ids), ("pdfs", pdf_ids), ("parquet", parquet_ids)):
        for sid in sorted(expected - set(ids)):
            report.fail(sid, f"missing from {source}")
        for sid in sorted(set(ids) - expected):
            report.fail(sid, f"in {source} but not in the manifest")

    # 2. Parquet order: statement_id sorted.
    all_ids = df["statement_id"]
    if not all_ids.equals(all_ids.sort()):
        report.fail("parquet", "statement_id is not sorted")

    by_id = {key[0]: part for key, part in df.group_by("statement_id", maintain_order=True)}

    for row in manifest:
        sid = row["statement_id"]
        problems_before = len(report.failures)
        label_path = out / "labels" / f"{sid}.json"
        if not label_path.exists() or sid not in by_id:
            continue  # already reported as missing above
        label = json.loads(label_path.read_text(encoding="utf-8"))
        part = by_id[sid]

        # 3. Counts agree, rows numbered 0..k-1.
        counts = {"manifest": int(row["transaction_count"]),
                  "label": label["transaction_count"], "parquet": part.height}  # fmt: skip
        if len(set(counts.values())) != 1:
            report.fail(sid, f"transaction counts disagree {counts}")
        if part["row"].to_list() != list(range(part.height)):
            report.fail(sid, "Parquet rows are not numbered 0..n-1 in order")

        # 4. Every running balance, exactly, in integer paise.
        balance = label["opening_balance_paise"]
        for i, (amount, after) in enumerate(
            part.select("amount_paise", "balance_paise").iter_rows()
        ):
            balance += amount
            if after != balance:
                report.fail(sid, f"row {i}: balance {after} but opening + amounts = {balance}")
                break
        else:
            if balance != label["closing_balance_paise"]:
                report.fail(sid, f"closing balance {label['closing_balance_paise']} != {balance}")

        # 5. The PDF opens and has pages.
        pdf_path = out / "pdfs" / f"{sid}.pdf"
        if pdf_path.exists():
            try:
                if len(PdfReader(pdf_path).pages) < 1:
                    report.fail(sid, "PDF has no pages")
            except Exception as e:  # any read error is a failure for this statement
                report.fail(sid, f"PDF does not open: {e}")

        if len(report.failures) == problems_before:
            report.reconciled += 1
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.verify",
        description="Check a written batch from its files alone.",
    )
    p.add_argument("out", type=Path, nargs="?", default=Path("data/synthetic"))
    args = p.parse_args(argv)

    report = verify_batch(args.out)
    print(f"Checked {report.statements} statements in {args.out}")
    print(f"  reconciliation: {report.reconciled}/{report.statements} "
          f"({report.reconciliation_rate:.0%})")  # fmt: skip
    if report.ok:
        print("  all checks passed")
        return 0
    print(f"  {len(report.failures)} problem(s):", file=sys.stderr)
    for failure in report.failures:
        print(f"    {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
