"""Turn one ParsedStatement into Polars tables with fixed schemas.

Three tables, one row each per transaction / statement / issue. The parse command
writes them; the scoring step reads them. Fixed schemas mean an empty statement still
produces correctly typed (empty) tables, so parts always merge cleanly.
"""

from dataclasses import fields

import polars as pl

from statementscreen.parsing.checks import CHECK_CODES
from statementscreen.parsing.models import ParsedStatement

ROW_SCHEMA = {
    "statement_id": pl.String,
    "row": pl.Int32,
    "posted": pl.Date,
    "value_date": pl.Date,
    "ref": pl.String,
    "narration": pl.String,
    "amount_paise": pl.Int64,
    "balance_paise": pl.Int64,
    "page": pl.Int32,
    "layout": pl.String,
}

STATEMENT_SCHEMA = {
    "statement_id": pl.String,
    "layout": pl.String,  # as detected from the PDF (None if it couldn't be)
    "status": pl.String,  # "ok" | "error"
    "error": pl.String,
    "pages": pl.Int32,
    "rows": pl.Int32,
    "issues": pl.Int32,
    "seconds": pl.Float64,  # wall-clock time to parse this one PDF
    "account_number": pl.String,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "opening_balance_paise": pl.Int64,
    "closing_balance_paise": pl.Int64,
    "total_debits_paise": pl.Int64,
    "total_credits_paise": pl.Int64,
    "dr_count": pl.Int32,
    "cr_count": pl.Int32,
}

ISSUE_SCHEMA = {
    "statement_id": pl.String,
    "kind": pl.String,  # "check" (a named self-check) or "parse" (reading problem)
    "code": pl.String,
    "page": pl.Int32,
    "row": pl.Int32,
    "detail": pl.String,
}


def rows_frame(s: ParsedStatement) -> pl.DataFrame:
    return pl.DataFrame(
        [{k: getattr(r, k) for k in ROW_SCHEMA} for r in s.rows], schema=ROW_SCHEMA, orient="row"
    )


def statement_frame(s: ParsedStatement, seconds: float) -> pl.DataFrame:
    h = s.header
    record = {
        "statement_id": s.statement_id,
        "layout": s.layout,
        "status": s.status,
        "error": s.error,
        "pages": s.pages,
        "rows": len(s.rows),
        "issues": len(s.issues),
        "seconds": seconds,
        **{f.name: getattr(h, f.name) for f in fields(h)},  # every header field
    }
    return pl.DataFrame([record], schema=STATEMENT_SCHEMA, orient="row")


def issues_frame(s: ParsedStatement) -> pl.DataFrame:
    records = [
        {"statement_id": i.statement_id, "kind": "check" if i.code in CHECK_CODES else "parse",
         "code": i.code, "page": i.page, "row": i.row, "detail": i.detail}
        for i in s.issues
    ]  # fmt: skip
    return pl.DataFrame(records, schema=ISSUE_SCHEMA, orient="row")
