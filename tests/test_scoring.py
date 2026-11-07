"""Scoring: a clean batch scores perfectly, and every kind of damage is counted and listed."""

from dataclasses import fields
from pathlib import Path

import polars as pl
import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.parsing.models import StatementHeader
from statementscreen.parsing.pdf import parse_pdf
from statementscreen.parsing.tables import (
    STATEMENT_SCHEMA,
    issues_frame,
    rows_frame,
    statement_frame,
)
from statementscreen.scoring import build_report, load_parsed, load_truth
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=11), 8), out)
    return out


@pytest.fixture(scope="module")
def parsed(batch: Path, tmp_path_factory) -> dict[str, pl.DataFrame]:
    """Parse every PDF and write the three tables (a stand-in for the step 8 command)."""
    out = tmp_path_factory.mktemp("parsed")
    rows, statements, issues = [], [], []
    for pdf in sorted((batch / "pdfs").glob("*.pdf")):
        s = parse_pdf(pdf)
        rows.append(rows_frame(s))
        statements.append(statement_frame(s, seconds=0.0))
        issues.append(issues_frame(s))
    pl.concat(rows).write_parquet(out / "rows.parquet")
    pl.concat(statements).write_parquet(out / "statements.parquet")
    pl.concat(issues).write_parquet(out / "issues.parquet")
    return load_parsed(out)


def _report(parsed: dict[str, pl.DataFrame], batch: Path, **kw) -> dict:
    truth, manifest = load_truth(batch)
    return build_report(parsed, truth, manifest, **kw)


def test_statement_table_has_every_header_field() -> None:
    assert {f.name for f in fields(StatementHeader)} <= set(STATEMENT_SCHEMA)


def test_a_clean_batch_scores_perfectly(parsed, batch) -> None:
    r = _report(parsed, batch)
    o = r["overall"]
    assert o["statements"] == 8 and o["truth_rows"] == o["parsed_rows"] == o["exact_rows"] > 0
    assert o["exact_row_rate"] == 1.0 and o["row_count_match_rate"] == 1.0
    assert o["fully_clean_rate"] == 1.0 and o["layout_detected_correctly"] == 8
    assert o["field_accuracy"] == {"posted": 1.0, "amount_paise": 1.0, "balance_paise": 1.0}
    n = o["narration"]
    assert n["wrong"] == 0
    assert n["equal_ignoring_whitespace"] + n["cut_short_by_layout"] == o["truth_rows"]
    assert set(r["by_layout"]) == {"columns", "drcr"}  # both layouts are in this batch
    assert all(layout["meets_target"] for layout in r["by_layout"].values())
    assert r["self_check_flags"]["count"] == 0 and r["parse_issues"]["count"] == 0
    assert r["failures"] == {"count": 0, "first": []}
    assert r["parse_errors"] == [] and r["row_count_mismatches"] == []
    assert r["manifest_statements_not_parsed"] == 0


def test_a_wrong_amount_is_listed_with_expected_and_got(parsed, batch) -> None:
    rows = parsed["rows"].with_columns(
        amount_paise=pl.when((pl.col("statement_id") == "stmt_0003") & (pl.col("row") == 5))
        .then(pl.col("amount_paise") + 1)
        .otherwise(pl.col("amount_paise"))
    )
    original = parsed["rows"].filter(
        (pl.col("statement_id") == "stmt_0003") & (pl.col("row") == 5)
    )["amount_paise"][0]
    r = _report(parsed | {"rows": rows}, batch)
    assert r["overall"]["exact_rows"] == r["overall"]["truth_rows"] - 1
    assert r["overall"]["field_accuracy"]["amount_paise"] < 1.0
    assert r["failures"]["first"] == [
        {"statement_id": "stmt_0003", "row": 5, "field": "amount_paise",
         "expected": original, "got": original + 1}
    ]  # fmt: skip
    assert r["overall"]["fully_clean_statements"] == 7


def test_a_missing_row_shows_as_a_row_count_mismatch(parsed, batch) -> None:
    last = parsed["rows"].filter(pl.col("statement_id") == "stmt_0001")["row"].max()
    rows = parsed["rows"].filter(~((pl.col("statement_id") == "stmt_0001")
                                   & (pl.col("row") == last)))  # fmt: skip
    r = _report(parsed | {"rows": rows}, batch)
    assert r["overall"]["row_count_match_rate"] == 7 / 8
    [mismatch] = r["row_count_mismatches"]
    assert mismatch["statement_id"] == "stmt_0001"
    assert mismatch["truth_rows"] == mismatch["parsed_rows"] + 1
    assert r["failures"]["first"] == [
        {"statement_id": "stmt_0001", "row": last, "field": "row",
         "expected": "present", "got": "missing"}
    ]  # fmt: skip


def test_statements_not_parsed_are_counted_not_scored(parsed, batch) -> None:
    keep = pl.col("statement_id") != "stmt_0008"
    r = _report({name: df.filter(keep) for name, df in parsed.items()}, batch)
    assert r["manifest_statements_not_parsed"] == 1
    assert r["overall"]["statements"] == 7 and r["overall"]["exact_row_rate"] == 1.0


def test_every_self_check_flag_is_listed(parsed, batch) -> None:
    flags = pl.DataFrame(
        [{"statement_id": "stmt_0002", "kind": "check", "code": "BALANCE_CHAIN", "page": 1,
          "row": r, "detail": "test"} for r in range(60)],
        schema=parsed["issues"].schema, orient="row",
    )  # fmt: skip
    r = _report(parsed | {"issues": pl.concat([parsed["issues"], flags])}, batch)
    assert r["self_check_flags"]["count"] == 60
    assert len(r["self_check_flags"]["all"]) == 60  # all of them, not a sample of 50
    assert r["self_check_flags"]["by_code"] == {"BALANCE_CHAIN": 60}
    assert r["overall"]["fully_clean_statements"] == 7  # rows exact, but flagged


def test_failures_are_capped_but_fully_counted(parsed, batch) -> None:
    rows = parsed["rows"].with_columns(balance_paise=pl.col("balance_paise") + 1)
    r = _report(parsed | {"rows": rows}, batch, max_failures=10)
    assert r["failures"]["count"] == r["overall"]["truth_rows"]
    assert len(r["failures"]["first"]) == 10
    assert r["overall"]["exact_row_rate"] == 0.0
    assert not any(layout["meets_target"] for layout in r["by_layout"].values())


def test_first_contact_is_carried_alongside(parsed, batch) -> None:
    fc = {"exact_row_rate": 0.0, "statements": 103}
    assert _report(parsed, batch, first_contact=fc)["first_contact"] == fc
