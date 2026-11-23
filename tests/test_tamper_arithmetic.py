"""Arithmetic and parse findings (tamper/detect/arithmetic.py) on hand-built statements."""

import ast
from datetime import date
from pathlib import Path

import pytest

from statementscreen.parsing.checks import run_checks
from statementscreen.parsing.models import Issue, ParsedRow, ParsedStatement, StatementHeader
from statementscreen.tamper.detect import findings as findings_module
from statementscreen.tamper.detect.arithmetic import arithmetic_findings
from statementscreen.tamper.detect.findings import Finding

DETECT_DIR = Path(findings_module.__file__).parent


def _row(i: int, amount: int, balance: int, page: int = 1) -> ParsedRow:
    return ParsedRow("s", i, date(2025, 1, 2 + i), None, None, f"row {i}", amount, balance,
                     page, "columns")  # fmt: skip


def _statement(rows: list[ParsedRow], opening: int = 10_000) -> ParsedStatement:
    credits = sum(r.amount_paise for r in rows if r.amount_paise > 0)
    debits = -sum(r.amount_paise for r in rows if r.amount_paise < 0)
    header = StatementHeader(
        account_number="1", period_start=date(2025, 1, 1), period_end=date(2025, 1, 31),
        opening_balance_paise=opening, closing_balance_paise=rows[-1].balance_paise,
        total_debits_paise=debits, total_credits_paise=credits,
        dr_count=sum(r.amount_paise < 0 for r in rows),
        cr_count=sum(r.amount_paise > 0 for r in rows),
    )  # fmt: skip
    s = ParsedStatement("s", "columns", 1, "ok", header=header, rows=rows)
    s.issues = run_checks(s)
    return s


def test_consistent_statement_has_no_findings():
    s = _statement([_row(0, 5_000, 15_000), _row(1, -2_000, 13_000)])
    assert arithmetic_findings(s) == []


def test_broken_running_balance_is_strong():
    s = _statement([_row(0, 5_000, 15_000), _row(1, -2_000, 14_000)])  # should be 13,000
    found = [f for f in arithmetic_findings(s) if f.code == "BALANCE_CHAIN"]
    assert len(found) == 1
    assert found[0].strength == "strong"
    assert (found[0].page, found[0].row) == (1, 1)
    assert "transaction 2" in found[0].reason


def test_date_order_is_weak():
    rows = [_row(0, 5_000, 15_000), _row(1, -2_000, 13_000)]
    rows[1].posted = date(2025, 1, 1)
    found = [f for f in arithmetic_findings(_statement(rows)) if f.code == "DATE_ORDER"]
    assert [f.strength for f in found] == ["weak"]


def test_parser_issue_is_weak_parse_finding():
    s = _statement([_row(0, 5_000, 15_000)])
    s.issues.append(Issue("s", "ORPHAN_LINE", 1, 1, "stray text"))
    (f,) = arithmetic_findings(s)
    assert (f.code, f.group, f.strength) == ("ORPHAN_LINE", "parse", "weak")
    assert "stray text" in f.reason


def test_check_not_run_is_info():
    s = _statement([_row(0, 5_000, 15_000)])
    s.header.opening_balance_paise = None
    s.issues = run_checks(s)
    codes = {(f.code, f.strength) for f in arithmetic_findings(s)}
    assert ("CHECK_NOT_RUN", "info") in codes


def test_parse_failure_is_weak_and_checks_listed_as_not_run():
    s = ParsedStatement("s", None, 1, "error", "unknown layout")
    s.issue("UNKNOWN_LAYOUT", 1, None, "no known table header")
    found = arithmetic_findings(s)
    assert [(f.code, f.strength) for f in found] == [("PARSE_FAILED", "weak"),
                                                     ("CHECK_NOT_RUN", "info")]  # fmt: skip
    assert "no known table header" in found[0].reason


def test_every_finding_has_a_reason():
    rows = [_row(0, 5_000, 15_000), _row(1, -2_000, 99_000)]
    s = _statement(rows)
    s.header.closing_balance_paise = 1
    s.header.cr_count = 7
    s.issues = run_checks(s)  # re-run on the changed header
    found = arithmetic_findings(s)
    assert {"BALANCE_CHAIN", "CLOSING_MISMATCH", "SUMMARY_COUNTS"} <= {f.code for f in found}
    assert all(f.reason.strip() for f in found)


def test_finding_refuses_empty_reason_or_unknown_strength():
    with pytest.raises(ValueError):
        Finding("X", "arithmetic", None, None, "strong", "  ")
    with pytest.raises(ValueError):
        Finding("X", "arithmetic", None, None, "medium", "why")


def test_detector_never_imports_generator_or_tamper_generator():
    """The detector works from the PDF alone: no synth.*, tamper.copy, tamper.rebuild,
    and no reading of answer keys, labels or parquet (checked by tamper_pdf tests too)."""
    forbidden = ("statementscreen.synth", "statementscreen.tamper.copy",
                 "statementscreen.tamper.rebuild", "statementscreen.tamper.sets")  # fmt: skip
    for path in sorted(DETECT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"
