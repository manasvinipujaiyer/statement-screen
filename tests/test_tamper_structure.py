"""Structure findings (tamper/detect/structure.py): hand-built page lines, then real
generated PDFs and their page_remove / page_swap copies (written to temp folders only)."""

import csv
from datetime import date
from pathlib import Path

import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.parsing.models import ParsedRow, ParsedStatement, StatementHeader
from statementscreen.parsing.pdf import parse_pdf
from statementscreen.parsing.words import Word
from statementscreen.tamper import copy as tc
from statementscreen.tamper.detect.findings import rupees
from statementscreen.tamper.detect.structure import (
    carryover_findings,
    header_findings,
    page_number_findings,
    structure_findings,
)
from statementscreen.writer import write_batch

# ---------------------------------------------------------------- hand-built lines


def _line(text: str, top: float) -> list[Word]:
    words, x = [], 36.0
    for t in text.split():
        words.append(Word(t, x, x + 5 * len(t), top))
        x += 5 * len(t) + 3
    return words


def _pages(footers: list[str | None], running: list[str | None] | None = None) -> list:
    pages = []
    for i, footer in enumerate(footers):
        lines = [_line("VISTARIKA BANK Account No : 123", 40)] if i == 0 else []
        if running and i > 0 and running[i] is not None:
            lines.append(_line(running[i], 30))
        if footer is not None:
            lines.append(_line(f"SYNTHETIC DOCUMENT {footer}", 800))
        pages.append(lines)
    return pages


def _parsed(rows: list[ParsedRow] | None = None) -> ParsedStatement:
    h = StatementHeader(account_number="123", period_start=date(2025, 1, 1),
                        period_end=date(2025, 3, 31))  # fmt: skip
    return ParsedStatement("s", "columns", 3, "ok", header=h, rows=rows or [])


RUN = "Vistarika Bank | A/c 123 | 01/01/2025 to 31/03/2025"


def test_complete_page_numbers_give_nothing():
    assert page_number_findings(_pages(["Page 1 of 3", "Page 2 of 3", "Page 3 of 3"])) == []


def test_missing_page_number_is_strong():
    found = page_number_findings(_pages(["Page 1 of 3", None, "Page 3 of 3"]))
    assert [(f.code, f.page, f.strength) for f in found] == [("PAGE_NUMBER", 2, "strong")]
    assert "no 'Page X of Y'" in found[0].reason


def test_removed_page_gives_out_of_order_and_wrong_count():
    found = page_number_findings(_pages(["Page 1 of 3", "Page 3 of 3"]))  # page 2 removed
    reasons = [f.reason for f in found]
    assert all(f.strength == "strong" for f in found)
    assert any("says 'Page 3 of 3' but is page 2" in r for r in reasons)
    assert any("the file has 2 pages" in r for r in reasons)


def test_pages_disagreeing_on_count_is_strong():
    found = page_number_findings(_pages(["Page 1 of 3", "Page 2 of 4", "Page 3 of 3"]))
    assert any("disagree" in f.reason for f in found)


def test_matching_running_headers_give_nothing():
    pages = _pages(["Page 1 of 3", "Page 2 of 3", "Page 3 of 3"], [None, RUN, RUN])
    assert header_findings(pages, _parsed()) == []


def test_other_account_or_missing_header_is_strong():
    other = RUN.replace("A/c 123", "A/c 999")
    pages = _pages(["Page 1 of 3", "Page 2 of 3", "Page 3 of 3"], [None, other, None])
    found = header_findings(pages, _parsed())
    expected = [("HEADER_MISMATCH", 2, "strong"), ("HEADER_MISMATCH", 3, "strong")]
    assert [(f.code, f.page, f.strength) for f in found] == expected
    assert "account 999" in found[0].reason and "no running header" in found[1].reason


def test_other_period_is_strong():
    other = RUN.replace("31/03/2025", "30/04/2025")
    pages = _pages(["Page 1 of 2", "Page 2 of 2"], [None, other])
    (f,) = header_findings(pages, _parsed())
    assert "period" in f.reason and f.strength == "strong"


def test_header_check_not_run_without_page1_account():
    p = _parsed()
    p.header.account_number = None
    pages = _pages(["Page 1 of 2", "Page 2 of 2"], [None, RUN])
    found = header_findings(pages, p)
    assert [(f.code, f.strength) for f in found] == [("CHECK_NOT_RUN", "info")]


def _row(i: int, amount: int, balance: int, page: int) -> ParsedRow:
    return ParsedRow("s", i, date(2025, 1, 2), None, None, "x", amount, balance, page, "columns")


def test_balance_carried_between_pages():
    good = [_row(0, 500, 10_500, 1), _row(1, -200, 10_300, 2)]
    assert carryover_findings(_parsed(good)) == []
    bad = [_row(0, 500, 10_500, 1), _row(1, -200, 9_000, 2)]
    (f,) = carryover_findings(_parsed(bad))
    assert (f.code, f.page, f.row, f.strength) == ("PAGE_CARRYOVER", 2, 1, "strong")
    assert rupees(10_300) in f.reason  # what it should have been


def test_rupees_uses_indian_grouping():
    assert rupees(12_345_678) == "INR 1,23,456.78"
    assert rupees(-5) == "INR -0.05"


# ---------------------------------------------------------------- real PDFs


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 6), folder)
    return folder


def _statements(batch: Path) -> list[tuple[Path, str, int]]:
    """(pdf, layout, pages), longest first."""
    out = []
    with (batch / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            pdf = batch / "pdfs" / f"{row['statement_id']}.pdf"
            out.append((pdf, row["layout"], parse_pdf(pdf).pages))
    return sorted(out, key=lambda t: -t[2])


def test_genuine_pdfs_have_no_structure_findings(batch):
    for pdf, _, _ in _statements(batch):
        assert structure_findings(pdf, parse_pdf(pdf)) == [], pdf.name


def test_page_remove_copy_is_caught(batch, tmp_path):
    pdf, _, pages = _statements(batch)[0]
    assert pages >= 3, "the test batch needs a statement of 3+ pages"
    data, _ = tc.tamper_copy(pdf, "page_remove", seed=1)
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(data)
    found = structure_findings(copy, parse_pdf(copy))
    codes = {f.code for f in found if f.strength == "strong"}
    assert {"PAGE_NUMBER", "PAGE_CARRYOVER"} <= codes


def test_page_swap_copy_is_caught(batch, tmp_path):
    stmts = _statements(batch)
    for pdf, layout, _ in stmts:
        for donor, donor_layout, _ in stmts:
            if donor == pdf or donor_layout != layout:
                continue
            try:
                data, _ = tc.tamper_copy(pdf, "page_swap", seed=1, donor_pdf=donor)
            except tc.NotApplicable:
                continue
            copy = tmp_path / "copy.pdf"
            copy.write_bytes(data)
            found = structure_findings(copy, parse_pdf(copy))
            assert any(f.code == "HEADER_MISMATCH" and f.strength == "strong" for f in found)
            return
    raise AssertionError("no page_swap pair in the test batch")
