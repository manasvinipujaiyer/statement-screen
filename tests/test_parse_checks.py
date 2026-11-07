"""Self-checks: each one fires alone, on exactly the broken thing, and changes nothing."""

import copy
from datetime import date

import pytest

from statementscreen.parsing.checks import CHECK_CODES, run_checks
from statementscreen.parsing.models import ParsedRow, ParsedStatement, StatementHeader
from statementscreen.parsing.pdf import parse_pdf
from statementscreen.synth import generate_statement
from statementscreen.synth.render import render_statement


def _clean() -> ParsedStatement:
    """Opening 1,000.00; +500.00, -200.00, -100.00; closing 1,200.00. Everything agrees."""
    rows = [
        (date(2025, 1, 3), 50_000, 150_000, 1),
        (date(2025, 1, 10), -20_000, 130_000, 1),
        (date(2025, 1, 10), -10_000, 120_000, 2),  # same day is fine; page 2
    ]
    s = ParsedStatement("stmt_9999", "columns", 2, "ok")
    s.header = StatementHeader(
        account_number="1234",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 1, 31),
        opening_balance_paise=100_000,
        closing_balance_paise=120_000,
        total_debits_paise=30_000,
        total_credits_paise=50_000,
        dr_count=2,
        cr_count=1,
    )
    s.rows = [
        ParsedRow("stmt_9999", i, d, None, None, f"txn {i}", amt, bal, page, "columns")
        for i, (d, amt, bal, page) in enumerate(rows)
    ]
    return s


def _codes(s: ParsedStatement) -> list[tuple[str, int | None, int | None]]:
    return [(i.code, i.page, i.row) for i in run_checks(s)]


def test_a_consistent_statement_has_no_flags() -> None:
    assert run_checks(_clean()) == []


def test_checks_never_change_the_statement() -> None:
    s = _clean()
    s.rows[1].balance_paise += 1
    s.header.closing_balance_paise = 0
    before = copy.deepcopy(s)
    run_checks(s)
    assert s == before


def test_balance_chain_flags_only_the_broken_row() -> None:
    s = _clean()
    # Row 1's balance jumps by one paisa and later rows carry on from it. The chain flags
    # row 1 only (row 2 follows on correctly from what was printed). A jump like this
    # always breaks the summary identity too, because the amounts no longer add up to
    # closing - opening, so that check fires as well.
    s.rows[1].balance_paise = 130_001
    s.rows[2].balance_paise = 120_001
    s.header.closing_balance_paise = 120_001
    assert _codes(s) == [("BALANCE_CHAIN", 1, 1), ("SUMMARY_IDENTITY", None, None)]


def test_first_row_is_chained_to_the_opening_balance() -> None:
    s = _clean()
    s.header.opening_balance_paise = 99_999
    s.header.closing_balance_paise = 119_999  # keeps the summary identity true...
    # ...so the closing no longer matches the last row (1,200.00): the edited opening
    # shows up as the first chain link plus the closing mismatch.
    assert _codes(s) == [("BALANCE_CHAIN", 1, 0), ("CLOSING_MISMATCH", 2, 2)]


def test_edited_closing_balance() -> None:
    s = _clean()
    s.header.closing_balance_paise = 120_100
    # The rows end at 1,200.00 and opening + credits - debits is also 1,200.00, so both
    # the closing check and the identity catch the edited closing.
    assert _codes(s) == [("CLOSING_MISMATCH", 2, 2), ("SUMMARY_IDENTITY", None, None)]


def test_closing_without_rows_is_compared_with_the_opening() -> None:
    s = _clean()
    s.rows = []
    s.header.total_credits_paise = s.header.total_debits_paise = 0
    s.header.dr_count = s.header.cr_count = 0
    s.header.closing_balance_paise = 100_000
    assert _codes(s) == []
    s.header.closing_balance_paise = 100_001
    assert _codes(s) == [("CLOSING_MISMATCH", None, None), ("SUMMARY_IDENTITY", None, None)]


def test_summary_totals() -> None:
    s = _clean()
    # Both printed totals are 10.00 too high. Their difference is unchanged, so the
    # identity and the closing still agree, and only the totals check notices.
    s.header.total_credits_paise = 51_000
    s.header.total_debits_paise = 31_000
    assert _codes(s) == [("SUMMARY_TOTALS", None, None), ("SUMMARY_TOTALS", None, None)]


def test_summary_counts() -> None:
    s = _clean()
    s.header.dr_count = 3
    assert _codes(s) == [("SUMMARY_COUNTS", None, None)]


def test_date_order() -> None:
    s = _clean()
    s.rows[2].posted = date(2025, 1, 9)  # before row 1's 10 Jan
    assert _codes(s) == [("DATE_ORDER", 2, 2)]


@pytest.mark.parametrize("bad", [date(2024, 12, 31), date(2025, 2, 1)])
def test_dates_out_of_period(bad: date) -> None:
    s = _clean()
    s.rows[0].posted = bad
    expected = [("DATE_OUT_OF_PERIOD", 1, 0)]
    if bad > s.rows[1].posted:  # a date after the period also runs backwards into row 1
        expected = [("DATE_ORDER", 1, 1)] + expected
    assert _codes(s) == expected


def test_missing_values_report_check_not_run_never_a_pass() -> None:
    s = _clean()
    s.header = StatementHeader()  # nothing read from the header or summary
    flags = run_checks(s)
    assert {i.code for i in flags} == {"CHECK_NOT_RUN"}
    named = " ".join(i.detail for i in flags)
    for check in CHECK_CODES:
        if check not in ("DATE_ORDER", "CHECK_NOT_RUN"):  # date order needs no header value
            assert check in named, check


@pytest.mark.parametrize("layout", ["columns", "drcr"])
def test_clean_rendered_pdfs_pass_every_check(tmp_path, layout: str) -> None:
    s = generate_statement(21, "salaried", months=6, hard_cases=["emi_bounce"])
    r = parse_pdf(render_statement(s, tmp_path / "stmt_0021.pdf", layout))
    assert r.status == "ok" and r.rows
    assert [i for i in r.issues if i.code in CHECK_CODES] == []
