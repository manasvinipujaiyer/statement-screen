"""Self-checks on a parsed statement, using only what the parser read from the PDF.

Each check is separate and has its own code, so a later tamper detector (Week 6) can use
each one as its own signal. A check only reports: it never changes a row or a header
field. A check that can't run because a value it needs is missing reports CHECK_NOT_RUN,
naming the check, so a skipped check is never mistaken for a passed one.

Codes:
    BALANCE_CHAIN       a row's balance != previous balance + its amount (row 0 uses the
                        printed opening balance)
    SUMMARY_IDENTITY    opening + total credits - total debits != closing (printed values)
    SUMMARY_TOTALS      parsed rows' credit/debit totals != the printed totals
    SUMMARY_COUNTS      parsed rows' Cr/Dr counts != the printed counts
    DATE_ORDER          a row's posting date is earlier than the row before it
    DATE_OUT_OF_PERIOD  a row's posting date is outside the printed statement period
    CHECK_NOT_RUN       a check needed a value the parser couldn't read
"""

from statementscreen.parsing.models import Issue, ParsedStatement

CHECK_CODES = (
    "BALANCE_CHAIN",
    "SUMMARY_IDENTITY",
    "SUMMARY_TOTALS",
    "SUMMARY_COUNTS",
    "DATE_ORDER",
    "DATE_OUT_OF_PERIOD",
    "CHECK_NOT_RUN",
)


def check_balance_chain(s: ParsedStatement) -> list[Issue]:
    out: list[Issue] = []
    previous = s.header.opening_balance_paise
    for r in s.rows:
        if previous is None:  # only possible on row 0, when no opening balance was read
            out.append(Issue(s.statement_id, "CHECK_NOT_RUN", r.page, r.row,
                             "BALANCE_CHAIN: no opening balance for the first row"))  # fmt: skip
        elif previous + r.amount_paise != r.balance_paise:
            out.append(Issue(
                s.statement_id, "BALANCE_CHAIN", r.page, r.row,
                f"previous {previous} + amount {r.amount_paise} = "
                f"{previous + r.amount_paise}, printed balance {r.balance_paise} (paise)",
            ))  # fmt: skip
        previous = r.balance_paise  # carry on from what was printed, so one bad row = one flag
    return out


def check_summary_identity(s: ParsedStatement) -> list[Issue]:
    h = s.header
    needed = {
        "opening balance": h.opening_balance_paise,
        "closing balance": h.closing_balance_paise,
        "total credits": h.total_credits_paise,
        "total debits": h.total_debits_paise,
    }
    missing = [name for name, v in needed.items() if v is None]
    if missing:
        return [Issue(s.statement_id, "CHECK_NOT_RUN", None, None,
                      f"SUMMARY_IDENTITY: not read: {', '.join(missing)}")]  # fmt: skip
    assert h.opening_balance_paise is not None and h.closing_balance_paise is not None
    assert h.total_credits_paise is not None and h.total_debits_paise is not None
    expected = h.opening_balance_paise + h.total_credits_paise - h.total_debits_paise
    if expected != h.closing_balance_paise:
        return [Issue(s.statement_id, "SUMMARY_IDENTITY", None, None,
                      f"opening + credits - debits = {expected}, "
                      f"printed closing {h.closing_balance_paise} (paise)")]  # fmt: skip
    return []


def check_summary_totals(s: ParsedStatement) -> list[Issue]:
    """Totals are printed as positive numbers; parsed debits are negative, so flip them."""
    h = s.header
    out: list[Issue] = []
    parsed = {
        "total credits": (sum(r.amount_paise for r in s.rows if r.amount_paise > 0),
                          h.total_credits_paise),
        "total debits": (-sum(r.amount_paise for r in s.rows if r.amount_paise < 0),
                         h.total_debits_paise),
    }  # fmt: skip
    for name, (got, printed) in parsed.items():
        if printed is None:
            out.append(Issue(s.statement_id, "CHECK_NOT_RUN", None, None,
                             f"SUMMARY_TOTALS: no printed {name}"))  # fmt: skip
        elif got != printed:
            out.append(Issue(s.statement_id, "SUMMARY_TOTALS", None, None,
                             f"{name}: rows sum to {got}, printed {printed} (paise)"))  # fmt: skip
    return out


def check_summary_counts(s: ParsedStatement) -> list[Issue]:
    h = s.header
    out: list[Issue] = []
    counts = {
        "Cr count": (sum(r.amount_paise > 0 for r in s.rows), h.cr_count),
        "Dr count": (sum(r.amount_paise < 0 for r in s.rows), h.dr_count),
    }
    for name, (got, printed) in counts.items():
        if printed is None:
            out.append(Issue(s.statement_id, "CHECK_NOT_RUN", None, None,
                             f"SUMMARY_COUNTS: no printed {name}"))  # fmt: skip
        elif got != printed:
            out.append(Issue(s.statement_id, "SUMMARY_COUNTS", None, None,
                             f"{name}: {got} rows parsed, printed {printed}"))  # fmt: skip
    return out


def check_date_order(s: ParsedStatement) -> list[Issue]:
    return [
        Issue(s.statement_id, "DATE_ORDER", r.page, r.row,
              f"{r.posted.isoformat()} comes after {prev.posted.isoformat()} (row {prev.row})")
        for prev, r in zip(s.rows, s.rows[1:], strict=False)
        if r.posted < prev.posted
    ]  # fmt: skip


def check_dates_in_period(s: ParsedStatement) -> list[Issue]:
    start, end = s.header.period_start, s.header.period_end
    if start is None or end is None:
        if not s.rows:
            return []  # no dates to check, so nothing was skipped
        return [Issue(s.statement_id, "CHECK_NOT_RUN", None, None,
                      "DATE_OUT_OF_PERIOD: no statement period read")]  # fmt: skip
    return [
        Issue(s.statement_id, "DATE_OUT_OF_PERIOD", r.page, r.row,
              f"{r.posted.isoformat()} outside {start.isoformat()} to {end.isoformat()}")
        for r in s.rows
        if not start <= r.posted <= end
    ]  # fmt: skip


CHECKS = (
    check_balance_chain,
    check_summary_identity,
    check_summary_totals,
    check_summary_counts,
    check_date_order,
    check_dates_in_period,
)


def run_checks(s: ParsedStatement) -> list[Issue]:
    """All self-check flags for one parsed statement, in check order. Changes nothing."""
    return [issue for check in CHECKS for issue in check(s)]
