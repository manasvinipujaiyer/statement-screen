"""Arithmetic and parse findings: the Week 2 self-checks and parser issues, unchanged.

parse_pdf already runs the Week 2 self-checks (parsing/checks.py) and lists every
problem it met while reading as an Issue. This module only turns those issues into
findings with a strength and a plain reason; it re-checks nothing and changes nothing.

    strong  BALANCE_CHAIN, CLOSING_MISMATCH, SUMMARY_IDENTITY, SUMMARY_TOTALS,
            SUMMARY_COUNTS: printed numbers that contradict each other
    weak    DATE_ORDER, DATE_OUT_OF_PERIOD: unusual dates, not proof of editing
    weak    parser issues (ORPHAN_LINE, BAD_ROW, ...): part of a page couldn't be read
    weak    PARSE_FAILED: the PDF couldn't be parsed at all (not evidence of editing on
            its own); the arithmetic checks are then listed as not run (info)
    info    CHECK_NOT_RUN: a self-check needed a value the parser couldn't read
"""

from statementscreen.parsing.models import Issue, ParsedStatement
from statementscreen.tamper.detect.findings import Finding, where

# Self-check code -> (strength, what it means in plain words)
SELF_CHECKS = {
    "BALANCE_CHAIN": ("strong", "the balance doesn't follow from the previous balance "
                                "and this transaction's amount"),
    "CLOSING_MISMATCH": ("strong", "the last transaction's balance differs from the "
                                   "printed closing balance"),
    "SUMMARY_IDENTITY": ("strong", "opening balance + total credits - total debits "
                                   "doesn't equal the printed closing balance"),
    "SUMMARY_TOTALS": ("strong", "the transactions don't add up to the printed total"),
    "SUMMARY_COUNTS": ("strong", "the number of transactions differs from the printed count"),
    "DATE_ORDER": ("weak", "this transaction is dated before the one above it"),
    "DATE_OUT_OF_PERIOD": ("weak", "this transaction is dated outside the statement period"),
    "CHECK_NOT_RUN": ("info", "a self-check couldn't run"),
}  # fmt: skip
# Parser issue code -> plain words. All weak.
PARSER_ISSUES = {
    "ORPHAN_LINE": "a line inside the table couldn't be read as part of a transaction",
    "BAD_ROW": "a transaction line couldn't be read",
    "UNPLACED_WORD": "a word in a transaction line sits outside every column",
    "BAD_AMOUNT": "a printed amount couldn't be read",
    "BAD_DATE": "a printed date couldn't be read",
    "UNREADABLE_PAGE": "a page couldn't be read",
}
# Issue codes that mean the whole PDF couldn't be parsed.
FATAL = ("UNREADABLE_PDF", "UNKNOWN_LAYOUT", "LAYOUT_NOT_SUPPORTED")


def _finding(issue: Issue) -> Finding:
    if issue.code in SELF_CHECKS:
        strength, meaning = SELF_CHECKS[issue.code]
        group = "arithmetic"
    elif issue.code in PARSER_ISSUES:
        strength, meaning, group = "weak", PARSER_ISSUES[issue.code], "parse"
    else:  # a code added to the parser later: listed, never dropped
        strength, meaning, group = "weak", "the parser reported a problem", "parse"
    reason = f"{where(issue.page, issue.row)}{meaning} ({issue.detail})"
    return Finding(issue.code, group, issue.page, issue.row, strength, reason)


def arithmetic_findings(parsed: ParsedStatement) -> list[Finding]:
    """Findings from what parse_pdf reported, in the order it reported them."""
    if parsed.status == "error":
        fatal = next((i for i in parsed.issues if i.code in FATAL), None)
        detail = fatal.detail if fatal else (parsed.error or "unknown error")
        return [
            Finding("PARSE_FAILED", "parse", None, None, "weak",
                    f"the PDF couldn't be parsed ({detail}); this alone isn't evidence "
                    "of editing"),
            Finding("CHECK_NOT_RUN", "arithmetic", None, None, "info",
                    "the arithmetic checks didn't run because the PDF couldn't be parsed"),
        ]  # fmt: skip
    return [_finding(i) for i in parsed.issues]
