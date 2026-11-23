"""Structure findings: page numbering, the header repeated on every page, and the
balance carried from one page to the next. Findings are strong; a check that couldn't
run (page 1's account number or period wasn't read) is listed as info, never as a pass.

    PAGE_NUMBER      a page has no "Page X of Y", X isn't this page's position in the
                     file, the pages disagree on Y, or Y isn't the file's page count
    HEADER_MISMATCH  a page after the first has no running header ("<bank> | A/c <no> |
                     <start> to <end>"), or its bank, account number or period differs
                     from page 1. (The holder's name is printed on page 1 only, so it
                     can't be compared across pages.)
    PAGE_CARRYOVER   the first transaction on a page doesn't follow from the last balance
                     on the page before. This is the same place a BALANCE_CHAIN finding
                     points to; the verdict counts one place once.

Everything is read from the PDF (words grouped into lines with the parser's own helpers)
and from parse_pdf's result: nothing from the generator.
"""

import re
from pathlib import Path

import pdfplumber

from statementscreen.parsing.models import ParsedStatement
from statementscreen.parsing.values import parse_dmony, parse_dmy
from statementscreen.parsing.words import Line, group_lines, text_of, to_words
from statementscreen.tamper.detect.findings import Finding, rupees, where

PAGE_OF = re.compile(r"Page (\d+) of (\d+)$")
RUNNING_HEADER = re.compile(r"(?P<bank>.+?) \| A/c (?P<account>\S+) \| "
                            r"(?P<start>\S+) to (?P<end>\S+)")  # fmt: skip
PERIOD_DATES = {"columns": lambda text, period: parse_dmy(text), "drcr": parse_dmony}


def read_page_lines(pdf_path: Path) -> list[list[Line]]:
    """Every page's words grouped into lines, one page at a time (each closed after)."""
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                out.append(group_lines(to_words(page.extract_words())))
            finally:
                page.close()
    return out


def _strong(code: str, page: int | None, reason: str, row: int | None = None) -> Finding:
    return Finding(code, "structure", page, row, "strong", f"{where(page, row)}{reason}")


# ---------------------------------------------------------------- page numbering


def page_number_findings(pages: list[list[Line]]) -> list[Finding]:
    n = len(pages)
    out: list[Finding] = []
    totals: dict[int, list[int]] = {}  # printed Y -> pages that print it
    for p, lines in enumerate(pages, start=1):
        found = [m for line in lines if (m := PAGE_OF.search(text_of(line)))]
        if not found:
            out.append(_strong("PAGE_NUMBER", p, "no 'Page X of Y' on this page"))
            continue
        x, y = int(found[-1].group(1)), int(found[-1].group(2))  # the lowest one: the footer
        totals.setdefault(y, []).append(p)
        if x != p:
            out.append(_strong("PAGE_NUMBER", p,
                               f"says 'Page {x} of {y}' but is page {p} of the file"))  # fmt: skip
    if len(totals) > 1:
        seen = "; ".join(f"'of {y}' on page(s) {', '.join(map(str, ps))}"
                         for y, ps in sorted(totals.items()))  # fmt: skip
        out.append(_strong("PAGE_NUMBER", None, f"pages disagree on the page count: {seen}"))
    for y in sorted(totals):
        if y != n:
            out.append(_strong("PAGE_NUMBER", None,
                               f"pages say 'of {y}' but the file has {n} pages"))  # fmt: skip
    return out


# ---------------------------------------------------------------- running header


def header_findings(pages: list[list[Line]], parsed: ParsedStatement) -> list[Finding]:
    if len(pages) < 2:
        return []
    h = parsed.header
    not_run = None
    if parsed.status == "error" or parsed.layout not in PERIOD_DATES:
        not_run = "HEADER_MISMATCH: the PDF couldn't be parsed"
    elif h.account_number is None or h.period_start is None or h.period_end is None:
        not_run = "HEADER_MISMATCH: page 1's account number or period wasn't read"
    if not_run:
        return [Finding("CHECK_NOT_RUN", "structure", None, None, "info", not_run)]
    assert h.period_start is not None and h.period_end is not None and parsed.layout
    page1_text = " ".join(text_of(line) for line in pages[0]).upper()
    period = (h.period_start, h.period_end)
    to_date = PERIOD_DATES[parsed.layout]
    out: list[Finding] = []
    for p, lines in enumerate(pages[1:], start=2):
        m = next((m for line in lines if (m := RUNNING_HEADER.fullmatch(text_of(line)))), None)
        if m is None:
            out.append(_strong("HEADER_MISMATCH", p, "no running header (bank | A/c | period)"))
            continue
        problems = []
        if m["bank"].upper() not in page1_text:
            problems.append(f"bank '{m['bank']}' doesn't appear on page 1")
        if m["account"] != h.account_number:
            problems.append(f"account {m['account']}, page 1 says {h.account_number}")
        try:
            dates = (to_date(m["start"], period), to_date(m["end"], period))
        except ValueError:
            problems.append(f"period '{m['start']} to {m['end']}' couldn't be read")
        else:
            if dates != period:
                page1 = f"{h.period_start.isoformat()} to {h.period_end.isoformat()}"
                problems.append(f"period {m['start']} to {m['end']} differs from page 1's {page1}")
        out += [_strong("HEADER_MISMATCH", p, problem) for problem in problems]
    return out


# ---------------------------------------------------------------- balance between pages


def carryover_findings(parsed: ParsedStatement) -> list[Finding]:
    out: list[Finding] = []
    for prev, r in zip(parsed.rows, parsed.rows[1:], strict=False):
        if r.page == prev.page or prev.balance_paise + r.amount_paise == r.balance_paise:
            continue
        out.append(_strong(
            "PAGE_CARRYOVER", r.page,
            f"the first transaction on this page doesn't follow from page {prev.page}'s "
            f"last balance: {rupees(prev.balance_paise)} + {rupees(r.amount_paise)} = "
            f"{rupees(prev.balance_paise + r.amount_paise)}, printed {rupees(r.balance_paise)}",
            row=r.row,
        ))  # fmt: skip
    return out


def structure_findings(pdf_path: Path, parsed: ParsedStatement,
                       pages: list[list[Line]] | None = None) -> list[Finding]:  # fmt: skip
    """All structure findings for one PDF (pages: its lines, if already read)."""
    pages = read_page_lines(Path(pdf_path)) if pages is None else pages
    numbering = page_number_findings(pages)
    return numbering + header_findings(pages, parsed) + carryover_findings(parsed)
