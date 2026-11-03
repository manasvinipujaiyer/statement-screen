"""Draw a ``Statement`` as a PDF using ReportLab, in one of several layouts.

ReportLab's coordinate system starts at the *bottom-left* corner of the page and
measures in points (1 pt = 1/72 inch). We move *down* the page by subtracting from y.

Layouts
-------
Real banks differ in fonts, date formats, page size, how amounts are shown and
where the summary goes. If every synthetic PDF looked the same, a model would learn
that one look instead of learning to read statements. Each ``Layout`` below is just
settings; the drawing code is shared.

Determinism
-----------
PDFs normally embed the creation time and a random document ID, so two renders of the
same statement differ byte-for-byte. ``invariant=True`` makes ReportLab use fixed
values instead, so the same statement + layout always gives an identical file.

Every page carries a small "SYNTHETIC" footer so a generated file can never be
mistaken for a real bank document.
"""

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from statementscreen.synth.models import Statement, Transaction

MARGIN = 54  # 0.75 inch
TABLE_BOTTOM = 70  # stop above the footer
FOOTER_TEXT = "SYNTHETIC DOCUMENT - GENERATED FOR TESTING - NOT A REAL BANK STATEMENT"


@dataclass(frozen=True)
class Column:
    key: str  # what goes in it: date, description, debit, credit, amount, amount_crdr, balance
    header: str
    x: float  # position as a fraction of the usable width (right edge if right-aligned)
    align: str = "right"


@dataclass(frozen=True)
class Layout:
    name: str
    page_size: tuple[float, float]
    font: str
    font_bold: str
    size: float  # body font size
    row_h: float
    date_fmt: str
    columns: tuple[Column, ...]
    summary: str  # "box" (top-left box) | "lines" (text under the header) | "end" (after table)
    header_align: str  # "left" | "center"


LAYOUTS: dict[str, Layout] = {
    # Two money columns, US dates, summary box. The original layout.
    "classic": Layout(
        name="classic",
        page_size=LETTER,
        font="Helvetica",
        font_bold="Helvetica-Bold",
        size=8.5,
        row_h=14,
        date_fmt="%m/%d",
        columns=(
            Column("date", "Date", 0.0, "left"),
            Column("description", "Description", 0.083, "left"),
            Column("debit", "Withdrawals", 0.66),
            Column("credit", "Deposits", 0.82),
            Column("balance", "Balance", 1.0),
        ),
        summary="box",
        header_align="left",
    ),
    # One signed amount column, day-month-year dates, serif font, A4 paper.
    "ledger": Layout(
        name="ledger",
        page_size=A4,
        font="Times-Roman",
        font_bold="Times-Bold",
        size=9,
        row_h=13,
        date_fmt="%d %b %Y",
        columns=(
            Column("date", "Date", 0.0, "left"),
            Column("description", "Transaction Details", 0.14, "left"),
            Column("amount", "Amount", 0.80),
            Column("balance", "Balance", 1.0),
        ),
        summary="lines",
        header_align="center",
    ),
    # Monospaced, ISO dates, CR/DR suffixes instead of signs, summary at the end.
    "compact": Layout(
        name="compact",
        page_size=LETTER,
        font="Courier",
        font_bold="Courier-Bold",
        size=7.5,
        row_h=11,
        date_fmt="%Y-%m-%d",
        columns=(
            Column("date", "DATE", 0.0, "left"),
            Column("description", "DESCRIPTION", 0.13, "left"),
            Column("amount_crdr", "AMOUNT", 0.80),
            Column("balance", "BALANCE", 1.0),
        ),
        summary="end",
        header_align="left",
    ),
}
LAYOUT_NAMES = tuple(LAYOUTS)


def _fmt(x: Decimal) -> str:
    return f"{x:,.2f}"  # 1234.5 -> "1,234.50"


def _cell(t: Transaction, key: str, layout: Layout) -> str:
    """The text for one transaction in one column."""
    if key == "date":
        return f"{t.posted:{layout.date_fmt}}"
    if key == "description":
        return t.description
    if key == "debit":
        return _fmt(-t.amount) if t.amount < 0 else ""
    if key == "credit":
        return _fmt(t.amount) if t.amount > 0 else ""
    if key == "amount":
        return _fmt(t.amount)
    if key == "amount_crdr":
        return f"{_fmt(abs(t.amount))} {'CR' if t.amount > 0 else 'DR'}"
    if key == "balance":
        return _fmt(t.balance)
    raise ValueError(f"unknown column key {key!r}")


class _Page:
    """Geometry helpers for one layout."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.w, self.h = layout.page_size
        self.usable = self.w - 2 * MARGIN
        self.table_top_first = self.h - (230 if layout.summary in ("box", "lines") else 140)
        self.table_top_other = self.h - 110

    def x(self, col: Column) -> float:
        return MARGIN + col.x * self.usable

    def description_width(self) -> float:
        cols = self.layout.columns
        i = next(i for i, c in enumerate(cols) if c.key == "description")
        next_col = cols[i + 1]
        # The next column is right-aligned at its x, so leave room for its widest value.
        return self.x(next_col) - self.x(cols[i]) - 75

    def rows_that_fit(self, top: float) -> int:
        return int((top - TABLE_BOTTOM) // self.layout.row_h) - 1  # minus the header row


def _fit(text: str, max_w: float, font: str, size: float) -> str:
    """Cut text so it fits in max_w points, like a bank's fixed-width description column."""
    while stringWidth(text, font, size) > max_w:
        text = text[:-1]
    return text


def _paginate(n: int, page: _Page, tail_rows: int) -> list[range]:
    """Split n transactions into per-page index ranges, leaving tail_rows free on the last page."""
    pages, start, first = [], 0, True
    while start < n or not pages:
        per_page = page.rows_that_fit(page.table_top_first if first else page.table_top_other)
        pages.append(range(start, min(start + per_page, n)))
        start += per_page
        first = False
    last_top = page.table_top_first if len(pages) == 1 else page.table_top_other
    if len(pages[-1]) + tail_rows > page.rows_that_fit(last_top):
        pages.append(range(n, n))  # the summary gets a page of its own
    return pages


def _totals(s: Statement) -> list[tuple[str, Decimal]]:
    deposits = sum((t.amount for t in s.transactions if t.amount > 0), Decimal(0))
    withdrawals = sum((-t.amount for t in s.transactions if t.amount < 0), Decimal(0))
    return [
        ("Opening balance", s.opening_balance),
        ("Total deposits", deposits),
        ("Total withdrawals", withdrawals),
        ("Closing balance", s.closing_balance),
    ]


def _draw_header(c: Canvas, s: Statement, page: _Page) -> None:
    lay, top = page.layout, page.h - MARGIN
    period = f"{s.period_start:%B %d, %Y} - {s.period_end:%B %d, %Y}"
    details = [s.account_holder, f"Checking Account {s.account_number}", period]

    if lay.header_align == "center":
        mid = page.w / 2
        c.setFont(lay.font_bold, 16)
        c.drawCentredString(mid, top - 12, s.bank_name.upper())
        c.setFont(lay.font, 10)
        c.drawCentredString(mid, top - 28, "Statement of Account")
        c.setFont(lay.font, 9)
        for i, line in enumerate(details):
            c.drawString(MARGIN, top - 50 - i * 12, line)
    else:
        c.setFont(lay.font_bold, 16)
        c.drawString(MARGIN, top - 12, s.bank_name)
        c.setFont(lay.font, 10)
        c.drawString(MARGIN, top - 28, "Account Statement")
        c.setFont(lay.font, 9)
        for i, line in enumerate(details):
            c.drawRightString(page.w - MARGIN, top - 10 - i * 12, line)


def _draw_summary_box(c: Canvas, s: Statement, page: _Page) -> None:
    lay = page.layout
    box_top, box_h = page.h - 110, 90
    c.rect(MARGIN, box_top - box_h, 240, box_h)
    c.setFont(lay.font_bold, 9)
    c.drawString(MARGIN + 8, box_top - 16, "Account Summary")
    c.setFont(lay.font, 9)
    for i, (label, value) in enumerate(_totals(s)):
        y = box_top - 32 - i * 14
        c.drawString(MARGIN + 8, y, label)
        c.drawRightString(MARGIN + 232, y, _fmt(value))


def _draw_summary_lines(c: Canvas, s: Statement, page: _Page, top: float) -> None:
    lay = page.layout
    c.setFont(lay.font_bold, 9)
    c.drawString(MARGIN, top, "SUMMARY")
    c.setFont(lay.font, 9)
    for i, (label, value) in enumerate(_totals(s)):
        y = top - 14 - i * 12
        c.drawString(MARGIN, y, label)
        c.drawRightString(MARGIN + 200, y, _fmt(value))


def _draw_table(c: Canvas, s: Statement, rows: range, top: float, page: _Page) -> float:
    """Draw the column headers and the given rows; return the y just below the last row."""
    lay = page.layout
    c.setFont(lay.font_bold, lay.size)
    for col in lay.columns:
        draw = c.drawRightString if col.align == "right" else c.drawString
        draw(page.x(col), top, col.header)
    c.line(MARGIN, top - 4, page.w - MARGIN, top - 4)

    c.setFont(lay.font, lay.size)
    desc_w = page.description_width()
    y = top - lay.row_h
    for i in rows:
        t = s.transactions[i]
        for col in lay.columns:
            text = _cell(t, col.key, lay)
            if col.key == "description":
                text = _fit(text, desc_w, lay.font, lay.size)
            draw = c.drawRightString if col.align == "right" else c.drawString
            draw(page.x(col), y, text)
        y -= lay.row_h
    return y


def _draw_footer(c: Canvas, page_no: int, total: int, page: _Page) -> None:
    c.setFont(page.layout.font, 7)
    c.drawString(MARGIN, 40, FOOTER_TEXT)
    c.drawRightString(page.w - MARGIN, 40, f"Page {page_no} of {total}")


def render_statement(s: Statement, path: str | Path, layout: str = "classic") -> Path:
    """Write the statement to ``path`` as a PDF in the named layout and return the path."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUT_NAMES}, got {layout!r}")
    page = _Page(LAYOUTS[layout])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = Canvas(str(path), pagesize=page.layout.page_size, invariant=True)
    c.setTitle(f"{s.bank_name} statement {s.period_start:%Y-%m}")
    c.setCreator("StatementScreen synthetic generator")

    tail_rows = 7 if page.layout.summary == "end" else 0
    pages = _paginate(len(s.transactions), page, tail_rows)
    for page_no, rows in enumerate(pages, start=1):
        if page_no == 1:
            _draw_header(c, s, page)
            if page.layout.summary == "box":
                _draw_summary_box(c, s, page)
            elif page.layout.summary == "lines":
                _draw_summary_lines(c, s, page, page.h - 150)  # below the account details
            top = page.table_top_first
        else:
            top = page.table_top_other
        bottom = _draw_table(c, s, rows, top, page) if len(rows) or page_no == 1 else top
        if page_no == len(pages) and page.layout.summary == "end":
            _draw_summary_lines(c, s, page, bottom - 10)
        _draw_footer(c, page_no, len(pages), page)
        c.showPage()  # finish this page and start a new one
    c.save()
    return path
