"""Draw a ``Statement`` as a PDF using ReportLab.

ReportLab's coordinate system starts at the *bottom-left* corner of the page and
measures in points (1 pt = 1/72 inch). A US Letter page is 612 x 792 pt, so
``y = 792`` is the top edge and we move *down* the page by subtracting from y.

Every page carries a small "SYNTHETIC" footer so a generated file can never be
mistaken for a real bank document.
"""

from decimal import Decimal
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from statementscreen.synth.models import Statement

PAGE_W, PAGE_H = LETTER
MARGIN = 54  # 0.75 inch
ROW_H = 14
FONT, FONT_BOLD, SIZE = "Helvetica", "Helvetica-Bold", 8.5

# Table columns: (header, x position, alignment). Right-aligned columns use x as their right edge.
COLUMNS = [
    ("Date", MARGIN, "left"),
    ("Description", MARGIN + 42, "left"),
    ("Withdrawals", PAGE_W - MARGIN - 170, "right"),
    ("Deposits", PAGE_W - MARGIN - 90, "right"),
    ("Balance", PAGE_W - MARGIN, "right"),
]
DESC_MAX_W = COLUMNS[2][1] - 60 - COLUMNS[1][1]  # leave a gap before the Withdrawals column

TABLE_TOP_FIRST = PAGE_H - 230  # page 1 has the header + summary box above the table
TABLE_TOP_OTHER = PAGE_H - 110
TABLE_BOTTOM = 70  # stop above the footer


def _fmt(x: Decimal) -> str:
    return f"{x:,.2f}"  # 1234.5 -> "1,234.50"


def _fit(text: str, max_w: float) -> str:
    """Cut text so it fits in max_w points, like a bank's fixed-width description column."""
    while stringWidth(text, FONT, SIZE) > max_w:
        text = text[:-1]
    return text


def _rows_that_fit(top: float) -> int:
    # The header row takes one line; the rest are transactions.
    return int((top - TABLE_BOTTOM) // ROW_H) - 1


def _paginate(n: int) -> list[range]:
    """Split n transactions into per-page index ranges."""
    pages, start, first = [], 0, True
    while start < n or not pages:
        per_page = _rows_that_fit(TABLE_TOP_FIRST if first else TABLE_TOP_OTHER)
        pages.append(range(start, min(start + per_page, n)))
        start += per_page
        first = False
    return pages


def _draw_header(c: Canvas, s: Statement) -> None:
    top = PAGE_H - MARGIN
    c.setFont(FONT_BOLD, 16)
    c.drawString(MARGIN, top - 12, s.bank_name)
    c.setFont(FONT, 10)
    c.drawString(MARGIN, top - 28, "Account Statement")

    c.setFont(FONT, 9)
    right = PAGE_W - MARGIN
    c.drawRightString(right, top - 10, s.account_holder)
    c.drawRightString(right, top - 22, f"Checking Account {s.account_number}")
    c.drawRightString(right, top - 34, f"{s.period_start:%B %d, %Y} - {s.period_end:%B %d, %Y}")


def _draw_summary(c: Canvas, s: Statement) -> None:
    deposits = sum((t.amount for t in s.transactions if t.amount > 0), Decimal(0))
    withdrawals = sum((-t.amount for t in s.transactions if t.amount < 0), Decimal(0))
    lines = [
        ("Opening balance", s.opening_balance),
        ("Total deposits", deposits),
        ("Total withdrawals", withdrawals),
        ("Closing balance", s.closing_balance),
    ]
    box_top, box_h = PAGE_H - 110, 90
    c.rect(MARGIN, box_top - box_h, 240, box_h)
    c.setFont(FONT_BOLD, 9)
    c.drawString(MARGIN + 8, box_top - 16, "Account Summary")
    c.setFont(FONT, 9)
    for i, (label, value) in enumerate(lines):
        y = box_top - 32 - i * 14
        c.drawString(MARGIN + 8, y, label)
        c.drawRightString(MARGIN + 232, y, _fmt(value))


def _draw_table(c: Canvas, s: Statement, rows: range, top: float) -> None:
    c.setFont(FONT_BOLD, SIZE)
    for name, x, align in COLUMNS:
        (c.drawRightString if align == "right" else c.drawString)(x, top, name)
    c.line(MARGIN, top - 4, PAGE_W - MARGIN, top - 4)

    c.setFont(FONT, SIZE)
    y = top - ROW_H
    for i in rows:
        t = s.transactions[i]
        c.drawString(COLUMNS[0][1], y, f"{t.posted:%m/%d}")
        c.drawString(COLUMNS[1][1], y, _fit(t.description, DESC_MAX_W))
        if t.amount < 0:
            c.drawRightString(COLUMNS[2][1], y, _fmt(-t.amount))
        else:
            c.drawRightString(COLUMNS[3][1], y, _fmt(t.amount))
        c.drawRightString(COLUMNS[4][1], y, _fmt(t.balance))
        y -= ROW_H


def _draw_footer(c: Canvas, page: int, total: int) -> None:
    c.setFont(FONT, 7)
    c.drawString(
        MARGIN, 40, "SYNTHETIC DOCUMENT - GENERATED FOR TESTING - NOT A REAL BANK STATEMENT"
    )
    c.drawRightString(PAGE_W - MARGIN, 40, f"Page {page} of {total}")


def render_statement(s: Statement, path: str | Path) -> Path:
    """Write the statement to ``path`` as a PDF and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    c = Canvas(str(path), pagesize=LETTER)
    c.setTitle(f"{s.bank_name} statement {s.period_start:%Y-%m}")
    c.setCreator("StatementScreen synthetic generator")

    pages = _paginate(len(s.transactions))
    for page_no, rows in enumerate(pages, start=1):
        if page_no == 1:
            _draw_header(c, s)
            _draw_summary(c, s)
            _draw_table(c, s, rows, TABLE_TOP_FIRST)
        else:
            _draw_table(c, s, rows, TABLE_TOP_OTHER)
        _draw_footer(c, page_no, len(pages))
        c.showPage()  # finish this page and start a new one
    c.save()
    return path
