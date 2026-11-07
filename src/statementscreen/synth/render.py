"""Draw a ``Statement`` as an A4 PDF in one of two fictional Indian bank layouts.

ReportLab's coordinate system starts at the *bottom-left* corner of the page and
measures in points (1 pt = 1/72 inch). An A4 page is 595 x 842 pt; we move *down* the
page by subtracting from y.

The two layouts differ in the ways a parser has to cope with:

========================  ===========================  ==========================
                          ``columns``                  ``drcr``
========================  ===========================  ==========================
Dates                     03/02/2025 (DD/MM/YYYY)      03-Feb-25 (DD-Mon-YY)
Money columns             Withdrawal Amt / Deposit Amt one Amount with Dr / Cr
Balance                   1,23,456.78                  1,23,456.78 Cr
Extra columns             Chq./Ref.No., Value Dt       none
Long narrations           wrap onto up to 3 lines      cut to one line
Font                      Helvetica                    Times
Summary                   box on page 1                block after the last row
========================  ===========================  ==========================

Both use Indian digit grouping (12,34,567.89) and write the currency as "INR": the
built-in PDF fonts have no rupee glyph.

Determinism and the tamper detector
-----------------------------------
PDFs normally embed the creation time and a random document ID. ``invariant=True``
makes ReportLab use fixed values instead, so the same statement + layout always gives a
byte-identical file. Side effect: every synthetic PDF has the *same* uniform metadata.
The Week 6 tamper detector must therefore NOT treat "uniform/fixed metadata" as a sign
of a genuine statement: it's an artefact of how this dataset was made.

Every page carries a "SYNTHETIC" footer so a generated file can never be mistaken for a
real bank document.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from statementscreen.synth.models import Statement, Transaction

PAGE_W, PAGE_H = A4
MARGIN = 36  # 0.5 inch: Indian statements pack seven columns onto A4
TABLE_BOTTOM = 56  # stop above the footer
FOOTER_TEXT = "SYNTHETIC DOCUMENT - GENERATED FOR TESTING - NOT A REAL BANK STATEMENT"


def inr(x: Decimal) -> str:
    """Indian digit grouping: 1234567.8 -> '12,34,567.80' (last 3 digits, then pairs)."""
    sign = "-" if x < 0 else ""
    whole, frac = f"{abs(x):.2f}".split(".")
    if len(whole) > 3:
        head, groups = whole[:-3], [whole[-3:]]
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        whole = ",".join([head, *groups])
    return f"{sign}{whole}.{frac}"


@dataclass(frozen=True)
class Column:
    key: str  # what goes in it, see _cell()
    header: str
    x: float  # points from the left margin; the right edge if right-aligned
    align: str = "right"
    width: float = 0  # for left-aligned text columns: how wide the text may get


@dataclass(frozen=True)
class Layout:
    name: str
    font: str
    font_bold: str
    size: float
    line_h: float
    date_fmt: str
    columns: tuple[Column, ...]
    narration_lines: int  # 1 = cut to fit; more = wrap onto up to this many lines
    summary: str  # "box" (page 1) | "end" (after the last transaction)


LAYOUTS: dict[str, Layout] = {
    "columns": Layout(
        name="columns",
        font="Helvetica",
        font_bold="Helvetica-Bold",
        size=7.2,
        line_h=9.5,
        date_fmt="%d/%m/%Y",
        columns=(
            Column("date", "Date", 0, "left", 46),
            Column("narration", "Narration", 50, "left", 178),
            Column("ref", "Chq./Ref.No.", 232, "left", 64),
            Column("value_date", "Value Dt", 300, "left", 32),
            Column("debit", "Withdrawal Amt.", 392),
            Column("credit", "Deposit Amt.", 458),
            Column("balance", "Closing Balance", PAGE_W - 2 * MARGIN),
        ),
        narration_lines=3,
        summary="box",
    ),
    "drcr": Layout(
        name="drcr",
        font="Times-Roman",
        font_bold="Times-Bold",
        size=8.5,
        line_h=12,
        date_fmt="%d-%b-%y",
        columns=(
            Column("date", "Txn Date", 0, "left", 44),
            Column("narration", "Description", 52, "left", 300),
            Column("amount_drcr", "Amount (INR)", 430),
            Column("balance_cr", "Balance (INR)", PAGE_W - 2 * MARGIN),
        ),
        narration_lines=1,
        summary="end",
    ),
}
LAYOUT_NAMES = tuple(LAYOUTS)


def _reference(t: Transaction) -> str:
    """The Chq./Ref.No. column: the first long number in the narration (UTR/RRN), or zeros."""
    m = re.search(r"\d{10,}", t.description)
    return m.group(0)[-16:].rjust(16, "0") if m else "0" * 16


def _cell(t: Transaction, key: str, lay: Layout) -> str:
    if key == "date":
        return f"{t.posted:{lay.date_fmt}}"
    if key == "value_date":
        return f"{t.posted:%d/%m/%y}"
    if key == "narration":
        return t.description
    if key == "ref":
        return _reference(t)
    if key == "debit":
        return inr(-t.amount) if t.amount < 0 else ""
    if key == "credit":
        return inr(t.amount) if t.amount > 0 else ""
    if key == "balance":
        return inr(t.balance)
    if key == "amount_drcr":
        return f"{inr(abs(t.amount))} {'Cr' if t.amount > 0 else 'Dr'}"
    if key == "balance_cr":
        return f"{inr(t.balance)} Cr"
    raise ValueError(f"unknown column key {key!r}")


def _fit(text: str, max_w: float, font: str, size: float) -> str:
    """Cut text so it fits in max_w points."""
    while text and stringWidth(text, font, size) > max_w:
        text = text[:-1]
    return text


def _wrap(text: str, max_w: float, font: str, size: float, max_lines: int) -> list[str]:
    """Split text into lines that fit, preferring to break after '/', '-' or a space."""
    lines: list[str] = []
    while text and len(lines) < max_lines:
        if stringWidth(text, font, size) <= max_w:
            lines.append(text)
            break
        piece = _fit(text, max_w, font, size)
        cut = max(piece.rfind("/"), piece.rfind("-"), piece.rfind(" "))
        if cut >= len(piece) // 2:  # a natural break point, not too early
            piece = piece[: cut + 1]
        lines.append(piece.rstrip())
        text = text[len(piece) :].lstrip()
    return lines


def _totals(s: Statement) -> list[tuple[str, str]]:
    debits = [t.amount for t in s.transactions if t.amount < 0]
    credits = [t.amount for t in s.transactions if t.amount > 0]
    return [
        ("Opening Balance", inr(s.opening_balance)),
        ("Dr Count", str(len(debits))),
        ("Cr Count", str(len(credits))),
        ("Debits", inr(-sum(debits, Decimal(0)))),
        ("Credits", inr(sum(credits, Decimal(0)))),
        ("Closing Bal", inr(s.closing_balance)),
    ]


class _Renderer:
    def __init__(self, s: Statement, lay: Layout, c: Canvas) -> None:
        self.s, self.lay, self.c = s, lay, c
        self.left, self.right = MARGIN, PAGE_W - MARGIN
        self.narr = next(col for col in lay.columns if col.key == "narration")
        self.top_first = PAGE_H - (262 if lay.summary == "box" else 196)
        self.top_other = PAGE_H - 74

    # ------------------------------------------------------------------ layout maths

    def rows(self) -> list[list[str]]:
        """The narration lines for every transaction (1 line in drcr, up to 3 in columns)."""
        lay = self.lay
        out = []
        for t in self.s.transactions:
            if lay.narration_lines == 1:
                out.append([_fit(t.description, self.narr.width, lay.font, lay.size)])
            else:
                out.append(_wrap(t.description, self.narr.width, lay.font, lay.size,
                                 lay.narration_lines) or [""])  # fmt: skip
        return out

    def paginate(self, rows: list[list[str]]) -> list[range]:
        """Split rows into pages by height, leaving room for an end-of-statement summary."""
        lay = self.lay
        pages: list[range] = []
        start, top = 0, self.top_first
        used = 0.0
        avail = top - TABLE_BOTTOM - lay.line_h * 2  # minus the column-header row
        for i, lines in enumerate(rows):
            h = len(lines) * lay.line_h
            if used + h > avail:
                pages.append(range(start, i))
                start, used = i, 0.0
                avail = self.top_other - TABLE_BOTTOM - lay.line_h * 2
            used += h
        pages.append(range(start, len(rows)))
        if lay.summary == "end" and used + lay.line_h * 9 > avail:
            pages.append(range(len(rows), len(rows)))  # the summary gets its own page
        return pages

    # ------------------------------------------------------------------ drawing

    def text(self, x: float, y: float, s: str, bold: bool = False, size: float | None = None,
             align: str = "left") -> None:  # fmt: skip
        self.c.setFont(self.lay.font_bold if bold else self.lay.font, size or self.lay.size)
        draw = {"left": self.c.drawString, "right": self.c.drawRightString,
                "center": self.c.drawCentredString}[align]  # fmt: skip
        draw(x, y, s)

    def details(self) -> list[tuple[str, str]]:
        s, fmt = self.s, self.lay.date_fmt
        return [
            ("Account No", s.account_number),
            ("Account Type", s.account_type),
            ("Customer ID", s.customer_id),
            ("Branch", s.branch),
            ("IFSC", s.ifsc),
            ("MICR", s.micr),
            ("Currency", "INR"),
            ("Statement Period", f"{s.period_start:{fmt}} to {s.period_end:{fmt}}"),
        ]

    def header_columns(self) -> None:
        s, top = self.s, PAGE_H - MARGIN
        self.text(self.left, top - 14, s.bank_name, bold=True, size=15)
        self.text(self.left, top - 28, "Statement of Account", size=9)
        self.text(self.left, top - 52, s.account_holder, bold=True, size=8.5)
        for i, line in enumerate(_wrap(s.address, 230, self.lay.font, 8, 3)):
            self.text(self.left, top - 64 - i * 10, line, size=8)
        x_label, x_value = self.right - 210, self.right - 130
        for i, (label, value) in enumerate(self.details()):
            y = top - 10 - i * 11
            self.text(x_label, y, label, size=8)
            self.text(x_value, y, f": {value}", size=8)

    def summary_box(self) -> None:
        y_top, h = PAGE_H - 150, 62
        self.c.rect(self.left, y_top - h, self.right - self.left, h)
        self.text(self.left + 6, y_top - 12, "STATEMENT SUMMARY", bold=True, size=8)
        totals = _totals(self.s)
        col_w = (self.right - self.left) / len(totals)
        for i, (label, value) in enumerate(totals):
            x = self.left + i * col_w + col_w / 2
            self.text(x, y_top - 32, label, size=8, align="center")
            self.text(x, y_top - 46, value, bold=True, size=8, align="center")

    def header_drcr(self) -> None:
        s, top, mid = self.s, PAGE_H - MARGIN, PAGE_W / 2
        self.text(mid, top - 14, s.bank_name.upper(), bold=True, size=15, align="center")
        self.text(mid, top - 28, "ACCOUNT STATEMENT", size=9.5, align="center")
        left = [("Name", s.account_holder)] + [
            ("Address", line) for line in _wrap(s.address, 200, self.lay.font, 8.5, 2)
        ]
        rows = left + self.details()
        half = (len(rows) + 1) // 2
        for i, (label, value) in enumerate(rows):
            x = self.left if i < half else mid + 10
            y = top - 52 - (i % half) * 12
            self.text(x, y, label, bold=True, size=8.5)
            self.text(x + 78, y, f": {value}", size=8.5)

    def summary_end(self, y: float) -> None:
        self.text(self.left, y, "Statement Summary", bold=True, size=9)
        for i, (label, value) in enumerate(_totals(self.s)):
            row_y = y - 14 - i * 12
            self.text(self.left, row_y, label, size=8.5)
            self.text(self.left + 190, row_y, value, size=8.5, align="right")
        self.text(self.left, y - 14 - 6 * 12 - 6,
                  "This is a computer generated statement.", size=7.5)  # fmt: skip

    def running_header(self) -> None:
        """Pages 2+ repeat the account number and period, as real statements do."""
        s, fmt = self.s, self.lay.date_fmt
        line = (f"{s.bank_name}  |  A/c {s.account_number}  |  "
                f"{s.period_start:{fmt}} to {s.period_end:{fmt}}")  # fmt: skip
        self.text(self.left, PAGE_H - MARGIN - 10, line, size=8)

    def table(self, rows: list[list[str]], idx: range, top: float) -> float:
        lay = self.lay
        for col in lay.columns:
            x = self.left + col.x
            self.text(x, top, col.header, bold=True, align=col.align)
        self.c.line(self.left, top - 3, self.right, top - 3)

        y = top - lay.line_h - 2
        for i in idx:
            t = self.s.transactions[i]
            for col in lay.columns:
                x = self.left + col.x
                if col.key == "narration":
                    for k, line in enumerate(rows[i]):
                        self.text(x, y - k * lay.line_h, line)
                else:
                    value = _cell(t, col.key, lay)
                    if col.width:
                        value = _fit(value, col.width, lay.font, lay.size)
                    self.text(x, y, value, align=col.align)
            y -= len(rows[i]) * lay.line_h
        return y

    def footer(self, page_no: int, total: int) -> None:
        self.text(self.left, 28, FOOTER_TEXT, size=6.5)
        self.text(self.right, 28, f"Page {page_no} of {total}", size=6.5, align="right")

    def draw(self) -> None:
        rows = self.rows()
        pages = self.paginate(rows)
        for page_no, idx in enumerate(pages, start=1):
            if page_no == 1:
                if self.lay.summary == "box":
                    self.header_columns()
                    self.summary_box()
                else:
                    self.header_drcr()
                top = self.top_first
            else:
                self.running_header()
                top = self.top_other
            bottom = self.table(rows, idx, top) if (len(idx) or page_no == 1) else top
            if page_no == len(pages) and self.lay.summary == "end":
                self.summary_end(bottom - 14)
            self.footer(page_no, len(pages))
            self.c.showPage()  # finish this page and start a new one


def render_statement(s: Statement, path: str | Path, layout: str = "columns") -> Path:
    """Write the statement to ``path`` as an A4 PDF in the named layout and return the path."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUT_NAMES}, got {layout!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # invariant=True: no timestamp or random ID, so output is byte-identical (see docstring).
    c = Canvas(str(path), pagesize=A4, invariant=True)
    c.setTitle(f"{s.bank_name} statement {s.period_start:%Y-%m}")
    c.setCreator("StatementScreen synthetic generator")
    _Renderer(s, LAYOUTS[layout], c).draw()
    c.save()
    return path
