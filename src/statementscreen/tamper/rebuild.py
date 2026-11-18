"""Read a synthetic statement PDF back into what the renderer needs to draw it again.

Used only by the tamper generator (to re-render a statement with an edit). The detector
never uses it.

  rows      dates, amounts and balances from the Week 2 parser (100% exact rows)
  printed   each row's narration lines and reference, and the address lines, exactly as
            printed, never from the parser's joined narration (which drops the space at
            a wrap; drcr narrations are cut)
  header    holder, account number, customer ID, branch, IFSC, MICR, account type and
            period, read at our two layouts' fixed positions; the bank name from the PDF's
            Title (the drcr layout prints it in capitals)

Text is read as RUNS: each run is one string the renderer drew, rebuilt from pdfplumber's
characters in drawing order (a run ends where the baseline changes or the next character
doesn't start where the previous one ended). A run's start is its first character's
text-matrix origin, the very numbers the renderer drew it at. Runs never mix two strings,
even where one overlaps another's column. Rendering the result with no edit must give a
byte-identical PDF (rebuild_check checks this on every dev PDF).
"""

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pdfplumber

from statementscreen.parsing.models import ParsedRow
from statementscreen.parsing.pdf import parse_pdf
from statementscreen.synth.models import IncomeTruth, Statement, Transaction
from statementscreen.synth.render import LAYOUTS, MARGIN, PAGE_H, PAGE_W, Printed

DETAIL_LABELS = ("Account No", "Account Type", "Customer ID", "Branch", "IFSC", "MICR",
                 "Currency", "Statement Period")  # fmt: skip
DATE_PATTERNS = {"columns": re.compile(r"\d{2}/\d{2}/\d{4}"),
                 "drcr": re.compile(r"\d{2}-[A-Z][a-z]{2}-\d{2}")}  # fmt: skip
LEFT, RIGHT, TOP = MARGIN, PAGE_W - MARGIN, PAGE_H - MARGIN
NARRATION_X = {"columns": LEFT + 50, "drcr": LEFT + 52}  # where narration lines start
REF_X = LEFT + 232
FOOTER_Y = 50  # the footer is drawn at y = 28; table rows are always above this
TOL = 0.01


class RebuildError(ValueError):
    """The PDF can't be read back exactly; nothing is guessed."""


@dataclass(frozen=True)
class Run:
    x: float  # start (text-matrix origin), PDF points
    y: float  # baseline, PDF points
    text: str
    font: str


@dataclass
class ReadBack:
    statement: Statement
    layout: str
    printed: Printed
    rows: list[ParsedRow]  # the parser's rows (with the page each starts on)
    pages: int


# ---------------------------------------------------------------- runs


def runs(page) -> list[Run]:
    """The page's drawn strings, in drawing order."""
    out: list[Run] = []
    text: list[str] = []
    start = prev = None
    for ch in page.chars:
        new_line = prev is not None and abs(ch["matrix"][5] - prev["matrix"][5]) > TOL
        if prev is not None and (new_line or abs(ch["x0"] - prev["x1"]) > TOL):
            out.append(Run(start["matrix"][4], start["matrix"][5], "".join(text),
                           start["fontname"]))  # fmt: skip
            text, start = [], None
        if start is None:
            start = ch
        text.append(ch["text"])
        prev = ch
    if start is not None:
        out.append(Run(start["matrix"][4], start["matrix"][5], "".join(text),
                       start["fontname"]))  # fmt: skip
    return out


def _at(found: list[Run], x: float, y: float) -> Run | None:
    hits = [r for r in found if abs(r.x - x) <= TOL and abs(r.y - y) <= TOL]
    if len(hits) > 1:
        raise RebuildError(f"{len(hits)} strings start at ({x:.2f}, {y:.2f})")
    return hits[0] if hits else None


def _value(run: Run | None, label: str) -> str:
    if run is None or not run.text.startswith(": "):
        raise RebuildError(f"{label}: expected ': value', got {run and run.text!r}")
    return run.text[2:]


# ---------------------------------------------------------------- header


def _header_columns(found: list[Run]) -> tuple[str, list[str], dict[str, str]]:
    holder = _at(found, LEFT, TOP - 52)
    if holder is None:
        raise RebuildError("no account holder line")
    address = []
    for i in range(3):
        line = _at(found, LEFT, TOP - 64 - i * 10)
        if line is None:
            break
        address.append(line.text)
    details = {}
    for i, label in enumerate(DETAIL_LABELS):
        y = TOP - 10 - i * 11
        seen = _at(found, RIGHT - 210, y)
        if seen is None or seen.text != label:
            raise RebuildError(f"expected header label {label!r}, found {seen}")
        details[label] = _value(_at(found, RIGHT - 130, y), label)
    return holder.text, address, details


def _header_drcr(found: list[Run]) -> tuple[str, list[str], dict[str, str]]:
    mid = PAGE_W / 2
    pairs: list[tuple[str, str]] = []
    for x in (LEFT, mid + 10):  # the left column, then the right one
        for i in range(7):
            y = TOP - 52 - i * 12
            label = _at(found, x, y)
            if label is not None:
                pairs.append((label.text, _value(_at(found, x + 78, y), label.text)))
    holder = next((v for k, v in pairs if k == "Name"), None)
    if holder is None:
        raise RebuildError("no 'Name' in the drcr header")
    details = {k: v for k, v in pairs if k in DETAIL_LABELS}
    missing = [label for label in DETAIL_LABELS if label not in details]
    if missing:
        raise RebuildError(f"header labels not found: {missing}")
    return holder, [v for k, v in pairs if k == "Address"], details


# ---------------------------------------------------------------- table


def _printed_rows(pages_runs: list[list[Run]], layout: str,
                  rows: list[ParsedRow]) -> tuple[list[tuple[str, ...]], list[str]]:  # fmt: skip
    """Per row: its printed narration line(s) and its printed reference."""
    lay = LAYOUTS[layout]
    pattern = DATE_PATTERNS[layout]
    narrations: list[tuple[str, ...]] = []
    refs: list[str] = []
    for page_no, found in enumerate(pages_runs, start=1):
        dated = sorted((r.y for r in found if abs(r.x - LEFT) <= TOL and r.y > FOOTER_Y
                        and pattern.fullmatch(r.text)), reverse=True)  # fmt: skip
        page_rows = [r for r in rows if r.page == page_no]
        if len(dated) != len(page_rows):
            raise RebuildError(f"page {page_no}: {len(dated)} dated lines but "
                               f"{len(page_rows)} parsed rows")  # fmt: skip
        lines = sorted((r for r in found if abs(r.x - NARRATION_X[layout]) <= TOL),
                       key=lambda r: -r.y)  # fmt: skip
        for k, y in enumerate(dated):
            below = dated[k + 1] if k + 1 < len(dated) else FOOTER_Y
            mine = [r.text for r in lines if below < r.y <= y + TOL]
            if lay.narration_lines == 1:
                mine = [r.text for r in lines if abs(r.y - y) <= TOL]
            if len(mine) > lay.narration_lines:
                raise RebuildError(f"page {page_no}: a row with {len(mine)} narration lines")
            narrations.append(tuple(mine) or ("",))
            ref = _at(found, REF_X, y) if layout == "columns" else None
            refs.append(ref.text if ref else "")
    return narrations, refs


# ---------------------------------------------------------------- the statement


def _paise(p: int) -> Decimal:
    return Decimal(p) / 100


def read_back(pdf_path: Path) -> ReadBack:
    parsed = parse_pdf(Path(pdf_path))
    if parsed.status != "ok" or parsed.layout not in LAYOUTS:
        raise RebuildError(f"{pdf_path}: not parsed ({parsed.error or parsed.layout})")
    if parsed.header.opening_balance_paise is None:
        raise RebuildError(f"{pdf_path}: no opening balance")
    layout = parsed.layout
    lay = LAYOUTS[layout]
    with pdfplumber.open(pdf_path) as pdf:
        title = (pdf.metadata or {}).get("Title", "")
        pages_runs = [runs(page) for page in pdf.pages]
    header = _header_columns if layout == "columns" else _header_drcr
    holder, address, details = header(pages_runs[0])
    narrations, refs = _printed_rows(pages_runs, layout, parsed.rows)
    m = re.fullmatch(r"(.+) statement \d{4}-\d{2}", title)
    if not m:
        raise RebuildError(f"{pdf_path}: unexpected Title {title!r}")
    start_text, _, end_text = details["Statement Period"].partition(" to ")
    transactions = [
        Transaction(r.posted, " ".join(lines), _paise(r.amount_paise), _paise(r.balance_paise),
                    "unknown")
        for r, lines in zip(parsed.rows, narrations, strict=True)
    ]  # fmt: skip
    statement = Statement(
        bank_name=m.group(1), account_holder=holder, account_number=details["Account No"],
        period_start=datetime.strptime(start_text, lay.date_fmt).date(),
        period_end=datetime.strptime(end_text, lay.date_fmt).date(),
        opening_balance=_paise(parsed.header.opening_balance_paise),
        income=IncomeTruth("unknown", (), "unknown", None, Decimal(0)),
        transactions=transactions, ifsc=details["IFSC"], micr=details["MICR"],
        branch=details["Branch"], customer_id=details["Customer ID"],
        address=" ".join(address), account_type=details["Account Type"],
    )  # fmt: skip
    printed = Printed(tuple(narrations), tuple(refs) if layout == "columns" else None,
                      tuple(address))  # fmt: skip
    return ReadBack(statement, layout, printed, parsed.rows, len(pages_runs))
