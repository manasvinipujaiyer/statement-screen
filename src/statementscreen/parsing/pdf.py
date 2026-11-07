"""Parse one statement PDF into rows, header fields and issues, from the PDF alone.

For each page (opened, read and closed one at a time):

1. Read the words and group them into lines by their vertical position.
2. Find this page's table header line; its words give the column positions.
3. The table is everything below the header, up to the footer ("SYNTHETIC DOCUMENT ...
   Page X of Y") or a "Statement Summary" line. Everything else (running header,
   account details, summary, footer) is outside the table.
4. Inside the table, a line with a date in the date column starts a new transaction;
   a line with words only in the narration column continues the previous narration;
   any other line is reported as ORPHAN_LINE.

Header fields come from page 1 and summary fields from page 1 or the last page, read by
their labels. Anything the parser can't place or read becomes an Issue, never a guess.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pdfplumber

from statementscreen.parsing.layouts import (
    LAYOUTS,
    Geometry,
    LayoutSpec,
    UnknownLayoutError,
    detect_layout,
    find_header,
)
from statementscreen.parsing.models import ParsedRow, ParsedStatement, StatementHeader
from statementscreen.parsing.values import (
    is_dmy,
    parse_amount,
    parse_dmy,
    parse_dmy_short,
    to_paise,
)
from statementscreen.parsing.words import Line, Word, group_lines, is_footer, text_of, to_words

VALUE_GAP = 12.0  # points: a wider gap between words ends a "Label : value" value

SUMMARY_LABELS = {
    "Opening Balance": "opening_balance_paise",
    "Dr Count": "dr_count",
    "Cr Count": "cr_count",
    "Debits": "total_debits_paise",
    "Credits": "total_credits_paise",
    "Closing Bal": "closing_balance_paise",
}


class LayoutNotSupportedError(ValueError):
    pass


@dataclass
class _RowDraft:
    """A transaction being assembled: its first line plus any continuation lines."""

    page: int
    posted: date
    value_date: date | None
    ref: str | None
    narration: str
    amount_paise: int
    balance_paise: int
    extra: list[str] = field(default_factory=list)


def _join_narration(head: str, more: list[str]) -> str:
    """Join wrapped narration lines. The renderer breaks after '/' or '-' (keep no space)
    or at a space (which it drops, so put one back). A mid-word break gains a space,
    which is why scoring also compares narrations with whitespace removed."""
    text = head
    for part in more:
        text = text + part if text.endswith(("/", "-")) else f"{text} {part}"
    return text


# ---------------------------------------------------------------------- page structure


def _split_page(lines: list[Line], spec: LayoutSpec):
    """(geometry, table lines, other lines) for one page. No header -> no table."""
    found = find_header(lines, spec)
    if found is None:
        return None, [], lines
    i, geometry = found
    rest = lines[i + 1 :]
    stop = next(
        (k for k, line in enumerate(rest)
         if is_footer(line) or text_of(line).startswith("Statement Summary")),
        len(rest),
    )  # fmt: skip
    return geometry, rest[:stop], lines[:i] + rest[stop:]


# ---------------------------------------------------------------------- columns rows


def _cells(line: Line, geometry: Geometry) -> tuple[dict[str, list[Word]], list[Word]]:
    cells: dict[str, list[Word]] = {}
    unplaced: list[Word] = []
    for w in line:
        key = geometry.column_for(w)
        (cells.setdefault(key, []) if key else unplaced).append(w)
    return cells, unplaced


def _start_row_columns(cells: dict[str, list[Word]], page: int) -> _RowDraft:
    """Read a 'columns' row's first line. Raises ValueError if a value can't be read."""
    posted = parse_dmy(text_of(cells["date"]))
    value_text = text_of(cells.get("value_date", []))
    value_date = parse_dmy_short(value_text, near=posted) if value_text else None
    debit = parse_amount(text_of(cells.get("debit", [])))
    credit = parse_amount(text_of(cells.get("credit", [])))
    balance = parse_amount(text_of(cells.get("balance", [])))
    if (debit is None) == (credit is None):
        raise ValueError("expected exactly one of withdrawal or deposit")
    if balance is None:
        raise ValueError("missing closing balance")
    amount = credit if credit is not None else -debit
    return _RowDraft(
        page=page,
        posted=posted,
        value_date=value_date,
        ref=text_of(cells["ref"]) if cells.get("ref") else None,
        narration=text_of(cells.get("narration", [])),
        amount_paise=to_paise(amount),
        balance_paise=to_paise(balance),
    )


def _is_row_start(cells: dict[str, list[Word]], spec: LayoutSpec) -> bool:
    date_words = cells.get("date")
    return bool(date_words) and len(date_words) == 1 and is_dmy(date_words[0].text)


# ---------------------------------------------------------------------- header fields


def _label_value(lines: list[Line], label: str) -> str | None:
    """The value printed after 'label :' on the same line (stops at a wide gap)."""
    tokens = label.split()
    for line in lines:
        texts = [w.text for w in line]
        for i in range(len(texts) - len(tokens)):
            if texts[i : i + len(tokens)] == tokens and texts[i + len(tokens)] == ":":
                value = line[i + len(tokens) + 1 :]
                if not value:
                    return None
                out = [value[0]]
                for w in value[1:]:
                    if w.x0 - out[-1].x1 > VALUE_GAP:
                        break
                    out.append(w)
                return text_of(out)
    return None


def _summary_values(lines: list[Line]) -> dict[str, str]:
    """Printed summary values by label: on the same line to the right of the label, or
    else on the next line, under the label (closest centre)."""
    found: dict[str, str] = {}
    for li, line in enumerate(lines):
        texts = [w.text for w in line]
        for label in SUMMARY_LABELS:
            tokens = label.split()
            for i in range(len(texts) - len(tokens) + 1):
                if texts[i : i + len(tokens)] != tokens or label in found:
                    continue
                label_words = line[i : i + len(tokens)]
                right = line[i + len(tokens) :]
                if right and re.fullmatch(r"[\d,.-]+", right[0].text):
                    found[label] = right[0].text
                elif li + 1 < len(lines):
                    centre = (label_words[0].x0 + label_words[-1].x1) / 2
                    below = min(lines[li + 1], key=lambda w: abs(w.center - centre))
                    found[label] = below.text
    return found


def _read_header(lines: list[Line], summary_lines: list[Line], result: ParsedStatement):
    h = StatementHeader()
    h.account_number = _label_value(lines, "Account No")
    period = _label_value(lines, "Statement Period")
    if period and " to " in period:
        start, end = period.split(" to ", 1)
        try:
            h.period_start, h.period_end = parse_dmy(start), parse_dmy(end)
        except ValueError as e:
            result.issue("BAD_DATE", 1, None, f"statement period {period!r}: {e}")
    values = _summary_values(summary_lines)
    for label, attr in SUMMARY_LABELS.items():
        text = values.get(label)
        if text is None:
            continue
        try:
            if attr.endswith("_count"):
                setattr(h, attr, int(text))
            else:
                setattr(h, attr, to_paise(parse_amount(text)))
        except ValueError as e:
            result.issue("BAD_AMOUNT", None, None, f"summary {label} {text!r}: {e}")
    result.header = h


# ---------------------------------------------------------------------- public API


def parse_pdf(path: str | Path, force_layout: str | None = None) -> ParsedStatement:
    """Parse one statement PDF. ``force_layout`` skips detection (used for first_contact)."""
    path = Path(path)
    sid = path.stem
    try:
        pdf = pdfplumber.open(path)
    except Exception as e:  # any failure to open is reported, not raised
        result = ParsedStatement(sid, None, 0, "error", f"unreadable PDF: {e}")
        result.issue("UNREADABLE_PDF", None, None, str(e))
        return result

    with pdf:
        result = ParsedStatement(sid, None, len(pdf.pages), "ok")
        spec: LayoutSpec | None = None
        first_page_other: list[Line] = []
        last_page_other: list[Line] = []
        drafts: list[_RowDraft] = []

        for page_no, page in enumerate(pdf.pages, start=1):
            try:
                lines = group_lines(to_words(page.extract_words()))
            except Exception as e:
                result.issue("UNREADABLE_PAGE", page_no, None, str(e))
                continue
            finally:
                page.close()  # one page at a time: free its cached objects now

            if spec is None:
                try:
                    name = force_layout or detect_layout(lines)
                except UnknownLayoutError as e:
                    result.status, result.error = "error", str(e)
                    result.issue("UNKNOWN_LAYOUT", page_no, None, str(e))
                    return result
                if name != "columns":
                    result.layout = name
                    result.status, result.error = "error", f"no parser for layout {name!r} yet"
                    result.issue("LAYOUT_NOT_SUPPORTED", page_no, None, result.error)
                    return result
                spec, result.layout = LAYOUTS[name], name

            geometry, table, other = _split_page(lines, spec)
            # Keep only what's outside the table on page 1 (header, maybe a summary box)
            # and on the latest page (an end-of-statement summary, if the layout has one).
            if page_no == 1:
                first_page_other = other
            last_page_other = other
            if geometry is None:
                continue

            for line in table:
                cells, unplaced = _cells(line, geometry)
                if _is_row_start(cells, spec):
                    try:
                        draft = _start_row_columns(cells, page_no)
                    except (ValueError, KeyError) as e:
                        result.issue("BAD_ROW", page_no, len(drafts), f"{e}: {text_of(line)!r}")
                        continue
                    drafts.append(draft)
                    for w in unplaced:
                        result.issue("UNPLACED_WORD", page_no, len(drafts) - 1, w.text)
                elif set(cells) == {"narration"} and not unplaced and drafts:
                    drafts[-1].extra.append(text_of(cells["narration"]))
                else:
                    result.issue("ORPHAN_LINE", page_no, len(drafts), text_of(line))

        # Summary fields can be on page 1 (summary box) or the last page (summary block).
        summary_lines = first_page_other + (last_page_other if result.pages > 1 else [])
        _read_header(first_page_other, summary_lines, result)

    result.rows = [
        ParsedRow(sid, i, d.posted, d.value_date, d.ref, _join_narration(d.narration, d.extra),
                  d.amount_paise, d.balance_paise, d.page, result.layout or "")
        for i, d in enumerate(drafts)
    ]  # fmt: skip
    return result
