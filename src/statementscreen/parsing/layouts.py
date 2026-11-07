"""Recognise a statement layout from its table-header words, and map columns by position.

A layout is described only by the *text* of its column headers, in order, and whether
each column's values line up with the header's left edge (text) or right edge (money).
No coordinates are hard-coded: column positions are read from the header words on
every page.

Detection looks for a line containing every header of exactly one layout. Anything
else (no match, or more than one) is an "unknown layout" error, never a guess.
"""

from dataclasses import dataclass

from statementscreen.parsing.words import Line, Word

POSITION_TOLERANCE = 3.0  # points of slack when lining a value up with its header


class UnknownLayoutError(ValueError):
    def __init__(self, detail: str = "") -> None:
        super().__init__("unknown layout" + (f": {detail}" if detail else ""))


@dataclass(frozen=True)
class ColumnSpec:
    key: str  # what the column holds: date, narration, ref, value_date, debit, credit, ...
    header: str  # header text as printed, e.g. "Withdrawal Amt."
    align: str  # "left" (text starts at the header's left edge) or "right" (money)


@dataclass(frozen=True)
class LayoutSpec:
    name: str
    columns: tuple[ColumnSpec, ...]


LAYOUTS: dict[str, LayoutSpec] = {
    "columns": LayoutSpec(
        "columns",
        (
            ColumnSpec("date", "Date", "left"),
            ColumnSpec("narration", "Narration", "left"),
            ColumnSpec("ref", "Chq./Ref.No.", "left"),
            ColumnSpec("value_date", "Value Dt", "left"),
            ColumnSpec("debit", "Withdrawal Amt.", "right"),
            ColumnSpec("credit", "Deposit Amt.", "right"),
            ColumnSpec("balance", "Closing Balance", "right"),
        ),
    ),
    "drcr": LayoutSpec(
        "drcr",
        (
            ColumnSpec("date", "Txn Date", "left"),
            ColumnSpec("narration", "Description", "left"),
            ColumnSpec("amount", "Amount (INR)", "right"),
            ColumnSpec("balance", "Balance (INR)", "right"),
        ),
    ),
}


@dataclass(frozen=True)
class Span:
    column: ColumnSpec
    x0: float  # left edge of the header text
    x1: float  # right edge of the header text


@dataclass(frozen=True)
class Geometry:
    """Where each column is on one page, read from that page's header line."""

    spans: tuple[Span, ...]
    header_top: float

    def column_for(self, word: Word) -> str | None:
        """Which column a word belongs to, or None if it lines up with none."""
        # Money columns are right-aligned: the value ends where the header ends.
        for s in self.spans:
            if s.column.align == "right" and abs(word.x1 - s.x1) <= POSITION_TOLERANCE:
                return s.column.key
        # Text columns run from their header's left edge to the next column's left edge.
        ordered = sorted(self.spans, key=lambda s: s.x0)
        for here, nxt in zip(ordered, [*ordered[1:], None], strict=True):
            if here.column.align != "left":
                continue
            right = nxt.x0 if nxt is not None else float("inf")
            if here.x0 - POSITION_TOLERANCE <= word.x0 < right:
                return here.column.key
        return None


def _match_phrase(line: Line, start: int, phrase: str) -> int | None:
    """Index just past ``phrase`` if the line's words from ``start`` spell it, else None."""
    tokens = phrase.split()
    for i in range(start, len(line) - len(tokens) + 1):
        if [w.text for w in line[i : i + len(tokens)]] == tokens:
            return i + len(tokens)
    return None


def header_geometry(line: Line, spec: LayoutSpec) -> Geometry | None:
    """If ``line`` holds every header of ``spec`` in order, where each one sits."""
    spans, pos = [], 0
    for column in spec.columns:
        end = _match_phrase(line, pos, column.header)
        if end is None:
            return None
        n = len(column.header.split())
        spans.append(Span(column, line[end - n].x0, line[end - 1].x1))
        pos = end
    return Geometry(tuple(spans), line[0].top)


def find_header(lines: list[Line], spec: LayoutSpec) -> tuple[int, Geometry] | None:
    """The first line on the page that is ``spec``'s table header, with its geometry."""
    for i, line in enumerate(lines):
        geometry = header_geometry(line, spec)
        if geometry is not None:
            return i, geometry
    return None


def detect_layout(lines: list[Line]) -> str:
    """The one layout whose full set of column headers appears on this page."""
    matches = [name for name, spec in LAYOUTS.items() if find_header(lines, spec)]
    if len(matches) != 1:
        found = "none" if not matches else ", ".join(matches)
        raise UnknownLayoutError(f"table headers matched {found}")
    return matches[0]
