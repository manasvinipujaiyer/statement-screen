"""Font and layout findings, from pdfplumber's character data.

Reading (read_layout): for every page, the money cells in the transaction table and
every pair of visible characters drawn on top of each other.

- Columns are found from the page's own table header, with the parser's layout specs
  (header text only, no coordinates); nothing comes from the generator. A money cell is
  a word whose right edge lines up with a right-aligned (money) column, plus the amount
  just before it when that word is a Dr/Cr marker.
- What is "usual" for a column (its font, size and right edge) is taken from the same
  document: the majority of its cells. Nothing is compared with a known generator.

Findings (layout_findings), with tolerances in config.py (measured on genuine train):
    AMOUNT_FONT       strong  a money cell's characters use a font or size other than
                              the column's usual one in this document
    CHAR_OVERLAP      strong  two visible characters on a transaction-table line overlap
                              by more than a share of the narrower one's width (text drawn
                              over text); weak outside the table (header, summary, footer),
                              where genuine drcr headers can collide (a long address)
    AMOUNT_ALIGNMENT  strong  a money cell's right edge is off the column's usual edge
    DIGIT_SPACING     weak    the gaps between the characters of one number are uneven
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pdfplumber

from statementscreen.parsing.layouts import LAYOUTS, UnknownLayoutError, detect_layout, find_header
from statementscreen.parsing.values import is_amount
from statementscreen.parsing.words import (
    LINE_TOLERANCE,
    Line,
    Word,
    group_lines,
    is_footer,
    text_of,
    to_words,
)
from statementscreen.tamper.detect import config
from statementscreen.tamper.detect.findings import Finding, where

MARKERS = ("Dr", "Cr")
MARKER_GAP = 4.0  # points: a Dr/Cr this close after an amount belongs to it (as the parser)
NUMBER_CHARS = set("0123456789,.-")
RECORD_OVERLAP = 0.05  # pairs overlapping more than this share are kept for reporting


@dataclass(frozen=True)
class Char:
    text: str
    font: str
    size: float
    x0: float
    x1: float


@dataclass(frozen=True)
class Cell:
    page: int
    column: str  # the parser's column key: debit, credit, amount or balance
    text: str
    right: float  # right edge of the cell (points from the left)
    chars: tuple[Char, ...]

    @property
    def number_chars(self) -> tuple[Char, ...]:
        """The characters of the amount itself (no Dr/Cr marker, no spaces)."""
        return tuple(c for c in self.chars if c.text in NUMBER_CHARS)


@dataclass(frozen=True)
class Overlap:
    page: int
    top: float
    a: str
    b: str
    share: float  # overlap width / the narrower character's width
    x: float  # where the overlap starts (points from the left)
    in_table: bool  # on a line of the transaction table (below its header, above the
    #                 footer or summary)


@dataclass
class LayoutRead:
    layout: str | None
    pages: int = 0
    cells: list[Cell] = field(default_factory=list)
    overlaps: list[Overlap] = field(default_factory=list)  # share > RECORD_OVERLAP
    max_overlap: float = 0.0  # largest share seen on any pair
    problem: str | None = None


# ---------------------------------------------------------------- reading


def _table_lines(lines: list[Line], start: int) -> list[Line]:
    """The lines below the table header at `start`, up to the footer or the summary."""
    out = []
    for line in lines[start + 1 :]:
        if is_footer(line) or text_of(line).startswith("Statement Summary"):
            break
        out.append(line)
    return out


def _cell_words(line: Line, k: int) -> list[Word]:
    """The aligned word at index k, plus the amount before it if it's a Dr/Cr marker."""
    w = line[k]
    if w.text in MARKERS and k > 0:
        prev = line[k - 1]
        if is_amount(prev.text) and w.x0 - prev.x1 <= MARKER_GAP:
            return [prev, w]
    return [w]


def _chars_in(chars: list[dict], words: list[Word]) -> tuple[Char, ...]:
    x0, x1, top = words[0].x0, words[-1].x1, words[0].top
    return tuple(
        Char(c["text"], str(c["fontname"]), round(float(c["size"]), 2), float(c["x0"]),
             float(c["x1"]))
        for c in sorted(chars, key=lambda c: c["x0"])
        if c["text"].strip() and abs(float(c["top"]) - top) <= LINE_TOLERANCE
        and float(c["x0"]) >= x0 - 0.5 and float(c["x1"]) <= x1 + 0.5
    )  # fmt: skip


def _overlaps(page_no: int, chars: list[dict],
              table_tops: list[float]) -> tuple[list[Overlap], float]:  # fmt: skip
    """Pairs of visible characters on one line whose boxes overlap horizontally.
    table_tops: the tops of this page's transaction-table lines."""
    visible = sorted((c for c in chars if c["text"].strip()),
                     key=lambda c: (round(float(c["top"])), float(c["x0"])))  # fmt: skip
    lines: list[list[dict]] = []
    for c in visible:
        if lines and abs(float(c["top"]) - float(lines[-1][0]["top"])) <= 1.0:
            lines[-1].append(c)
        else:
            lines.append([c])
    found, worst = [], 0.0
    for line in lines:
        line.sort(key=lambda c: float(c["x0"]))
        for i, a in enumerate(line):
            for b in line[i + 1 :]:
                if float(b["x0"]) >= float(a["x1"]):
                    break
                width = min(float(a["x1"]) - float(a["x0"]), float(b["x1"]) - float(b["x0"]))
                if width <= 0:
                    continue
                share = (min(float(a["x1"]), float(b["x1"])) - float(b["x0"])) / width
                worst = max(worst, share)
                if share > RECORD_OVERLAP:
                    top = float(a["top"])
                    in_table = any(abs(top - t) <= LINE_TOLERANCE for t in table_tops)
                    found.append(Overlap(page_no, round(top, 2), a["text"], b["text"],
                                         round(share, 4), round(float(b["x0"]), 2),
                                         in_table))  # fmt: skip
    return found, worst


def read_layout(pdf_path: Path) -> LayoutRead:
    """Money cells and overlapping characters of one PDF, one page at a time."""
    out = LayoutRead(layout=None)
    with pdfplumber.open(pdf_path) as pdf:
        out.pages = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            try:
                chars = list(page.chars)
                lines = group_lines(to_words(page.extract_words()))
            finally:
                page.close()
            if out.layout is None and page_no == 1:
                try:
                    out.layout = detect_layout(lines)
                except UnknownLayoutError as e:
                    out.problem = str(e)
            header = find_header(lines, LAYOUTS[out.layout]) if out.layout else None
            table = _table_lines(lines, header[0]) if header else []
            found, worst = _overlaps(page_no, chars, [line[0].top for line in table])
            out.overlaps += found
            out.max_overlap = max(out.max_overlap, worst)
            if header is None:
                continue  # unknown layout, or e.g. a page holding only the summary
            geometry = header[1]
            money = {s.column.key for s in geometry.spans if s.column.align == "right"}
            for line in table:
                for k, w in enumerate(line):
                    key = geometry.column_for(w)
                    if key not in money:
                        continue
                    words = _cell_words(line, k)
                    out.cells.append(Cell(page_no, key, text_of(words), w.x1,
                                          _chars_in(chars, words)))  # fmt: skip
    return out


# ---------------------------------------------------------------- measurements


def usual_font(cells: list[Cell]) -> dict[str, tuple[str, float]]:
    """Per column: the (font, size) most of its number characters use in this document."""
    counts: dict[str, Counter] = {}
    for cell in cells:
        for c in cell.number_chars:
            counts.setdefault(cell.column, Counter())[(c.font, c.size)] += 1
    return {col: cnt.most_common(1)[0][0] for col, cnt in counts.items()}


def usual_right(cells: list[Cell]) -> dict[str, float]:
    """Per column: the median right edge of its cells in this document."""
    edges: dict[str, list[float]] = {}
    for cell in cells:
        edges.setdefault(cell.column, []).append(cell.right)
    return {col: median(xs) for col, xs in edges.items()}


def spacing_spread(cell: Cell) -> float | None:
    """Largest minus smallest gap between neighbouring characters of the number."""
    chars = cell.number_chars
    if len(chars) < 3:
        return None
    gaps = [b.x0 - a.x1 for a, b in zip(chars, chars[1:], strict=False)]
    return max(gaps) - min(gaps)


def measure(read: LayoutRead) -> dict:
    """The largest values each finding would test, for setting tolerances on genuine PDFs."""
    fonts, rights = usual_font(read.cells), usual_right(read.cells)
    off_font = sum(any((c.font, c.size) != fonts[cell.column] for c in cell.number_chars)
                   for cell in read.cells)  # fmt: skip
    align = [abs(cell.right - rights[cell.column]) for cell in read.cells]
    spreads = [s for cell in read.cells if (s := spacing_spread(cell)) is not None]
    return {
        "layout": read.layout,
        "problem": read.problem,
        "cells": len(read.cells),
        "cells_off_usual_font": off_font,
        "max_alignment_off": max(align, default=0.0),
        "max_spacing_spread": max(spreads, default=0.0),
        "max_overlap_share": read.max_overlap,
        "overlaps_recorded": len(read.overlaps),
    }


# ---------------------------------------------------------------- findings


def _finding(code: str, strength: str, page: int | None, reason: str) -> Finding:
    return Finding(code, "layout", page, None, strength, f"{where(page, None)}{reason}")


def _overlap_findings(read: LayoutRead) -> list[Finding]:
    """One finding per line with characters drawn over others (share > OVERLAP_SHARE)."""
    by_line: dict[tuple[int, float], list[Overlap]] = {}
    for o in read.overlaps:
        if o.share > config.OVERLAP_SHARE:
            key = next((k for k in by_line if k[0] == o.page
                        and abs(k[1] - o.top) <= LINE_TOLERANCE), (o.page, o.top))  # fmt: skip
            by_line.setdefault(key, []).append(o)
    out = []
    for (page, _), pairs in sorted(by_line.items()):
        in_table = any(o.in_table for o in pairs)
        shown = ", ".join(f"'{o.a}' and '{o.b}'" for o in pairs[:4])
        more = f" and {len(pairs) - 4} more" if len(pairs) > 4 else ""
        place = ("on a transaction line" if in_table
                 else "outside the transaction table (header, summary or footer)")  # fmt: skip
        out.append(_finding(
            "CHAR_OVERLAP", "strong" if in_table else "weak", page,
            f"text drawn over text {place} near x={pairs[0].x:.1f}: {shown}{more} "
            f"occupy the same place",
        ))  # fmt: skip
    return out


def layout_findings(read: LayoutRead) -> list[Finding]:
    """Font, alignment, spacing and overlap findings for one PDF's layout read."""
    out = _overlap_findings(read)
    if read.layout is None:
        return out + [Finding("CHECK_NOT_RUN", "layout", None, None, "info",
                              "AMOUNT_FONT, AMOUNT_ALIGNMENT, DIGIT_SPACING: layout not "
                              f"recognised ({read.problem})")]  # fmt: skip
    fonts, rights = usual_font(read.cells), usual_right(read.cells)
    for cell in read.cells:
        if cell.column not in fonts:  # the column had no number characters at all
            continue
        odd = sorted({(c.font, c.size) for c in cell.number_chars} - {fonts[cell.column]})
        if odd:
            usual = f"{fonts[cell.column][0]} {fonts[cell.column][1]}"
            used = ", ".join(f"{f} {s}" for f, s in odd)
            out.append(_finding("AMOUNT_FONT", "strong", cell.page,
                                f"{cell.column} '{cell.text}' uses {used}; the column's usual "
                                f"font is {usual}"))  # fmt: skip
        off = cell.right - rights[cell.column]
        if abs(off) > config.ALIGNMENT_TOLERANCE_PT:
            side = "right" if off > 0 else "left"
            out.append(_finding("AMOUNT_ALIGNMENT", "strong", cell.page,
                                f"{cell.column} '{cell.text}' ends {abs(off):.2f} pt "
                                f"{side} of the column's edge"))  # fmt: skip
        spread = spacing_spread(cell)
        if spread is not None and spread > config.SPACING_TOLERANCE_PT:
            out.append(_finding("DIGIT_SPACING", "weak", cell.page,
                                f"{cell.column} '{cell.text}': gaps between its characters "
                                f"differ by {spread:.2f} pt"))  # fmt: skip
    return out
