"""Make one tampered copy of a synthetic statement PDF, with its answer key.

    pdf_bytes, key = tamper_copy(path, "inplace_edit", seed=7)

Independent of any split or batch folder: it reads only the PDF (and, for page_swap,
the donor PDF). Every random choice (row, digit, factor, page, incremental or full save)
comes from numpy's default_rng([seed, type number]), so the same call gives the same copy.

Types (TYPES):
  rerender_inconsistent   one row's amount x1.1-3; re-rendered (rebuild.read_back + the
                          project's renderer); balances kept, so they no longer add up
  rerender_consistent     every salary credit (found by the Week 3 rules on the parsed
                          rows) x one factor of 1.3-2; every later balance and the summary
                          recalculated: the arithmetic is correct
  inplace_edit            one digit of one amount changed in the page's content stream
                          (same length, so same width and alignment); balances untouched
  inplace_edit_consistent as above, plus every later balance on that page and the
                          summary's totals (x moved to keep right or centre alignment).
                          "Consistent" means consistent where the page allows: only that
                          page's balances are rewritten, so when a page follows the edited
                          one, its first balance no longer follows from this page's last
                          (a PAGE_CARRYOVER break). Only an edit on the last page leaves
                          the whole statement's arithmetic correct. The key's changes list
                          every balance rewritten, so which case a copy is can be read off
  page_swap               one middle page replaced by the same page of a donor statement
  page_remove             one middle page deleted
  overlay (HELD OUT)      a white box over one amount and new text drawn on top; refused
                          unless allow_held_out=True (only the Week 6 step 6 command)
  resaved_genuine         control: re-saved through pypdf with no change

Every copy written by pypdf (all but the re-renders) is saved incrementally or as a full
rewrite, chosen by the seed (about half each) and recorded in the key, and is stamped
Producer = pypdf as a re-saving tool would (an unchanged incremental save would
otherwise append nothing). No modification date is added. The answer key lists the
source, type, seed, pages changed and every string changed (page, row, field, old and
new); the detector never reads it.
"""

import hashlib
import io
import re
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pdfplumber
import polars as pl
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, TextStringObject
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from statementscreen.parsing.pdf import parse_pdf
from statementscreen.patterns.cli import process_statement
from statementscreen.synth.render import LAYOUTS, MARGIN, PAGE_H, PAGE_W, inr, render_statement
from statementscreen.tamper.rebuild import DATE_PATTERNS, ReadBack, read_back, runs

TYPES = (
    "rerender_inconsistent",
    "rerender_consistent",
    "inplace_edit",
    "inplace_edit_consistent",
    "page_swap",
    "page_remove",
    "overlay",
    "resaved_genuine",
)
HELD_OUT = "overlay"
CONTROL = "resaved_genuine"
# What each type needs from a source statement (the set builder records this).
APPLIES_TO = {
    "rerender_inconsistent": "any statement",
    "rerender_consistent": "a statement where the Week 3 rules find a salary",
    "inplace_edit": "any statement",
    "inplace_edit_consistent": "any statement",
    "page_swap": "3+ pages, and a donor of the same layout that has the chosen page",
    "page_remove": "3+ pages",
    "overlay": "any statement",
    "resaved_genuine": "any statement",
}
LEFT = MARGIN
# Right edge of each money column, per layout (the renderer draws them right-aligned).
RIGHT_EDGE = {"columns": {"debit": LEFT + 392, "credit": LEFT + 458,
                          "balance": PAGE_W - MARGIN},
              "drcr": {"amount": LEFT + 430, "balance": PAGE_W - MARGIN}}  # fmt: skip
TOTAL_LABELS = ("Opening Balance", "Dr Count", "Cr Count", "Debits", "Credits", "Closing Bal")
TOL = 0.01


class NotApplicable(ValueError):
    """This type can't be applied to this statement (see APPLIES_TO)."""


class HeldOut(PermissionError):
    """Overlay copies are made only by the one-time Week 6 step 6 command."""


# ---------------------------------------------------------------- small helpers


def _rng(seed: int, tamper_type: str) -> np.random.Generator:
    return np.random.default_rng([seed, TYPES.index(tamper_type)])


def _money(paise: int) -> Decimal:
    return Decimal(paise) / 100


def _round(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def cell_text(layout: str, field: str, amount: Decimal) -> str:
    """A money cell exactly as the renderer prints it."""
    if field in ("debit", "credit"):
        return inr(abs(amount))
    if field == "amount":
        return f"{inr(abs(amount))} {'Cr' if amount > 0 else 'Dr'}"
    if field == "balance":
        return inr(amount) + (" Cr" if layout == "drcr" else "")
    raise ValueError(field)


def amount_field(layout: str, amount: Decimal) -> str:
    if layout == "drcr":
        return "amount"
    return "credit" if amount > 0 else "debit"


def change_one_digit(text: str, rng: np.random.Generator) -> str:
    """One digit replaced by a different one; same length; never a leading zero."""
    positions = [i for i, ch in enumerate(text) if ch.isdigit()]
    i = positions[int(rng.integers(len(positions)))]
    choices = [d for d in "0123456789" if d != text[i] and not (d == "0" and i == positions[0])]
    return text[:i] + str(rng.choice(choices)) + text[i + 1 :]


def _parse_money(text: str) -> Decimal:
    return Decimal(re.sub(r"[^\d.\-]", "", text))


def _row_lines(pdf_path: Path, layout: str) -> list[tuple[int, float]]:
    """(page, baseline y) of every transaction row, in order."""
    pattern = DATE_PATTERNS[layout]
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for n, page in enumerate(pdf.pages, start=1):
            ys = sorted((r.y for r in runs(page) if abs(r.x - LEFT) <= TOL and r.y > 50
                         and pattern.fullmatch(r.text)), reverse=True)  # fmt: skip
            out += [(n, y) for y in ys]
    return out


def _key(pdf_path: Path, tamper_type: str, seed: int) -> dict:
    return {"source": Path(pdf_path).stem,
            "source_sha256": hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest(),
            "tamper_type": tamper_type, "seed": seed, "pages_changed": [], "changes": [],
            "incremental_save": None, "donor": None}  # fmt: skip


# ---------------------------------------------------------------- re-render


def _render(rb: ReadBack) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        return render_statement(rb.statement, Path(tmp) / "x.pdf", rb.layout,
                                rb.printed).read_bytes()  # fmt: skip


def _totals(rb: ReadBack) -> dict[str, str]:
    t = rb.statement.transactions
    debits = -sum((x.amount for x in t if x.amount < 0), Decimal(0))
    credits = sum((x.amount for x in t if x.amount > 0), Decimal(0))
    return {"Debits": inr(debits), "Credits": inr(credits),
            "Closing Bal": inr(t[-1].balance if t else rb.statement.opening_balance)}  # fmt: skip


def _summary_page(rb: ReadBack) -> int:
    return 1 if LAYOUTS[rb.layout].summary == "box" else rb.pages


def _summary_changes(rb: ReadBack, before: dict[str, str], key: dict) -> None:
    after = _totals(rb)
    for label in ("Debits", "Credits", "Closing Bal"):
        if before[label] != after[label]:
            key["changes"].append({"page": _summary_page(rb), "row": None,
                                   "field": f"summary {label}", "old": before[label],
                                   "new": after[label]})  # fmt: skip


def _rerender_inconsistent(pdf_path: Path, rng, key: dict) -> bytes:
    rb = read_back(pdf_path)
    before = _totals(rb)
    i = int(rng.integers(len(rb.statement.transactions)))
    t = rb.statement.transactions[i]
    new = _round(t.amount * Decimal(str(round(rng.uniform(1.1, 3.0), 4))))
    field = amount_field(rb.layout, t.amount)
    rb.statement.transactions[i] = t.__class__(t.posted, t.description, new, t.balance,
                                               t.category)  # fmt: skip
    page = rb.rows[i].page
    key["changes"].append({"page": page, "row": i, "field": field,
                           "old": cell_text(rb.layout, field, t.amount),
                           "new": cell_text(rb.layout, field, new)})  # fmt: skip
    _summary_changes(rb, before, key)
    key["pages_changed"] = sorted({c["page"] for c in key["changes"]})
    return _render(rb)


def salary_rows(rb: ReadBack) -> list[int]:
    """Rows the Week 3 rules flag as salary, on the statement's parsed rows."""
    rows = pl.DataFrame({"statement_id": ["s"] * len(rb.rows),
                         "row": [r.row for r in rb.rows], "posted": [r.posted for r in rb.rows],
                         "narration": [r.narration for r in rb.rows],
                         "amount_paise": [r.amount_paise for r in rb.rows]})  # fmt: skip
    s = rb.statement
    flagged = process_statement(rows, s.period_start, s.period_end, rb.layout)["rows"]
    return sorted(flagged.filter(pl.col("is_salary"))["row"].to_list())


def _rerender_consistent(pdf_path: Path, rng, key: dict) -> bytes:
    rb = read_back(pdf_path)
    salary = salary_rows(rb)
    if not salary:
        raise NotApplicable("no salary found by the Week 3 rules")
    before = _totals(rb)
    factor = Decimal(str(round(rng.uniform(1.3, 2.0), 4)))
    key["factor"] = float(factor)
    tx = rb.statement.transactions
    balance = rb.statement.opening_balance
    for i, t in enumerate(tx):
        amount = _round(t.amount * factor) if i in salary else t.amount
        balance += amount
        if amount != t.amount:
            field = amount_field(rb.layout, t.amount)
            key["changes"].append({"page": rb.rows[i].page, "row": i, "field": field,
                                   "old": cell_text(rb.layout, field, t.amount),
                                   "new": cell_text(rb.layout, field, amount)})  # fmt: skip
        if balance != t.balance:
            key["changes"].append({"page": rb.rows[i].page, "row": i, "field": "balance",
                                   "old": cell_text(rb.layout, "balance", t.balance),
                                   "new": cell_text(rb.layout, "balance", balance)})  # fmt: skip
        tx[i] = t.__class__(t.posted, t.description, amount, balance, t.category)
    _summary_changes(rb, before, key)
    key["pages_changed"] = sorted({c["page"] for c in key["changes"]})
    return _render(rb)


# ---------------------------------------------------------------- content-stream edits


@dataclass
class Edit:
    """Replace the string `old` drawn at baseline y with `new`, keeping its alignment."""

    old: str
    new: str
    y: float | None  # None: any baseline (the drcr summary, whose y varies)
    anchor: float  # the right edge (align "right") or centre (align "centre")
    align: str
    font: str
    size: float


def _width(text: str, font: str, size: float) -> float:
    return stringWidth(text, font, size)


def _apply_edits(writer: PdfWriter, page_no: int, edits: list[Edit]) -> None:
    page = writer.pages[page_no - 1]
    content = page.get_contents()
    if not isinstance(content, ContentStream):
        content = ContentStream(content, writer)
    ops = content.operations
    done = [0] * len(edits)
    last_tm = None
    for k, (operands, operator) in enumerate(ops):
        if operator == b"Tm":
            last_tm = k
        elif operator == b"Tj" and last_tm is not None:
            x, y = float(ops[last_tm][0][4]), float(ops[last_tm][0][5])
            text = str(operands[0])
            for n, e in enumerate(edits):
                if text != e.old or (e.y is not None and abs(y - e.y) > TOL):
                    continue
                w = _width(text, e.font, e.size)
                at = x + w if e.align == "right" else x + w / 2
                if abs(at - e.anchor) > 0.05:
                    continue
                new_w = _width(e.new, e.font, e.size)
                new_x = e.anchor - new_w if e.align == "right" else e.anchor - new_w / 2
                tm = list(ops[last_tm][0])
                tm[4] = FloatObject(round(new_x, 4)) if abs(new_x - x) > 1e-6 else tm[4]
                ops[last_tm] = (tm, b"Tm")
                ops[k] = ([TextStringObject(e.new)], b"Tj")
                done[n] += 1
                break
    missing = [e.old for e, d in zip(edits, done, strict=True) if d != 1]
    if missing:
        raise RuntimeError(f"page {page_no}: strings not found exactly once: {missing}")
    page.replace_contents(content)


def _writer(pdf_path: Path, incremental: bool) -> PdfWriter:
    if incremental:
        return PdfWriter(PdfReader(io.BytesIO(Path(pdf_path).read_bytes())), incremental=True)
    return PdfWriter(clone_from=PdfReader(io.BytesIO(Path(pdf_path).read_bytes())))


def _save(writer: PdfWriter) -> bytes:
    """Every pypdf-written copy (edited or the control, full or incremental) is stamped
    the same way a tool stamps a re-save: Producer = pypdf, nothing else (no ModDate).
    Without a change, pypdf's incremental mode appends nothing and returns the original."""
    writer.add_metadata({"/Producer": "pypdf"})
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _cell_edit(layout: str, field: str, old: str, new: str, y: float) -> Edit:
    lay = LAYOUTS[layout]
    return Edit(old, new, y, RIGHT_EDGE[layout][field], "right", lay.font, lay.size)


def _summary_edit(layout: str, label: str, old: str, new: str) -> Edit:
    lay = LAYOUTS[layout]
    if lay.summary == "box":  # columns: centred values in the page-1 box, bold, size 8
        col_w = (PAGE_W - 2 * MARGIN) / len(TOTAL_LABELS)
        centre = LEFT + TOTAL_LABELS.index(label) * col_w + col_w / 2
        return Edit(old, new, PAGE_H - 150 - 46, centre, "centre", lay.font_bold, 8)
    return Edit(old, new, None, LEFT + 190, "right", lay.font, 8.5)  # drcr: after the rows


def _inplace(pdf_path: Path, rng, key: dict, consistent: bool) -> bytes:
    parsed = parse_pdf(Path(pdf_path))
    layout = parsed.layout
    lines = _row_lines(pdf_path, layout)
    if len(lines) != len(parsed.rows):
        raise RuntimeError(f"{len(lines)} row lines vs {len(parsed.rows)} parsed rows")
    i = int(rng.integers(len(parsed.rows)))
    r = parsed.rows[i]
    page, y = lines[i]
    field = amount_field(layout, _money(r.amount_paise))
    old = cell_text(layout, field, _money(r.amount_paise))
    new = change_one_digit(old, rng)
    edits = {page: [_cell_edit(layout, field, old, new, y)]}
    key["changes"].append({"page": page, "row": i, "field": field, "old": old, "new": new})
    if consistent:
        delta = _parse_money(new) - _parse_money(old)
        if _money(r.amount_paise) < 0:
            delta = -delta  # a bigger withdrawal lowers every later balance
        for j in range(i, len(parsed.rows)):
            if lines[j][0] != page:
                break
            bal = _money(parsed.rows[j].balance_paise)
            b_old, b_new = cell_text(layout, "balance", bal), cell_text(layout, "balance",
                                                                        bal + delta)  # fmt: skip
            edits[page].append(_cell_edit(layout, "balance", b_old, b_new, lines[j][1]))
            key["changes"].append({"page": page, "row": j, "field": "balance", "old": b_old,
                                   "new": b_new})  # fmt: skip
        h = parsed.header
        label = "Credits" if _money(r.amount_paise) > 0 else "Debits"
        total = _money(h.total_credits_paise if label == "Credits" else h.total_debits_paise)
        moved = abs(_parse_money(new) - _parse_money(old))
        bigger = _parse_money(new) > _parse_money(old)
        s_page = 1 if LAYOUTS[layout].summary == "box" else parsed.pages
        summary = [(label, inr(total), inr(total + moved if bigger else total - moved))]
        if lines[-1][0] == page:  # the edited page is the last: the closing balance moves
            closing = _money(h.closing_balance_paise)
            summary.append(("Closing Bal", inr(closing), inr(closing + delta)))
        for lab, s_old, s_new in summary:
            edits.setdefault(s_page, []).append(_summary_edit(layout, lab, s_old, s_new))
            key["changes"].append({"page": s_page, "row": None, "field": f"summary {lab}",
                                   "old": s_old, "new": s_new})  # fmt: skip
    incremental = bool(rng.random() < 0.5)
    writer = _writer(pdf_path, incremental)
    for p, page_edits in sorted(edits.items()):
        _apply_edits(writer, p, page_edits)
    key["incremental_save"] = incremental
    key["pages_changed"] = sorted(edits)
    return _save(writer)


# ---------------------------------------------------------------- pages


def _middle_page(n_pages: int, rng) -> int:
    if n_pages < 3:
        raise NotApplicable(f"needs 3+ pages, has {n_pages}")
    return int(rng.integers(2, n_pages))  # 2 .. n_pages - 1 (1-based)


def _rows_on(pdf_path: Path, page: int) -> list[int]:
    return [r.row for r in parse_pdf(Path(pdf_path)).rows if r.page == page]


def _page_remove(pdf_path: Path, rng, key: dict) -> bytes:
    n = len(PdfReader(str(pdf_path)).pages)
    k = _middle_page(n, rng)
    incremental = bool(rng.random() < 0.5)
    writer = _writer(pdf_path, incremental)
    writer.remove_page(k - 1)
    key["changes"] = [{"page": k, "row": r, "field": "row removed", "old": None, "new": None}
                      for r in _rows_on(pdf_path, k)]  # fmt: skip
    key |= {"pages_changed": [k], "incremental_save": incremental}
    return _save(writer)


def _page_swap(pdf_path: Path, donor_pdf: Path | None, rng, key: dict) -> bytes:
    if donor_pdf is None:
        raise NotApplicable("page_swap needs a donor PDF")
    if Path(donor_pdf).resolve() == Path(pdf_path).resolve():
        raise NotApplicable("the donor is the source itself")
    if parse_pdf(Path(donor_pdf)).layout != parse_pdf(Path(pdf_path)).layout:
        raise NotApplicable("the donor has another layout")
    n = len(PdfReader(str(pdf_path)).pages)
    k = _middle_page(n, rng)
    donor = PdfReader(str(donor_pdf))
    if len(donor.pages) < k:
        raise NotApplicable(f"the donor has {len(donor.pages)} pages, needs page {k}")
    incremental = bool(rng.random() < 0.5)
    writer = _writer(pdf_path, incremental)
    writer.insert_page(donor.pages[k - 1], k - 1)
    writer.remove_page(k)
    key["changes"] = [{"page": k, "row": r, "field": "row replaced", "old": None, "new": None}
                      for r in _rows_on(pdf_path, k)]  # fmt: skip
    key |= {"pages_changed": [k], "incremental_save": incremental,
            "donor": Path(donor_pdf).stem, "donor_page": k}  # fmt: skip
    return _save(writer)


# ---------------------------------------------------------------- overlay (held out)


def _overlay(pdf_path: Path, rng, key: dict) -> bytes:
    parsed = parse_pdf(Path(pdf_path))
    layout = parsed.layout
    lay = LAYOUTS[layout]
    lines = _row_lines(pdf_path, layout)
    i = int(rng.integers(len(parsed.rows)))
    r = parsed.rows[i]
    page, y = lines[i]
    field = amount_field(layout, _money(r.amount_paise))
    old = cell_text(layout, field, _money(r.amount_paise))
    new = change_one_digit(old, rng)
    right = RIGHT_EDGE[layout][field]
    width = _width(old, lay.font, lay.size)
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4, invariant=True)
    c.setFillColorRGB(1, 1, 1)
    c.rect(right - width - 1, y - lay.size * 0.3, width + 2, lay.size * 1.25, stroke=0, fill=1)
    c.setFillColorRGB(0, 0, 0)
    c.setFont(lay.font, lay.size)
    c.drawRightString(right, y, new)
    c.save()
    incremental = bool(rng.random() < 0.5)
    writer = _writer(pdf_path, incremental)
    writer.pages[page - 1].merge_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])
    key["changes"].append({"page": page, "row": i, "field": field, "old": old, "new": new})
    key |= {"pages_changed": [page], "incremental_save": incremental}
    return _save(writer)


# ---------------------------------------------------------------- the entry point


def tamper_copy(pdf_path: Path, tamper_type: str, seed: int, donor_pdf: Path | None = None,
                allow_held_out: bool = False) -> tuple[bytes, dict]:  # fmt: skip
    """One tampered copy (PDF bytes) of pdf_path, and its answer key."""
    if tamper_type not in TYPES:
        raise ValueError(f"unknown tamper type {tamper_type!r}; choose from {TYPES}")
    if tamper_type == HELD_OUT and not allow_held_out:
        raise HeldOut("overlay copies are made only by the one-time step 6 command")
    pdf_path = Path(pdf_path)
    rng = _rng(seed, tamper_type)
    key = _key(pdf_path, tamper_type, seed)
    if tamper_type == "rerender_inconsistent":
        data = _rerender_inconsistent(pdf_path, rng, key)
    elif tamper_type == "rerender_consistent":
        data = _rerender_consistent(pdf_path, rng, key)
    elif tamper_type in ("inplace_edit", "inplace_edit_consistent"):
        data = _inplace(pdf_path, rng, key, consistent=tamper_type.endswith("consistent"))
    elif tamper_type == "page_swap":
        data = _page_swap(pdf_path, donor_pdf, rng, key)
    elif tamper_type == "page_remove":
        data = _page_remove(pdf_path, rng, key)
    elif tamper_type == "overlay":
        data = _overlay(pdf_path, rng, key)
    else:  # resaved_genuine
        incremental = bool(rng.random() < 0.5)
        data = _save(_writer(pdf_path, incremental))
        key["incremental_save"] = incremental
    key["copy_sha256"] = hashlib.sha256(data).hexdigest()
    return data, key


__all__ = ["APPLIES_TO", "CONTROL", "HELD_OUT", "TYPES", "HeldOut", "NotApplicable",
           "tamper_copy"]  # fmt: skip
