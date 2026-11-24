"""Font and layout findings (tamper/detect/layout.py) on hand-drawn PDFs and genuine ones.

The hand-drawn PDFs have a 'columns'-style table header (so the parser's layout specs
recognise it) and a few rows; each test draws one thing wrong."""

import csv
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.tamper.detect.layout import layout_findings, measure, read_layout
from statementscreen.writer import write_batch

FONT, SIZE = "Helvetica", 8
DEBIT_RIGHT, CREDIT_RIGHT, BALANCE_RIGHT = 440, 520, 590
HEADER_Y = 700


def _draw(path: Path, wrong: str | None = None) -> Path:
    """A one-page table of 6 rows; `wrong` names the one thing drawn wrong, if any."""
    c = Canvas(str(path), pagesize=A4, invariant=True)
    c.setFont(FONT, SIZE)
    c.drawString(36, 780, "Statement of Account")
    if wrong == "header_overlap":
        c.drawString(36, 780, "XX")  # over the 'St' of the title, outside the table
    for text, x in (("Date", 36), ("Narration", 90), ("Chq./Ref.No.", 200), ("Value Dt", 300)):
        c.drawString(x, HEADER_Y, text)
    for text, x in (("Withdrawal Amt.", DEBIT_RIGHT), ("Deposit Amt.", CREDIT_RIGHT),
                    ("Closing Balance", BALANCE_RIGHT)):  # fmt: skip
        c.drawRightString(x, HEADER_Y, text)
    for i in range(6):
        y = HEADER_Y - 14 - 12 * i
        c.drawString(36, y, f"0{i + 1}/01/2025")
        c.drawString(90, y, "UPI-SHOP")
        c.drawString(300, y, f"0{i + 1}/01/25")
        c.drawRightString(DEBIT_RIGHT, y, "1,234.00")
        balance = f"98,76{i}.00"
        if i == 3 and wrong == "font":
            c.setFont("Helvetica-Bold", SIZE)
            c.drawRightString(BALANCE_RIGHT, y, balance)
            c.setFont(FONT, SIZE)
        elif i == 3 and wrong == "alignment":
            c.drawRightString(BALANCE_RIGHT - 2, y, balance)  # 2 pt left of the edge
        elif i == 3 and wrong == "spacing":
            head, tail = balance[:4], balance[4:]
            x0 = BALANCE_RIGHT - stringWidth(balance, FONT, SIZE) - 1.0
            c.drawString(x0, y, head)
            c.drawString(x0 + stringWidth(head, FONT, SIZE) + 1.0, y, tail)  # a 1 pt gap
        else:
            c.drawRightString(BALANCE_RIGHT, y, balance)
        if i == 3 and wrong == "table_overlap":
            c.drawRightString(BALANCE_RIGHT, y, "98,99" + balance[5:])  # over the balance
    c.drawRightString(BALANCE_RIGHT, 28, "Page 1 of 1")
    c.save()
    return path


def _codes(path: Path) -> list[tuple[str, str]]:
    return [(f.code, f.strength) for f in layout_findings(read_layout(path))]


def test_clean_table_has_no_findings_and_reads_every_money_cell(tmp_path):
    read = read_layout(_draw(tmp_path / "clean.pdf"))
    assert read.layout == "columns"
    assert len(read.cells) == 12  # 6 withdrawals + 6 balances
    assert layout_findings(read) == []
    m = measure(read)
    # positions are floats: allow rounding noise (~1e-13), far below the 0.5 pt tolerances
    assert m["max_alignment_off"] < 1e-6 and m["max_spacing_spread"] < 1e-6


def test_overlapping_characters_on_a_table_line_are_strong(tmp_path):
    found = layout_findings(read_layout(_draw(tmp_path / "x.pdf", "table_overlap")))
    overlap = [f for f in found if f.code == "CHAR_OVERLAP"]
    assert [f.strength for f in overlap] == ["strong"]
    assert "transaction line" in overlap[0].reason


def test_overlap_outside_the_table_is_weak(tmp_path):
    assert _codes(_draw(tmp_path / "x.pdf", "header_overlap")) == [("CHAR_OVERLAP", "weak")]


def test_amount_in_another_font_is_strong(tmp_path):
    found = layout_findings(read_layout(_draw(tmp_path / "x.pdf", "font")))
    assert [(f.code, f.strength) for f in found] == [("AMOUNT_FONT", "strong")]
    assert "Helvetica-Bold" in found[0].reason


def test_misaligned_amount_is_strong(tmp_path):
    found = layout_findings(read_layout(_draw(tmp_path / "x.pdf", "alignment")))
    assert [(f.code, f.strength) for f in found] == [("AMOUNT_ALIGNMENT", "strong")]
    assert "2.00 pt left" in found[0].reason


def test_uneven_digit_spacing_is_weak(tmp_path):
    assert ("DIGIT_SPACING", "weak") in _codes(_draw(tmp_path / "x.pdf", "spacing"))


def test_unknown_layout_lists_checks_as_not_run(tmp_path):
    path = tmp_path / "blank.pdf"
    c = Canvas(str(path), pagesize=A4, invariant=True)
    c.drawString(36, 700, "no table here")
    c.save()
    assert _codes(path) == [("CHECK_NOT_RUN", "info")]


def test_every_layout_finding_has_a_reason(tmp_path):
    for wrong in ("table_overlap", "header_overlap", "font", "alignment", "spacing"):
        found = layout_findings(read_layout(_draw(tmp_path / f"{wrong}.pdf", wrong)))
        assert found and all(f.reason.strip() for f in found), wrong


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 6), folder)
    return folder


def test_genuine_pdfs_have_no_strong_layout_findings(batch):
    with (batch / "manifest.csv").open() as f:
        ids = [row["statement_id"] for row in csv.DictReader(f)]
    for sid in ids:
        read = read_layout(batch / "pdfs" / f"{sid}.pdf")
        assert read.cells, sid
        assert not [f for f in layout_findings(read) if f.strength == "strong"], sid
