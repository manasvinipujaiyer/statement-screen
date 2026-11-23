"""File-history findings (tamper/detect/history.py) on genuine PDFs and re-saved copies,
plus the rule that metadata alone never gives a strong finding (all checks run)."""

import csv
import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.parsing.pdf import parse_pdf
from statementscreen.tamper.detect.arithmetic import arithmetic_findings
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.history import font_names, history_findings
from statementscreen.tamper.detect.layout import layout_findings, read_layout
from statementscreen.tamper.detect.structure import structure_findings
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 6), folder)
    return folder


def _genuine(batch: Path, layout: str) -> Path:
    with (batch / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            if row["layout"] == layout:
                return batch / "pdfs" / f"{row['statement_id']}.pdf"
    raise AssertionError(f"no {layout} statement in the test batch")


def _rewrite(src: Path, dst: Path, metadata: dict, incremental: bool = False) -> Path:
    reader = PdfReader(io.BytesIO(src.read_bytes()))
    writer = PdfWriter(reader, incremental=True) if incremental else PdfWriter(clone_from=reader)
    writer.add_metadata(metadata)
    with dst.open("wb") as f:
        writer.write(f)
    return dst


def _all_findings(path: Path) -> list[Finding]:
    parsed = parse_pdf(path)
    found = arithmetic_findings(parsed) + structure_findings(path, parsed)
    return found + layout_findings(read_layout(path)) + history_findings(path, parsed.layout)


def _codes(found: list[Finding]) -> set[tuple[str, str]]:
    return {(f.code, f.strength) for f in found}


@pytest.mark.parametrize("layout", ["columns", "drcr"])
def test_genuine_pdf_has_no_weak_or_strong_history_findings(batch, layout):
    pdf = _genuine(batch, layout)
    found = history_findings(pdf, layout)
    assert not [f for f in found if f.strength != "info"], found


def test_incremental_save_is_weak(batch, tmp_path):
    copy = _rewrite(_genuine(batch, "columns"), tmp_path / "inc.pdf", {"/Producer": "pypdf"},
                    incremental=True)  # fmt: skip
    found = history_findings(copy, "columns")
    assert ("INCREMENTAL_UPDATE", "weak") in _codes(found)
    assert not [f for f in found if f.strength == "strong"]


@pytest.mark.parametrize("layout", ["columns", "drcr"])
def test_changing_only_the_producer_raises_no_strong_finding(batch, tmp_path, layout):
    """Spec: a genuine PDF with only its producer string changed (a full re-save, so no
    incremental marker either) must not get a strong finding from ANY check."""
    src = _genuine(batch, layout)
    copy = _rewrite(src, tmp_path / "p.pdf", {"/Producer": "SomeEditor 9"})
    assert PdfReader(copy).metadata.producer == "SomeEditor 9"
    found = _all_findings(copy)
    assert not [f for f in found if f.strength == "strong"], found
    # a full re-save adds no weak finding either: the copy's weak findings are the source's
    weak = sorted(f.reason for f in found if f.strength == "weak")
    assert weak == sorted(f.reason for f in _all_findings(src) if f.strength == "weak")


def test_modification_date_alone_is_weak(batch, tmp_path):
    copy = _rewrite(_genuine(batch, "drcr"), tmp_path / "m.pdf",
                    {"/CreationDate": "D:20250101090000+05'30'",
                     "/ModDate": "D:20250315120000+05'30'"})  # fmt: skip
    found = _all_findings(copy)
    assert ("MODDATE_AFTER_CREATION", "weak") in _codes(found)
    assert not [f for f in found if f.strength == "strong"], found


def test_equal_or_earlier_modification_date_gives_nothing(batch, tmp_path):
    copy = _rewrite(_genuine(batch, "columns"), tmp_path / "m.pdf",
                    {"/CreationDate": "D:20250101090000+05'30'",
                     "/ModDate": "D:20250101090000+05'30'"})  # fmt: skip
    assert "MODDATE_AFTER_CREATION" not in {f.code for f in history_findings(copy, "columns")}


def test_an_added_font_is_weak(batch, tmp_path):
    src = _genuine(batch, "columns")
    stamp = io.BytesIO()
    c = Canvas(stamp, pagesize=A4, invariant=True)
    c.setFont("Courier", 6)
    c.drawString(36, 8, "x")  # below the footer: no overlap with anything
    c.save()
    reader = PdfReader(io.BytesIO(src.read_bytes()))
    writer = PdfWriter(clone_from=reader)
    writer.pages[0].merge_page(PdfReader(io.BytesIO(stamp.getvalue())).pages[0])
    copy = tmp_path / "f.pdf"
    with copy.open("wb") as f:
        writer.write(f)
    assert "Courier" in font_names(PdfReader(copy))
    found = history_findings(copy, "columns")
    assert ("EXTRA_FONTS", "weak") in _codes(found)


def test_unknown_layout_lists_font_check_as_not_run(batch):
    found = history_findings(_genuine(batch, "columns"), None)
    assert any(f.code == "CHECK_NOT_RUN" and "EXTRA_FONTS" in f.reason for f in found)


def test_producer_creator_differ_is_info_only(batch):
    found = [f for f in history_findings(_genuine(batch, "columns"), "columns")
             if f.code == "PRODUCER_CREATOR_DIFFER"]  # fmt: skip
    assert all(f.strength == "info" for f in found)
    assert all("never counted" in f.reason for f in found)


def test_every_history_finding_has_a_reason(batch, tmp_path):
    copy = _rewrite(_genuine(batch, "columns"), tmp_path / "all.pdf",
                    {"/Producer": "pypdf", "/CreationDate": "D:20250101000000Z",
                     "/ModDate": "D:20250201000000Z"}, incremental=True)  # fmt: skip
    found = history_findings(copy, "columns")
    assert len(found) >= 2
    assert all(f.reason.strip() for f in found)
