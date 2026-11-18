"""Reading a statement PDF back and redrawing it with no edit gives the same bytes."""

from pathlib import Path

import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.tamper.rebuild import read_back
from statementscreen.tamper.rebuild_check import check, rebuild_bytes
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    """6 statements with every applicable hard case: both layouts, several pages each."""
    folder = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=11, hard_case_rate=1.0), 6), folder)
    return folder


def test_every_statement_rebuilds_byte_identical(batch) -> None:
    pdfs = sorted((batch / "pdfs").glob("*.pdf"))
    result = check(pdfs)
    assert result["differ"] == [], result["differ"]
    assert result["byte_identical"] == 6


def test_the_rebuild_reads_printed_text_not_the_parsers_join(batch) -> None:
    for pdf in sorted((batch / "pdfs").glob("*.pdf")):
        rb = read_back(pdf)
        assert len(rb.printed.narration_lines) == len(rb.statement.transactions)
        if rb.layout == "columns":
            assert len(rb.printed.refs) == len(rb.statement.transactions)
            assert any(len(lines) > 1 for lines in rb.printed.narration_lines)  # wrapped rows
        assert rb.printed.address_lines


def test_rebuild_bytes_returns_a_pdf(batch) -> None:
    pdf = sorted((batch / "pdfs").glob("*.pdf"))[0]
    assert rebuild_bytes(pdf)[:5] == b"%PDF-"
