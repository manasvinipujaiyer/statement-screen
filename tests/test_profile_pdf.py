"""profile_pdf: one PDF in, one profile out, with nothing else to read."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.profile import pdf
from statementscreen.profile.build import to_json
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def lone_pdf(tmp_path_factory) -> Path:
    """A generated statement's PDF, copied alone into an empty folder."""
    batch = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=11), 1), batch)
    alone = tmp_path_factory.mktemp("alone") / "statement.pdf"
    shutil.copy(batch / "pdfs" / "stmt_0001.pdf", alone)
    assert [p.name for p in alone.parent.iterdir()] == ["statement.pdf"]
    return alone


def test_profiles_a_pdf_with_nothing_else_to_read(lone_pdf, monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError(f"profile_pdf tried to read a table: {args[:1]}")

    for name in ("read_parquet", "scan_parquet", "read_csv", "scan_csv"):
        monkeypatch.setattr(pl, name, refuse)  # no truth, manifest or other table at all
    p = pdf.profile_pdf(lone_pdf)
    assert p["statement_id"] == "statement"  # named after the file, nothing else
    assert p["statement_months"] >= 6
    assert all(f["reason"].strip() for f in p["findings"].values())
    assert p["findings"]["customer_type"]["value"] in {"salaried", "gig", "self_employed",
                                                        "unknown"}  # fmt: skip


def test_the_same_pdf_gives_byte_identical_json(lone_pdf) -> None:
    assert to_json(pdf.profile_pdf(lone_pdf)) == to_json(pdf.profile_pdf(lone_pdf))


def test_timing_is_recorded(lone_pdf) -> None:
    p, stats = pdf.profile_pdf_timed(lone_pdf)
    assert stats["seconds"] > 0 and stats["peak_memory_mb"] > 0
    assert "seconds" not in to_json(p)  # timings stay out of the profile itself


def test_an_unreadable_pdf_gives_a_profile_with_a_warning(tmp_path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 this is not really a PDF")
    p = pdf.profile_pdf(broken)
    assert p["warnings"][0].startswith("could not read the statement")
    assert p["findings"]["estimated_monthly_income_paise"]["value"] is None
    assert "no recurring income found" in p["warnings"]
