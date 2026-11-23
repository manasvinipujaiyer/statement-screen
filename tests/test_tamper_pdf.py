"""tamper_pdf on real generated PDFs, tampered copies and a file that isn't a PDF.
Everything is written to temp folders only."""

import csv
import io
import shutil
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.tamper import copy as tc
from statementscreen.tamper.detect.pdf import collect_findings, main, tamper_pdf, tamper_pdf_timed
from statementscreen.tamper.detect.verdict import NONE, SUSPICIOUS, TAMPERED, to_json
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


def _show(verdict: dict) -> str:
    return f"{verdict['label']}: {verdict['summary']} | " + "; ".join(
        f"{r['strength']} {r['code']} x{r['count']}" for r in verdict["reasons"]
    )


@pytest.mark.parametrize("layout", ["columns", "drcr"])
def test_a_genuine_pdf_has_no_evidence_of_tampering(batch, layout):
    v = tamper_pdf(_genuine(batch, layout))
    assert v["label"] == NONE, _show(v)
    assert all(r["reason"].strip() for r in v["reasons"]) and v["summary"].strip()


@pytest.mark.parametrize("layout", ["columns", "drcr"])
def test_changing_only_the_producer_does_not_make_a_genuine_pdf_tampered(batch, tmp_path, layout):
    src = _genuine(batch, layout)
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(src.read_bytes())))
    writer.add_metadata({"/Producer": "SomeEditor 9"})
    copy = tmp_path / src.name
    with copy.open("wb") as f:
        writer.write(f)
    v = tamper_pdf(copy)
    assert v["label"] == NONE, _show(v)
    assert not [f for f in v["findings"] if f["strength"] == "strong"]


def test_tampered_copies_are_caught(batch, tmp_path):
    cases = [
        ("rerender_inconsistent", _genuine(batch, "columns"), "BALANCE_CHAIN"),
        ("inplace_edit", _genuine(batch, "drcr"), "BALANCE_CHAIN"),
    ]
    for tamper_type, src, code in cases:
        data, _ = tc.tamper_copy(src, tamper_type, seed=3)
        copy = tmp_path / f"{tamper_type}.pdf"
        copy.write_bytes(data)
        v = tamper_pdf(copy)
        assert v["label"] == TAMPERED, f"{tamper_type}: {_show(v)}"
        assert code in {r["code"] for r in v["reasons"]}, f"{tamper_type}: {_show(v)}"


def test_a_resaved_genuine_pdf_is_not_tampered(batch, tmp_path):
    src = _genuine(batch, "columns")
    for seed in (1, 2, 3, 4):  # both full and incremental saves occur across seeds
        data, key = tc.tamper_copy(src, "resaved_genuine", seed=seed)
        copy = tmp_path / f"resaved_{seed}.pdf"
        copy.write_bytes(data)
        v = tamper_pdf(copy)
        assert v["label"] != TAMPERED, f"seed {seed} {key['incremental_save']}: {_show(v)}"


def test_a_file_that_is_not_a_pdf_is_suspicious(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is not a pdf at all")
    v = tamper_pdf(bad)
    assert v["label"] == SUSPICIOUS, _show(v)
    codes = {r["code"] for r in v["reasons"]}
    assert "PARSE_FAILED" in codes
    # the checks that couldn't run are listed, never skipped quietly
    assert "CHECK_NOT_RUN" in codes
    assert all(r["reason"].strip() for r in v["reasons"])


def test_the_verdict_is_byte_identical_across_two_runs(batch):
    pdf = _genuine(batch, "drcr")
    first, second = to_json(tamper_pdf(pdf)), to_json(tamper_pdf(pdf))
    assert first == second
    verdict, stats = tamper_pdf_timed(pdf)
    assert to_json(verdict) == first  # timings are returned apart from the verdict
    assert stats["seconds"] > 0 and stats["peak_memory_mb"] > 0


def test_tamper_pdf_works_on_a_pdf_alone_in_an_empty_folder(batch, tmp_path, monkeypatch):
    """No split, manifest, labels, answer key or parquet within reach: only the file."""
    src = _genuine(batch, "columns")
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copy(src, alone / src.name)
    expected = to_json(tamper_pdf(src))
    monkeypatch.chdir(alone)
    assert sorted(p.name for p in Path.cwd().iterdir()) == [src.name]
    assert to_json(tamper_pdf(Path(src.name))) == expected


def test_collect_findings_gives_the_findings_the_verdict_uses(batch):
    pdf = _genuine(batch, "columns")
    found = collect_findings(pdf)
    assert all(f.reason.strip() for f in found)
    assert {f.group for f in found} <= {"parse", "arithmetic", "structure", "layout",
                                        "history", "anomaly"}  # fmt: skip


def test_the_command_line_prints_one_verdict(batch, capsys):
    assert main([str(_genuine(batch, "columns"))]) == 0
    out = capsys.readouterr()
    assert out.out.startswith('{"findings":') and NONE in out.err
    assert main([]) == 2 and main(["no_such_file.pdf"]) == 2
