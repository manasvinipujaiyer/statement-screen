"""The parse and score commands, end to end on a small generated batch."""

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from statementscreen import parse, score
from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=5), 5), out)
    return out


@pytest.fixture(scope="module")
def parsed(batch: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("parsed") / "parsed"
    assert parse.main([str(batch / "pdfs"), "--out", str(out)]) == 0
    return out


def test_parse_writes_merged_tables_and_cleans_up(parsed: Path) -> None:
    assert sorted(p.name for p in parsed.iterdir()) == [
        "issues.parquet", "parse_run.json", "rows.parquet", "statements.parquet"]  # fmt: skip
    statements = pl.read_parquet(parsed / "statements.parquet")
    assert statements["statement_id"].to_list() == [f"stmt_{i:04d}" for i in range(1, 6)]
    assert (statements["status"] == "ok").all() and (statements["seconds"] > 0).all()
    rows = pl.read_parquet(parsed / "rows.parquet")
    assert rows.height == statements["rows"].sum()
    assert rows.select("statement_id", "row").equals(rows.select("statement_id", "row").sort(
        "statement_id", "row"))  # fmt: skip
    assert pl.read_parquet(parsed / "issues.parquet").height == 0


def test_parse_run_records_time_and_memory(parsed: Path) -> None:
    run = json.loads((parsed / "parse_run.json").read_text())
    assert run["pdfs"] == 5 and run["status"] == {"ok": 5}
    assert run["seconds_per_pdf"]["mean"] > 0 and run["peak_memory_mb"] > 0


def test_unreadable_and_crashing_pdfs_are_listed_not_skipped(batch, tmp_path, capsys,
                                                             monkeypatch) -> None:  # fmt: skip
    pdfs = tmp_path / "pdfs"
    shutil.copytree(batch / "pdfs", pdfs)
    (pdfs / "stmt_9998.pdf").write_bytes(b"not a pdf")
    real = parse.parse_pdf

    def crash_on_one(path, *a, **kw):
        if Path(path).stem == "stmt_0002":
            raise RuntimeError("boom")
        return real(path, *a, **kw)

    monkeypatch.setattr(parse, "parse_pdf", crash_on_one)
    out = tmp_path / "parsed"
    assert parse.main([str(pdfs), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "ERROR stmt_9998.pdf: unreadable PDF" in printed
    assert "ERROR stmt_0002.pdf: parser crashed" in printed
    statements = pl.read_parquet(out / "statements.parquet")
    assert statements.height == 6  # every PDF has a row, including the two failures
    errors = statements.filter(pl.col("status") == "error")["statement_id"].to_list()
    assert errors == ["stmt_0002", "stmt_9998"]
    codes = pl.read_parquet(out / "issues.parquet")["code"].to_list()
    assert sorted(codes) == ["PARSER_CRASH", "UNREADABLE_PDF"]


def test_limit_parses_the_first_n_by_name(batch, tmp_path) -> None:
    out = tmp_path / "parsed"
    assert parse.main([str(batch / "pdfs"), "--out", str(out), "--limit", "2"]) == 0
    ids = pl.read_parquet(out / "statements.parquet")["statement_id"].to_list()
    assert ids == ["stmt_0001", "stmt_0002"]


def test_parse_refuses_to_overwrite_without_permission(batch, tmp_path) -> None:
    out = tmp_path / "parsed"
    args = [str(batch / "pdfs"), "--out", str(out), "--limit", "1"]
    assert parse.main(args) == 0
    assert parse.main(args) == 2  # not empty, no --overwrite
    assert parse.main([*args, "--overwrite"]) == 0
    (out / "notes.txt").write_text("mine")
    assert parse.main([*args, "--overwrite"]) == 2  # a foreign file: never deleted
    assert (out / "notes.txt").exists()


def test_score_writes_the_report(parsed, batch, tmp_path) -> None:
    report_path = tmp_path / "report.json"
    fc = tmp_path / "fc.json"
    fc.write_text(json.dumps({"exact_row_rate": 0.0}))
    assert score.main([str(parsed), str(batch), "--out", str(report_path),
                       "--first-contact", str(fc)]) == 0  # fmt: skip
    report = json.loads(report_path.read_text())
    assert report["overall"]["exact_row_rate"] == 1.0
    assert report["first_contact"] == {"exact_row_rate": 0.0}
    assert report["parse_run"]["pdfs"] == 5 and report["batch"] == batch.name
    text = report_path.read_text()
    assert text == json.dumps(report, indent=2, sort_keys=True) + "\n"


def test_score_explains_missing_inputs(parsed, tmp_path, capsys) -> None:
    assert score.main([str(parsed), str(tmp_path), "--out", str(tmp_path / "r.json")]) == 2
    assert "missing transactions.parquet, manifest.csv" in capsys.readouterr().err
