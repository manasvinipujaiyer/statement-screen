"""verify must pass a clean batch and catch every kind of damage, naming the statement."""

import shutil
from pathlib import Path

import polars as pl
import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.verify import main, verify_batch
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def clean_batch(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("clean")
    write_batch(plan_batch(BatchConfig(seed=21), 5), out)
    return out


@pytest.fixture
def copy(clean_batch: Path, tmp_path: Path) -> Path:
    """A throwaway copy to damage; the clean batch stays clean."""
    target = tmp_path / "copy"
    shutil.copytree(clean_batch, target)
    return target


def _edit_parquet(out: Path, change) -> None:
    path = out / "transactions.parquet"
    df = change(pl.read_parquet(path))
    df.write_parquet(path)


def test_clean_batch_passes_with_full_reconciliation(clean_batch: Path, capsys) -> None:
    report = verify_batch(clean_batch)
    assert report.ok, report.failures
    assert report.statements == report.reconciled == 5
    assert report.reconciliation_rate == 1.0
    assert main([str(clean_batch)]) == 0
    assert "5/5 (100%)" in capsys.readouterr().out


def test_one_altered_amount_fails_and_names_the_statement(copy: Path) -> None:
    def bump(df: pl.DataFrame) -> pl.DataFrame:
        # Change one amount in stmt_0003 by 1 rupee, leaving balances as they were.
        target = (pl.col("statement_id") == "stmt_0003") & (pl.col("row") == 10)
        return df.with_columns(
            pl.when(target).then(pl.col("amount_paise") + 100).otherwise(pl.col("amount_paise"))
            .alias("amount_paise")
        )  # fmt: skip

    _edit_parquet(copy, bump)
    report = verify_batch(copy)
    assert not report.ok
    assert report.reconciled == 4  # the damaged statement is reported, never dropped
    assert report.statements == 5
    assert any(f.startswith("stmt_0003: row 10") for f in report.failures), report.failures
    assert main([str(copy)]) == 1


def test_missing_pdf_is_caught(copy: Path) -> None:
    (copy / "pdfs" / "stmt_0002.pdf").unlink()
    report = verify_batch(copy)
    assert "stmt_0002: missing from pdfs" in report.failures


def test_broken_pdf_is_caught(copy: Path) -> None:
    (copy / "pdfs" / "stmt_0004.pdf").write_bytes(b"not a pdf at all")
    assert any(f.startswith("stmt_0004: PDF") for f in verify_batch(copy).failures)


def test_count_mismatch_is_caught(copy: Path) -> None:
    def drop_last_row_of_stmt_0001(df: pl.DataFrame) -> pl.DataFrame:
        last = df.filter(pl.col("statement_id") == "stmt_0001")["row"].max()
        return df.filter(~((pl.col("statement_id") == "stmt_0001") & (pl.col("row") == last)))

    _edit_parquet(copy, drop_last_row_of_stmt_0001)
    failures = verify_batch(copy).failures
    assert any(f.startswith("stmt_0001: transaction counts disagree") for f in failures)
    assert any(f.startswith("stmt_0001: closing balance") for f in failures)


def test_unsorted_parquet_is_caught(copy: Path) -> None:
    _edit_parquet(copy, lambda df: df.sort("statement_id", descending=True, maintain_order=True))
    assert "parquet: statement_id is not sorted" in verify_batch(copy).failures


def test_extra_label_not_in_manifest_is_caught(copy: Path) -> None:
    shutil.copy(copy / "labels" / "stmt_0001.json", copy / "labels" / "stmt_0099.json")
    assert "stmt_0099: in labels but not in the manifest" in verify_batch(copy).failures


def test_missing_output_file_is_caught(copy: Path) -> None:
    (copy / "transactions.parquet").unlink()
    assert verify_batch(copy).failures == ["batch: missing transactions.parquet"]
