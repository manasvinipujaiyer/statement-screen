"""Batch-writing tests. They render real PDFs, so batches are kept small for CI."""

import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from pypdf import PdfReader

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.generate import main
from statementscreen.writer import MANIFEST_COLUMNS, TRANSACTION_SCHEMA, write_batch


def _run_cli_subprocess(out: Path, n: int, seed: int, hashseed: str) -> None:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    cmd = [sys.executable, "-m", "statementscreen.generate", "--n", str(n), "--seed", str(seed),
           "--out", str(out)]  # fmt: skip
    subprocess.run(cmd, check=True, env=env, capture_output=True)


def _hashes(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _manifest(out: Path) -> list[dict]:
    with (out / "manifest.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def small_batch(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=11), 6), out)
    return out


def test_all_outputs_are_written_and_parts_cleaned_up(small_batch: Path) -> None:
    assert sorted(p.name for p in small_batch.iterdir()) == [
        "labels", "manifest.csv", "pdfs", "transactions.parquet"]  # fmt: skip
    ids = [f"stmt_{i:04d}" for i in range(1, 7)]
    assert sorted(p.stem for p in (small_batch / "pdfs").iterdir()) == ids
    assert sorted(p.stem for p in (small_batch / "labels").iterdir()) == ids


def test_manifest_labels_parquet_and_pdfs_agree(small_batch: Path) -> None:
    rows = _manifest(small_batch)
    assert list(rows[0]) == list(MANIFEST_COLUMNS)
    df = pl.read_parquet(small_batch / "transactions.parquet")
    counts = dict(df.group_by("statement_id").len().iter_rows())
    for row in rows:
        sid = row["statement_id"]
        label = json.loads((small_batch / "labels" / f"{sid}.json").read_text(encoding="utf-8"))
        assert label["statement_id"] == sid
        assert int(row["transaction_count"]) == label["transaction_count"] == counts[sid]
        assert row["hard_case_tags"] == "|".join(label["hard_case_tags"])
        assert label["pdf"] == f"pdfs/{sid}.pdf"
        assert len(PdfReader(small_batch / label["pdf"]).pages) >= 1


def test_parquet_schema_sorting_and_paise(small_batch: Path) -> None:
    df = pl.read_parquet(small_batch / "transactions.parquet")
    assert dict(df.schema) == TRANSACTION_SCHEMA
    ids = df["statement_id"].to_list()
    assert ids == sorted(ids)
    for sid, part in df.group_by("statement_id", maintain_order=True):
        assert part["row"].to_list() == list(range(part.height))
        label = json.loads((small_batch / "labels" / f"{sid[0]}.json").read_text("utf-8"))
        balance = label["opening_balance_paise"]
        for amount, after in part.select("amount_paise", "balance_paise").iter_rows():
            balance += amount
            assert after == balance >= 0
        assert balance == label["closing_balance_paise"]


def test_no_timestamps_or_absolute_paths_in_outputs(small_batch: Path) -> None:
    today = datetime.date.today()
    forbidden = [str(small_batch).encode(), str(Path.home()).encode(),
                 today.isoformat().encode(), today.strftime("%Y%m%d").encode()]  # fmt: skip
    for path in small_batch.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert not [f for f in forbidden if f in data], path.name


def test_determinism_across_processes_and_hash_seeds(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _run_cli_subprocess(a, 20, 42, hashseed="1")
    _run_cli_subprocess(b, 20, 42, hashseed="98765")
    ha, hb = _hashes(a), _hashes(b)
    assert len(ha) == 20 * 2 + 2  # 20 PDFs, 20 labels, manifest, Parquet
    assert ha == hb


def test_different_main_seed_gives_different_output(tmp_path: Path) -> None:
    write_batch(plan_batch(BatchConfig(seed=1), 3), tmp_path / "s1")
    write_batch(plan_batch(BatchConfig(seed=2), 3), tmp_path / "s2")
    h1, h2 = _hashes(tmp_path / "s1"), _hashes(tmp_path / "s2")
    assert all(h1[name] != h2[name] for name in h1)


def test_first_statements_identical_whatever_n(tmp_path: Path) -> None:
    write_batch(plan_batch(BatchConfig(seed=5), 5), tmp_path / "n5")
    write_batch(plan_batch(BatchConfig(seed=5), 10), tmp_path / "n10")
    h5, h10 = _hashes(tmp_path / "n5"), _hashes(tmp_path / "n10")
    for i in range(1, 6):
        for kind, ext in (("pdfs", "pdf"), ("labels", "json")):
            name = f"{kind}/stmt_{i:04d}.{ext}"
            assert h5[name] == h10[name], name
    small = pl.read_parquet(tmp_path / "n5" / "transactions.parquet")
    big = pl.read_parquet(tmp_path / "n10" / "transactions.parquet")
    first5 = big.filter(pl.col("statement_id") <= "stmt_0005")
    assert small.equals(first5)


def test_batch_has_both_layouts_and_6_to_12_months(tmp_path: Path) -> None:
    write_batch(plan_batch(BatchConfig(seed=42), 12), tmp_path)
    rows = _manifest(tmp_path)
    assert {r["layout"] for r in rows} == {"columns", "drcr"}
    assert all(6 <= int(r["months"]) <= 12 for r in rows)
    for r in rows:
        label = json.loads((tmp_path / "labels" / f"{r['statement_id']}.json").read_text("utf-8"))
        start = datetime.date.fromisoformat(label["period_start"])
        end = datetime.date.fromisoformat(label["period_end"])
        assert (end.year - start.year) * 12 + end.month - start.month + 1 == int(r["months"])


def test_cli_refuses_a_non_empty_folder_and_overwrite_replaces(tmp_path: Path, capsys) -> None:
    assert main(["--n", "2", "--out", str(tmp_path)]) == 0
    first = _hashes(tmp_path)
    assert main(["--n", "2", "--out", str(tmp_path)]) == 2  # refused
    assert "not empty" in capsys.readouterr().err
    assert main(["--n", "2", "--out", str(tmp_path), "--overwrite"]) == 0
    assert _hashes(tmp_path) == first  # same seed, same files
