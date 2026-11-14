"""The patterns command, end to end on a small generated and parsed batch."""

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from statementscreen import parse, splits
from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.patterns import cli
from statementscreen.writer import write_batch

N = 8


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict[str, Path]:
    batch = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=11), N), batch)
    parsed = tmp_path_factory.mktemp("parsed") / "parsed"
    assert parse.main([str(batch / "pdfs"), "--out", str(parsed)]) == 0
    split_dir = tmp_path_factory.mktemp("splits")
    splits.write(*splits.build(batch / "manifest.csv"), split_dir)
    return {"batch": batch, "parsed": parsed, "splits": split_dir}


@pytest.fixture
def alone(world, tmp_path) -> Path:
    """The parsed folder copied alone into an empty folder: no truth files anywhere."""
    copy = tmp_path / "only_parsed"
    shutil.copytree(world["parsed"], copy)
    return copy


def _run(parsed: Path, out: Path, split_dir: Path, split: str = "train", *extra: str) -> int:
    return cli.main([str(parsed), "--split", split, "--out", str(out),
                     "--splits-dir", str(split_dir), *extra])  # fmt: skip


def test_runs_on_parser_output_alone(world, alone, tmp_path) -> None:
    truth_names = {"transactions.parquet", "manifest.csv", "labels"}
    assert not any(p.name in truth_names for p in alone.parent.rglob("*"))
    out = tmp_path / "patterns"
    assert _run(alone, out, world["splits"]) == 0
    assert sorted(p.name for p in (out / "train").iterdir()) == [
        "group_features.parquet", "group_rules.parquet", "patterns_run.json", "rows.parquet",
        "statement_results.parquet"]  # fmt: skip
    results = pl.read_parquet(out / "train" / "statement_results.parquet")
    assert results.height == len(set(pl.read_parquet(out / "train" / "rows.parquet")[
        "statement_id"]))  # fmt: skip
    assert results["salary_reason"].null_count() == 0  # a reason whether or not it fired

    train = splits.load_split_for_parsed(
        "train", pl.read_parquet(alone / "statements.parquet")["statement_id"].to_list(),
        world["splits"])  # fmt: skip
    parsed_rows = pl.read_parquet(alone / "rows.parquet").filter(
        pl.col("statement_id").is_in(train))  # fmt: skip
    rows = pl.read_parquet(out / "train" / "rows.parquet")
    assert rows.height == parsed_rows.height
    assert rows.select("statement_id", "row").rows() == parsed_rows.select(
        "statement_id", "row").rows()  # fmt: skip
    assert rows["group_id"].null_count() == 0

    feats = pl.read_parquet(out / "train" / "group_features.parquet")
    assert set(feats["statement_id"]) == set(train)
    assert feats["n_txn"].sum() == rows.height  # every row is in exactly one group
    assert set(feats["direction"]) <= {"credit", "debit"}

    run = json.loads((out / "train" / "patterns_run.json").read_text())
    assert run["statements"] == len(train) and run["rows"] == rows.height
    assert run["seconds_per_statement"]["mean"] > 0 and run["peak_memory_mb"] > 0
    assert sum(sum(c.values()) for c in run["styles"].values()) == rows.height
    assert run["config"]["group_threshold"] == cli.DEFAULT.group_threshold


def test_test_split_is_refused(world, alone, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(splits.OPEN_TEST_ENV, raising=False)
    assert _run(alone, tmp_path / "p", world["splits"], "test") == 2
    assert not (tmp_path / "p").exists()  # refused before anything was written


def test_a_previous_run_is_not_overwritten_by_accident(world, alone, tmp_path) -> None:
    out = tmp_path / "patterns"
    assert _run(alone, out, world["splits"]) == 0
    assert _run(alone, out, world["splits"]) == 2
    assert _run(alone, out, world["splits"], "train", "--overwrite") == 0


def test_a_parse_of_another_batch_is_refused(world, alone, tmp_path) -> None:
    other = tmp_path / "other_splits"
    meta = json.loads((world["splits"] / "split_v1.json").read_text())
    shutil.copytree(world["splits"], other)
    meta["statement_ids_sha256"] = "0" * 64
    (other / "split_v1.json").write_text(json.dumps(meta))
    assert _run(alone, tmp_path / "p", other) == 2


def test_score_patterns_end_to_end(world, alone, tmp_path, monkeypatch) -> None:
    from statementscreen import score_patterns

    monkeypatch.chdir(tmp_path)  # so a stray default mlruns/ would show up here
    out = tmp_path / "patterns"
    assert _run(alone, out, world["splits"]) == 0
    report_path = tmp_path / "report.json"
    tracking = tmp_path / "tracking"
    argv = [str(out), str(world["batch"]), "--split", "train", "--out", str(report_path),
            "--splits-dir", str(world["splits"]), "--tracking-dir", str(tracking)]  # fmt: skip
    assert score_patterns.main(argv) == 0
    assert not (tmp_path / "mlruns").exists()  # tests never write to mlruns/
    assert (tracking / "mlflow.db").exists()  # the run went to the temp folder
    assert any((tracking / "artifacts").rglob("report.json"))
    report = json.loads(report_path.read_text())
    assert report["rows_without_truth"] == 0
    s = report["scoring_run"]  # step 8: scoring records its own time and memory
    assert s["seconds_per_statement"] > 0 and s["peak_memory_mb"] > 0
    assert set(report["overall"]["transactions"]) == {"is_salary", "is_emi", "is_bounce",
                                                       "is_rent"}  # fmt: skip
    assert {"by_layout", "by_customer_type", "by_hard_case"} <= set(report["breakdown"])
    assert report["overall"]["grouping"]["precision"] is not None
