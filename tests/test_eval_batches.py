"""Evaluation batches can't be used to get round the split guard."""

import json
import shutil
from pathlib import Path

import pytest

from statementscreen import eval_batches as eb
from statementscreen import parse, splits
from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.patterns import cli
from statementscreen.splits import SplitError
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict[str, Path]:
    """A small reference batch with its split, an evaluation batch with other seeds, and
    a parse of the first reference statements, which share the evaluation batch's ids."""
    root = tmp_path_factory.mktemp("eval_world")
    reference, shift = root / "reference", root / "shift"
    write_batch(plan_batch(BatchConfig(seed=11), 6), reference)
    write_batch(plan_batch(BatchConfig(seed=12), 3), shift)
    for batch, limit in ((reference, 6), (shift, 3)):
        assert parse.main([str(batch / "pdfs"), "--out", str(root / f"parsed_{batch.name}"),
                           "--limit", str(limit)]) == 0  # fmt: skip
    assert parse.main([str(reference / "pdfs"), "--out", str(root / "parsed_impostor"),
                       "--limit", "3"]) == 0  # fmt: skip
    split_dir = root / "splits"
    splits.write(*splits.build(reference / "manifest.csv"), split_dir)
    eb.register("shift", shift, root / "parsed_shift", reference / "manifest.csv", split_dir)
    return {"root": root, "reference": reference, "shift": shift, "splits": split_dir,
            "parsed_reference": root / "parsed_reference", "parsed_shift": root / "parsed_shift",
            "parsed_impostor": root / "parsed_impostor"}  # fmt: skip


def _patterns(world, parsed: Path, out: Path, *which: str) -> int:
    return cli.main([str(parsed), *which, "--out", str(out), "--splits-dir", str(world["splits"])])


def test_a_registered_batch_runs_whole_on_parser_output_alone(world, tmp_path) -> None:
    alone = tmp_path / "only_parsed"
    shutil.copytree(world["parsed_shift"], alone)
    out = tmp_path / "patterns"
    assert _patterns(world, alone, out, "--eval", "shift") == 0
    run = json.loads((out / "shift" / "patterns_run.json").read_text())
    assert run["evaluation_batch"] is True and run["split"] == "shift"
    assert run["statements"] == 3  # the whole batch
    assert run["style_examples"] == "not shown outside train"
    entry = json.loads(eb.eval_path("shift", world["splits"]).read_text())
    assert entry["statements"] == 3 and entry["batch"] == "shift"


def test_the_reference_batch_is_refused_as_an_evaluation_batch(world) -> None:
    with pytest.raises(SplitError, match="is the reference batch"):
        eb.register("ref", world["reference"], world["parsed_reference"],
                    world["reference"] / "manifest.csv", world["splits"])  # fmt: skip
    assert not eb.eval_path("ref", world["splits"]).exists()


def test_a_batch_sharing_statements_with_the_reference_is_refused(world, tmp_path) -> None:
    overlap = tmp_path / "overlap"
    write_batch(plan_batch(BatchConfig(seed=11), 2), overlap)  # same seeds as the reference
    with pytest.raises(SplitError, match="shares 2 statements"):
        eb.register("overlap", overlap, world["parsed_impostor"],
                    world["reference"] / "manifest.csv", world["splits"])  # fmt: skip


def test_reference_must_be_the_split_manifest(world) -> None:
    with pytest.raises(SplitError, match="is not the manifest"):
        eb.register("x", world["shift"], world["parsed_shift"], world["shift"] / "manifest.csv",
                    world["splits"])  # fmt: skip


def test_registration_is_never_overwritten(world) -> None:
    with pytest.raises(SplitError, match="refusing to overwrite"):
        eb.register("shift", world["shift"], world["parsed_shift"],
                    world["reference"] / "manifest.csv", world["splits"])  # fmt: skip


def test_reference_statements_under_the_same_ids_are_refused(world, tmp_path) -> None:
    """The impostor parse has stmt_0001-0003 like the shift batch, but reference rows."""
    assert eb.ids_sha256(eb.parsed_ids(world["parsed_impostor"])) == eb.ids_sha256(
        eb.parsed_ids(world["parsed_shift"])
    )
    with pytest.raises(SplitError, match="different rows"):
        eb.load_eval_for_parsed("shift", world["parsed_impostor"], world["splits"])
    assert _patterns(world, world["parsed_impostor"], tmp_path / "p", "--eval", "shift") == 2
    assert not (tmp_path / "p").exists()  # refused before anything was written


def test_a_parse_of_the_reference_batch_is_refused_by_eval(world, tmp_path) -> None:
    with pytest.raises(SplitError, match="parse of the reference batch"):
        eb.load_eval_for_parsed("shift", world["parsed_reference"], world["splits"])
    assert _patterns(world, world["parsed_reference"], tmp_path / "p", "--eval", "shift") == 2


def test_the_test_split_stays_refused(world, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(splits.OPEN_TEST_ENV, raising=False)
    assert _patterns(world, world["parsed_reference"], tmp_path / "p", "--split", "test") == 2
    assert not (tmp_path / "p").exists()


def test_unregistered_names_and_mixed_options_are_refused(world, tmp_path) -> None:
    assert _patterns(world, world["parsed_shift"], tmp_path / "p", "--eval", "nope") == 2
    with pytest.raises(SystemExit):  # argparse: --split and --eval together
        _patterns(world, world["parsed_shift"], tmp_path / "p", "--split", "dev", "--eval", "x")


def test_score_patterns_on_an_evaluation_batch(world, tmp_path) -> None:
    from statementscreen import score_patterns

    out = tmp_path / "patterns"
    assert _patterns(world, world["parsed_shift"], out, "--eval", "shift") == 0
    report_path = tmp_path / "rules.json"
    argv = [str(out), str(world["shift"]), "--eval", "shift", "--out", str(report_path),
            "--splits-dir", str(world["splits"]), "--no-mlflow"]  # fmt: skip
    assert score_patterns.main(argv) == 0
    report = json.loads(report_path.read_text())
    assert report["split"] == "shift" and report["statements"] == 3
    assert report["rows_without_truth"] == 0
    wrong = [str(out), str(world["reference"]), "--eval", "shift", "--out", str(report_path),
             "--splits-dir", str(world["splits"]), "--no-mlflow"]  # fmt: skip
    assert score_patterns.main(wrong) == 2  # the reference batch's truth is refused


def test_scoring_checks_the_manifest(world) -> None:
    assert eb.load_eval("shift", world["shift"] / "manifest.csv", world["splits"]) == [
        "stmt_0001", "stmt_0002", "stmt_0003"]  # fmt: skip
    with pytest.raises(SplitError, match="is not evaluation batch"):
        eb.load_eval("shift", world["reference"] / "manifest.csv", world["splits"])
