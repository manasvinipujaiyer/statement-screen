"""The one-time test and shift scoring: provenance from each run's own record, completeness,
the pre-registration guard, and write-once results. A miniature world stands in for the real
one: the registered counts are shrunk to match it (the real ones are 300, 300 and 800)."""

import json
from pathlib import Path

import polars as pl
import pytest

from statementscreen.tamper import compare_rules as cr
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import preregister as pre
from statementscreen.tamper import provenance as prov
from statementscreen.tamper import run as driver
from statementscreen.tamper import score_tamper as cli
from statementscreen.tamper.detect.findings import Finding

STRONG = Finding("BALANCE_CHAIN", "arithmetic", 1, 0, "strong", "balance broken")
COPIES = [f"copy_{n:04d}" for n in range(1, 5)]
TEST_IDS = [f"stmt_{n:04d}" for n in range(1, 6)]
SHIFT_IDS = ["shift_0001", "shift_0002"]


def _run(folder: Path, items: list[str], findings: dict | None = None, **extra) -> None:
    """A finished detector run, as tamper/run.py would have written it."""
    findings = findings or {}
    folder.mkdir(parents=True)
    rows = [{"item": i, **f.to_dict()} for i, fs in findings.items() for f in fs]
    pl.DataFrame(rows, schema=driver.FINDING_SCHEMA).write_parquet(folder / "findings.parquet")
    n = len(items)
    pl.DataFrame(
        {
            "item": items,
            "seconds": [1.0] * n,
            "peak_memory_mb": [100.0] * n,
            "n_findings": [len(findings.get(i, [])) for i in items],
            "crashed": [False] * n,
        },
        schema=driver.ITEM_SCHEMA,
    ).write_parquet(folder / "items.parquet")
    summary = {"fingerprint": fp.fingerprint(), "pdfs": n, "crashed": []} | extra
    (folder / "detect_run.json").write_text(json.dumps(summary), encoding="utf-8")


def _keys(set_dir: Path) -> None:
    (set_dir / "keys").mkdir(parents=True)
    for n, copy_id in enumerate(COPIES, start=1):
        key = {
            "copy_id": copy_id,
            "source": f"stmt_{n:04d}",
            "source_layout": "columns" if n % 2 else "drcr",
            "tamper_type": "inplace_edit",
            "incremental_save": bool(n % 2),
            "pages_changed": [1],
            "changes": [{"page": 1}],
            "donor": None,
        }
        (set_dir / "keys" / f"{copy_id}.json").write_text(json.dumps(key), encoding="utf-8")


def _build(set_dir: Path, **change) -> None:
    build = {
        "set": "test",
        "per_type": 4,
        "made": {"inplace_edit": 4},
        "short": {},
        "complete": True,
    } | change
    (set_dir / "build_run.json").write_text(json.dumps(build), encoding="utf-8")


def _manifest(path: Path, ids: list[str]) -> Path:
    rows = ["statement_id,layout,customer_type"]
    rows += [f"{s},{'columns' if n % 2 else 'drcr'},salaried" for n, s in enumerate(ids)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def world(tmp_path, monkeypatch, pushed):
    """Test copies (4, one type), genuine test (5), genuine shift (2), all complete."""
    pdfs = {"test": "pdfs/test", "shift": "pdfs/shift"}
    monkeypatch.setattr(prov, "EXPECTED_COUNTS", {"test": 5, "shift": 2, "copies": 4})
    monkeypatch.setattr(prov, "EXPECTED_SOURCES", pdfs)
    monkeypatch.setattr(prov, "TYPES", ("inplace_edit",))
    monkeypatch.setattr(prov, "COPIES_PER_TYPE", 4)
    set_dir = tmp_path / "test"
    _keys(set_dir)
    _build(set_dir)
    _run(
        set_dir / "detect",
        COPIES,
        {c: [STRONG] for c in COPIES[:3]},
        source=str(set_dir / "copies"),
    )
    _run(tmp_path / "genuine_test", TEST_IDS, split="test", population="test", source=pdfs["test"])
    _run(tmp_path / "genuine_shift", SHIFT_IDS, population="shift", source=pdfs["shift"])
    out, table = tmp_path / "reports" / "test.json", tmp_path / "reports" / "test.csv"
    monkeypatch.setattr(pre, "FIRST_REPORT", str(out))
    monkeypatch.setattr(pre, "FIRST_TABLE", str(table))
    prereg = tmp_path / "prereg.json"
    data = pre.build(head=pre.git("rev-parse", "HEAD"), dirty="")
    prereg.write_text(json.dumps(data), encoding="utf-8")
    return {
        "tmp": tmp_path,
        "set": set_dir,
        "genuine_test": tmp_path / "genuine_test",
        "genuine_shift": tmp_path / "genuine_shift",
        "manifest": _manifest(tmp_path / "manifest.csv", TEST_IDS),
        "shift_manifest": _manifest(tmp_path / "shift_manifest.csv", SHIFT_IDS),
        "prereg": prereg,
        "out": out,
        "table": table,
    }


def _argv(w: dict, *extra: str, prereg: bool = True, table: bool = True) -> list[str]:
    argv = [
        *("--copies", str(w["set"])),
        *("--genuine", f"test={w['genuine_test']}"),
        *("--genuine", f"shift={w['genuine_shift']}"),
        *("--manifest", str(w["manifest"])),
        *("--genuine-manifest", f"shift={w['shift_manifest']}"),
        *("--out", str(w["out"])),
    ]
    if prereg:
        argv += ["--preregistration", str(w["prereg"])]
    if table:
        argv += ["--table", str(w["table"])]
    return [*argv, *extra]


def _only(w: dict, spec: str, *extra: str) -> list[str]:
    """Score one genuine population on its own (no copies, no pre-registration)."""
    return [
        *("--genuine", spec),
        *("--manifest", str(w["manifest"])),
        *("--out", str(w["tmp"] / "only.json")),
        *extra,
    ]


# ---------------------------------------------------------------- the one-time scoring


def test_the_one_time_scoring_runs_once_and_records_its_inputs(world):
    assert cli.main(_argv(world)) == 0
    report = json.loads(world["out"].read_text(encoding="utf-8"))
    assert world["table"].is_file() and "post_hoc" not in report
    assert set(report["inputs"]) == {"copies", "genuine_test", "genuine_shift", "preregistration"}
    assert report["inputs"]["preregistration"]["sha256"] == prov.file_sha256(world["prereg"])
    assert set(report["inputs"]["copies"]["sha256"]) == {"keys", "build_run.json", "detect"}
    assert report["inputs"]["genuine_shift"]["manifest"] == str(world["shift_manifest"])


def test_check_populations_returns_the_registered_ones(world):
    specs = [f"test={world['genuine_test']}", f"shift={world['genuine_shift']}"]
    assert prov.check_populations(world["set"], None, specs) == ["copies", "test", "shift"]


def test_the_test_and_shift_populations_need_a_preregistration(world, capsys):
    assert cli.main(_argv(world, prereg=False)) == 2
    assert "pre-registration" in capsys.readouterr().err
    assert not world["out"].exists()


def test_scoring_refuses_a_detector_or_rule_version_that_differs(world, capsys):
    data = json.loads(world["prereg"].read_text(encoding="utf-8"))
    data["detector_fingerprint"]["model"] = "a different model"
    world["prereg"].write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(_argv(world)) == 2
    err = capsys.readouterr().err
    assert "model differs" in err and "code differs" not in err
    data["detector_fingerprint"] = fp.fingerprint()
    world["prereg"].write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(_argv(world, "--rule-version", "v1")) == 2
    assert "rule version" in capsys.readouterr().err
    assert not world["out"].exists()


def test_an_edited_preregistration_is_refused_against_the_pushed_version(
    world, monkeypatch, capsys
):
    pushed_version = world["prereg"].read_text(encoding="utf-8")
    monkeypatch.setattr(pre, "pushed_text", lambda path, root=pre.ROOT: pushed_version)
    data = json.loads(pushed_version)
    data["rules"] = ["an uncommitted edit"]
    world["prereg"].write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(_argv(world)) == 2
    assert "differs from the pushed version" in capsys.readouterr().err
    assert not world["out"].exists()


# ---------------------------------------------------------------- B1: the data decides


def test_a_test_run_cannot_be_scored_under_another_name(world, capsys):
    spec = f"dev={world['genuine_test']}"
    assert cli.main(_only(world, spec)) == 2
    assert "recorded as 'test'" in capsys.readouterr().err
    assert not (world["tmp"] / "only.json").exists()


def test_a_dev_run_cannot_be_scored_as_test(world, capsys):
    _run(world["tmp"] / "genuine_dev", TEST_IDS, split="dev", source="pdfs/dev")
    spec = f"test={world['tmp'] / 'genuine_dev'}"
    assert cli.main(_only(world, spec, "--preregistration", str(world["prereg"]))) == 2
    assert "recorded as 'dev'" in capsys.readouterr().err


def test_a_run_that_records_no_population_is_refused(world, capsys):
    _run(world["tmp"] / "anonymous", TEST_IDS, source="pdfs/test")
    assert cli.main(_only(world, f"dev={world['tmp'] / 'anonymous'}")) == 2
    assert "does not record which population" in capsys.readouterr().err


def test_the_shift_batch_can_only_be_scored_as_shift(world, capsys):
    _run(world["tmp"] / "mislabelled", SHIFT_IDS, population="dev", source="pdfs/shift")
    assert cli.main(_only(world, f"dev={world['tmp'] / 'mislabelled'}")) == 2
    assert "are the shift batch" in capsys.readouterr().err


def test_a_test_folder_must_have_been_built_as_test(world, capsys):
    _build(world["set"], set="dev")
    assert cli.main(_argv(world)) == 2
    assert "named test but its build record says" in capsys.readouterr().err


# ---------------------------------------------------------------- S2, S3: complete or refused


def test_a_limit_trial_run_is_never_scored(world, capsys):
    _run(world["tmp"] / "trial", TEST_IDS, split="test", limit=3, source="pdfs/test")
    assert cli.main(_only(world, f"test={world['tmp'] / 'trial'}")) == 2
    assert "used --limit" in capsys.readouterr().err


def test_an_incomplete_registered_population_is_refused(world, capsys):
    _run(world["tmp"] / "short", TEST_IDS[:4], split="test", source="pdfs/test")
    argv = _only(world, f"test={world['tmp'] / 'short'}", "--preregistration", str(world["prereg"]))
    assert cli.main(argv) == 2
    assert "must cover 5 PDFs" in capsys.readouterr().err


def test_a_registered_population_must_come_from_its_registered_folder(world, capsys):
    _run(world["tmp"] / "elsewhere", TEST_IDS, split="test", source="pdfs/other")
    argv = _only(
        world, f"test={world['tmp'] / 'elsewhere'}", "--preregistration", str(world["prereg"])
    )
    assert cli.main(argv) == 2
    assert "must come from" in capsys.readouterr().err


def test_a_build_that_came_up_short_is_refused(world, capsys):
    _build(world["set"], short={"inplace_edit": 2}, made={"inplace_edit": 2}, complete=False)
    assert cli.main(_argv(world)) == 2
    assert "incomplete" in capsys.readouterr().err and not world["out"].exists()


def test_a_build_with_the_wrong_counts_is_refused(world, capsys):
    _build(world["set"], made={"inplace_edit": 3})
    assert cli.main(_argv(world)) == 2
    assert "must hold 4 copies" in capsys.readouterr().err


def test_a_copies_run_made_on_another_folder_is_refused(world, capsys):
    run_file = world["set"] / "detect" / "detect_run.json"
    run = json.loads(run_file.read_text(encoding="utf-8"))
    run["source"] = "somewhere/else"
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert cli.main(_argv(world)) == 2
    assert "was not made on" in capsys.readouterr().err


# ---------------------------------------------------------------- S7: write once, then post-hoc


def test_the_first_result_is_never_overwritten(world, capsys):
    assert cli.main(_argv(world)) == 0
    first = world["out"].read_bytes()
    assert cli.main(_argv(world)) == 2
    assert "never overwritten" in capsys.readouterr().err
    assert cli.main(_argv(world, "--overwrite")) == 2
    assert "--overwrite is not allowed" in capsys.readouterr().err
    assert world["out"].read_bytes() == first


def test_a_second_run_without_post_hoc_goes_to_another_file_and_is_refused(world, capsys):
    assert cli.main(_argv(world)) == 0
    other = world["tmp"] / "reports" / "again.json"
    argv = _argv(world)
    argv[argv.index(str(world["out"]))] = str(other)
    assert cli.main([*argv, "--table", str(world["tmp"] / "again.csv")]) == 2
    assert not other.exists() and "writes" in capsys.readouterr().err


def test_a_post_hoc_rerun_has_its_own_file_and_is_labelled(world, capsys):
    assert cli.main(_argv(world)) == 0
    first = world["out"].read_bytes()
    post_out = world["tmp"] / "reports" / "test_post_hoc.json"
    post_table = world["tmp"] / "reports" / "test_post_hoc.csv"
    argv = _argv(world, "--post-hoc", table=False)
    argv[argv.index(str(world["out"]))] = str(post_out)
    assert cli.main([*argv, "--table", str(post_table)]) == 0
    assert json.loads(post_out.read_text(encoding="utf-8"))["post_hoc"] is True
    assert "POST-HOC RE-RUN" in capsys.readouterr().out
    assert world["out"].read_bytes() == first  # the first result is untouched
    assert cli.main([*argv, "--table", str(post_table)]) == 2  # nor is the post-hoc file replaced


def test_post_hoc_needs_its_own_file_name_and_a_first_result(world, capsys):
    post_out = world["tmp"] / "reports" / "test_post_hoc.json"
    argv = _argv(world, "--post-hoc", table=False)
    argv[argv.index(str(world["out"]))] = str(post_out)
    assert cli.main(argv) == 2  # no first result yet
    assert "needs the first result" in capsys.readouterr().err
    assert cli.main(_argv(world)) == 0
    named_wrong = world["tmp"] / "reports" / "second.json"
    argv[argv.index(str(post_out))] = str(named_wrong)
    assert cli.main(argv) == 2 and "needs post_hoc" in capsys.readouterr().err


def test_post_hoc_applies_only_to_the_registered_populations(world, capsys):
    _run(world["tmp"] / "genuine_dev", TEST_IDS, split="dev", source="pdfs/dev")
    argv = _only(world, f"dev={world['tmp'] / 'genuine_dev'}", "--post-hoc")
    assert cli.main(argv) == 2
    assert "applies only to the test and shift populations" in capsys.readouterr().err


# ---------------------------------------------------------------- B2: compare_rules


def test_compare_rules_never_reads_test_or_shift(world, capsys):
    out = world["tmp"] / "comparison.json"
    base = [*("--copies", str(world["set"])), *("--manifest", str(world["manifest"]))]
    argv = [*base, "--genuine", f"test={world['genuine_test']}", "--out", str(out)]
    assert cr.main(argv) == 2
    captured = capsys.readouterr()
    assert "never reads test or shift" in captured.err and captured.out == ""
    assert not out.exists()


def test_compare_rules_cannot_be_fooled_by_a_name(world, capsys):
    out = world["tmp"] / "comparison.json"
    argv = [
        *("--copies", str(world["tmp"] / "nonexistent")),
        *("--genuine", f"dev={world['genuine_test']}"),
        *("--manifest", str(world["manifest"])),
        *("--out", str(out)),
    ]
    assert cr.main(argv) == 2 and not out.exists()
    argv[1] = str(world["set"])
    assert cr.main(argv) == 2
    assert "recorded as 'test'" in capsys.readouterr().err and not out.exists()
