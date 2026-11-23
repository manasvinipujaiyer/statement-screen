"""The scoring command (tamper/score_tamper.py): refusals, the stop rule and the outputs.
Detector runs, keys and manifests are written by hand into temp folders."""

import csv
import hashlib
import json
from pathlib import Path

import polars as pl

from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import run as driver
from statementscreen.tamper import score_tamper as cli
from statementscreen.tamper.detect.findings import Finding

STRONG = Finding("BALANCE_CHAIN", "arithmetic", 1, 0, "strong", "balance broken")
COPIES = [f"copy_{n:04d}" for n in range(1, 5)]
STATEMENTS = [f"stmt_{n:04d}" for n in range(1, 6)]


def _write_run(folder: Path, items: list[str], findings: dict[str, list[Finding]], **extra):
    """A finished detector run, as tamper/run.py would have written it."""
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
            "sha256": [f"sha_{i}" for i in items],
        },
        schema=driver.ITEM_SCHEMA,
    ).write_parquet(folder / "items.parquet")
    summary = {
        "fingerprint": fp.fingerprint(),
        "pdfs": n,
        "seconds_per_pdf": {"mean": 1.0, "p95": 1.0, "max": 1.0},
        "peak_memory_mb": 100.0,
    } | extra
    (folder / "detect_run.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_keys(set_dir: Path) -> None:
    keys = set_dir / "keys"
    keys.mkdir(parents=True)
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
        (keys / f"{copy_id}.json").write_text(json.dumps(key), encoding="utf-8")


def _manifest(path: Path) -> Path:
    rows = ["statement_id,layout,customer_type"]
    rows += [f"{s},{'columns' if n % 2 else 'drcr'},salaried" for n, s in enumerate(STATEMENTS)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_build(set_dir: Path, name: str) -> None:
    """The build's own record (build_run.json), as tamper.build would have written it."""
    build = {"set": name, "per_type": 4, "made": {"inplace_edit": 4}, "short": {}}
    (set_dir / "build_run.json").write_text(json.dumps(build | {"complete": True}), "utf-8")


def _setup(
    tmp_path: Path, genuine_findings: dict | None = None, genuine_split: str = "dev"
) -> dict:
    """copies 1-3 caught (strong), copy 4 missed; five genuine statements."""
    set_dir = tmp_path / "dev"
    _write_keys(set_dir)
    _write_build(set_dir, "dev")
    _write_run(set_dir / "detect", COPIES, {c: [STRONG] for c in COPIES[:3]})
    _write_run(tmp_path / "genuine_dev", STATEMENTS, genuine_findings or {}, split=genuine_split)
    return {
        "set": set_dir,
        "genuine": tmp_path / "genuine_dev",
        "manifest": _manifest(tmp_path / "manifest.csv"),
        "out": tmp_path / "reports" / "dev.json",
        "table": tmp_path / "table.csv",
    }


def _argv(s: dict, *extra: str) -> list[str]:
    return [
        *("--copies", str(s["set"])),
        *("--genuine", f"dev={s['genuine']}"),
        *("--manifest", str(s["manifest"])),
        *("--out", str(s["out"])),
        *extra,
    ]


def test_scores_the_run_and_writes_the_report_and_the_table(tmp_path, capsys):
    s = _setup(tmp_path)
    assert cli.main(_argv(s, "--table", str(s["table"]), "--table-extended")) == 0
    report = json.loads(s["out"].read_text(encoding="utf-8"))
    d = report["detection"]["inplace_edit"]["with_history"]["tampered"]
    assert (d["count"], d["total"], d["rate"]) == (3, 4, 0.75)
    g = report["false_alarms"]["genuine"]["dev"]["with_history"]["tampered_or_suspicious"]
    assert g["count"] == 0 and g["total"] == 5
    assert set(report["inputs"]) == {"copies", "genuine_dev"}
    assert report["inputs"]["copies"]["seconds_per_pdf"]["mean"] == 1.0
    assert [m["item"] for m in report["misses"]] == ["copy_0004"]
    with s["table"].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["copy"] for r in rows] == COPIES and "verdict_without_history" in rows[0]
    out = capsys.readouterr().out
    assert "inplace_edit" in out and "genuine dev: the anomaly threshold was set" in out


def test_a_report_is_not_overwritten_without_the_flag(tmp_path):
    s = _setup(tmp_path)
    assert cli.main(_argv(s)) == 0
    assert cli.main(_argv(s)) == 2
    assert cli.main(_argv(s, "--overwrite")) == 0


def test_findings_from_a_different_detector_are_refused_naming_the_part(tmp_path, capsys):
    s = _setup(tmp_path)
    run_file = s["set"] / "detect" / "detect_run.json"
    run = json.loads(run_file.read_text(encoding="utf-8"))
    run["fingerprint"]["model"] = "a different model"
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert cli.main(_argv(s)) == 2
    err = capsys.readouterr().err
    assert "model differs" in err and "code differs" not in err
    assert not s["out"].exists()


def test_a_run_that_does_not_cover_the_copies_is_refused(tmp_path, capsys):
    s = _setup(tmp_path)
    (s["set"] / "detect" / "items.parquet").unlink()
    _write_run_items_only(s["set"] / "detect", COPIES[:3])
    assert cli.main(_argv(s)) == 2
    assert "1 not run" in capsys.readouterr().err


def _write_run_items_only(folder: Path, items: list[str]) -> None:
    n = len(items)
    pl.DataFrame(
        {
            "item": items,
            "seconds": [1.0] * n,
            "peak_memory_mb": [100.0] * n,
            "n_findings": [0] * n,
            "crashed": [False] * n,
            "sha256": [f"sha_{i}" for i in items],
        },
        schema=driver.ITEM_SCHEMA,
    ).write_parquet(folder / "items.parquet")


def test_a_missing_run_and_bad_arguments_are_refused(tmp_path, capsys):
    s = _setup(tmp_path)
    assert cli.main(["--manifest", str(s["manifest"]), "--out", str(s["out"])]) == 2
    assert "nothing to score" in capsys.readouterr().err
    bad = ["--genuine", "no_equals_sign", "--manifest", str(s["manifest"]), "--out", str(s["out"])]
    assert cli.main(bad) == 2 and "NAME=RUN_DIR" in capsys.readouterr().err
    empty = ["--copies", str(tmp_path / "none"), "--manifest", str(s["manifest"])]
    assert cli.main([*empty, "--out", str(s["out"])]) == 2
    assert "no finished detector run" in capsys.readouterr().err


def test_a_genuine_statement_missing_from_the_manifest_is_refused(tmp_path, capsys):
    s = _setup(tmp_path)
    s["manifest"].write_text("statement_id,layout,customer_type\nstmt_0001,columns,x\n")
    assert cli.main(_argv(s)) == 2
    assert "not in the manifest" in capsys.readouterr().err


def test_a_strong_finding_on_a_genuine_pdf_stops_with_exit_code_3(tmp_path, capsys):
    s = _setup(tmp_path, {"stmt_0003": [STRONG]})
    assert cli.main(_argv(s)) == 3
    err = capsys.readouterr().err
    assert "STOP" in err and "stmt_0003" in err and "BALANCE_CHAIN" in err
    report = json.loads(s["out"].read_text(encoding="utf-8"))  # still written, for inspection
    assert [c["item"] for c in report["genuine_with_strong_findings"]] == ["stmt_0003"]


def test_the_genuine_train_row_is_labelled_in_sample(tmp_path, capsys):
    s = _setup(tmp_path, genuine_split="train")
    argv = _argv(s)
    argv[argv.index(f"dev={s['genuine']}")] = f"train={s['genuine']}"
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "genuine train (in-sample for anomaly)" in out and "without anomaly" in out


def _add_extra(tmp_path: Path) -> Path:
    """A second genuine population whose statements are only in its own manifest."""
    _write_run(tmp_path / "genuine_extra", ["extra_0001", "extra_0002"], {}, population="extra")
    manifest = tmp_path / "extra_manifest.csv"
    manifest.write_text(
        "statement_id,layout,customer_type\nextra_0001,columns,x\nextra_0002,drcr,x\n",
        encoding="utf-8",
    )
    return manifest


def test_each_population_can_have_its_own_manifest(tmp_path):
    s = _setup(tmp_path)
    extra_manifest = _add_extra(tmp_path)
    argv = _argv(
        s,
        *("--genuine", f"extra={tmp_path / 'genuine_extra'}"),
        *("--genuine-manifest", f"extra={extra_manifest}"),
    )
    assert cli.main(argv) == 0
    report = json.loads(s["out"].read_text(encoding="utf-8"))
    assert report["inputs"]["genuine_dev"]["manifest"] == str(s["manifest"])
    assert report["inputs"]["genuine_extra"]["manifest"] == str(extra_manifest)
    assert report["false_alarms"]["genuine"]["extra"]["with_history"]["tampered"]["total"] == 2
    assert report["false_alarms"]["genuine"]["dev"]["with_history"]["tampered"]["total"] == 5


def test_a_population_is_not_found_in_another_populations_manifest(tmp_path, capsys):
    s = _setup(tmp_path)
    _add_extra(tmp_path)  # these ids are not in the default manifest
    argv = _argv(s, "--genuine", f"extra={tmp_path / 'genuine_extra'}")
    assert cli.main(argv) == 2
    assert "not in the manifest" in capsys.readouterr().err


def test_a_population_with_no_manifest_at_all_is_refused(tmp_path, capsys):
    s = _setup(tmp_path)
    argv = ["--genuine", f"dev={s['genuine']}", "--out", str(s["out"])]
    assert cli.main(argv) == 2
    assert "no manifest" in capsys.readouterr().err


def test_bad_genuine_manifest_arguments_are_refused(tmp_path, capsys):
    s = _setup(tmp_path)
    m = str(s["manifest"])
    for extra, message in (
        (["--genuine-manifest", "nopath"], "NAME=PATH"),
        (["--genuine-manifest", f"ghost={m}"], "not given with --genuine"),
        (["--genuine-manifest", f"dev={m}", "--genuine-manifest", f"dev={m}"], "twice"),
        (["--genuine", f"dev={s['genuine']}"], "twice"),
    ):
        assert cli.main(_argv(s, *extra)) == 2
        assert message in capsys.readouterr().err


def test_the_report_records_the_hashes_of_its_inputs(tmp_path):
    s = _setup(tmp_path)
    assert cli.main(_argv(s)) == 0
    inputs = json.loads(s["out"].read_text(encoding="utf-8"))["inputs"]
    assert set(inputs["copies"]["sha256"]) == {"keys", "build_run.json", "detect"}
    assert set(inputs["copies"]["sha256"]["detect"]) == {*driver.OUTPUTS, "pdfs"}
    assert set(inputs["genuine_dev"]["sha256"]) == {"manifest", "detect"}
    assert (
        inputs["genuine_dev"]["sha256"]["manifest"]
        == hashlib.sha256(s["manifest"].read_bytes()).hexdigest()
    )


def test_a_crashed_pdf_is_shown_as_could_not_check_in_the_summary(tmp_path, capsys):
    s = _setup(tmp_path)
    run_file = s["genuine"] / "detect_run.json"
    run = json.loads(run_file.read_text(encoding="utf-8"))
    run["crashed"] = ["stmt_0002"]
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert cli.main(_argv(s)) == 0
    out = capsys.readouterr().out
    assert "could not check 1/5 (20.0%)" in out and "could not check 0/4" in out
    assert "Checks that did not run" in out and "parse 1" in out
    report = json.loads(s["out"].read_text(encoding="utf-8"))
    assert [c["item"] for c in report["could_not_check_cases"]] == ["stmt_0002"]
