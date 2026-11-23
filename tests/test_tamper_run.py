"""The detector driver (tamper/run.py) and the detector fingerprint (tamper/fingerprint.py).

Most tests swap in a fast fake detector, so they test the driver's own logic (chunks,
resume, refusals, the split guard) without reading any PDF. One test runs the real
detector on a few PDFs alone in a folder. Everything is in temp folders."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from statementscreen import splits
from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.splits import OPEN_TEST_ENV
from statementscreen.tamper import copy as tc
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import run as driver
from statementscreen.tamper.detect.findings import Finding
from statementscreen.writer import write_batch

NAMES = [f"x_{n:02d}" for n in range(1, 6)]


def _pdfs(folder: Path, names: list[str]) -> Path:
    """Empty files named like PDFs: enough for a fake detector, which never opens them."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / f"{name}.pdf").write_bytes(b"")
    return folder


@pytest.fixture
def fake(monkeypatch):
    """A fast detector: one weak finding per PDF. `calls` lists the PDFs it was asked for."""
    state = {"calls": [], "interrupt_at": None, "crash_at": None}

    def collect(path):
        stem = Path(path).stem
        state["calls"].append(stem)
        if stem == state["interrupt_at"]:
            raise KeyboardInterrupt
        if stem == state["crash_at"]:
            raise RuntimeError("boom")
        return [Finding("DIGIT_SPACING", "layout", 1, None, "weak", f"fake finding for {stem}")]

    monkeypatch.setattr(driver, "collect_findings", collect)
    return state


def _argv(folder: Path, out: Path, *extra: str) -> list[str]:
    return [str(folder), "--out", str(out), *extra]


def _items(out: Path) -> list[str]:
    return pl.read_parquet(out / "items.parquet")["item"].to_list()


def test_writes_the_three_files_and_lists_every_pdf(tmp_path, fake):
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    assert driver.main(_argv(folder, out)) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(driver.OUTPUTS)  # no _parts left
    assert _items(out) == NAMES
    findings = pl.read_parquet(out / "findings.parquet")
    assert findings.columns == list(driver.FINDING_SCHEMA) and findings.height == 5
    run = json.loads((out / "detect_run.json").read_text(encoding="utf-8"))
    assert run["pdfs"] == 5 and run["crashed"] == [] and run["items_with_findings"] == 5
    assert set(run["fingerprint"]) == {"code", "config", "model", "uv_lock"}
    assert set(run["seconds_per_pdf"]) == {"mean", "p95", "max"}


def test_an_interrupted_run_resumes_without_repeating_finished_chunks(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(driver, "CHUNK", 2)  # chunks: x_01 x_02 | x_03 x_04 | x_05
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    fake["interrupt_at"] = "x_04"
    with pytest.raises(KeyboardInterrupt):
        driver.main(_argv(folder, out))
    assert not (out / "findings.parquet").exists() and (out / "_parts").exists()
    fake["interrupt_at"] = None
    fake["calls"].clear()
    assert driver.main(_argv(folder, out, "--resume")) == 0
    assert fake["calls"] == ["x_03", "x_04", "x_05"]  # the finished chunk was not repeated
    assert _items(out) == NAMES
    run = json.loads((out / "detect_run.json").read_text(encoding="utf-8"))
    assert run["resumed_chunks"] == 1


def test_resume_refuses_when_the_detector_changed(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setattr(driver, "CHUNK", 2)
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    fake["interrupt_at"] = "x_04"
    with pytest.raises(KeyboardInterrupt):
        driver.main(_argv(folder, out))
    monkeypatch.setattr(driver, "fingerprint", lambda: {**fp.fingerprint(), "code": "other"})
    assert driver.main(_argv(folder, out, "--resume")) == 2
    assert "code differ" in capsys.readouterr().err


def test_a_finished_run_is_not_overwritten_without_the_flag(tmp_path, fake):
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    assert driver.main(_argv(folder, out)) == 0
    assert driver.main(_argv(folder, out)) == 2
    assert driver.main(_argv(folder, out, "--resume")) == 2  # nothing left to resume
    assert driver.main(_argv(folder, out, "--overwrite")) == 0


def test_an_interrupted_run_needs_resume_or_overwrite(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(driver, "CHUNK", 2)
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    fake["interrupt_at"] = "x_04"
    with pytest.raises(KeyboardInterrupt):
        driver.main(_argv(folder, out))
    assert driver.main(_argv(folder, out)) == 2  # neither flag: refused
    fake["interrupt_at"] = None
    assert driver.main(_argv(folder, out, "--overwrite")) == 0
    assert _items(out) == NAMES


def test_a_crash_is_listed_and_the_other_pdfs_are_unaffected(tmp_path, fake):
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    fake["crash_at"] = "x_02"
    assert driver.main(_argv(folder, out)) == 0
    run = json.loads((out / "detect_run.json").read_text(encoding="utf-8"))
    assert run["crashed"] == ["x_02"]
    crashed = pl.read_parquet(out / "findings.parquet").filter(pl.col("item") == "x_02")
    assert crashed["code"].to_list() == ["CHECK_NOT_RUN"] and crashed["strength"][0] == "info"
    assert "boom" in crashed["reason"][0]
    assert _items(out) == NAMES


def test_limit_runs_the_first_pdfs_only(tmp_path, fake):
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    assert driver.main(_argv(folder, out, "--limit", "2")) == 0
    assert _items(out) == NAMES[:2]


def test_a_folder_without_pdfs_or_a_missing_folder_is_an_error(tmp_path, fake):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert driver.main(_argv(empty, tmp_path / "o1")) == 2
    assert driver.main(_argv(tmp_path / "missing", tmp_path / "o2")) == 2
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists()


def _toy_split(tmp_path: Path) -> tuple[Path, Path, Path]:
    """24 statements in 6 strata, split by the real splitter; returns (pdfs, manifest, splits)."""
    kinds = ["salaried", "gig", "self_employed"]
    rows = ["statement_id,layout,customer_type"]
    for i in range(24):
        layout = "columns" if i % 2 else "drcr"
        rows.append(f"stmt_{i:04d},{layout},{kinds[(i // 2) % 3]}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assigned, meta = splits.build(manifest, 42)
    splits.write(assigned, meta, tmp_path / "splits")
    pdfs = _pdfs(tmp_path / "pdfs", [f"stmt_{i:04d}" for i in range(24)])
    return pdfs, manifest, tmp_path / "splits"


def test_split_runs_only_that_splits_statements_and_test_stays_closed(tmp_path, fake, monkeypatch):
    monkeypatch.delenv(OPEN_TEST_ENV, raising=False)
    pdfs, manifest, splits_dir = _toy_split(tmp_path)
    split_args = ["--manifest", str(manifest), "--splits-dir", str(splits_dir)]
    out = tmp_path / "dev_out"
    assert driver.main(_argv(pdfs, out, "--split", "dev", *split_args)) == 0
    assert _items(out) == splits.load_split("dev", manifest, splits_dir)
    assert 0 < len(_items(out)) < 24
    closed = tmp_path / "test_out"
    assert driver.main(_argv(pdfs, closed, "--split", "test", *split_args)) == 2
    assert not closed.exists()  # refused before anything was written
    assert driver.main(_argv(pdfs, tmp_path / "o3", "--split", "dev")) == 2  # no --manifest


# ---------------------------------------------------------------- the real detector


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 6), folder)
    return folder


def test_the_real_detector_runs_on_pdfs_alone_in_a_folder(batch, tmp_path):
    sources = sorted((batch / "pdfs").glob("*.pdf"))
    folder = tmp_path / "copies"
    folder.mkdir()
    shutil.copy(sources[0], folder / "copy_0001.pdf")
    shutil.copy(sources[1], folder / "copy_0002.pdf")
    for n, kind in ((3, "rerender_inconsistent"), (4, "inplace_edit")):
        data, _ = tc.tamper_copy(sources[n], kind, seed=3)
        (folder / f"copy_{n:04d}.pdf").write_bytes(data)
    # nothing but the four PDFs: no keys, manifest, labels or parquet to read
    assert sorted(p.name for p in folder.iterdir()) == [f"copy_{n:04d}.pdf" for n in range(1, 5)]
    out = tmp_path / "detect"
    assert driver.main(_argv(folder, out)) == 0
    assert _items(out) == [f"copy_{n:04d}" for n in range(1, 5)]
    findings = pl.read_parquet(out / "findings.parquet")
    strong = set(findings.filter(pl.col("strength") == "strong")["item"].to_list())
    assert strong == {"copy_0003", "copy_0004"}  # the two tampered copies, not the genuine ones
    items = pl.read_parquet(out / "items.parquet")
    assert items["seconds"].min() > 0 and not items["crashed"].any()


# ---------------------------------------------------------------- the fingerprint


def _detector_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    detect = tmp_path / "detect"
    detect.mkdir()
    for name in ("a.py", "b.py", "verdict.py", "config.py"):
        (detect / name).write_text(f"# {name}")
    model, lock = tmp_path / "model.pkl", tmp_path / "uv.lock"
    model.write_bytes(b"model")
    lock.write_text("lock")
    return detect, model, lock


def test_the_fingerprint_names_which_part_changed(tmp_path):
    detect, model, lock = _detector_files(tmp_path)
    base = fp.fingerprint(detect, model, lock)
    assert fp.fingerprint(detect, model, lock) == base and set(base) == set(fp.PARTS)

    (detect / "verdict.py").write_text("# new verdict rules")  # rules are not part of it
    assert fp.differences(base, fp.fingerprint(detect, model, lock)) == []
    (detect / "a.py").write_text("# a check changed")
    assert fp.differences(base, fp.fingerprint(detect, model, lock)) == ["code"]
    (detect / "config.py").write_text("# a tolerance changed")
    assert fp.differences(base, fp.fingerprint(detect, model, lock)) == ["code", "config"]
    model.write_bytes(b"retrained")
    lock.write_text("new versions")
    now = fp.fingerprint(detect, model, lock)
    assert fp.differences(base, now) == ["code", "config", "model", "uv_lock"]


def test_a_missing_file_hashes_as_missing_and_counts_as_different(tmp_path):
    detect, model, lock = _detector_files(tmp_path)
    base = fp.fingerprint(detect, model, lock)
    model.unlink()
    now = fp.fingerprint(detect, model, lock)
    assert now["model"] == "missing" and fp.differences(base, now) == ["model"]


def test_check_refuses_stale_findings_and_says_which_part(tmp_path):
    detect, model, lock = _detector_files(tmp_path)
    recorded = fp.fingerprint(detect, model, lock)
    fp.check(recorded, fp.fingerprint(detect, model, lock))  # same detector: no error
    model.write_bytes(b"retrained")
    with pytest.raises(fp.StaleFindings) as err:
        fp.check(recorded, fp.fingerprint(detect, model, lock))
    assert "model differs" in str(err.value) and "Run the detector again" in str(err.value)
    assert "code differs" not in str(err.value)
    with pytest.raises(fp.StaleFindings, match="no detector fingerprint"):
        fp.check(None)


def test_the_real_fingerprint_is_stable_and_complete():
    first = fp.fingerprint()
    assert first == fp.fingerprint()
    assert set(first) == set(fp.PARTS) and "missing" not in first.values()


# ---------------------------------------------------------------- independence


def test_nothing_in_detect_imports_the_driver_or_the_split_loader():
    forbidden = ("statementscreen.tamper.run", "statementscreen.splits")
    for path in sorted(fp.DETECT_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"


def test_the_run_records_its_population_and_any_limit(tmp_path, fake):
    folder = _pdfs(tmp_path / "pdfs", NAMES)
    out = tmp_path / "out"
    assert driver.main(_argv(folder, out, "--population", "shift")) == 0
    run = json.loads((out / "detect_run.json").read_text(encoding="utf-8"))
    assert run["population"] == "shift" and run["limit"] is None and run["split"] is None
    trial = tmp_path / "trial"
    assert driver.main(_argv(folder, trial, "--population", "shift", "--limit", "2")) == 0
    run = json.loads((trial / "detect_run.json").read_text(encoding="utf-8"))
    assert run["limit"] == 2 and run["pdfs"] == 2


def test_a_population_that_contradicts_the_split_is_refused(tmp_path, fake, capsys):
    folder = _pdfs(tmp_path / "pdfs", NAMES)
    argv = _argv(folder, tmp_path / "out", "--split", "dev", "--population", "shift")
    assert driver.main(argv) == 2
    assert "contradicts" in capsys.readouterr().err


def test_every_pdf_gets_a_sha256_in_items(tmp_path, fake):
    folder, out = _pdfs(tmp_path / "pdfs", NAMES), tmp_path / "out"
    assert driver.main(_argv(folder, out)) == 0
    items = pl.read_parquet(out / "items.parquet")
    assert items["sha256"].to_list() == [hashlib.sha256(b"").hexdigest()] * len(NAMES)
    (folder / "x_03.pdf").write_bytes(b"changed")
    again = tmp_path / "again"
    assert driver.main(_argv(folder, again)) == 0
    items_again = pl.read_parquet(again / "items.parquet")
    hashes = dict(zip(items_again["item"], items_again["sha256"], strict=True))
    assert hashes["x_03"] == hashlib.sha256(b"changed").hexdigest()


def test_test_pdfs_are_not_read_without_the_split_or_the_env_var(
    tmp_path, fake, monkeypatch, capsys
):
    monkeypatch.setattr(driver.provenance, "split_test_ids", lambda splits_dir=None: {"x_02"})
    monkeypatch.delenv("STATEMENTSCREEN_OPEN_TEST", raising=False)
    folder = _pdfs(tmp_path / "pdfs", NAMES)
    assert driver.main(_argv(folder, tmp_path / "a")) == 2
    assert "are test statements" in capsys.readouterr().err
    relabelled = _argv(folder, tmp_path / "b", "--population", "dev")
    assert driver.main(relabelled) == 2
    assert "are test statements" in capsys.readouterr().err
    as_test = _argv(folder, tmp_path / "c", "--population", "test")
    assert driver.main(as_test) == 2
    assert "STATEMENTSCREEN_OPEN_TEST=1" in capsys.readouterr().err
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    # No test may set the real open-test variable (tests/test_stress.py enforces it), so the
    # allowed path is exercised with a stand-in variable name.
    monkeypatch.setattr(driver, "OPEN_TEST_ENV", "STAND_IN_FOR_THE_GATE")
    monkeypatch.setenv("STAND_IN_FOR_THE_GATE", "1")
    assert driver.main(as_test) == 0


def test_a_folder_without_test_statements_runs_under_any_population(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(driver.provenance, "split_test_ids", lambda splits_dir=None: {"nope"})
    folder = _pdfs(tmp_path / "pdfs", NAMES)
    assert driver.main(_argv(folder, tmp_path / "out", "--population", "dev")) == 0


def test_the_shift_batch_may_reuse_main_batch_ids_only_from_its_own_folder(
    tmp_path, fake, monkeypatch
):
    # the shift batch's ids (stmt_0001..) overlap the main batch's test ids
    monkeypatch.setattr(driver.provenance, "split_test_ids", lambda splits_dir=None: {"x_02"})
    folder = _pdfs(tmp_path / "shift_pdfs", NAMES)
    other = _pdfs(tmp_path / "main_pdfs", NAMES)
    monkeypatch.setattr(driver.provenance, "EXPECTED_SOURCES", {"shift": str(folder), "test": "t"})
    assert driver.main(_argv(folder, tmp_path / "a", "--population", "shift")) == 0
    # the same label on any other folder is not exempt
    assert driver.main(_argv(other, tmp_path / "b", "--population", "shift")) == 2
    assert not (tmp_path / "b").exists()
