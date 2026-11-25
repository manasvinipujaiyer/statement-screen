"""The Week 7 blind detector run and verdict step: what they refuse; neither opens the key."""

import builtins
import csv
import io
import json
import os
from pathlib import Path

import polars as pl
import pytest

from statementscreen.tamper import blind_detect as bd
from statementscreen.tamper import blind_verdicts as bv
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import KEY_NAME, BlindError, pdf_hashes, pdf_set_sha256
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.verdict import NONE, RULE_VERSION, SUSPICIOUS, TAMPERED
from statementscreen.tamper.run import FINDING_SCHEMA, ITEM_SCHEMA
from statementscreen.tamper.scoring import COULD_NOT_CHECK

STRONG = Finding("BALANCE_BREAK", "arithmetic", 1, 3, "strong", "a balance does not follow")
ANOMALY = Finding("ANOMALY_SCORE", "anomaly", None, None, "weak", "an unusual money pattern")
HISTORY = Finding("INCREMENTAL_SAVE", "history", None, None, "weak", "saved incrementally")
NOT_RUN = Finding("CHECK_NOT_RUN", "arithmetic", None, None, "info", "closing balance not read")


@pytest.fixture
def key_guard(monkeypatch):
    """Reading any path named sealed_key.json fails, whichever of the five ways it is opened.
    Returns the list of attempts, which must stay empty."""
    attempts: list[str] = []

    def guarded(real, has_mode=True):
        def wrapper(*args, **kwargs):
            first = args[0] if args else kwargs.get("file")
            is_path = isinstance(first, str | bytes | os.PathLike)
            mode = str(args[1] if has_mode and len(args) > 1 else kwargs.get("mode", "r"))
            writing = has_mode and any(c in mode for c in "wax")  # the test setup may write
            if is_path and not writing and Path(os.fsdecode(first)).name == KEY_NAME:
                attempts.append(str(first))
                raise PermissionError("the sealed key is not readable here")
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))
    monkeypatch.setattr(Path, "open", guarded(Path.open))
    monkeypatch.setattr(Path, "read_text", guarded(Path.read_text, has_mode=False))
    monkeypatch.setattr(Path, "read_bytes", guarded(Path.read_bytes, has_mode=False))
    return attempts


def _blind_folder(tmp_path: Path, n: int = 3) -> tuple[Path, dict, dict]:
    """tmp/blind/pdfs with n opaque PDFs, a key file beside them, and a matching prereg."""
    root = tmp_path / "blind"
    pdfs = root / "pdfs"
    pdfs.mkdir(parents=True)
    for k in range(1, n + 1):
        (pdfs / f"blind_{k:04d}.pdf").write_bytes(f"pdf {k}".encode())
    (root / KEY_NAME).write_text('{"secret": true}\n', encoding="utf-8")
    hashes = pdf_hashes(pdfs)
    prereg = {
        "sealed": {"pdfs": n, "pdf_set_sha256": pdf_set_sha256(hashes)},
        "detector_fingerprint": fp.fingerprint(),
        "rule_version": RULE_VERSION,
    }
    return pdfs, prereg, hashes


def _detect_output(folder: Path, hashes: dict, findings: dict | None = None, **info) -> Path:
    """What tamper/run.py writes, built by hand, so the verdict step can be tested alone."""
    findings = findings or {}
    folder.mkdir()
    items = [
        {
            "item": n,
            "seconds": 0.1,
            "peak_memory_mb": 1.0,
            "crashed": False,
            "sha256": h,
            "n_findings": len(findings.get(n, [])),
        }  # fmt: skip
        for n, h in sorted(hashes.items())
    ]
    rows = [{"item": n, **f.to_dict()} for n, fs in findings.items() for f in fs]
    pl.DataFrame(items, schema=ITEM_SCHEMA).write_parquet(folder / "items.parquet")
    pl.DataFrame(rows, schema=FINDING_SCHEMA).write_parquet(folder / "findings.parquet")
    run = {"population": "blind", "limit": None, "pdfs": len(hashes), "crashed": [],
           "fingerprint": fp.fingerprint(), **info}  # fmt: skip
    (folder / "detect_run.json").write_text(json.dumps(run), encoding="utf-8")
    return folder


# ---------------------------------------------------------------- the key guard itself


@pytest.mark.parametrize(
    "read",
    [
        lambda p: builtins.open(p),
        lambda p: io.open(p),  # noqa: UP020  (the test must call the io.open name itself)
        lambda p: Path(p).open(),
        lambda p: Path(p).read_text(),
        lambda p: Path(p).read_bytes(),
    ],
    ids=["builtins.open", "io.open", "Path.open", "Path.read_text", "Path.read_bytes"],
)
def test_the_guard_makes_every_way_of_reading_the_key_fail(tmp_path, key_guard, read):
    pdfs, _, _ = _blind_folder(tmp_path)
    with pytest.raises(PermissionError, match="not readable"):
        read(pdfs.parent / KEY_NAME)
    assert len(key_guard) == 1
    (tmp_path / "other.txt").write_text("fine", encoding="utf-8")
    other = read(tmp_path / "other.txt")  # any other file still opens
    assert other is not None
    if hasattr(other, "close"):
        other.close()


# ---------------------------------------------------------------- the verdict step


def test_the_verdict_step_passes_when_every_way_of_reading_the_key_fails(tmp_path, key_guard):
    pdfs, prereg, hashes = _blind_folder(tmp_path)
    detect = _detect_output(tmp_path / "blind" / "detect", hashes, {"blind_0001": [STRONG]})
    rows = bv.make_verdicts(prereg, detect, pdfs)
    digest = bv.write_verdicts(rows, tmp_path / "out" / "verdicts.csv")
    assert len(rows) == 3 and len(digest) == 64
    assert key_guard == []  # nothing even tried


def test_the_verdict_command_never_opens_the_key(tmp_path, key_guard, monkeypatch):
    pdfs, prereg, hashes = _blind_folder(tmp_path)
    detect = _detect_output(tmp_path / "blind" / "detect", hashes)
    monkeypatch.setattr(bv.bpr, "load_and_verify", lambda path: prereg)
    out = tmp_path / "out" / "verdicts.csv"
    argv = ["--detect", str(detect), "--pdfs", str(pdfs), "--out", str(out)]
    assert bv.main([*argv, "--preregistration", "unused.json"]) == 0
    assert out.is_file() and key_guard == []


def test_a_verdict_follows_the_findings_under_the_rule_version(tmp_path):
    pdfs, prereg, hashes = _blind_folder(tmp_path, n=5)
    findings = {
        "blind_0001": [STRONG],
        "blind_0002": [ANOMALY],
        "blind_0003": [NOT_RUN],
        "blind_0004": [STRONG, HISTORY],
    }
    detect = _detect_output(tmp_path / "detect", hashes, findings)
    rows = {r["name"]: r for r in bv.make_verdicts(prereg, detect, pdfs)}
    assert [rows[f"blind_000{k}"]["verdict"] for k in range(1, 6)] == [
        TAMPERED, SUSPICIOUS, COULD_NOT_CHECK, TAMPERED, NONE,
    ]  # fmt: skip
    assert rows["blind_0001"]["finding_codes"] == "BALANCE_BREAK"
    assert rows["blind_0003"]["checks_not_run"] == "arithmetic"
    assert rows["blind_0004"]["verdict_without_history"] == TAMPERED
    assert rows["blind_0001"]["pdf_sha256"] == hashes["blind_0001"]
    assert "a balance does not follow" in rows["blind_0001"]["reasons"]


def test_the_csv_and_its_hash_are_written_once(tmp_path):
    pdfs, prereg, hashes = _blind_folder(tmp_path)
    detect = _detect_output(tmp_path / "detect", hashes)
    out = tmp_path / "out" / "verdicts.csv"
    digest = bv.write_verdicts(bv.make_verdicts(prereg, detect, pdfs), out)
    assert out.with_suffix(".sha256").read_text(encoding="utf-8") == f"{digest}  verdicts.csv\n"
    with out.open(newline="", encoding="utf-8") as f:
        assert tuple(next(csv.reader(f))) == bv.COLUMNS
    with pytest.raises(BlindError, match="never overwritten"):
        bv.write_verdicts([], out)


def test_the_verdict_step_refuses_output_that_is_not_the_frozen_detectors(tmp_path):
    pdfs, prereg, hashes = _blind_folder(tmp_path)
    other = {**fp.fingerprint(), "model": "another model"}
    with pytest.raises(fp.StaleFindings, match="model differs"):
        bv.make_verdicts(prereg, _detect_output(tmp_path / "d1", hashes, fingerprint=other), pdfs)
    with pytest.raises(BlindError, match="not a full blind run"):
        bv.make_verdicts(prereg, _detect_output(tmp_path / "d2", hashes, limit=2), pdfs)
    with pytest.raises(BlindError, match="not a full blind run"):
        bv.make_verdicts(prereg, _detect_output(tmp_path / "d3", hashes, population="test"), pdfs)


def test_the_verdict_step_refuses_output_for_other_pdfs(tmp_path):
    pdfs, prereg, hashes = _blind_folder(tmp_path)
    missing = {k: v for k, v in hashes.items() if k != "blind_0003"}
    with pytest.raises(BlindError, match="does not cover exactly"):
        bv.make_verdicts(prereg, _detect_output(tmp_path / "d1", missing), pdfs)
    changed = {**hashes, "blind_0002": "0" * 64}
    with pytest.raises(BlindError, match="changed since the detector read them"):
        bv.make_verdicts(prereg, _detect_output(tmp_path / "d2", changed), pdfs)


def test_the_verdict_step_refuses_pdfs_that_are_not_the_sealed_set(tmp_path):
    pdfs, prereg, hashes = _blind_folder(tmp_path)
    detect = _detect_output(tmp_path / "detect", hashes)
    (pdfs / "blind_0001.pdf").write_bytes(b"changed")
    with pytest.raises(bv.bpr.BlindPreregError, match="not the sealed set"):
        bv.make_verdicts(prereg, detect, pdfs)


def test_the_verdict_command_reports_a_refused_preregistration(tmp_path, monkeypatch, capsys):
    def refused(path):
        raise bv.bpr.BlindPreregError("changed since the pre-registration: x.py changed")

    monkeypatch.setattr(bv.bpr, "load_and_verify", refused)
    assert bv.main(["--out", str(tmp_path / "v.csv")]) == 2
    assert "x.py changed" in capsys.readouterr().err
    assert not (tmp_path / "v.csv").exists()


# ---------------------------------------------------------------- the detector run


def test_the_detector_runs_once_on_the_sealed_pdfs_and_never_opens_the_key(
    tmp_path, key_guard, monkeypatch, capsys
):
    pdfs, prereg, hashes = _blind_folder(tmp_path)
    monkeypatch.setattr(bd.bpr, "load_and_verify", lambda path: prereg)
    out = tmp_path / "blind" / "detect"
    argv = ["--pdfs", str(pdfs), "--out", str(out), "--preregistration", "unused.json"]
    assert bd.main(argv) == 0
    assert "Ran the detector on 3 PDFs" in capsys.readouterr().out
    info = json.loads((out / "detect_run.json").read_text(encoding="utf-8"))
    assert info["population"] == bd.POPULATION == "blind" and info["pdfs"] == 3
    assert info["fingerprint"] == prereg["detector_fingerprint"]
    items = pl.read_parquet(out / "items.parquet")
    assert dict(zip(items["item"], items["sha256"], strict=True)) == hashes
    assert bd.main(argv) == 2  # a finished run is never overwritten
    assert "already holds a finished run" in capsys.readouterr().err
    assert key_guard == []


def test_the_detector_refuses_pdfs_that_are_not_the_sealed_set(tmp_path, monkeypatch, capsys):
    pdfs, prereg, _ = _blind_folder(tmp_path)
    (pdfs / "blind_0002.pdf").write_bytes(b"changed")
    monkeypatch.setattr(bd.bpr, "load_and_verify", lambda path: prereg)
    out = tmp_path / "blind" / "detect"
    assert bd.main(["--pdfs", str(pdfs), "--out", str(out)]) == 2
    assert "not the sealed set" in capsys.readouterr().err
    assert not out.exists()


def test_the_detector_refuses_when_the_preregistration_is_refused(tmp_path, monkeypatch, capsys):
    def refused(path):
        raise bd.bpr.BlindPreregError("has not been pushed")

    monkeypatch.setattr(bd.bpr, "load_and_verify", refused)
    out = tmp_path / "detect"
    assert bd.main(["--pdfs", str(tmp_path), "--out", str(out)]) == 2
    assert "has not been pushed" in capsys.readouterr().err
    assert not out.exists()
