"""The Week 7 blind pre-registration: what it records, and what verifying it refuses."""

import json
from pathlib import Path

import pytest

from statementscreen.tamper import blind_prereg as bp
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import pdf_hashes, pdf_set_sha256
from statementscreen.tamper.detect.verdict import RULE_VERSION

KEY = "a" * 64
SET = "b" * 64
HEAD = "c" * 40
LIVE_FINGERPRINT = fp.fingerprint  # the real function, kept before the fixture below replaces it


def _week6_fingerprint() -> dict:
    prereg = json.loads((bp.ROOT / bp.WEEK6_PREREG).read_text(encoding="utf-8"))
    return prereg["detector_fingerprint"]


@pytest.fixture(autouse=True)
def week6_detector(monkeypatch):
    """The logic tests below run against Week 6's recorded detector fingerprint, not the live
    one: uv.lock gained the service's web dependencies in Week 7 Part 2, so the live uv_lock
    part no longer equals Week 6's and never will again. The last test in this file still
    holds the live code, config and model to Week 6's record."""
    recorded = _week6_fingerprint()
    monkeypatch.setattr(fp, "fingerprint", lambda *args, **kwargs: dict(recorded))


def _build(**kwargs) -> dict:
    args = {"head": HEAD, "dirty": "", **kwargs}
    return bp.build(KEY, KEY, SET, **args)


def _write(tmp_path: Path, prereg: dict) -> Path:
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- what it records


def test_it_records_the_seal_the_fingerprint_the_rule_and_the_frozen_files():
    prereg = _build()
    week6 = json.loads((bp.ROOT / bp.WEEK6_PREREG).read_text(encoding="utf-8"))
    sealed = prereg["sealed"]
    assert prereg["frozen_commit"] == HEAD and prereg["rule_version"] == RULE_VERSION
    assert sealed["key_sha256"] == KEY and sealed["backup_sha256"] == KEY
    assert sealed["pdf_set_sha256"] == SET and sealed["pdfs"] == 200
    assert prereg["detector_fingerprint"] == week6["detector_fingerprint"]
    assert {s["main_seed"] for s in sealed["sources"].values()} == {7101, 7102}
    assert "src/statementscreen/tamper/blind_seal.py" in prereg["hashes"]["code"]
    assert set(prereg["hashes"]["artefacts"]) == set(bp.ARTEFACTS)


def test_it_names_every_command_the_expectation_and_the_rules():
    prereg = _build()
    commands = " ".join(prereg["commands"])
    for name in ("blind_detect", "blind_verdicts", "blind_score", "blind_log", "blind_verify"):
        assert name in commands
    assert "ALREADY RUN" in prereg["commands"][0]
    text = " ".join(prereg["expectations"])
    assert "No other numbers are pre-stated" in text and "rerender_consistent" in text
    rules = " ".join(prereg["rules"])
    assert "re-run the identical command once" in rules and "post-hoc" in rules
    assert "BEFORE the unseal step" in rules and "never read" in rules
    assert "log_run style limit" in " ".join(prereg["known_limits"])


def test_the_limits_carry_the_sentences_the_user_approved():
    limits = " ".join(_build()["known_limits"])
    assert "File sizes may differ between genuine and tampered PDFs;" in limits
    assert "the seal hides the mix from the scoring process, not from someone inspecting" in limits
    assert "stays on disk because the rebuild-and-verify check needs it" in limits
    assert "anyone can rebuild everything anyway" in limits


def test_uncommitted_changes_and_malformed_hashes_are_refused():
    with pytest.raises(ValueError, match="uncommitted"):
        bp.build(KEY, KEY, SET, head=HEAD, dirty=" M src/x.py")
    with pytest.raises(ValueError, match="64 lowercase hex"):
        bp.build("abc", KEY, SET, head=HEAD, dirty="")
    with pytest.raises(ValueError, match="different SHA-256"):
        bp.build(KEY, "d" * 64, SET, head=HEAD, dirty="")


def test_a_detector_that_differs_from_week_6_is_refused_when_writing():
    with pytest.raises(fp.StaleFindings, match="model differs"):
        _build(detector={**fp.fingerprint(), "model": "another model"})


# ---------------------------------------------------------------- verifying


def _verify(prereg: dict, **kwargs) -> None:
    kwargs.setdefault("ancestor", True)
    kwargs.setdefault("pushed", True)
    bp.verify(prereg, **kwargs)


def test_a_matching_preregistration_verifies():
    _verify(_build())


def test_a_changed_or_new_source_file_or_a_changed_artefact_is_refused():
    prereg = _build()
    first = next(iter(prereg["hashes"]["code"]))
    prereg["hashes"]["code"][first] = "0" * 64
    with pytest.raises(bp.BlindPreregError, match=f"{first} changed"):
        _verify(prereg)
    prereg = _build()
    del prereg["hashes"]["code"][first]
    with pytest.raises(bp.BlindPreregError, match=f"{first} is new"):
        _verify(prereg)
    prereg = _build()
    prereg["hashes"]["artefacts"]["uv.lock"] = "0" * 64
    with pytest.raises(bp.BlindPreregError, match="uv.lock changed"):
        _verify(prereg)


def test_a_different_detector_or_rule_version_is_refused():
    prereg = _build()
    prereg["detector_fingerprint"]["model"] = "another model"
    with pytest.raises(fp.StaleFindings, match="model differs"):
        _verify(prereg)
    with pytest.raises(bp.BlindPreregError, match="rule version"):
        _verify(_build(), rule_version="v1")


def test_a_detector_that_equals_the_recorded_one_but_not_week_6s_is_refused():
    changed = {**fp.fingerprint(), "config": "another config"}
    prereg = _build()
    prereg["detector_fingerprint"] = changed  # recorded and current agree with each other
    with pytest.raises(fp.StaleFindings, match="config differs"):
        _verify(prereg, detector=changed)


def test_a_commit_that_is_not_in_history_or_not_pushed_is_refused():
    with pytest.raises(bp.BlindPreregError, match="not in this history"):
        _verify(_build(), ancestor=False)
    with pytest.raises(bp.BlindPreregError, match="has not been pushed"):
        _verify(_build(), pushed=False)


# ---------------------------------------------------------------- the pushed copy


@pytest.mark.usefixtures("pushed")
def test_the_pushed_file_verifies_when_it_is_the_one_on_disk(tmp_path):
    assert bp.load_and_verify(_write(tmp_path, _build()))["frozen_commit"] == HEAD


def test_a_file_that_differs_from_the_pushed_one_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path, _build())
    monkeypatch.setattr(bp.pre, "pushed_text", lambda p, root=bp.ROOT: "{}\n")
    with pytest.raises(bp.BlindPreregError, match="differs from the pushed version"):
        bp.load_and_verify(path)


def test_a_file_that_is_not_pushed_or_missing_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path, _build())

    def not_pushed(p, root=bp.ROOT):
        raise bp.pre.PreregError("not on the pushed branch: commit and push it")

    monkeypatch.setattr(bp.pre, "pushed_text", not_pushed)
    with pytest.raises(bp.BlindPreregError, match="not on the pushed branch"):
        bp.load_and_verify(path)
    with pytest.raises(bp.BlindPreregError, match="no pre-registration"):
        bp.load_and_verify(tmp_path / "missing.json")


# ---------------------------------------------------------------- the sealed PDFs


def _pdfs(folder: Path, n: int = 3) -> dict:
    folder.mkdir()
    for k in range(1, n + 1):
        (folder / f"blind_{k:04d}.pdf").write_bytes(f"pdf {k}".encode())
    hashes = pdf_hashes(folder)
    return {"sealed": {"pdfs": n, "pdf_set_sha256": pdf_set_sha256(hashes)}}


def test_the_folder_must_hold_exactly_the_sealed_pdfs(tmp_path):
    prereg = _pdfs(tmp_path / "pdfs")
    assert len(bp.check_pdf_set(prereg, tmp_path / "pdfs")) == 3
    (tmp_path / "pdfs" / "blind_0002.pdf").write_bytes(b"changed")
    with pytest.raises(bp.BlindPreregError, match="not the sealed set"):
        bp.check_pdf_set(prereg, tmp_path / "pdfs")
    (tmp_path / "pdfs" / "blind_0003.pdf").unlink()
    with pytest.raises(bp.BlindPreregError, match="holds 2 PDFs"):
        bp.check_pdf_set(prereg, tmp_path / "pdfs")


# ---------------------------------------------------------------- the writer command


@pytest.fixture
def writer(tmp_path, monkeypatch):
    prereg = _pdfs(tmp_path / "pdfs")
    monkeypatch.setattr(bp, "EXPECTED_PDFS", 3)
    monkeypatch.setattr(bp.pre, "dirty_files", lambda root=bp.ROOT: "")
    monkeypatch.setattr(bp.pre, "git", lambda *args, **kwargs: HEAD)
    set_hash = prereg["sealed"]["pdf_set_sha256"]
    return {"pdfs": tmp_path / "pdfs", "out": tmp_path / "out.json", "set": set_hash}


def _argv(w: dict, **extra) -> list[str]:
    values = {
        "--key-sha256": KEY,
        "--backup-sha256": KEY,
        "--pdf-set-sha256": w["set"],
        "--pdfs": w["pdfs"],
        "--out": w["out"],
        **extra,
    }
    return [str(x) for pair in values.items() for x in pair]


def test_the_command_writes_the_file_once_and_lists_the_commands(writer, capsys):
    assert bp.main(_argv(writer)) == 0
    out = capsys.readouterr().out
    assert "blind_detect" in out and "commit and push this file" in out
    written = json.loads(writer["out"].read_text(encoding="utf-8"))
    assert written["sealed"]["pdf_set_sha256"] == writer["set"] and written["sealed"]["pdfs"] == 3
    assert bp.main(_argv(writer)) == 2
    assert "never overwritten" in capsys.readouterr().err


def test_the_command_refuses_pdfs_that_are_not_the_sealed_set(writer, capsys):
    assert bp.main(_argv(writer, **{"--pdf-set-sha256": "e" * 64})) == 2
    assert "do not match --pdf-set-sha256" in capsys.readouterr().err
    assert not writer["out"].exists()


# ---------------------------------------------------------------- the live detector


def test_the_live_detector_code_config_and_model_equal_week_6s():
    """Strict and live: no stand-in. uv_lock is left out on purpose (see week6_detector); the
    three parts that decide every finding must still be Week 6's."""
    live, recorded = LIVE_FINGERPRINT(), _week6_fingerprint()
    for part in ("code", "config", "model"):
        assert live[part] == recorded[part], f"{part} differs from Week 6's record"
