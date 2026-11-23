"""The Week 6 pre-registration (tamper/preregister.py): what it records and what it refuses."""

import json

import pytest

from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import preregister as pre
from statementscreen.tamper.detect.verdict import RULE_VERSION


def _built() -> dict:
    return pre.build(head="abc123", dirty="", detector=fp.fingerprint())


def test_a_dirty_tree_is_refused():
    with pytest.raises(ValueError, match="uncommitted changes"):
        pre.build(head="abc123", dirty=" M src/statementscreen/tamper/scoring.py")


def test_the_preregistration_records_what_the_scoring_will_check():
    data = _built()
    assert data["frozen_commit"] == "abc123" and data["rule_version"] == RULE_VERSION
    assert set(data["detector_fingerprint"]) == set(fp.PARTS)
    assert "src/statementscreen/tamper/detect/verdict.py" in data["hashes"]["code"]
    assert "models/week6_iforest.pkl" in data["hashes"]["artefacts"]
    assert data["dev_adoption"]["conditions"]["all_passed"] is True
    assert data["dev_adoption"]["decision"].startswith("v2 is adopted")


def test_the_plain_text_fields_say_what_was_agreed():
    data = _built()
    commands = " ".join(data["commands"])
    assert "--per-type 100" in commands and "--include-overlay" in commands
    assert "--seed 6" in commands and "--preregistration" in commands
    assert "--genuine-manifest shift=" in commands
    assert "2026" in data["seeds"]["bootstrap"] and "1,000" in data["seeds"]["bootstrap"]
    assert any("overlay: no expectation" in line for line in data["expectations"])
    assert "post-hoc" in data["stop_rule"] and "change nothing" in data["stop_rule"]
    assert any("reports/week6_tamper_test_copies.csv" in m for m in data["metrics"])


def test_a_matching_preregistration_verifies():
    pre.verify(_built(), ancestor=True, detector=fp.fingerprint())


def test_a_commit_that_is_not_in_history_is_refused():
    with pytest.raises(pre.PreregError, match="not in this branch"):
        pre.verify(_built(), ancestor=False, detector=fp.fingerprint())


def test_a_different_detector_is_refused_naming_the_part():
    other = {**fp.fingerprint(), "config": "different"}
    with pytest.raises(fp.StaleFindings, match="config differs"):
        pre.verify(_built(), ancestor=True, detector=other)


def test_a_different_rule_version_is_refused():
    with pytest.raises(pre.PreregError, match="rule version"):
        pre.verify(_built(), rule_version="v1", ancestor=True, detector=fp.fingerprint())


def test_a_changed_source_file_or_artefact_is_refused():
    data = _built()
    first = sorted(data["hashes"]["code"])[0]
    data["hashes"]["code"][first] = "0" * 64
    data["hashes"]["artefacts"]["uv.lock"] = "0" * 64
    with pytest.raises(pre.PreregError) as err:
        pre.verify(data, ancestor=True, detector=fp.fingerprint())
    assert f"{first} changed" in str(err.value) and "uv.lock changed" in str(err.value)


def test_a_new_source_file_is_refused():
    data = _built()
    del data["hashes"]["code"][sorted(data["hashes"]["code"])[0]]
    with pytest.raises(pre.PreregError, match="is new"):
        pre.verify(data, ancestor=True, detector=fp.fingerprint())


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(pre.PreregError, match="no pre-registration"):
        pre.load_and_verify(tmp_path / "nope.json")


def test_which_scoring_runs_need_a_preregistration(tmp_path):
    assert pre.needs_preregistration(None, ["test=x"])
    assert pre.needs_preregistration(None, ["dev=a", "shift=b"])
    assert pre.needs_preregistration(tmp_path / "test", [])
    assert not pre.needs_preregistration(tmp_path / "dev", ["dev=a", "train=b"])


def test_an_existing_preregistration_is_never_overwritten(tmp_path, capsys):
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps({"keep": "me"}), encoding="utf-8")
    assert pre.main(["--out", str(path)]) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"keep": "me"}
    assert "never overwritten" in capsys.readouterr().err
