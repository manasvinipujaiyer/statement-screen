"""The Week 6 pre-registration (tamper/preregister.py): what it records and what it refuses."""

import json
import subprocess
from pathlib import Path

import pytest

from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import preregister as pre
from statementscreen.tamper.detect.verdict import RULE_VERSION


def _built() -> dict:
    return pre.build(head="abc123", dirty="", detector=fp.fingerprint())


def _verify(data: dict, **kwargs) -> None:
    """verify() with the git questions answered: in this branch's history and pushed."""
    kwargs = {"ancestor": True, "pushed": True, "detector": fp.fingerprint()} | kwargs
    pre.verify(data, **kwargs)


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


def test_the_manifests_and_split_files_are_frozen_too():
    artefacts = _built()["hashes"]["artefacts"]
    for name in (
        "pyproject.toml",
        "data/week1_n2000/manifest.csv",
        "data/week4_shift_n300/manifest.csv",
        "splits/split_v1.csv",
        "splits/split_v1.json",
        "splits/eval_shift_karan_v1.json",
    ):
        assert name in artefacts


def test_the_required_counts_and_sources_are_recorded():
    expected = _built()["expected_inputs"]
    assert expected["counts"] == {"test": 300, "shift": 300, "copies": 800}
    assert expected["copies_per_type"] == 100
    assert expected["sources"]["shift"] == "data/week4_shift_n300/pdfs"


def test_the_plain_text_fields_say_what_was_agreed():
    data = _built()
    commands = " ".join(data["commands"])
    assert "--per-type 100" in commands and "--include-overlay" in commands
    assert "--seed 6" in commands and "--preregistration" in commands
    assert "--genuine-manifest shift=" in commands and "--population shift" in commands
    assert "2026" in data["seeds"]["bootstrap"] and "1,000" in data["seeds"]["bootstrap"]
    assert any("overlay: no expectation" in line for line in data["expectations"])
    assert "post-hoc" in data["stop_rule"] and "change nothing" in data["stop_rule"]
    assert data["commands"][-1].startswith("uv run python -m statementscreen.tamper.log_run")
    expectations = data["expectations"]
    assert any(line.startswith("shift: 'tampered' expected near 0") for line in expectations)
    assert any("no number pre-stated" in line for line in expectations)
    shortfall = [r for r in data["rules"] if r.startswith("If the builder cannot reach 100")]
    assert shortfall and "No change to seed, count or code" in shortfall[0]
    mlflow_rule = [r for r in data["rules"] if "local MLflow store" in r]
    assert mlflow_rule and "does not change any result" in mlflow_rule[0]
    assert any("reports/week6_tamper_test_copies.csv" in m for m in data["metrics"])


def test_the_week_5_history_is_stated_and_the_false_line_is_gone():
    rules = _built()["rules"]
    week5 = [r for r in rules if r.startswith("Test statements were opened in Week 5")]
    assert week5 and "shift batch was scored in Weeks 4 and 5" in week5[0]
    assert "never been run on either" in week5[0]
    assert not any("were not looked at before scoring" in r for r in rules)


def test_the_new_limits_are_stated():
    limits = " ".join(_built()["known_limits"])
    assert "come from the anomaly score alone" in limits and "in-sample on dev" in limits
    assert "income_to_debits and salary_to_spending" in limits
    assert "can never give 'tampered' on its own" in limits
    assert "in-table character-overlap maximum" in limits and "drcr" in limits


def test_the_one_time_and_crash_rules_are_stated():
    rules = " ".join(_built()["rules"])
    assert "never overwritten" in rules and "--post-hoc" in rules
    assert "Checks that did not run" in rules
    assert "Any strong finding still gives 'tampered'" in rules
    assert "turns 'no evidence of tampering' into 'could not check'" in rules
    assert "the only exemption is the anomaly check" in rules
    assert "Every arithmetic 'not checked' note" in rules
    assert "PARSE_FAILED alone is 'suspicious'" in rules
    assert "no printed closing balance" not in rules
    assert "identifies populations by what they contain" in rules and "seed 6" in rules
    assert "git has ever recorded" in rules and "combined hash of all PDFs" in rules
    assert "--split test (or --population test)" in rules
    assert "the pushed version of this file" in rules


def test_a_matching_preregistration_verifies():
    _verify(_built())


def test_a_commit_that_is_not_in_history_is_refused():
    with pytest.raises(pre.PreregError, match="not in this branch"):
        _verify(_built(), ancestor=False)


def test_a_commit_that_is_not_pushed_is_refused():
    with pytest.raises(pre.PreregError, match="has not been pushed"):
        _verify(_built(), pushed=False)


def test_a_different_detector_is_refused_naming_the_part():
    other = {**fp.fingerprint(), "config": "different"}
    with pytest.raises(fp.StaleFindings, match="config differs"):
        _verify(_built(), detector=other)


def test_a_different_rule_version_is_refused():
    with pytest.raises(pre.PreregError, match="rule version"):
        _verify(_built(), rule_version="v1")


def test_a_changed_source_file_or_artefact_is_refused():
    data = _built()
    first = sorted(data["hashes"]["code"])[0]
    data["hashes"]["code"][first] = "0" * 64
    data["hashes"]["artefacts"]["uv.lock"] = "0" * 64
    data["hashes"]["artefacts"]["data/week4_shift_n300/manifest.csv"] = "0" * 64
    with pytest.raises(pre.PreregError) as err:
        _verify(data)
    message = str(err.value)
    assert f"{first} changed" in message and "uv.lock changed" in message
    assert "data/week4_shift_n300/manifest.csv changed" in message


def test_a_new_source_file_is_refused():
    data = _built()
    del data["hashes"]["code"][sorted(data["hashes"]["code"])[0]]
    with pytest.raises(pre.PreregError, match="is new"):
        _verify(data)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(pre.PreregError, match="no pre-registration"):
        pre.load_and_verify(tmp_path / "nope.json")


def test_an_existing_preregistration_is_never_overwritten(tmp_path, capsys):
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps({"keep": "me"}), encoding="utf-8")
    assert pre.main(["--out", str(path)]) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"keep": "me"}
    assert "never overwritten" in capsys.readouterr().err


# ---------------------------------------------------------------- the pushed version (real git)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _repo(tmp_path: Path, push: bool = True) -> tuple[Path, Path]:
    """A throwaway repository with a bare remote and one committed pre-registration."""
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    (work / "reports").mkdir()
    prereg = work / "reports" / "prereg.json"
    prereg.write_text('{"a": 1}\n', encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "pre-registration")
    _git(work, "remote", "add", "origin", str(remote))
    if push:
        _git(work, "push", "-u", "origin", "HEAD")
    return work, prereg


def test_the_pushed_version_is_read_from_the_remote_branch(tmp_path):
    work, prereg = _repo(tmp_path)
    assert pre.pushed_text(prereg, work) == '{"a": 1}\n'
    prereg.write_text('{"a": 2}\n', encoding="utf-8")  # an uncommitted edit changes nothing
    assert pre.pushed_text(prereg, work) == '{"a": 1}\n'


def test_a_working_tree_edit_is_refused_against_the_pushed_version(tmp_path):
    work, prereg = _repo(tmp_path)
    prereg.write_text('{"a": 2}\n', encoding="utf-8")
    with pytest.raises(pre.PreregError, match="differs from the pushed version"):
        pre.load_and_verify(prereg, root=work)


def test_a_branch_with_no_upstream_is_refused(tmp_path):
    work, prereg = _repo(tmp_path, push=False)
    with pytest.raises(pre.PreregError, match="no upstream"):
        pre.pushed_text(prereg, work)


def test_a_file_that_is_committed_but_not_pushed_is_refused(tmp_path):
    work, _ = _repo(tmp_path)
    later = work / "reports" / "later.json"
    later.write_text("{}\n", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "not pushed")
    with pytest.raises(pre.PreregError, match="not on the pushed branch"):
        pre.pushed_text(later, work)


def test_a_commit_is_pushed_only_when_it_is_on_the_upstream_branch(tmp_path):
    work, _ = _repo(tmp_path)
    first = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True
    ).stdout.strip()
    (work / "reports" / "later.json").write_text("{}\n", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "not pushed")
    second = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert pre.is_ancestor(first, "@{u}", work) is True
    assert pre.is_ancestor(second, "@{u}", work) is False


def _frozen_repo(tmp_path: Path) -> Path:
    """A repository laid out like the project: code and frozen artefacts, the pre-registration
    and an unrelated file, all committed."""
    work = tmp_path / "frozen"
    subprocess.run(["git", "init", str(work)], check=True, capture_output=True)
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    names = ("src/a.py", "models/m.pkl", "splits/s.csv", "uv.lock", "notes.txt")
    for name in (*names, "reports/week6_preregistration.json"):
        file = work / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("x\n", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "everything")
    return work


def test_a_clean_tree_is_clean_and_the_preregistration_itself_is_ignored(tmp_path):
    work = _frozen_repo(tmp_path)
    assert pre.dirty_files(work) == ""
    (work / "reports" / "week6_preregistration.json").unlink()  # about to be replaced
    assert pre.dirty_files(work) == ""


def test_a_changed_tracked_file_or_an_untracked_code_file_is_dirty(tmp_path):
    work = _frozen_repo(tmp_path)
    (work / "notes.txt").write_text("changed\n", encoding="utf-8")
    assert "notes.txt" in pre.dirty_files(work)
    _git(work, "checkout", "notes.txt")
    (work / "src" / "b.py").write_text("new\n", encoding="utf-8")
    assert "src/b.py" in pre.dirty_files(work)


def test_untracked_files_outside_the_frozen_folders_are_not_dirty(tmp_path):
    work = _frozen_repo(tmp_path)
    (work / "data").mkdir()
    (work / "data" / "x.csv").write_text("x\n", encoding="utf-8")
    (work / "scratch.txt").write_text("x\n", encoding="utf-8")
    assert pre.dirty_files(work) == ""


def test_a_path_git_has_ever_recorded_has_history_even_after_it_is_deleted(tmp_path):
    work, prereg = _repo(tmp_path)
    assert pre.has_git_history(prereg, work) is True
    prereg.unlink()
    assert pre.has_git_history(prereg, work) is True
    assert pre.has_git_history(work / "reports" / "never.json", work) is False
    assert pre.has_git_history(tmp_path / "outside.json", work) is False


def test_the_upstream_commit_is_the_pushed_one(tmp_path):
    work, _ = _repo(tmp_path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert pre.upstream_sha(work) == head
    (work / "reports" / "later.json").write_text("{}\n", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-m", "not pushed")
    assert pre.upstream_sha(work) == head  # still the pushed commit, not HEAD


def test_no_upstream_means_no_upstream_commit(tmp_path):
    work, _ = _repo(tmp_path, push=False)
    with pytest.raises(pre.PreregError, match="no upstream"):
        pre.upstream_sha(work)


def test_the_builder_recovery_and_post_freeze_rules_are_stated():
    data = _built()
    rules = " ".join(data["rules"])
    assert "replaced by the next source by seed, like NotApplicable" in rules
    assert "RebuildError or CopyError" in rules and "RuntimeError included" in rules
    assert "If more than 5 sources fail across the test build" in rules and "STOP" in rules
    assert "re-run once" in rules and "a second failure means STOP" in rules
    assert (
        "An interrupted registered detector run may be continued with --resume, which "
        "re-checks the detector fingerprint and skips finished chunks; this is not a re-run."
    ) in rules
    assert "Any code change after this freeze needs a new pre-registration" in rules
    assert "discloses it as a post-freeze change" in rules
    assert "labelled post-hoc only if a first result already exists" in rules


def test_the_metric_is_called_checks_that_did_not_run():
    metrics = " ".join(_built()["metrics"])
    assert "checks that did not run" in metrics and "at least one applicable check" in metrics


def test_the_extra_limits_are_stated():
    limits = " ".join(_built()["known_limits"])
    assert "rebuild check" in limits and "300 dev statements" in limits
    assert "deleted and rebuilt data/tamper/test" in limits
    assert "stand-ins in the test suite" in limits
    assert "log_run does not prevent a second MLflow entry for the same report" in limits
    assert "the committed report is the record of truth" in limits
    assert "The generator always prints every summary label" in limits
    assert "per-layout list of expected labels" in limits
