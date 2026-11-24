"""The service's code version: the git commit (with +dirty) and the detector fingerprint."""

import json
from pathlib import Path

from statementscreen.service import version


def _git(answers: dict):
    """A fake version._git: answers[first argument] is its output (None means git failed)."""

    def fake(*args: str, root: Path) -> str | None:
        return answers[args[0]]

    return fake


def test_the_environment_variable_wins_over_git(monkeypatch):
    monkeypatch.setattr(version, "_git", _git({"rev-parse": "c" * 40, "status": ""}))
    env = {version.COMMIT_ENV: " abc123 "}
    assert version.git_commit(env=env) == "abc123"


def test_a_clean_tree_gives_the_bare_commit_and_a_dirty_one_gets_a_suffix(monkeypatch):
    head = "c" * 40
    monkeypatch.setattr(version, "_git", _git({"rev-parse": head, "status": ""}))
    assert version.git_commit(env={}) == head
    monkeypatch.setattr(version, "_git", _git({"rev-parse": head, "status": " M src/x.py"}))
    assert version.git_commit(env={}) == f"{head}+dirty"
    monkeypatch.setattr(version, "_git", _git({"rev-parse": head, "status": "?? new.py"}))
    assert version.git_commit(env={}) == f"{head}+dirty"  # an untracked file counts


def test_an_empty_variable_is_ignored_and_git_decides(monkeypatch):
    monkeypatch.setattr(version, "_git", _git({"rev-parse": "d" * 40, "status": ""}))
    assert version.git_commit(env={version.COMMIT_ENV: "  "}) == "d" * 40


def test_when_git_cannot_say_the_commit_is_unknown_or_its_state_is(monkeypatch):
    monkeypatch.setattr(version, "_git", _git({"rev-parse": None, "status": None}))
    assert version.git_commit(env={}) == "unknown"
    monkeypatch.setattr(version, "_git", _git({"rev-parse": "e" * 40, "status": None}))
    assert version.git_commit(env={}) == f"{'e' * 40}+status-unknown"


def test_the_code_version_carries_the_commit_the_fingerprint_and_the_versions():
    detector = {"code": "1", "config": "2", "model": "3", "uv_lock": "4"}
    v = version.code_version(env={version.COMMIT_ENV: "abc"}, detector=detector)
    assert v["git_commit"] == "abc" and v["detector_fingerprint"] == detector
    assert v["rule_version"] == "v2" and v["profile_version"] == 1


def test_the_services_detector_code_config_and_model_are_week_6s():
    """Strict and live. uv_lock is left out: it gained the service's dependencies in Week 7
    Part 2. These three parts decide every finding, and they are identical to Week 6's."""
    recorded = json.loads(
        (version.ROOT / "reports/week6_preregistration.json").read_text(encoding="utf-8")
    )["detector_fingerprint"]
    live = version.code_version(env={version.COMMIT_ENV: "x"})["detector_fingerprint"]
    for part in ("code", "config", "model"):
        assert live[part] == recorded[part], f"{part} differs from Week 6's record"
