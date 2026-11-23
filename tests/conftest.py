"""Fixtures shared by the Week 6 tamper tests."""

from pathlib import Path

import pytest

from statementscreen.tamper import preregister as pre


@pytest.fixture
def pushed(monkeypatch):
    """Stand in for the remote: the pre-registration counts as pushed exactly as it is on disk,
    and its commit as being on the pushed branch. The real git logic is tested on a throwaway
    repository in test_tamper_preregister.py."""
    monkeypatch.setattr(
        pre, "pushed_text", lambda path, root=pre.ROOT: Path(path).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(pre, "is_ancestor", lambda commit, ref="HEAD", root=pre.ROOT: True)
