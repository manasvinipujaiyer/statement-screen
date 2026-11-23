"""Fixtures shared by the Week 6 tamper tests."""

from pathlib import Path

import pytest

from statementscreen.tamper import preregister as pre
from statementscreen.tamper import provenance as prov


@pytest.fixture(autouse=True)
def closed_ids(monkeypatch):
    """The test split and the shift batch hold two ids no fixture uses, so the content checks
    in provenance never read the real split files. A test that needs particular sets patches
    them again; the one test of the real files calls monkeypatch.undo() first."""
    monkeypatch.setattr(prov, "split_test_ids", lambda splits_dir=None: {"closed_test_stmt"})
    monkeypatch.setattr(prov, "shift_batch_ids", lambda manifest=None: {"closed_shift_stmt"})


@pytest.fixture
def pushed(monkeypatch):
    """Stand in for the remote: the pre-registration counts as pushed exactly as it is on disk,
    its commit as being on the pushed branch, and the upstream as a fixed commit. The real git
    logic is tested on a throwaway repository in test_tamper_preregister.py."""
    monkeypatch.setattr(
        pre, "pushed_text", lambda path, root=pre.ROOT: Path(path).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(pre, "is_ancestor", lambda commit, ref="HEAD", root=pre.ROOT: True)
    monkeypatch.setattr(pre, "upstream_sha", lambda root=pre.ROOT: "f" * 40)
