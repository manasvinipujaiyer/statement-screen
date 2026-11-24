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


DATA_ARTEFACTS = ("data/week1_n2000/manifest.csv", "data/week4_shift_n300/manifest.csv")


def real_data_present() -> bool:
    """The two generated manifests (git-ignored, so absent from a fresh checkout such as CI)."""
    return all((pre.ROOT / a).is_file() for a in DATA_ARTEFACTS)


@pytest.fixture(autouse=True)
def stand_in_manifests(monkeypatch, tmp_path_factory):
    """On a checkout without data/, hash small stand-in manifests (written to a pytest temp
    folder, never into data/) where the pre-registration hashes the two real ones. The recorded
    names stay the real ones; only the bytes behind them differ. With the real files present
    nothing is patched. Tests that need the real bytes are skipped (see needs_real_data)."""
    if real_data_present():
        return
    folder = tmp_path_factory.mktemp("stand_in_manifests")
    stand_in = {}
    for n, name in enumerate(DATA_ARTEFACTS):
        path = folder / f"manifest_{n}.csv"
        path.write_text(f"statement_id,layout\nstand_in_{n},columns\n", encoding="utf-8")
        stand_in[(pre.ROOT / name).resolve()] = path
    real_sha256 = pre.sha256

    def sha256(path: Path) -> str:
        return real_sha256(stand_in.get(Path(path).resolve(), path))

    monkeypatch.setattr(pre, "sha256", sha256)


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
