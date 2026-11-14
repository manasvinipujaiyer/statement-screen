"""The locked train/dev/test split and its guarded loaders."""

import csv
import json
from pathlib import Path

import pytest

from statementscreen import splits
from statementscreen.splits import SplitError

LAYOUTS = ("columns", "drcr")
TYPES = ("salaried", "gig", "self_employed")


def _manifest(path: Path, n: int = 120) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["statement_id", "customer_type", "layout", "months"])
        for i in range(1, n + 1):
            w.writerow([f"stmt_{i:04d}", TYPES[i % 3], LAYOUTS[(i // 3) % 2], 6])
    return path


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    return _manifest(tmp_path / "manifest.csv")


@pytest.fixture
def locked(manifest: Path, tmp_path: Path) -> Path:
    out = tmp_path / "splits"
    assert splits.main([str(manifest), "--out", str(out)]) == 0
    return out


def test_same_seed_same_split_other_seed_differs(manifest: Path) -> None:
    a, meta_a = splits.build(manifest, 42)
    b, meta_b = splits.build(manifest, 42)
    c, _ = splits.build(manifest, 7)
    assert a == b and meta_a == meta_b
    assert a != c


def test_splits_are_disjoint_and_cover_every_statement(manifest: Path) -> None:
    assigned, meta = splits.build(manifest)
    assert set(assigned) == {f"stmt_{i:04d}" for i in range(1, 121)}
    assert set(assigned.values()) == set(splits.SPLITS)
    assert sum(meta["counts"][s]["statements"] for s in splits.SPLITS) == 120
    assert meta["counts"]["train"]["statements"] == 84  # 70% of 20 per stratum, x6


def test_every_split_has_both_layouts_and_all_customer_types(manifest: Path) -> None:
    _, meta = splits.build(manifest)
    for name in splits.SPLITS:
        assert set(meta["counts"][name]["by_layout"]) == set(LAYOUTS)
        assert set(meta["counts"][name]["by_customer_type"]) == set(TYPES)


def test_files_record_seed_and_manifest_hash_and_are_locked(manifest, locked) -> None:
    meta = json.loads((locked / "split_v1.json").read_text())
    assert meta["seed"] == 42
    assert meta["manifest_sha256"] == splits.sha256_file(manifest)
    with (locked / "split_v1.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 120 and set(rows[0]) == {"statement_id", "split"}
    assert splits.main([str(manifest), "--out", str(locked)]) == 2  # refuses to overwrite


def test_load_split_returns_members(manifest, locked, monkeypatch) -> None:
    monkeypatch.delenv(splits.OPEN_TEST_ENV, raising=False)
    train = splits.load_split("train", manifest, locked)
    dev = splits.load_split("dev", manifest, locked)
    assert len(train) == 84 and len(dev) == 18 and not set(train) & set(dev)


def test_mismatched_manifest_hash_is_refused(manifest, locked, tmp_path) -> None:
    other = _manifest(tmp_path / "other.csv", n=121)
    with pytest.raises(SplitError, match="different batch"):
        splits.load_split("train", other, locked)


def test_test_split_refused_without_env(manifest, locked, monkeypatch) -> None:
    monkeypatch.delenv(splits.OPEN_TEST_ENV, raising=False)
    for value in (None, "0", "yes"):
        if value is not None:
            monkeypatch.setenv(splits.OPEN_TEST_ENV, value)
        with pytest.raises(SplitError, match="closed"):
            splits.load_split("test", manifest, locked)
        with pytest.raises(SplitError, match="closed"):
            splits.load_split_for_parsed("test", [], locked)
    monkeypatch.setenv(splits.OPEN_TEST_ENV, "1")
    assert len(splits.load_split("test", manifest, locked)) == 18


def test_test_refusal_happens_before_any_file_is_read(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(splits.OPEN_TEST_ENV, raising=False)
    with pytest.raises(SplitError, match="closed"):
        splits.load_split("test", tmp_path / "missing.csv", tmp_path / "nowhere")


def test_unknown_split_name_is_refused(manifest, locked) -> None:
    with pytest.raises(SplitError, match="unknown split"):
        splits.load_split("validation", manifest, locked)


def test_parsed_loader_checks_ids_without_the_manifest(manifest, locked) -> None:
    ids = [f"stmt_{i:04d}" for i in range(1, 121)]
    manifest.unlink()  # pattern detection never has the manifest
    assert len(splits.load_split_for_parsed("dev", ids, locked)) == 18
    with pytest.raises(SplitError, match="1 missing"):
        splits.load_split_for_parsed("dev", ids[:-1], locked)
    with pytest.raises(SplitError, match="1 not in the split"):
        splits.load_split_for_parsed("dev", [*ids, "stmt_9999"], locked)
