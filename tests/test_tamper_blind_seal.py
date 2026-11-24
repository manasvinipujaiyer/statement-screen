"""Sealing the Week 7 blind set: opaque names, one file time, the key, the backup, and the rules.

Most tests replace tamper_copy with a fake that records its calls, so they need no real PDFs
and no data/. One test at the end seals a few really generated statements."""

import hashlib
import json
from pathlib import Path

import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.splits import SplitError
from statementscreen.tamper import blind_seal as bs
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import (
    FIXED_TIME,
    KEY_NAME,
    NAME,
    pdf_hashes,
    pdf_set_sha256,
)
from statementscreen.tamper.copy import CopyError, NotApplicable
from statementscreen.tamper.rebuild import RebuildError
from statementscreen.writer import write_batch

PER_BATCH = 12


def _source(root: Path, folder: str, name: str) -> bs.Source:
    base = root / folder
    (base / "pdfs").mkdir(parents=True)
    rows = ["statement_id,layout"]
    for k in range(1, PER_BATCH + 1):
        sid = f"stmt_{k:04d}"  # both batches use the same ids, as the real ones do
        (base / "pdfs" / f"{sid}.pdf").write_bytes(f"%PDF {folder} {sid} body".encode())
        rows.append(f"{sid},{'columns' if k % 2 else 'drcr'}")
    (base / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return bs.Source(name, base, 1, "default")


@pytest.fixture
def world(tmp_path):
    src = tmp_path / "blind_src"
    sources = [
        _source(src, "default_v1", "blind_default_v1"),
        _source(src, "karan_v1", "blind_karan_v1"),
    ]
    return {"root": tmp_path, "src": src, "sources": sources, "out": tmp_path / "blind",
            "backup": tmp_path / "backup"}  # fmt: skip


def _fake_copy(monkeypatch, behave=None, calls=None):
    def fake(pdf_path, tamper_type, seed, donor_pdf=None, allow_held_out=False):
        assert allow_held_out
        if calls is not None:
            calls.append((Path(pdf_path), tamper_type, donor_pdf))
        if behave is not None:
            behave(tamper_type)
        data = Path(pdf_path).read_bytes() + f"|{tamper_type}|{seed}".encode()
        return data, {"source": Path(pdf_path).stem, "tamper_type": tamper_type, "seed": seed}

    monkeypatch.setattr(bs, "tamper_copy", fake)


def _seal(world, **kwargs):
    return bs.seal(world["sources"], world["out"], world["backup"], **kwargs)


def _key(world) -> dict:
    return json.loads((world["out"] / KEY_NAME).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- what is written


def test_names_follow_one_pattern_and_are_a_shuffle_of_one_to_n(world, monkeypatch):
    _fake_copy(monkeypatch)
    _seal(world, seed=5)
    names = sorted(p.stem for p in (world["out"] / "pdfs").iterdir())
    assert all(NAME.fullmatch(n) for n in names)
    assert names == [f"blind_{k:04d}" for k in range(1, 2 * PER_BATCH + 1)]
    assert sorted(_key(world)["pdfs"]) == names


def test_every_pdf_and_the_pdf_folder_get_one_fixed_file_time(world, monkeypatch):
    _fake_copy(monkeypatch)
    _seal(world, seed=5)
    pdfs = world["out"] / "pdfs"
    assert {int(p.stat().st_mtime) for p in (*pdfs.iterdir(), pdfs)} == {FIXED_TIME}


def test_a_genuine_pdf_is_a_byte_copy_of_its_source_and_a_tampered_one_is_not(world, monkeypatch):
    _fake_copy(monkeypatch)
    _seal(world, seed=5, p_genuine=0.5)
    folders = {s.name: s.folder for s in world["sources"]}
    kinds = set()
    for name, entry in _key(world)["pdfs"].items():
        data = (world["out"] / "pdfs" / f"{name}.pdf").read_bytes()
        source = (folders[entry["sub_batch"]] / "pdfs" / f"{entry['source']}.pdf").read_bytes()
        assert (data == source) == (entry["truth"] == "genuine")
        kinds.add(entry["truth"] == "genuine")
    assert kinds == {True, False}


def test_the_key_records_the_seed_the_sources_and_every_pdf_hash(world, monkeypatch):
    _fake_copy(monkeypatch)
    summary = _seal(world, seed=99)
    key = _key(world)
    text = (world["out"] / KEY_NAME).read_text(encoding="utf-8")
    assert key["seed"] == 99 and set(key["sources"]) == {"blind_default_v1", "blind_karan_v1"}
    assert key["sources"]["blind_default_v1"]["statements"] == PER_BATCH
    for name, entry in key["pdfs"].items():
        file_hash = hashlib.sha256((world["out"] / "pdfs" / f"{name}.pdf").read_bytes())
        assert entry["pdf_sha256"] == file_hash.hexdigest()
    assert summary["key_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert pdf_set_sha256(pdf_hashes(world["out"] / "pdfs")) == summary["pdf_set_sha256"]
    assert key["pdf_set_sha256"] == summary["pdf_set_sha256"]


def test_the_backup_is_the_same_bytes_as_the_key(world, monkeypatch):
    _fake_copy(monkeypatch)
    summary = _seal(world, seed=5)
    assert (world["backup"] / KEY_NAME).read_bytes() == (world["out"] / KEY_NAME).read_bytes()
    assert summary["backup_sha256"] == summary["key_sha256"]


def test_the_pdf_set_hash_does_not_depend_on_order():
    a = {"blind_0001": "aa", "blind_0002": "bb", "blind_0003": "cc"}
    b = dict(reversed(list(a.items())))
    assert pdf_set_sha256(a) == pdf_set_sha256(b)
    assert pdf_set_sha256(a) != pdf_set_sha256({**a, "blind_0003": "cd"})


def test_the_pdf_folder_must_hold_only_opaque_names(tmp_path):
    (tmp_path / "blind_0001.pdf").write_bytes(b"x")
    (tmp_path / "stmt_0001.pdf").write_bytes(b"y")
    with pytest.raises(bs.BlindError, match="not an opaque blind PDF name"):
        pdf_hashes(tmp_path)


# ---------------------------------------------------------------- the seed and the draws


def test_the_same_seed_gives_the_same_key_and_another_seed_a_different_one(
    world, tmp_path, monkeypatch
):
    _fake_copy(monkeypatch)
    _seal(world, seed=11)
    first = (world["out"] / KEY_NAME).read_text(encoding="utf-8")
    other = {**world, "out": tmp_path / "again", "backup": tmp_path / "again_backup"}
    bs.seal(world["sources"], other["out"], other["backup"], seed=11)
    assert (other["out"] / KEY_NAME).read_text(encoding="utf-8") == first
    third = {**world, "out": tmp_path / "third", "backup": tmp_path / "third_backup"}
    bs.seal(world["sources"], third["out"], third["backup"], seed=12)
    assert (third["out"] / KEY_NAME).read_text(encoding="utf-8") != first


def test_no_seed_means_a_fresh_one_from_the_operating_system(world, tmp_path, monkeypatch):
    _fake_copy(monkeypatch)
    _seal(world)
    other = {**world, "out": tmp_path / "again", "backup": tmp_path / "again_backup"}
    bs.seal(world["sources"], other["out"], other["backup"])
    assert _key(world)["seed"] != json.loads((other["out"] / KEY_NAME).read_text())["seed"]


def test_the_genuine_probability_is_applied_per_source(world, monkeypatch):
    _fake_copy(monkeypatch)
    _seal(world, seed=3, p_genuine=1.0)
    assert {e["truth"] for e in _key(world)["pdfs"].values()} == {"genuine"}
    other = world["root"] / "none"
    bs.seal(world["sources"], other / "blind", other / "backup", seed=3, p_genuine=0.0)
    sealed = json.loads((other / "blind" / KEY_NAME).read_text(encoding="utf-8"))
    truths = {e["truth"] for e in sealed["pdfs"].values()}
    assert truths <= set(bs.TAMPER_TYPES) and "genuine" not in truths


def test_the_seven_types_include_overlay_and_not_the_control():
    assert len(bs.TAMPER_TYPES) == 7
    assert "overlay" in bs.TAMPER_TYPES and "resaved_genuine" not in bs.TAMPER_TYPES


# ---------------------------------------------------------------- failures and redraws


def test_a_source_that_cannot_take_its_type_gets_another_type_and_the_redraw_is_recorded(
    world, monkeypatch
):
    def behave(tamper_type):
        if tamper_type == "overlay":
            raise NotApplicable("no")
        if tamper_type == "inplace_edit":
            raise CopyError("cannot locate")

    _fake_copy(monkeypatch, behave)
    monkeypatch.setattr(bs, "MAX_BUILD_ERRORS", 1000)
    _seal(world, seed=4, p_genuine=0.0)
    key = _key(world)
    truths = {e["truth"] for e in key["pdfs"].values()}
    assert truths <= set(bs.TAMPER_TYPES) - {"overlay", "inplace_edit"}
    assert key["redraws"] and {r["type"] for r in key["redraws"]} <= {"overlay", "inplace_edit"}
    build_errors = [r for r in key["redraws"] if r["reason"] == "build error"]
    assert key["build_errors"] == build_errors


def test_exactly_five_build_errors_are_accepted_and_six_are_a_stop(world, tmp_path, monkeypatch):
    def failing_first(n):
        count = {"calls": 0}

        def behave(tamper_type):
            count["calls"] += 1
            if count["calls"] <= n:
                raise CopyError("cannot locate")

        return behave

    _fake_copy(monkeypatch, failing_first(5))
    assert _seal(world, seed=6, p_genuine=0.0)["build_errors"] == 5
    stopped = {**world, "out": tmp_path / "stop", "backup": tmp_path / "stop_backup"}
    _fake_copy(monkeypatch, failing_first(6))
    with pytest.raises(bs.SealError, match="STOP"):
        bs.seal(world["sources"], stopped["out"], stopped["backup"], seed=6, p_genuine=0.0)
    assert not stopped["out"].exists() and not stopped["backup"].exists()


def test_a_rebuild_error_is_a_build_error_like_a_copy_error(world, monkeypatch):
    def behave(tamper_type):
        if tamper_type == "inplace_edit":
            raise RebuildError("cannot rebuild")

    _fake_copy(monkeypatch, behave)
    monkeypatch.setattr(bs, "MAX_BUILD_ERRORS", 1000)
    _seal(world, seed=4, p_genuine=0.0)
    key = _key(world)
    assert "inplace_edit" not in {e["truth"] for e in key["pdfs"].values()}
    assert key["build_errors"]
    assert all(e["detail"].startswith("RebuildError") for e in key["build_errors"])


def test_the_key_is_written_atomically_and_leaves_no_temporary_file(world, monkeypatch):
    _fake_copy(monkeypatch)
    _seal(world, seed=5)
    assert sorted(p.name for p in world["out"].iterdir()) == ["pdfs", KEY_NAME]


def test_a_tampered_source_is_never_switched_to_genuine(world, monkeypatch):
    def behave(tamper_type):
        raise NotApplicable("nothing applies")

    _fake_copy(monkeypatch, behave)
    with pytest.raises(bs.SealError, match="never switched to genuine"):
        _seal(world, seed=7, p_genuine=0.0)
    assert not world["out"].exists()


def test_page_swap_donors_come_from_the_same_sub_batch_and_layout(world, monkeypatch):
    def behave(tamper_type):
        if tamper_type != "page_swap":
            raise NotApplicable("only page_swap works here")

    calls = []
    _fake_copy(monkeypatch, behave, calls)
    _seal(world, seed=8, p_genuine=0.0)
    swaps = [(src, donor) for src, kind, donor in calls if kind == "page_swap" and donor]
    assert len(swaps) >= 2 * PER_BATCH
    for source, donor in swaps:
        assert donor.parent == source.parent and donor != source
        layout = "columns" if int(source.stem[-4:]) % 2 else "drcr"
        assert layout == ("columns" if int(donor.stem[-4:]) % 2 else "drcr")


def test_a_blind_set_is_never_overwritten_and_the_backup_neither(world, monkeypatch):
    _fake_copy(monkeypatch)
    _seal(world, seed=5)
    with pytest.raises(bs.SealError, match="never overwritten"):
        _seal(world, seed=5)
    other = {**world, "out": world["root"] / "other"}
    with pytest.raises(bs.SealError, match="a backup is never overwritten"):
        bs.seal(world["sources"], other["out"], world["backup"], seed=5)
    assert not other["out"].exists()


# ---------------------------------------------------------------- the command


def _argv(world, **extra) -> list[str]:
    reference = world["root"] / "week6_reference.json"
    if not reference.exists():
        reference.write_text(json.dumps({"detector_fingerprint": fp.fingerprint()}))
    argv = {"--blind-src": world["src"], "--out": world["out"], "--backup-dir": world["backup"],
            "--detector-reference": reference, **extra}  # fmt: skip
    return [str(x) for pair in argv.items() for x in pair]


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(bs, "load_eval", lambda name, manifest, splits_dir=None: [])


def test_the_command_prints_hashes_and_counts_but_never_the_mix(
    world, registered, monkeypatch, capsys
):
    _fake_copy(monkeypatch)
    assert bs.main(_argv(world)) == 0
    shown = capsys.readouterr()
    text = shown.out + shown.err
    key = _key(world)
    key_sha = hashlib.sha256((world["out"] / KEY_NAME).read_bytes()).hexdigest()
    assert f"key SHA-256:     {key_sha}" in text
    assert f"Sealed {2 * PER_BATCH} PDFs" in text and "build errors: 0" in text
    assert key["pdf_set_sha256"] in text
    for word in (*bs.TAMPER_TYPES, "genuine", "redraw", str(key["seed"])):
        assert word not in text


def test_a_detector_that_differs_from_week_6_refuses_and_writes_nothing(world, registered, capsys):
    bad = {k: "0" * 64 for k in fp.PARTS}
    reference = world["root"] / "week6_reference.json"
    reference.write_text(json.dumps({"detector_fingerprint": bad}))
    assert bs.main(_argv(world)) == 2
    assert "differs" in capsys.readouterr().err
    assert not world["out"].exists() and not world["backup"].exists()


def test_sources_that_are_not_the_registered_ones_refuse(world, monkeypatch, capsys):
    def refuse(name, manifest, splits_dir=None):
        raise SplitError("manifest is not the registered batch")

    monkeypatch.setattr(bs, "load_eval", refuse)
    assert bs.main(_argv(world)) == 2
    assert "not the registered batch" in capsys.readouterr().err
    assert not world["out"].exists()


def test_a_backup_folder_inside_the_repository_is_refused(world, registered, capsys):
    inside = bs.ROOT / "backup_must_not_be_here"
    argv = _argv(world, **{"--backup-dir": inside})
    assert bs.main(argv) == 2
    assert "inside the repository" in capsys.readouterr().err
    assert not world["out"].exists() and not inside.exists()


# ---------------------------------------------------------------- real statements


def test_a_few_really_generated_statements_can_be_sealed(tmp_path):
    sources = []
    for n, name in enumerate(("blind_default_v1", "blind_karan_v1")):
        folder = tmp_path / "src" / name
        write_batch(plan_batch(BatchConfig(seed=7 + n), 3), folder)
        sources.append(bs.Source(name, folder, 7 + n, "default"))
    out, backup = tmp_path / "blind", tmp_path / "backup"
    summary = bs.seal(sources, out, backup, seed=3, p_genuine=0.0)
    key = json.loads((out / KEY_NAME).read_text(encoding="utf-8"))
    assert summary["pdfs"] == 6 and len(pdf_hashes(out / "pdfs")) == 6
    for name, entry in key["pdfs"].items():
        source = tmp_path / "src" / entry["sub_batch"] / "pdfs" / f"{entry['source']}.pdf"
        assert entry["truth"] in bs.TAMPER_TYPES
        assert (out / "pdfs" / f"{name}.pdf").read_bytes() != source.read_bytes()
