"""The set builder: opaque names, balanced and unique sources, replacements counted, and
overlay only by the one-time command. Everything here is in temp folders."""

import json
from collections import Counter

import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.splits import OPEN_TEST_ENV
from statementscreen.tamper import build
from statementscreen.tamper.copy import HELD_OUT, TYPES
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    batch = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 6), batch)
    ids = [f"stmt_{i:04d}" for i in range(1, 7)]
    layout = build.layouts_of(batch / "manifest.csv", ids)
    out = tmp_path_factory.mktemp("tamper") / "dev"
    summary = build.build_set(batch / "pdfs", ids, layout, out, per_type=2, seed=1)
    keys = [json.loads(p.read_text()) for p in sorted((out / "keys").glob("*.json"))]
    return {"out": out, "summary": summary, "keys": keys, "layout": layout}


def test_every_type_but_overlay_gets_its_copies(built) -> None:
    s = built["summary"]
    assert s["types"] == [t for t in TYPES if t != HELD_OUT]
    assert s["made"] == {t: 2 for t in s["types"]} and s["short"] == {}
    assert s["complete"] is True
    assert not any(k["tamper_type"] == HELD_OUT for k in built["keys"])
    assert set(s["applies_to"]) == set(s["types"])


def test_copies_have_opaque_names_and_one_key_each(built) -> None:
    out = built["out"]
    copies = sorted(p.stem for p in (out / "copies").glob("*.pdf"))
    assert copies == sorted(k["copy_id"] for k in built["keys"])
    assert copies == [f"copy_{n:04d}" for n in range(1, 15)]
    first_type = [k["tamper_type"] for k in sorted(built["keys"], key=lambda k: k["copy_id"])]
    assert first_type != sorted(first_type)  # numbering is shuffled, not grouped by type


def test_sources_are_unique_within_a_type_and_balanced(built) -> None:
    by_type: dict[str, list[dict]] = {}
    for k in built["keys"]:
        by_type.setdefault(k["tamper_type"], []).append(k)
    for t, keys in by_type.items():
        assert len({k["source"] for k in keys}) == len(keys), t
        if t not in ("rerender_consistent",):  # sources limited to salaried statements
            assert Counter(k["source_layout"] for k in keys) == {"columns": 1, "drcr": 1}, t


def test_page_swap_donors_share_the_layout_and_are_never_the_source(built) -> None:
    for k in built["keys"]:
        if k["tamper_type"] == "page_swap":
            assert k["donor"] != k["source"]
            assert built["layout"][k["donor"]] == built["layout"][k["source"]]


def test_overlay_is_refused_unless_it_is_the_one_time_test_command(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(OPEN_TEST_ENV, raising=False)
    for argv in (["--set", "dev", "--include-overlay"], ["--set", "test", "--include-overlay"]):
        assert build.main([str(tmp_path), *argv, "--per-type", "1",
                           "--out", str(tmp_path / "out")]) == 2  # fmt: skip
    assert not (tmp_path / "out").exists()  # refused before anything was written


def test_train_gets_no_copies_and_nothing_is_written(tmp_path) -> None:
    for name in ("train", "other"):
        argv = [str(tmp_path), "--set", name, "--per-type", "1", "--out", str(tmp_path / "out")]
        assert build.main(argv) == 2
    assert not (tmp_path / "out").exists()  # refused before anything was read or written


def test_a_test_set_is_never_overwritten(tmp_path) -> None:
    argv = [str(tmp_path), "--set", "test", "--per-type", "1", "--overwrite"]
    assert build.main(argv) == 2
