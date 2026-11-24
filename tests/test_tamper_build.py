"""The set builder: opaque names, balanced and unique sources, replacements counted, and
overlay only by the one-time command. Everything here is in temp folders."""

import json
from collections import Counter
from pathlib import Path

import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.splits import OPEN_TEST_ENV, SplitError
from statementscreen.tamper import build
from statementscreen.tamper.copy import HELD_OUT, TYPES, CopyError
from statementscreen.tamper.rebuild import RebuildError
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
    return {
        "out": out,
        "summary": summary,
        "keys": keys,
        "layout": layout,
        "batch": batch,
        "ids": ids,
    }


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


def _flaky_build(built, tmp_path, monkeypatch, fail_first, exc=RebuildError, per_type=2):
    """build_set with tamper_copy raising `exc` on its first `fail_first` calls."""
    real, calls = build.tamper_copy, []

    def flaky(path, tamper_type, seed, donor=None, allow_held_out=False):
        calls.append(Path(path).stem)
        if len(calls) <= fail_first:
            raise exc("boom")
        return real(path, tamper_type, seed, donor, allow_held_out=allow_held_out)

    monkeypatch.setattr(build, "tamper_copy", flaky)
    summary = build.build_set(
        built["batch"] / "pdfs", built["ids"], built["layout"], tmp_path / "out", per_type, seed=1
    )
    return summary, calls


def test_a_source_that_cannot_be_rebuilt_is_replaced_and_recorded(built, tmp_path, monkeypatch):
    summary, calls = _flaky_build(built, tmp_path, monkeypatch, fail_first=2)
    errors = summary["build_errors"]
    assert [e["source"] for e in errors] == calls[:2]  # each failed source id, in order
    assert all(e["error"] == "RebuildError: boom" and e["type"] in TYPES for e in errors)
    assert summary["stopped"] is None
    assert sum(summary["sources_replaced"].values()) >= 2  # replaced by the next source
    assert any((tmp_path / "out" / "keys").glob("*.json"))  # the build carried on


def test_a_copy_error_is_treated_the_same_way_and_records_the_donor(built, tmp_path, monkeypatch):
    summary, _ = _flaky_build(built, tmp_path, monkeypatch, fail_first=1, exc=CopyError)
    assert [e["error"] for e in summary["build_errors"]] == ["CopyError: boom"]
    assert "donor" in summary["build_errors"][0]


def test_a_plain_runtime_error_or_a_project_error_fails_loudly(built, tmp_path, monkeypatch):
    for exc in (
        RuntimeError,
        SplitError,
        RecursionError,
    ):  # all are RuntimeErrors, none is a build error
        with pytest.raises(exc):
            _flaky_build(built, tmp_path / exc.__name__, monkeypatch, fail_first=1, exc=exc)


def test_more_than_five_failed_sources_is_a_stop(built, tmp_path, monkeypatch, capsys):
    summary, calls = _flaky_build(built, tmp_path, monkeypatch, fail_first=99, per_type=1)
    assert len(summary["build_errors"]) == build.MAX_BUILD_ERRORS + 1 == len(calls)
    assert summary["stopped"].startswith("more than 5 sources failed")
    assert summary["complete"] is False and summary["short"]
    assert "STOP" in capsys.readouterr().out


def test_exactly_five_failed_sources_is_not_a_stop(built, tmp_path, monkeypatch):
    summary, _ = _flaky_build(built, tmp_path, monkeypatch, fail_first=5, per_type=1)
    assert len(summary["build_errors"]) == 5 and summary["stopped"] is None


def test_any_other_error_still_fails_loudly(built, tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        _flaky_build(built, tmp_path, monkeypatch, fail_first=1, exc=KeyError)


def test_a_non_empty_test_folder_says_how_to_recover_not_to_overwrite(tmp_path, capsys) -> None:
    partial = tmp_path / "out" / "test"
    (partial / "copies").mkdir(parents=True)
    argv = [str(tmp_path), "--set", "test", "--per-type", "1", "--out", str(tmp_path / "out")]
    assert build.main(argv) == 2
    err = capsys.readouterr().err
    assert "never overwritten" in err and "delete that folder by hand" in err
    assert "pass --overwrite" not in err and (partial / "copies").is_dir()
