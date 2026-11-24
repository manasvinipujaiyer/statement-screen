"""tamper_copy: each type changes exactly what its answer key says, deterministically.
Overlay copies exist only in memory here (never written outside a temp folder)."""

import io
from collections import Counter
from pathlib import Path

import pdfplumber
import pytest

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.tamper import copy as tc
from statementscreen.tamper.rebuild import runs
from statementscreen.writer import write_batch


@pytest.fixture(scope="module")
def batch(tmp_path_factory) -> Path:
    """6 statements, both layouts, 4 of them salaried (the recorded fingerprint batch)."""
    folder = tmp_path_factory.mktemp("batch")
    write_batch(plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 6), folder)
    return folder


def _pages(data: bytes) -> list[list[str]]:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return [[r.text for r in runs(page)] for page in pdf.pages]


def _pdf(batch: Path, layout: str | None = None, salaried: bool = False) -> Path:
    import csv

    with (batch / "manifest.csv").open() as f:
        for row in csv.DictReader(f):
            if (layout is None or row["layout"] == layout) and (
                    not salaried or row["customer_type"] == "salaried"):  # fmt: skip
                return batch / "pdfs" / f"{row['statement_id']}.pdf"
    raise AssertionError("no such statement in the test batch")


def _diff(before: list[str], after: list[str]) -> tuple[Counter, Counter]:
    b, a = Counter(before), Counter(after)
    return b - a, a - b  # removed, added


# ---------------------------------------------------------------- exactly what the key says


@pytest.mark.parametrize("tamper_type", ["rerender_inconsistent", "rerender_consistent",
                                         "inplace_edit", "inplace_edit_consistent"])  # fmt: skip
def test_edits_change_exactly_the_strings_in_the_key(batch, tamper_type) -> None:
    for layout in ("columns", "drcr"):
        src = _pdf(batch, layout, salaried=True)
        data, key = tc.tamper_copy(src, tamper_type, seed=3)
        before, after = _pages(src.read_bytes()), _pages(data)
        assert len(before) == len(after)
        assert key["changes"], "every edit changes something"
        for n, (b, a) in enumerate(zip(before, after, strict=True), start=1):
            removed, added = _diff(b, a)
            mine = [c for c in key["changes"] if c["page"] == n]
            assert removed == Counter(c["old"] for c in mine), (layout, n)
            assert added == Counter(c["new"] for c in mine), (layout, n)
        assert key["pages_changed"] == sorted({c["page"] for c in key["changes"]})


def test_overlay_draws_one_new_string_over_the_old_one(batch) -> None:
    src = _pdf(batch, "columns")
    with pytest.raises(tc.HeldOut):
        tc.tamper_copy(src, "overlay", seed=3)  # refused without allow_held_out
    data, key = tc.tamper_copy(src, "overlay", seed=3, allow_held_out=True)  # memory only
    (change,) = key["changes"]
    before, after = _pages(src.read_bytes()), _pages(data)
    for n, (b, a) in enumerate(zip(before, after, strict=True), start=1):
        removed, added = _diff(b, a)
        assert removed == Counter()  # the old text is still in the page, under a white box
        assert added == (Counter([change["new"]]) if n == change["page"] else Counter())


def test_page_remove_deletes_one_middle_page_and_nothing_else(batch) -> None:
    src = _pdf(batch)
    data, key = tc.tamper_copy(src, "page_remove", seed=3)
    (k,) = key["pages_changed"]
    before, after = _pages(src.read_bytes()), _pages(data)
    assert 1 < k < len(before)
    assert after == before[: k - 1] + before[k:]
    assert key["changes"] and all(c["page"] == k for c in key["changes"])


def test_page_swap_takes_the_same_page_from_a_donor_of_the_same_layout(batch) -> None:
    src = _pdf(batch, "columns")
    donor = next(p for p in sorted((batch / "pdfs").glob("*.pdf"))
                 if p != src and tc.parse_pdf(p).layout == "columns")  # fmt: skip
    with pytest.raises(tc.NotApplicable):
        tc.tamper_copy(src, "page_swap", seed=3)  # no donor
    with pytest.raises(tc.NotApplicable):
        tc.tamper_copy(src, "page_swap", seed=3, donor_pdf=src)  # never the source itself
    data, key = tc.tamper_copy(src, "page_swap", seed=3, donor_pdf=donor)
    (k,) = key["pages_changed"]
    before, after, other = _pages(src.read_bytes()), _pages(data), _pages(donor.read_bytes())
    assert after[k - 1] == other[k - 1]
    assert after[: k - 1] == before[: k - 1] and after[k:] == before[k:]
    assert key["donor"] == donor.stem and key["donor_page"] == k


def test_resaved_genuine_changes_no_text(batch) -> None:
    src = _pdf(batch)
    data, key = tc.tamper_copy(src, "resaved_genuine", seed=3)
    assert _pages(data) == _pages(src.read_bytes()) and key["changes"] == []
    with pdfplumber.open(src) as a, pdfplumber.open(io.BytesIO(data)) as b:
        assert [p.extract_text() for p in a.pages] == [p.extract_text() for p in b.pages]


# ---------------------------------------------------------------- determinism and seeds


def test_the_same_seed_gives_the_same_copy(batch) -> None:
    src = _pdf(batch)
    for tamper_type in ("inplace_edit", "rerender_inconsistent", "resaved_genuine"):
        a = tc.tamper_copy(src, tamper_type, seed=5)
        b = tc.tamper_copy(src, tamper_type, seed=5)
        assert a[1] == b[1], tamper_type
        assert a[0] == b[0], tamper_type


def test_other_seeds_give_other_edits(batch) -> None:
    src = _pdf(batch)
    edits = {tuple(sorted((c["row"], c["new"]) for c in tc.tamper_copy(src, "inplace_edit",
                                                                         seed=s)[1]["changes"]))
             for s in range(1, 6)}  # fmt: skip
    assert len(edits) > 1


def test_about_half_of_the_pypdf_saves_are_incremental(batch) -> None:
    src = _pdf(batch)
    flags = []
    for seed in range(1, 11):
        data, key = tc.tamper_copy(src, "resaved_genuine", seed=seed)
        flags.append(key["incremental_save"])
        eofs = data.count(b"%%EOF")
        assert eofs == (2 if key["incremental_save"] else 1), (seed, eofs)
        assert data != src.read_bytes()  # a re-save is never the original file itself
        if key["incremental_save"]:
            assert data.startswith(src.read_bytes())  # the original, then an appended update
    assert True in flags and False in flags


def test_unknown_types_and_not_applicable_are_refused(batch) -> None:
    with pytest.raises(ValueError, match="unknown tamper type"):
        tc.tamper_copy(_pdf(batch), "scribble", seed=1)
    assert set(tc.APPLIES_TO) == set(tc.TYPES)
