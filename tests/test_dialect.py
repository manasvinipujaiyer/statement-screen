"""Narration dialects: loading the files, and keeping the default output byte-identical."""

import json
from pathlib import Path

import pytest

from statementscreen.batch_hash import fingerprint
from statementscreen.synth.dialect import (
    KINDS,
    DialectError,
    available_dialects,
    load_dialect,
    parse_dialect,
)

FIXTURE = Path(__file__).parent / "data" / "default_batch_sha256.json"
KARAN = Path(__file__).parents[1] / "src/statementscreen/synth/dialects/karan_v1.txt"


# ---------------------------------------------------------------- the default stays identical


def test_default_output_is_byte_identical_to_before_dialects(tmp_path) -> None:
    """The fingerprint was recorded with the generator as it was before dialects existed."""
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    now = fingerprint(tmp_path)
    assert set(now) == set(recorded) == {"mixed", "self_employed"}
    for name in now:
        assert now[name]["plans"] == recorded[name]["plans"]
        assert now[name]["sha256"] == recorded[name]["sha256"], name
    assert all("self_employed" in line for line in now["self_employed"]["plans"])


# ---------------------------------------------------------------- karan_v1


def test_every_karan_line_loads() -> None:
    d = load_dialect("karan_v1")
    lines = [ln for ln in KARAN.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(d.formats) == len(lines) == 26
    assert [f.line for f in d.formats] == list(range(1, 27))
    assert d.missing_kinds == ()  # all 16 kinds have at least one format
    assert {f.kind for f in d.formats} == set(KINDS)


def test_karan_lines_are_used_exactly_as_written() -> None:
    d = load_dialect("karan_v1")
    for f, raw in zip(d.formats, KARAN.read_text(encoding="utf-8").splitlines(), strict=True):
        assert raw.split("|", 2)[2].strip() == f.template


def test_dialect_names() -> None:
    assert load_dialect("default") is None
    assert "karan_v1" in available_dialects()
    for bad in ("nope", "../karan_v1", "KARAN_V1"):
        with pytest.raises(DialectError, match="unknown narration dialect"):
            load_dialect(bad)


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("salary | CR", "expected 'kind"),
        ("wages | CR | PAY {employer}", "unknown kind"),
        ("salary | IN | PAY {employer}", "must be CR or DR"),
        ("emi_return | DR | RTN {lender}", "generator posts"),
        ("salary | CR | PAY {employer} {amount}", "unknown placeholder"),
        ("salary | CR | PAY {employer", "unbalanced"),
        ("salary | CR | PAY {employer!r}", "unknown placeholder"),
    ],
)
def test_bad_lines_are_refused_with_their_line_number(line: str, message: str) -> None:
    with pytest.raises(DialectError, match=message) as e:
        parse_dialect(f"interest | CR | INT\n{line}\n", "test")
    assert "test line 2" in str(e.value)


def test_placeholders_are_kept_in_order() -> None:
    d = parse_dialect("rent | DR | NEFT DR-{ifsc}-{payee}-HOUSE RENT {month}-{ref}\n", "t")
    assert d.formats[0].fields == ("ifsc", "payee", "month", "ref")
    assert d.missing_kinds == tuple(k for k in KINDS if k != "rent")
