"""The known-style match report for a dialect batch."""

import json

import pytest

from statementscreen import dialect_report as dr
from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.synth.dialect import load_dialect
from statementscreen.writer import write_batch

KARAN = load_dialect("karan_v1")


@pytest.fixture(scope="module")
def karan_batch(tmp_path_factory):
    folder = tmp_path_factory.mktemp("karan")
    write_batch(plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 6), folder, dialect=KARAN)
    return folder


def test_openings_and_recognition_on_hand_written_narrations() -> None:
    assert dr.opening_style("NEFT CR-VSTK0A1B2C3-ACME-HOLDER-VSTKN25001123456") == "neft_cr_dash"
    assert dr.opening_style("SAL CREDIT ACME JAN25 REF 1") is None
    assert dr.fully_recognised("NACH/DR/VSTK7000000000000001/ACME FINANCE")
    assert not dr.fully_recognised("SAL CREDIT ACME JAN25 REF 1")  # unknown style
    assert not dr.fully_recognised("UPI/DR/not-a-number/ACME")  # known opening, bad fields


def test_lines_are_grouped_from_their_templates() -> None:
    group = {f.line: dr.format_group(f) for f in KARAN.formats}
    assert group[1] == dr.KNOWN  # NEFT CR-{ifsc}-...
    assert group[5] == dr.KNOWN  # ACH D- {lender} ...
    assert group[2] == dr.NEW  # SAL CREDIT ...
    assert group[4] == dr.NEW  # NACH DR/... (ours is NACH/DR/)
    assert group[20] == dr.NEW  # INT.PD {month} (ours is INT.PD:)
    assert set(group.values()) == {dr.KNOWN, dr.NEW}


def test_shares_are_computed_and_saved_to_the_report(karan_batch, tmp_path) -> None:
    out = tmp_path / "report.json"
    assert dr.main([str(karan_batch), "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    lines = report["lines"]
    total = sum(e["narrations"] for e in lines)
    assert total == report["dialect_narrations"] > 0
    assert report["opening_match_share"] == round(sum(e["opening_match"] for e in lines) / total, 4)
    assert report["fully_recognised_share"] == round(
        sum(e["fully_recognised"] for e in lines) / total, 4
    )
    assert 0 <= report["fully_recognised_share"] <= report["opening_match_share"] <= 1
    groups = report["lines_by_group"]
    assert sorted(groups["known_opening"] + groups["new_opening"]) == list(range(1, 27))
    for e in lines:  # recognised implies a known opening
        assert e["fully_recognised"] <= e["opening_match"] <= e["narrations"]


def test_a_default_batch_is_refused(tmp_path) -> None:
    write_batch(plan_batch(BatchConfig(seed=42), 1), tmp_path)
    assert dr.main([str(tmp_path), "--out", str(tmp_path / "r.json")]) == 2  # no formats file
