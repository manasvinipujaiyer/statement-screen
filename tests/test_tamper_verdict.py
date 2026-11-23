"""The verdict rules (tamper/detect/verdict.py) on hand-built findings."""

import json
import random

from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.verdict import (
    NONE,
    SUSPICIOUS,
    TAMPERED,
    decide,
    order,
    to_json,
)


def _f(code: str, strength: str, group: str = "layout", page: int | None = None) -> Finding:
    return Finding(code, group, page, None, strength, f"{code} reason")


def _label(*findings: Finding) -> str:
    return decide(list(findings))["label"]


def test_no_findings_is_no_evidence():
    v = decide([])
    assert v["label"] == NONE and v["reasons"] == [] and v["summary"].strip()


def test_any_strong_finding_is_tampered():
    assert _label(_f("BALANCE_CHAIN", "strong", "arithmetic")) == TAMPERED
    assert _label(_f("X", "weak"), _f("BALANCE_CHAIN", "strong", "arithmetic")) == TAMPERED


def test_one_weak_finding_is_not_enough():
    assert _label(_f("DIGIT_SPACING", "weak")) == NONE


def test_two_different_weak_findings_are_suspicious():
    assert _label(_f("DIGIT_SPACING", "weak"), _f("INCREMENTAL_UPDATE", "weak")) == SUSPICIOUS


def test_the_same_weak_code_many_times_counts_once():
    cells = [_f("DIGIT_SPACING", "weak", page=p) for p in range(1, 40)]
    assert _label(*cells) == NONE


def test_info_is_listed_but_never_counted():
    found = [_f("CHECK_NOT_RUN", "info"), _f("PRODUCER_CREATOR_DIFFER", "info", "history")]
    v = decide(found + [_f("DIGIT_SPACING", "weak")])
    assert v["label"] == NONE  # one weak sign; the two info findings add nothing
    assert [r["strength"] for r in v["reasons"]][-2:] == ["info", "info"]
    assert all("[not counted]" in r["reason"] for r in v["reasons"] if r["strength"] == "info")


def test_a_pdf_that_could_not_be_parsed_is_suspicious_on_its_own():
    v = decide([_f("PARSE_FAILED", "weak", "parse"), _f("CHECK_NOT_RUN", "info", "arithmetic")])
    assert v["label"] == SUSPICIOUS and "couldn't be parsed" in v["summary"]


def test_reasons_run_strong_then_weak_then_info_and_all_have_text():
    found = [
        _f("CHECK_NOT_RUN", "info", "history"),
        _f("DATE_ORDER", "weak", "arithmetic", page=1),
        _f("PAGE_NUMBER", "strong", "structure", page=2),
        _f("BALANCE_CHAIN", "strong", "arithmetic", page=3),
        _f("DIGIT_SPACING", "weak", "layout", page=1),
    ]
    v = decide(found)
    assert [r["strength"] for r in v["reasons"]] == ["strong", "strong", "weak", "weak", "info"]
    # within a strength: check group order (arithmetic, structure, layout, ...), then page
    assert [r["code"] for r in v["reasons"]] == [
        "BALANCE_CHAIN",
        "PAGE_NUMBER",
        "DATE_ORDER",
        "DIGIT_SPACING",
        "CHECK_NOT_RUN",
    ]
    assert all(r["reason"].strip() for r in v["reasons"]) and v["summary"].strip()


def test_repeated_codes_are_collapsed_with_a_count():
    v = decide([_f("PAGE_NUMBER", "strong", "structure", page=p) for p in (2, 3, 4)])
    (r,) = v["reasons"]
    assert r["count"] == 3 and "(and 2 more like it)" in r["reason"]
    assert len(v["findings"]) == 3  # the full list is kept


def test_the_verdict_does_not_depend_on_the_order_findings_arrive_in():
    found = [_f(c, s, g, p) for c, s, g, p in [
        ("BALANCE_CHAIN", "strong", "arithmetic", 1),
        ("DATE_ORDER", "weak", "arithmetic", 2),
        ("INCREMENTAL_UPDATE", "weak", "history", None),
        ("CHECK_NOT_RUN", "info", "anomaly", None),
        ("PAGE_NUMBER", "strong", "structure", 3),
    ]]  # fmt: skip
    want = to_json(decide(found))
    for seed in range(5):
        shuffled = found[:]
        random.Random(seed).shuffle(shuffled)
        assert to_json(decide(shuffled)) == want
    assert order(found) == order(list(reversed(found)))


def test_the_verdict_json_round_trips_and_carries_the_rule_version():
    v = decide([_f("BALANCE_CHAIN", "strong", "arithmetic")])
    assert json.loads(to_json(v)) == v and v["rule_version"] == "v1"
