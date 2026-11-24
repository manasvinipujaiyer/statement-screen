"""The scoring module (tamper/scoring.py) on a tiny hand-built example with known rates."""

import csv
import json

import polars as pl
import pytest

from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.scoring import (
    COPY_COLUMNS,
    Item,
    bootstrap_ci,
    build_report,
    copy_rows,
    genuine_items,
    layouts_from_manifest,
    load_copies,
    load_findings,
    score_item,
    write_copy_table,
)

STRONG = Finding("BALANCE_CHAIN", "arithmetic", 1, 0, "strong", "balance broken")
WEAK_LAYOUT = Finding("DIGIT_SPACING", "layout", 1, None, "weak", "uneven gaps")
WEAK_HISTORY = Finding("INCREMENTAL_UPDATE", "history", None, None, "weak", "saved twice")
VERDICTS = {
    "tampered": (STRONG,),
    "suspicious": (WEAK_LAYOUT, WEAK_HISTORY),  # two weak codes; one is file history
    "none": (),
}


def _items(kind, outcomes, layouts=("columns", "drcr"), mode="n/a", prefix=None):
    """One Item per outcome name in `outcomes`, alternating layouts, each its own source."""
    prefix = prefix or kind
    return [
        Item(
            item=f"{prefix}_{n:02d}",
            source=f"src_{prefix}_{n:02d}",
            kind=kind,
            layout=layouts[n % len(layouts)],
            save_mode=mode,
            findings=VERDICTS[o],
            facts={"pages_changed": [1], "fields_changed": 1, "donor": None},
        )
        for n, o in enumerate(outcomes)
    ]


def _report(copies, genuine=None):
    return build_report(copies, genuine or {})


def test_detection_rates_are_exact_on_a_hand_built_type():
    copies = _items("inplace_edit", ["tampered"] * 6 + ["suspicious"] * 2 + ["none"] * 2)
    d = _report(copies)["detection"]["inplace_edit"]
    w = d["with_history"]
    assert d["n"] == 10 and w["labels"] == {
        "no evidence of tampering": 2,
        "suspicious": 2,
        "tampered": 6,
        "could not check": 0,
    }
    assert (w["tampered"]["count"], w["tampered"]["rate"]) == (6, 0.6)
    assert (w["tampered_or_suspicious"]["count"], w["tampered_or_suspicious"]["rate"]) == (8, 0.8)
    assert w["tampered"]["text"] == "6/10 (60.0%)"
    lo, hi = w["tampered"]["ci95"]
    assert 0 <= lo <= 0.6 <= hi <= 1


def test_without_file_history_the_suspicious_copies_are_no_longer_flagged():
    copies = _items("inplace_edit", ["tampered"] * 6 + ["suspicious"] * 2 + ["none"] * 2)
    wo = _report(copies)["detection"]["inplace_edit"]["without_history"]
    assert wo["tampered"]["count"] == 6
    assert wo["tampered_or_suspicious"]["count"] == 6  # one weak sign left: not enough


def test_a_type_that_depends_on_file_history_is_called_out():
    # every copy is "suspicious" only because of a file-history finding plus one weak layout one
    copies = _items("page_swap", ["suspicious"] * 10) + _items("page_remove", ["tampered"] * 10)
    notes = _report(copies)["history_dependence"]["detection"]
    assert len(notes) == 1 and notes[0].startswith("page_swap:")
    assert "10/10 (100.0%) to 0/10 (0.0%)" in notes[0]
    assert "depends mainly on file history" in notes[0]


def test_a_drop_of_less_than_half_is_not_called_out():
    copies = _items("inplace_edit", ["tampered"] * 6 + ["suspicious"] * 2 + ["none"] * 2)
    assert _report(copies)["history_dependence"]["detection"] == []


def test_false_alarms_on_genuine_pdfs_and_resaved_copies():
    genuine = _items("genuine", ["none"] * 19 + ["suspicious"], prefix="stmt")
    resaved = _items("resaved_genuine", ["none"] * 4, mode="incremental")
    rep = _report(resaved, {"dev": genuine})
    g = rep["false_alarms"]["genuine"]["dev"]["with_history"]
    assert g["tampered"]["count"] == 0 and g["tampered"]["rate"] == 0.0
    assert (g["tampered_or_suspicious"]["count"], g["tampered_or_suspicious"]["rate"]) == (1, 0.05)
    r = rep["false_alarms"]["resaved_genuine"]["with_history"]
    assert r["tampered"]["count"] == 0 and r["tampered_or_suspicious"]["count"] == 0
    (case,) = rep["false_alarm_cases"]
    assert case["population"] == "dev" and case["outcome"] == "suspicious"
    assert case["summary"].strip() and case["reasons"]


WEAK_ANOMALY = Finding("ANOMALY_SCORE", "anomaly", None, None, "weak", "unusual money pattern")


def test_genuine_populations_get_their_own_row_label_and_a_no_anomaly_recount():
    # two genuine statements look suspicious only because of the anomaly score plus one weak sign
    flagged = [
        Item(f"stmt_{n}", f"stmt_{n}", "genuine", "columns", findings=(WEAK_ANOMALY, WEAK_LAYOUT))
        for n in (1, 2)
    ]
    clean = _items("genuine", ["none"] * 8, prefix="stmt_c")
    rep = _report([], {"train": flagged + clean, "dev": clean})["false_alarms"]["genuine"]
    train, dev = rep["train"], rep["dev"]
    assert train["with_history"]["tampered_or_suspicious"]["text"] == "2/10 (20.0%)"
    assert train["without_anomaly"]["tampered_or_suspicious"]["text"] == "0/10 (0.0%)"
    assert "in-sample for anomaly" in train["population"]
    assert "by construction" in dev["population"] and dev["n"] == 8  # never combined with train


def test_a_strong_finding_on_a_genuine_pdf_is_listed_for_the_stop_rule():
    bad = Item("stmt_9", "stmt_9", "genuine", "drcr", findings=(STRONG, WEAK_LAYOUT))
    good = _items("genuine", ["none"] * 3, prefix="stmt_g")
    rep = _report([], {"dev": good + [bad]})
    (case,) = rep["genuine_with_strong_findings"]
    assert (case["population"], case["item"], case["codes"]) == ("dev", "stmt_9", ["BALANCE_CHAIN"])
    assert case["reasons"] == ["balance broken"]
    assert _report([], {"dev": good})["genuine_with_strong_findings"] == []


def test_codes_fired_counts_copies_per_strong_and_weak_code():
    copies = _items("inplace_edit", ["tampered"] * 3 + ["suspicious"] * 2 + ["none"])
    fired = _report(copies)["detection"]["inplace_edit"]["codes_fired"]
    assert fired == {
        "strong": {"BALANCE_CHAIN": 3},
        "weak": {"DIGIT_SPACING": 2, "INCREMENTAL_UPDATE": 2},
    }


def test_false_alarms_that_come_from_file_history_are_called_out():
    resaved = _items("resaved_genuine", ["suspicious"] * 4, mode="incremental")
    notes = _report(resaved)["history_dependence"]["false_alarms"]
    assert len(notes) == 1 and "come mainly from file history" in notes[0]


def test_save_mode_breakdown_for_library_written_copies_and_resaves():
    inc = _items("inplace_edit", ["tampered", "suspicious"], mode="incremental", prefix="a")
    full = _items("inplace_edit", ["tampered", "none"], mode="full", prefix="b")
    rep = _report(inc + full + _items("rerender_inconsistent", ["tampered"] * 2))
    modes = rep["detection"]["inplace_edit"]["by_save_mode"]
    assert modes["incremental"]["tampered_or_suspicious"]["text"] == "2/2 (100.0%)"
    assert modes["full"]["tampered_or_suspicious"]["text"] == "1/2 (50.0%)"
    assert modes["incremental"]["tampered_or_suspicious_without_history"]["count"] == 1
    assert "by_save_mode" not in rep["detection"]["rerender_inconsistent"]  # not library-written


def test_layout_breakdown_and_a_gap_over_15_points_is_flagged():
    # layouts alternate: columns on even positions (4 of 5 tampered), drcr on odd (1 of 5)
    outcomes = ["tampered", "none", "tampered", "none", "tampered", "none", "tampered"]
    outcomes += ["none", "none", "tampered"]
    rep = _report(_items("page_remove", outcomes))
    d = rep["detection"]["page_remove"]
    assert d["by_layout"]["columns"]["tampered"]["text"] == "4/5 (80.0%)"
    assert d["by_layout"]["drcr"]["tampered"]["text"] == "1/5 (20.0%)"
    (note, *_) = rep["layout_notes"]
    assert note.startswith("page_remove:") and "60.0 points" in note
    assert "counts per layout are small" in note


def test_a_small_layout_gap_is_not_flagged():
    d = _report(_items("page_remove", ["tampered"] * 10))
    assert d["layout_notes"] == []


def test_first_check_is_counted_per_type_by_verdict():
    copies = _items("page_remove", ["tampered", "tampered", "suspicious", "none"])
    fc = _report(copies)["detection"]["page_remove"]["first_check"]
    assert fc["tampered"] == {"arithmetic:BALANCE_CHAIN": 2}
    # a suspicious copy's first finding is the earlier check group's: layout before history
    assert fc["suspicious"] == {"layout:DIGIT_SPACING": 1}


def test_misses_list_everything_not_called_tampered_with_reasons():
    copies = _items("inplace_edit", ["tampered", "suspicious", "none"])
    misses = _report(copies)["misses"]
    assert [m["outcome"] for m in misses] == ["only suspicious", "not flagged"]
    assert all(m["summary"].strip() and m["type"] == "inplace_edit" for m in misses)
    assert misses[0]["reasons"] and misses[0]["pages_changed"] == [1]
    assert misses[1]["reasons"] == [] and "no strong or weak finding" in misses[1]["summary"]


def test_overlay_is_reported_apart_from_the_six_types():
    copies = _items("inplace_edit", ["tampered"] * 2) + _items("overlay", ["tampered", "none"])
    rep = _report(copies)
    assert "overlay" not in rep["detection"]
    assert rep["overlay_holdout"]["with_history"]["tampered"]["text"] == "1/2 (50.0%)"
    assert rep["pooled_detection"]["with_history"]["tampered"]["total"] == 2  # overlay left out
    assert [m["type"] for m in rep["overlay_misses"]] == ["overlay"] and rep["misses"] == []
    assert _report(_items("inplace_edit", ["tampered"]))["overlay_holdout"] is None


def test_unknown_tamper_type_is_refused():
    with pytest.raises(ValueError, match="unknown tamper types"):
        _report(_items("made_up", ["tampered"]))


def test_bootstrap_is_deterministic_and_resamples_sources_not_copies():
    sources = [f"s{i}" for i in range(20)]
    hits = [i % 4 == 0 for i in range(20)]
    assert bootstrap_ci(sources, hits) == bootstrap_ci(sources, hits)
    lo, hi = bootstrap_ci(sources, hits)
    assert lo < 0.25 < hi
    assert bootstrap_ci(["a", "b"], [True, True]) == [1.0, 1.0]
    assert bootstrap_ci(["a", "b"], [False, False]) == [0.0, 0.0]
    # two copies of ONE source are one cluster: every resample is the same, so no spread
    assert bootstrap_ci(["a", "a"], [True, False]) == [0.5, 0.5]
    assert bootstrap_ci([], []) is None


def test_the_pooled_rate_keeps_a_sources_copies_together():
    one_source = [
        Item("copy_01", "src_a", "inplace_edit", "columns", findings=VERDICTS["tampered"]),
        Item("copy_02", "src_a", "page_remove", "columns", findings=VERDICTS["none"]),
    ]
    pooled = _report(one_source)["pooled_detection"]["with_history"]["tampered"]
    assert pooled["text"] == "1/2 (50.0%)" and pooled["ci95"] == [0.5, 0.5]


def test_the_report_is_identical_across_two_builds():
    copies = _items("inplace_edit", ["tampered", "suspicious", "none"] * 4)
    genuine = {"dev": _items("genuine", ["none"] * 6, prefix="stmt")}
    a = json.dumps(_report(copies, genuine), sort_keys=True)
    assert a == json.dumps(_report(copies, genuine), sort_keys=True)


# ---------------------------------------------------------------- loading and the table


def test_loading_keys_findings_and_the_manifest(tmp_path):
    keys = tmp_path / "dev" / "keys"
    keys.mkdir(parents=True)
    specs = [("inplace_edit", True), ("rerender_consistent", None)]
    for n, (kind, incremental) in enumerate(specs):
        key = {
            "copy_id": f"copy_{n + 1:04d}",
            "source": f"stmt_{n + 1:04d}",
            "source_layout": "drcr",
            "tamper_type": kind,
            "incremental_save": incremental,
            "pages_changed": [2],
            "changes": [{"page": 2}, {"page": 2}],
            "donor": None,
        }
        (keys / f"copy_{n + 1:04d}.json").write_text(json.dumps(key), encoding="utf-8")
    cache = tmp_path / "findings.parquet"
    pl.DataFrame(
        {
            "item": ["copy_0001"],
            "group": ["arithmetic"],
            "code": ["BALANCE_CHAIN"],
            "page": [2],
            "row": [3],
            "strength": ["strong"],
            "reason": ["balance broken"],
        }
    ).write_parquet(cache)
    items = load_copies(tmp_path / "dev", load_findings(cache))
    assert [(i.item, i.kind, i.save_mode, i.layout) for i in items] == [
        ("copy_0001", "inplace_edit", "incremental", "drcr"),
        ("copy_0002", "rerender_consistent", "n/a", "drcr"),
    ]
    assert items[0].findings[0].code == "BALANCE_CHAIN" and items[1].findings == ()
    assert items[0].facts == {"pages_changed": [2], "fields_changed": 2, "donor": None}

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "statement_id,layout,customer_type\nstmt_0001,columns,salaried\n", encoding="utf-8"
    )
    layouts = layouts_from_manifest(manifest)
    (g,) = genuine_items(["stmt_0001"], layouts, {})
    assert (g.kind, g.layout, g.findings) == ("genuine", "columns", ())


def test_the_per_copy_table_has_the_audit_columns(tmp_path):
    copies = _items("inplace_edit", ["tampered", "none"], mode="full")
    rows = copy_rows(copies)
    assert tuple(rows[0]) == COPY_COLUMNS
    assert rows[0]["verdict"] == "tampered"
    assert rows[0]["first_check"] == "arithmetic:BALANCE_CHAIN"
    assert rows[1]["verdict"] == "no evidence of tampering" and rows[1]["first_check"] == ""
    assert "verdict_without_history" in copy_rows(copies, extended=True)[0]
    out = tmp_path / "reports" / "copies.csv"
    write_copy_table(out, rows)
    with out.open(newline="", encoding="utf-8") as f:
        read = list(csv.DictReader(f))
    assert read == rows and [r["copy"] for r in read] == ["inplace_edit_00", "inplace_edit_01"]


CRASH = Finding("CHECK_NOT_RUN", "parse", None, None, "info", "the detector crashed: ValueError")


def _crashed(item, kind, layout="columns"):
    return Item(item, f"src_{item}", kind, layout, findings=(CRASH,), crashed=True)


def test_a_crashed_pdf_is_could_not_check_and_never_no_evidence():
    copies = _items("inplace_edit", ["tampered"] * 3 + ["none"]) + [
        _crashed("crash_01", "inplace_edit")
    ]
    report = _report(copies)
    group = report["detection"]["inplace_edit"]
    w = group["with_history"]
    assert w["labels"]["could not check"] == 1 and w["labels"]["no evidence of tampering"] == 1
    assert group["could_not_check"]["count"] == 1 and group["could_not_check"]["total"] == 5
    assert w["tampered"]["count"] == 3 and w["tampered"]["total"] == 5
    assert w["tampered_or_suspicious"]["count"] == 3  # could not check is not flagged
    assert group["without_history"]["labels"]["could not check"] == 1
    outcomes = {c["item"]: c["outcome"] for c in report["misses"]}
    assert outcomes["crash_01"] == "could not check"
    assert [c["item"] for c in report["could_not_check_cases"]] == ["crash_01"]


def test_a_crashed_genuine_pdf_is_not_a_clean_one_and_not_a_false_alarm():
    clean = Item("stmt_1", "stmt_1", "genuine", "columns")
    genuine = {"dev": [clean, _crashed("stmt_2", "genuine")]}
    report = _report([], genuine)
    row = report["false_alarms"]["genuine"]["dev"]
    assert row["could_not_check"]["text"] == "1/2 (50.0%)"
    assert row["with_history"]["labels"]["no evidence of tampering"] == 1
    assert row["without_anomaly"]["labels"]["could not check"] == 1
    assert report["false_alarm_cases"] == []
    assert [(c["population"], c["item"]) for c in report["could_not_check_cases"]] == [
        ("dev", "stmt_2")
    ]


def test_the_pooled_row_and_the_table_carry_could_not_check():
    copies = _items("inplace_edit", ["tampered"]) + [_crashed("crash_01", "page_swap")]
    report = _report(copies)
    assert report["pooled_detection"]["could_not_check"]["count"] == 1
    rows = {r["copy"]: r for r in copy_rows(copies)}
    assert rows["crash_01"]["verdict"] == "could not check"
    assert rows["inplace_edit_00"]["verdict"] == "tampered"


def test_the_crashed_flag_comes_from_the_runs_crashed_list(tmp_path):
    keys = tmp_path / "keys"
    keys.mkdir()
    for n in (1, 2):
        key = {
            "copy_id": f"copy_{n:04d}",
            "source": f"stmt_{n:04d}",
            "source_layout": "columns",
            "tamper_type": "inplace_edit",
            "incremental_save": None,
            "pages_changed": [1],
            "changes": [],
            "donor": None,
        }
        (keys / f"copy_{n:04d}.json").write_text(json.dumps(key), encoding="utf-8")
    copies = load_copies(tmp_path, {}, frozenset({"copy_0002"}))
    assert [(c.item, c.crashed) for c in copies] == [("copy_0001", False), ("copy_0002", True)]
    items = genuine_items(["stmt_b", "stmt_a"], {"stmt_a": "drcr", "stmt_b": "columns"}, {},
                          frozenset({"stmt_b"}))  # fmt: skip
    assert [(i.item, i.crashed) for i in items] == [("stmt_a", False), ("stmt_b", True)]


def _crash_in(group):
    reason = f"{group} checks didn't run: ValueError: boom"
    return Finding("CHECK_NOT_RUN", group, None, None, "info", reason)


def _one(findings):
    return Item("c_1", "s_1", "inplace_edit", "columns", findings=tuple(findings))


def test_a_crashed_check_turns_no_evidence_into_could_not_check():
    report = _report([_one([_crash_in("layout")])])
    group = report["detection"]["inplace_edit"]
    assert group["with_history"]["labels"]["could not check"] == 1
    assert group["with_history"]["labels"]["no evidence of tampering"] == 0
    assert group["could_not_check"]["count"] == 1 and group["checks_not_run"] == {"layout": 1}
    assert [c["checks_not_run"] for c in report["could_not_check_cases"]] == [["layout"]]


def test_a_strong_finding_still_gives_tampered_when_a_check_crashed():
    group = _report([_one([STRONG, _crash_in("layout")])])["detection"]["inplace_edit"]
    assert group["with_history"]["tampered"]["count"] == 1
    assert group["could_not_check"]["count"] == 0 and group["checks_not_run"] == {"layout": 1}


def test_a_weak_verdict_stands_when_a_check_crashed():
    group = _report([_one([*VERDICTS["suspicious"], _crash_in("anomaly")])])["detection"][
        "inplace_edit"
    ]
    labels = group["with_history"]["labels"]
    assert labels["suspicious"] == 1 and labels["could not check"] == 0
    assert group["checks_not_run"] == {"anomaly": 1}


def test_the_without_views_ignore_the_crash_of_the_check_they_remove():
    history = score_item(_one([_crash_in("history")]))
    assert history.label == "could not check"
    assert history.label_without_history == "no evidence of tampering"
    assert history.label_without_anomaly == "could not check"
    anomaly = score_item(_one([_crash_in("anomaly")]))
    assert anomaly.label == "could not check"
    assert anomaly.label_without_anomaly == "no evidence of tampering"
    assert anomaly.label_without_history == "could not check"


def test_each_check_is_counted_and_a_pdf_can_crash_several():
    items = [
        Item("c_1", "s_1", "inplace_edit", "columns", findings=(_crash_in("layout"),)),
        Item("c_2", "s_2", "inplace_edit", "drcr", findings=(_crash_in("layout"), STRONG)),
        Item("c_3", "s_3", "inplace_edit", "columns", findings=(_crash_in("history"),
                                                                _crash_in("anomaly"))),
    ]  # fmt: skip
    group = _report(items)["detection"]["inplace_edit"]
    assert group["checks_not_run"] == {"anomaly": 1, "history": 1, "layout": 2}
    assert group["could_not_check"]["count"] == 2  # c_2 is tampered: it keeps its verdict


def _not_run(group, reason):
    return Finding("CHECK_NOT_RUN", group, None, None, "info", reason)


def test_the_only_check_that_does_not_apply_by_design_is_the_anomaly_check_with_no_transactions():
    note = _not_run("anomaly", "ANOMALY_SCORE: no parsed transactions to measure")
    scored = score_item(_one([note]))
    assert scored.label == "no evidence of tampering" and scored.unrun_checks == ()


def test_every_arithmetic_not_checked_note_counts_as_a_check_that_did_not_run():
    notes = [  # the parser's own wording: a value the generator always prints was not read
        "BALANCE_CHAIN: no opening balance for the first row",
        "CLOSING_MISMATCH: no closing or comparison balance read",
        "SUMMARY_IDENTITY: not read: closing balance",
        "SUMMARY_TOTALS: no printed total credits",
        "SUMMARY_COUNTS: no printed Cr count",
        "DATE_OUT_OF_PERIOD: no statement period read",
    ]
    for note in notes:
        finding = _not_run("arithmetic", f"a self-check couldn't run ({note})")
        scored = score_item(_one([finding]))
        assert scored.label == "could not check" and scored.unrun_checks == ("arithmetic",), note


def test_a_check_that_could_not_run_makes_no_evidence_could_not_check():
    cases = {
        "layout": "AMOUNT_FONT, AMOUNT_ALIGNMENT, DIGIT_SPACING: layout not recognised (x)",
        "history": "MODDATE_AFTER_CREATION, EXTRA_FONTS: pypdf couldn't read the file (x)",
        "structure": "HEADER_MISMATCH: page 1's account number or period wasn't read",
        "anomaly": "ANOMALY_SCORE: no model at week6_iforest.pkl",
    }
    for group, reason in cases.items():
        scored = score_item(_one([_not_run(group, reason)]))
        assert scored.label == "could not check" and scored.unrun_checks == (group,), group


def test_an_unparseable_pdf_is_suspicious_and_the_checks_it_stopped_are_counted():
    findings = [
        Finding("PARSE_FAILED", "parse", None, None, "weak", "the PDF couldn't be parsed (x)"),
        _not_run("arithmetic", "the arithmetic checks didn't run because the PDF couldn't be "
                 "parsed"),
        _not_run("structure", "HEADER_MISMATCH: the PDF couldn't be parsed"),
        _not_run("history", "EXTRA_FONTS: no font limit for layout None"),
        _not_run("layout", "AMOUNT_FONT: layout not recognised (x)"),
        _not_run("anomaly", "ANOMALY_SCORE: no parsed transactions to measure"),
    ]  # fmt: skip
    group = _report([_one(findings)])["detection"]["inplace_edit"]
    assert group["with_history"]["labels"]["suspicious"] == 1  # the earlier rule stands
    assert group["could_not_check"]["count"] == 0
    # on an unparseable PDF "nothing to measure" is a consequence of the failure: it counts too
    assert group["checks_not_run"] == {
        "anomaly": 1,
        "arithmetic": 1,
        "history": 1,
        "layout": 1,
        "structure": 1,
    }
