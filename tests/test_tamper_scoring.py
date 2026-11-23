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
