"""The rule comparison (tamper/compare_rules.py): the adoption rule applied as written, on
hand-built items where every count is known. Files are written to temp folders only."""

import json
from pathlib import Path

import polars as pl
import pytest

from statementscreen.tamper import compare_rules as cr
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import run as driver
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.verdict import RULES, TAMPERED
from statementscreen.tamper.scoring import Item

ANOMALY = Finding("ANOMALY_SCORE", "anomaly", None, None, "weak", "unusual money pattern")
STRONG = Finding("BALANCE_CHAIN", "arithmetic", 1, 0, "strong", "balance broken")


def _copies(flagged: int, total: int = 20) -> list[Item]:
    """rerender_consistent copies (the first `flagged` carry only an anomaly score), four
    caught in-place edits and four clean re-saves."""
    items = [
        Item(
            f"rc_{n:02d}",
            f"src_rc_{n:02d}",
            "rerender_consistent",
            "columns",
            findings=(ANOMALY,) if n < flagged else (),
        )
        for n in range(total)
    ]
    items += [
        Item(f"ie_{n}", f"src_ie_{n}", "inplace_edit", "columns", findings=(STRONG,))
        for n in range(4)
    ]
    items += [Item(f"rs_{n}", f"src_rs_{n}", "resaved_genuine", "drcr", "full") for n in range(4)]
    return items


def _genuine() -> dict[str, list[Item]]:
    """Genuine dev: ten statements, one of which the anomaly check flags."""
    items = [
        Item(f"stmt_{n}", f"stmt_{n}", "genuine", "columns", findings=(ANOMALY,) if n == 0 else ())
        for n in range(10)
    ]
    return {"dev": items}


def test_v2_is_adopted_when_all_three_conditions_hold():
    result = cr.compare(_copies(flagged=3), _genuine(), "v1", "v2")  # 0/20 -> 3/20: +15 points
    c = result["conditions"]
    assert c["a_rerender_consistent_gain"]["gain_percentage_points"] == 15.0
    assert (c["a_rerender_consistent_gain"]["old"], c["a_rerender_consistent_gain"]["new"]) == (
        "0/20 (0.0%)",
        "3/20 (15.0%)",
    )
    assert c["a_rerender_consistent_gain"]["passed"] and c["all_passed"]
    assert c["b_anomaly_alone_never_tampered"]["passed"]
    assert c["c_tampered_false_alarms_unchanged"]["passed"]
    assert result["selected"] == "v2" and "adopted" in result["decision"]


def test_side_by_side_shows_both_counts_and_the_genuine_dev_suspicious_rate():
    s = cr.compare(_copies(flagged=3), _genuine(), "v1", "v2")["side_by_side"]
    v1 = s["v1"]["detection"]["rerender_consistent"]
    v2 = s["v2"]["detection"]["rerender_consistent"]
    assert (v1["tampered"]["count"], v1["tampered_or_suspicious"]["count"]) == (0, 0)
    assert (v2["tampered"]["count"], v2["tampered_or_suspicious"]["count"]) == (0, 3)
    dev1, dev2 = s["v1"]["false_alarms"]["genuine_dev"], s["v2"]["false_alarms"]["genuine_dev"]
    assert dev1["suspicious"]["count"] == 0 and dev2["suspicious"]["count"] == 1  # 1 of 10
    assert dev2["tampered"]["count"] == 0
    assert s["v2"]["false_alarms"]["resaved_genuine"]["tampered_or_suspicious"]["count"] == 0


def test_a_gain_of_exactly_ten_points_passes_and_less_fails():
    exactly = cr.compare(_copies(flagged=2), _genuine(), "v1", "v2")  # 2/20 = +10.0 points
    assert exactly["conditions"]["a_rerender_consistent_gain"]["passed"]
    assert exactly["selected"] == "v2"
    short = cr.compare(_copies(flagged=1), _genuine(), "v1", "v2")  # 1/20 = +5.0 points
    assert not short["conditions"]["a_rerender_consistent_gain"]["passed"]
    assert short["selected"] == "v1" and "a_rerender_consistent_gain" in short["decision"]


def _summary(flagged: int, genuine_tampered: int, resaved_tampered: int) -> dict:
    def metric(count: int, total: int) -> dict:
        return {"count": count, "total": total, "text": f"{count}/{total}"}

    return {
        "detection": {"rerender_consistent": {"tampered_or_suspicious": metric(flagged, 60)}},
        "false_alarms": {
            "genuine_dev": {"tampered": metric(genuine_tampered, 300)},
            "resaved_genuine": {"tampered": metric(resaved_tampered, 60)},
        },
    }


CLEAN_ANOMALY = {"tampered_without_a_strong_finding": [], "lone_anomaly_score_gives": "suspicious"}


def test_more_tampered_false_alarms_means_v1_is_kept():
    old = _summary(0, genuine_tampered=0, resaved_tampered=0)
    assert cr.adoption(old, _summary(12, 0, 0), CLEAN_ANOMALY)["all_passed"]
    for new in (_summary(12, 1, 0), _summary(12, 0, 1)):  # a new 'tampered' false alarm
        result = cr.adoption(old, new, CLEAN_ANOMALY)
        assert not result["c_tampered_false_alarms_unchanged"]["passed"]
        assert not result["all_passed"]


def test_an_anomaly_score_that_can_make_tampered_blocks_adoption():
    old, new = _summary(0, 0, 0), _summary(12, 0, 0)
    bad_probe = {"tampered_without_a_strong_finding": [], "lone_anomaly_score_gives": "tampered"}
    orphan = {"tampered_without_a_strong_finding": ["copy_0001"], "lone_anomaly_score_gives": "x"}
    for anomaly in (bad_probe, orphan):
        result = cr.adoption(old, new, anomaly)
        assert not result["b_anomaly_alone_never_tampered"]["passed"]
        assert not result["all_passed"]


def test_condition_b_is_checked_against_the_rules_and_the_data(monkeypatch):
    items = _copies(flagged=3)
    ok = cr.anomaly_alone_check(items, "v2")
    assert ok["tampered_without_a_strong_finding"] == []
    assert ok["lone_anomaly_score_gives"] == "suspicious"
    monkeypatch.setitem(RULES, "v3", lambda findings: (TAMPERED, "a rule that overreaches"))
    bad = cr.anomaly_alone_check(items, "v3")  # everything is 'tampered', even without strong
    assert "rc_00" in bad["tampered_without_a_strong_finding"]
    assert bad["lone_anomaly_score_gives"] == "tampered"
    result = cr.compare(items, _genuine(), "v1", "v3")
    assert not result["conditions"]["b_anomaly_alone_never_tampered"]["passed"]
    assert result["selected"] == "v1"


def test_the_in_sample_note_is_always_in_the_result():
    result = cr.compare(_copies(flagged=3), _genuine(), "v1", "v2")
    assert "in-sample" in result["notes"][0] and "by construction" in result["notes"][0]


def test_unknown_versions_and_missing_populations_are_refused():
    with pytest.raises(cr.ScoreError, match="unknown rule version"):
        cr.compare(_copies(3), _genuine(), "v1", "v9")
    with pytest.raises(cr.ScoreError, match="genuine_dev"):
        cr.compare(_copies(3), {}, "v1", "v2")
    with pytest.raises(cr.ScoreError, match="resaved_genuine"):
        cr.compare([i for i in _copies(3) if i.kind != "resaved_genuine"], _genuine(), "v1", "v2")


# ---------------------------------------------------------------- the command


def _write_run(
    folder: Path, items: list[str], findings: dict[str, list[Finding]], **extra: str
) -> None:
    folder.mkdir(parents=True)
    rows = [{"item": i, **f.to_dict()} for i, fs in findings.items() for f in fs]
    pl.DataFrame(rows, schema=driver.FINDING_SCHEMA).write_parquet(folder / "findings.parquet")
    n = len(items)
    pl.DataFrame(
        {
            "item": items,
            "seconds": [1.0] * n,
            "peak_memory_mb": [100.0] * n,
            "n_findings": [len(findings.get(i, [])) for i in items],
            "crashed": [False] * n,
        },
        schema=driver.ITEM_SCHEMA,
    ).write_parquet(folder / "items.parquet")
    summary = {"fingerprint": fp.fingerprint(), "pdfs": n} | extra
    (folder / "detect_run.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_set(tmp_path: Path) -> dict:
    """10 rerender_consistent copies (3 with an anomaly score), 4 re-saves, 6 genuine."""
    set_dir = tmp_path / "dev"
    (set_dir / "keys").mkdir(parents=True)
    kinds = ["rerender_consistent"] * 10 + ["resaved_genuine"] * 4
    ids = [f"copy_{n:04d}" for n in range(1, 15)]
    for n, (copy_id, kind) in enumerate(zip(ids, kinds, strict=True), start=1):
        key = {
            "copy_id": copy_id,
            "source": f"stmt_{n:04d}",
            "source_layout": "columns",
            "tamper_type": kind,
            "incremental_save": None if kind == "rerender_consistent" else False,
            "pages_changed": [],
            "changes": [],
            "donor": None,
        }
        (set_dir / "keys" / f"{copy_id}.json").write_text(json.dumps(key), encoding="utf-8")
    build = {"set": "dev", "per_type": 10, "made": {}, "short": {}, "complete": True}
    (set_dir / "build_run.json").write_text(json.dumps(build), encoding="utf-8")
    _write_run(set_dir / "detect", ids, {ids[n]: [ANOMALY] for n in range(3)})
    statements = [f"stmt_{n:04d}" for n in range(1, 7)]
    _write_run(tmp_path / "genuine_dev", statements, {"stmt_0001": [ANOMALY]}, split="dev")
    manifest = tmp_path / "manifest.csv"
    rows = ["statement_id,layout,customer_type", *[f"{s},columns,salaried" for s in statements]]
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "copies": set_dir,
        "genuine": tmp_path / "genuine_dev",
        "manifest": manifest,
        "out": tmp_path / "reports" / "comparison.json",
    }


def _argv(s: dict, *extra: str) -> list[str]:
    return [
        *("--copies", str(s["copies"])),
        *("--genuine", f"dev={s['genuine']}"),
        *("--manifest", str(s["manifest"])),
        *("--out", str(s["out"])),
        *extra,
    ]


def test_the_command_compares_from_cached_findings_and_prints_the_selection(tmp_path, capsys):
    s = _write_set(tmp_path)
    assert cr.main(_argv(s)) == 0  # 0/10 -> 3/10 on rerender_consistent: +30 points
    result = json.loads(s["out"].read_text(encoding="utf-8"))
    assert result["selected"] == "v2" and result["old"] == "v1" and result["new"] == "v2"
    assert set(result["inputs"]) == {"copies", "genuine_dev"}
    out = capsys.readouterr().out
    assert "SELECTED: v2" in out and "[PASS]" in out and "in-sample" in out
    assert "rerender_consistent" in out and "genuine_dev" in out


def test_the_command_refuses_to_overwrite_and_stale_findings(tmp_path, capsys):
    s = _write_set(tmp_path)
    assert cr.main(_argv(s)) == 0
    assert cr.main(_argv(s)) == 2 and "pass --overwrite" in capsys.readouterr().err
    assert cr.main(_argv(s, "--overwrite")) == 0
    run_file = s["copies"] / "detect" / "detect_run.json"
    run = json.loads(run_file.read_text(encoding="utf-8"))
    run["fingerprint"]["config"] = "different"
    run_file.write_text(json.dumps(run), encoding="utf-8")
    assert cr.main(_argv(s, "--overwrite")) == 2
    assert "config differs" in capsys.readouterr().err
