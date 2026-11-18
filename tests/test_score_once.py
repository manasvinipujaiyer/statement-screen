"""Pre-registration and the one-time scoring: hashes, guards and profile intervals.
Nothing here opens the real test split."""

import json
from pathlib import Path

import polars as pl
import pytest

from statementscreen import preregister as pr
from statementscreen import score_once as so
from statementscreen import score_profile as sp


def small_table() -> pl.DataFrame:
    """The 4 hand-built statements of test_score_profile.py: incomes 100/200/300/0,
    estimates 110/200/none/50."""

    def profile(income, has_salary, employer, ctype, emis, emi_total, bounces=0) -> dict:
        values = {"estimated_monthly_income_paise": income, "has_salary": has_salary,
                  "employer": employer, "customer_type": ctype, "emi_count": emis,
                  "emi_total_monthly_paise": emi_total, "bounced_emis": bounces}  # fmt: skip
        findings = {k: {"value": v, "reason": "r"} for k, v in values.items()}
        return {"warnings": [], "findings": findings}

    cols = ["statement_id", "t_type", "t_income", "t_employer", "t_emi_count", "t_emi_total",
            "t_bounces", "layout", "tags"]  # fmt: skip
    truth = pl.DataFrame([
        ("s1", "salaried", 100, "Acme Tech Pvt Ltd", 1, 50, 0, "columns", ""),
        ("s2", "salaried", 200, "Bharat Mills", 0, 0, 1, "drcr", "emi_bounce"),
        ("s3", "gig", 300, None, 2, 70, 0, "columns", "emi_bounce"),
        ("s4", "self_employed", 0, None, 0, 0, 0, "drcr", ""),
    ], schema=cols, orient="row")  # fmt: skip
    profiles = {
        "s1": profile(110, True, "ACME TECH", "salaried", 1, 50),
        "s2": profile(200, True, "Bharat Textile Mills", "salaried", 0, 0, 1),
        "s3": profile(None, False, None, "self_employed", 1, 40),
        "s4": profile(50, False, None, "self_employed", 0, 0),
    }  # fmt: skip
    for sid, p in profiles.items():
        p["statement_id"] = sid
    return sp.statement_table(profiles, truth)


def _tree(root: Path) -> Path:
    """A fake repository: two source files and every pre-registered artefact."""
    (root / pr.CODE_DIR / "patterns").mkdir(parents=True)
    (root / pr.CODE_DIR / "a.py").write_text("A = 1\n")
    (root / pr.CODE_DIR / "patterns" / "rules.py").write_text("RULE = 2\n")
    for artefact in pr.ARTEFACTS:
        (root / artefact).parent.mkdir(parents=True, exist_ok=True)
        (root / artefact).write_bytes(artefact.encode())
    return root


def test_preregistration_refuses_uncommitted_changes(tmp_path) -> None:
    with pytest.raises(ValueError, match="uncommitted changes"):
        pr.build(_tree(tmp_path), head="abc", dirty=" M src/statementscreen/a.py")


def test_preregistration_hashes_code_configs_and_artefacts(tmp_path) -> None:
    prereg = pr.build(_tree(tmp_path), head="abc123", dirty="")
    h = prereg["hashes"]
    assert prereg["frozen_commit"] == "abc123"
    assert set(h["code"]) == {"src/statementscreen/a.py", "src/statementscreen/patterns/rules.py"}
    assert set(h["artefacts"]) == set(pr.ARTEFACTS)
    assert set(h["configs"]) == {"PatternConfig", "ProfileConfig"}
    assert set(prereg["sets"]) == {"test", "shift_karan_v1"}
    assert all("STATEMENTSCREEN_OPEN_TEST=1" in c for c in prereg["commands"] if "test" in c
               and "shift" not in c)  # fmt: skip


def test_verify_finds_any_change_since_preregistration(tmp_path) -> None:
    root = _tree(tmp_path)
    prereg = pr.build(root, head="abc", dirty="")
    assert so.verify(prereg, root) == []
    (root / pr.CODE_DIR / "patterns" / "rules.py").write_text("RULE = 3\n")
    (root / "data/models/is_emi.txt").write_bytes(b"retrained")
    (root / pr.CODE_DIR / "new.py").write_text("")
    problems = so.verify(prereg, root)
    assert len(problems) == 3
    assert any("rules.py" in p for p in problems) and any("is_emi.txt" in p for p in problems)


def test_a_preregistration_is_never_overwritten(tmp_path) -> None:
    out = tmp_path / "prereg.json"
    out.write_text("{}")
    assert pr.main(["--out", str(out)]) == 2
    assert out.read_text() == "{}"


def test_the_result_is_written_once_and_a_rerun_is_post_hoc(tmp_path) -> None:
    path = tmp_path / "result.json"
    so.write(path, {"first_result": True, "value": 1}, post_hoc=False)
    with pytest.raises(ValueError, match="one-time result"):
        so.write(path, {"first_result": True, "value": 2}, post_hoc=False)
    so.write(path, {"first_result": True, "value": 3}, post_hoc=True)
    saved = json.loads(path.read_text())
    assert saved["value"] == 1  # the first result is never changed
    assert saved["post_hoc_runs"] == [{"value": 3, "label": "post-hoc"}]


def test_profile_intervals_match_the_point_metrics_and_the_seed() -> None:
    table = small_table()
    a = so.profile_intervals(table, n=200, seed=7)
    assert a == so.profile_intervals(table, n=200, seed=7)
    # point values: same as score_profile on the hand example (missing estimate = 100%)
    assert a["income_median_ape"]["value"] == pytest.approx(0.10)
    assert a["income_within_10pct"]["value"] == pytest.approx(2 / 3)
    assert a["share_without_estimate"]["value"] == pytest.approx(1 / 3)
    assert a["employer_exact"]["value"] == 0.5
    assert a["customer_type_accuracy"]["value"] == pytest.approx(3 / 4)
    for m in a.values():
        if m["low"] is not None:
            assert m["low"] <= m["high"]


def test_dev_overstated_only_when_better_than_the_far_end_of_the_test_interval() -> None:
    def m(value, low=None, high=None):
        return {"value": value, "low": low, "high": high}

    dev = {name: m(0.9) for name in so.HEADLINE}
    test = {name: m(0.8, 0.85, 0.95) for name in so.HEADLINE}  # dev inside: not overstated
    test["income_within_10pct"] = m(0.7, 0.6, 0.8)  # dev 0.9 above 0.8: overstated
    test["income_median_ape"] = m(0.2, 0.95, 0.99)  # lower is better; dev 0.9 below 0.95
    assert so.dev_overstated(dev, test) == ["income_median_ape", "income_within_10pct"]
