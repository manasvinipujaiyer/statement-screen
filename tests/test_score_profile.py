"""Profile scoring: known answers on a tiny hand-built example."""

import polars as pl
import pytest

from statementscreen import score_profile as sp


def _profile(sid: str, income, has_salary=False, employer=None, ctype="unknown", emis=0,
             emi_total=0, bounces=0) -> dict:  # fmt: skip
    values = {"estimated_monthly_income_paise": income, "has_salary": has_salary,
              "employer": employer, "customer_type": ctype, "emi_count": emis,
              "emi_total_monthly_paise": emi_total, "bounced_emis": bounces}  # fmt: skip
    return {"statement_id": sid, "warnings": [],
            "findings": {k: {"value": v, "reason": "r"} for k, v in values.items()}}  # fmt: skip


def _truth(rows: list[tuple]) -> pl.DataFrame:
    cols = ["statement_id", "t_type", "t_income", "t_employer", "t_emi_count", "t_emi_total",
            "t_bounces", "layout", "tags"]  # fmt: skip
    return pl.DataFrame(rows, schema=cols, orient="row")


@pytest.fixture
def table() -> pl.DataFrame:
    truth = _truth([
        ("s1", "salaried", 100, "Acme Tech Pvt Ltd", 1, 50, 0, "columns", ""),
        ("s2", "salaried", 200, "Bharat Mills", 0, 0, 1, "drcr", "emi_bounce"),
        ("s3", "gig", 300, None, 2, 70, 0, "columns", "emi_bounce"),
        ("s4", "self_employed", 0, None, 0, 0, 0, "drcr", ""),
    ])  # fmt: skip
    profiles = {
        "s1": _profile("s1", 110, True, "ACME TECH", "salaried", 1, 50),
        "s2": _profile("s2", 200, True, "Bharat Textile Mills", "salaried", 0, 0, 1),
        "s3": _profile("s3", None, False, None, "self_employed", 1, 40),
        "s4": _profile("s4", 50, False, None, "self_employed"),
    }  # fmt: skip
    return sp.statement_table(profiles, truth)


def test_income_error_both_ways(table) -> None:
    inc = sp.income_metrics(table)
    # s1: |110-100|/100 = 10%; s2: 0%; s3: no estimate; s4: true income 0, never divided
    assert inc["estimated_only"] == {"statements": 2, "median_ape": pytest.approx(0.05),
                                     "within_5pct": 0.5, "within_10pct": 1.0}  # fmt: skip
    missing = inc["missing_as_100pct"]  # s3 counts as a 100% error
    assert missing["statements"] == 3 and missing["median_ape"] == pytest.approx(0.10)
    assert missing["within_5pct"] == pytest.approx(1 / 3)
    assert missing["within_10pct"] == pytest.approx(2 / 3)
    assert inc["share_without_estimate"] == pytest.approx(1 / 3)
    assert inc["true_income_zero"] == 1


def test_salary_employer_type_emi_and_bounce(table) -> None:
    m = sp.metrics(table)
    assert m["has_salary"]["precision"] == 1.0 and m["has_salary"]["recall"] == 1.0
    assert m["employer"]["exact_match_rate"] == 0.5  # 'acme tech' matches; 'bharat ...' doesn't
    assert m["customer_type"]["accuracy"] == pytest.approx(3 / 4)
    assert m["customer_type"]["confusion_true_to_predicted"]["gig"] == {"self_employed": 1}
    assert m["emi"]["count_exact_rate"] == pytest.approx(3 / 4)
    assert m["emi"]["total_monthly_abs_error_paise"]["max"] == 30
    assert m["bounces"]["count_exact_rate"] == 1.0


def test_worst_errors_count_a_missing_estimate_as_100pct(table) -> None:
    worst = sp.worst(table, n=2)
    assert [w["statement_id"] for w in worst] == ["s3", "s1"]
    assert worst[0]["error"] == 1.0


def test_breakdowns_split_the_same_metrics(table) -> None:
    b = sp.breakdowns(table)
    assert b["by_layout"]["columns"]["statements"] == 2
    assert b["by_hard_case"]["emi_bounce"]["statements"] == 2
    assert b["by_customer_type"]["salaried"]["income"]["estimated_only"]["statements"] == 2


def test_a_statement_without_a_profile_is_an_error(table) -> None:
    truth = _truth([("s9", "gig", 1, None, 0, 0, 0, "drcr", "")])
    with pytest.raises(ValueError, match="no profile"):
        sp.statement_table({}, truth)
