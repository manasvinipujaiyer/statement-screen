"""Recurring-payment features per group."""

from datetime import date

import polars as pl
import pytest

from statementscreen.patterns.cli import process_statement
from statementscreen.patterns.features import (
    circular_day_distance,
    day_arc,
    day_medoid,
    months_between,
    statement_months,
)


def test_31st_and_1st_are_one_day_apart() -> None:
    assert circular_day_distance(31, 1) == 1
    assert circular_day_distance(1, 31) == 1
    assert circular_day_distance(15, 15) == 0
    assert circular_day_distance(1, 16) == 15 and circular_day_distance(1, 17) == 15


def test_day_arc_wraps_round_the_month_end() -> None:
    assert day_arc([30, 31, 1]) == (30, 1, 2)
    assert day_arc([5, 6]) == (5, 6, 1)
    assert day_arc([5]) == (5, 5, 0)


def test_day_medoid_is_circular() -> None:
    assert day_medoid([30, 1, 31, 30]) == 30  # 30 and 31 tie at 3; earliest wins
    assert day_medoid([1, 2, 31, 1]) == 1


def test_months() -> None:
    assert months_between(date(2025, 1, 31), date(2025, 12, 1)) == 12
    assert months_between(date(2024, 11, 5), date(2025, 2, 5)) == 4
    assert statement_months(date(2025, 1, 1), date(2025, 6, 30), []) == 6
    assert statement_months(None, None, [date(2025, 3, 2), date(2025, 1, 9)]) == 3


SALARY = "NEFT/CR/VSTKN2503105666{}/NERANA TECHNOLOGIES PVT LTD/VSTK0384513/SALARY"


def test_monthly_salary_with_a_bonus_and_a_late_month() -> None:
    posted = [date(2025, 1, 31), date(2025, 2, 28), date(2025, 3, 15), date(2025, 3, 31),
              date(2025, 4, 30), date(2025, 5, 31), date(2025, 7, 1)]  # fmt: skip
    amounts = [8_002_800, 8_002_800, 16_005_600, 8_002_800, 8_002_800, 8_002_800, 8_002_800]
    stmt = pl.DataFrame({
        "statement_id": ["stmt_0001"] * 7,
        "row": list(range(7)),
        "posted": posted,
        "narration": [SALARY.format(i) for i in range(7)],
        "amount_paise": amounts,
    })  # fmt: skip
    tables = process_statement(stmt, date(2025, 1, 1), date(2025, 7, 31), "columns")
    rows, feats = tables["rows"], tables["group_features"]
    assert rows["group_id"].n_unique() == 1 and feats.height == 1
    f = feats.row(0, named=True)
    assert (f["direction"], f["main_channel"], f["group_name"]) == (
        "credit", "NEFT", "nerana technologies")  # fmt: skip
    assert (f["n_txn"], f["statement_months"], f["months_present"]) == (7, 7, 6)
    assert f["month_coverage"] == pytest.approx(6 / 7)
    assert f["months_missing_inside"] == 1  # June: its pay landed on 1 July
    assert f["max_txn_in_a_month"] == 2 and f["txn_per_month"] == pytest.approx(7 / 6)
    assert f["gap_mean_days"] == pytest.approx(151 / 6)
    assert f["dom_medoid"] == 31
    assert f["dom_share_on_time"] == pytest.approx(6 / 7)  # all but the bonus on the 15th
    assert (f["dom_arc_start"], f["dom_arc_end"], f["dom_arc_span"]) == (15, 1, 17)
    assert f["median_amount_paise"] == 8_002_800
    assert f["amount_share_near_median"] == pytest.approx(6 / 7)
    assert f["amount_cv"] > 0
