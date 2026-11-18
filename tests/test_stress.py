"""Stress perturbations: deterministic for a fixed seed, and each changes only what it should.
Also: no test in the suite opens the test split."""

import os
import re
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest

from statementscreen import stress
from statementscreen.splits import OPEN_TEST_ENV

PERIOD = (date(2025, 1, 1), date(2025, 6, 30))


def _rows() -> pl.DataFrame:
    days = [date(2025, 1, 2) + timedelta(days=9 * i) for i in range(20)]  # Jan to Jun
    return pl.DataFrame({
        "statement_id": ["stmt_0042"] * 20,
        "row": pl.Series(range(20), dtype=pl.Int32),
        "posted": days,
        "narration": [f"NEFT CR-KSVB0001234-ACME TECH PVT LTD-RAVI KUMAR-KSVBN25{i:03d}123 SALARY"
                      for i in range(20)],
        "amount_paise": [(-1) ** i * (100_000 + 1_001 * i) for i in range(20)],
    })  # fmt: skip


def _apply(name: str, sid: str = "stmt_0042"):
    return stress.PERTURBATIONS[name](_rows(), PERIOD, stress.rng_for(name, sid))


@pytest.mark.parametrize("name", list(stress.PERTURBATIONS))
def test_each_perturbation_is_deterministic(name) -> None:
    a, pa = _apply(name)
    b, pb = _apply(name)
    assert a.equals(b) and pa == pb


def test_another_statement_gets_another_stream() -> None:
    assert not _apply("typos", "stmt_0042")[0].equals(_apply("typos", "stmt_0043")[0])


def test_drop_month_removes_exactly_one_month() -> None:
    def months(frame: pl.DataFrame) -> set[tuple[int, int]]:
        return {(d.year, d.month) for d in frame["posted"]}

    before = _rows()
    after, period = _apply("drop_month")
    assert period == PERIOD
    gone = months(before) - months(after)
    assert len(gone) == 1
    [(y, m)] = list(gone)
    kept = before.filter(~((pl.col("posted").dt.year() == y) & (pl.col("posted").dt.month() == m)))
    assert after["narration"].to_list() == kept["narration"].to_list()
    assert after["row"].to_list() == list(range(after.height))  # renumbered, in order


def test_cut_3_months_keeps_the_first_three() -> None:
    after, period = _apply("cut_3_months")
    assert period == (date(2025, 1, 1), date(2025, 3, 31))
    assert max(after["posted"]) <= date(2025, 3, 31)
    assert after.height == _rows().filter(pl.col("posted") <= date(2025, 3, 31)).height


def test_text_perturbations_change_only_the_narration() -> None:
    before = _rows()
    for name in ("typos", "lower_case", "cut_20_chars"):
        after, period = _apply(name)
        assert period == PERIOD
        assert after.drop("narration").equals(before.drop("narration")), name
    lower, _ = _apply("lower_case")
    assert lower["narration"].to_list() == [t.lower() for t in before["narration"]]
    cut, _ = _apply("cut_20_chars")
    assert cut["narration"].to_list() == [t[:20] for t in before["narration"]]


def test_typos_touch_one_name_letter_and_never_a_format_word() -> None:
    before, (after, _) = _rows(), _apply("typos")
    changed = [(a, b) for a, b in zip(before["narration"], after["narration"], strict=True)
               if a != b]  # fmt: skip
    assert changed  # 30% of 20 narrations, so some change
    for a, b in changed:
        assert abs(len(a) - len(b)) <= 1
        assert "NEFT CR-KSVB0001234-" in b and "SALARY" in b  # codes and format words kept
        assert b.split("-")[-1] == a.split("-")[-1]  # the reference is never touched
    rng = stress.rng_for("typos", "stmt_0001")
    plain = "NEFT KSVB0001234 SALARY sharma12@ksvb"  # no name-like word: nothing to misspell
    assert stress.misspell_a_name(plain, rng) == plain


def test_shift_dates_moves_rows_by_at_most_three_days_inside_the_period() -> None:
    before, (after, period) = _rows(), _apply("shift_dates")
    assert after.drop("posted").equals(before.drop("posted")) and period == PERIOD
    for a, b in zip(before["posted"], after["posted"], strict=True):
        assert abs((b - a).days) <= 3 and PERIOD[0] <= b <= PERIOD[1]


def test_scale_amounts_multiplies_by_one_point_one() -> None:
    before, (after, _) = _rows(), _apply("scale_amounts")
    assert after.drop("amount_paise").equals(before.drop("amount_paise"))
    for a, b in zip(before["amount_paise"], after["amount_paise"], strict=True):
        # Exact arithmetic: a * 1.1 in floating point is off by ~1e-11, which matters
        # on a tie such as -115015 * 1.1 = -126516.5 exactly.
        assert abs(b - Fraction(a * 11, 10)) <= Fraction(1, 2)  # nearest paisa


# test_splits.py checks the gate itself (closed for "0" and "yes", open for "1") on a toy
# 120-statement split built in a temp folder; it never opens the real split. Kept as it is
# (Week 5 decision); every other test file must leave the variable alone.
GATE_TESTS = {"test_splits.py"}


def test_no_test_opens_the_test_split() -> None:
    assert os.environ.get(OPEN_TEST_ENV) != "1"
    opening = re.compile(r"setenv\([^)]*OPEN_TEST|environ\[[^\]]*OPEN_TEST[^\]]*\]\s*=")
    for path in Path(__file__).parent.glob("test_*.py"):
        if path.name not in GATE_TESTS:
            assert not opening.search(path.read_text(encoding="utf-8")), path.name
