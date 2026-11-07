from datetime import date
from decimal import Decimal

import pytest

from statementscreen.parsing.values import (
    parse_amount,
    parse_dmy,
    parse_dmy_short,
    to_paise,
)


@pytest.mark.parametrize(
    ("text", "paise"),
    [("1,25,000.00", 12_500_000), ("4,50,985.60", 45_098_560), ("0.75", 75),
     ("25,453.00", 2_545_300), ("1,61,030.55", 16_103_055), ("999.50", 99_950),
     ("12,34,567.80", 123_456_780), ("-2,50,00,000.00", -2_500_000_000)],
)  # fmt: skip
def test_indian_amounts_parse_to_exact_paise(text: str, paise: int) -> None:
    amount = parse_amount(text)
    assert isinstance(amount, Decimal)  # never a float
    assert to_paise(amount) == paise
    assert isinstance(to_paise(amount), int)


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_cell_gives_none(blank) -> None:
    assert parse_amount(blank) is None


@pytest.mark.parametrize(
    "bad",
    ["1,254", "1254.5", "12,5000.00", "1,23,4567.00", "1,234.567", "abc", "1 254.00",
     "125,000.00", "1.254,00"],
)  # fmt: skip
def test_non_indian_amounts_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_amount(bad)


def test_to_paise_refuses_fractions_of_a_paisa() -> None:
    with pytest.raises(ValueError):
        to_paise(Decimal("1.005"))


def test_dmy_dates() -> None:
    assert parse_dmy("01/01/2025") == date(2025, 1, 1)
    assert parse_dmy("29/02/2024") == date(2024, 2, 29)


@pytest.mark.parametrize("bad", ["31/02/2025", "1/1/2025", "01-01-2025", "2025/01/01",
                                 "01/13/2025", "01/01/25", "", "01/01/2025x"])  # fmt: skip
def test_malformed_dmy_dates_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_dmy(bad)


def test_short_value_date_takes_the_century_nearest_the_posting_date() -> None:
    assert parse_dmy_short("01/01/25", near=date(2025, 1, 1)) == date(2025, 1, 1)
    assert parse_dmy_short("31/12/99", near=date(2000, 1, 2)) == date(1999, 12, 31)
    with pytest.raises(ValueError):
        parse_dmy_short("01-01-25", near=date(2025, 1, 1))
