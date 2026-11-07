"""Small money and date helpers shared by the generator and the hard-case injectors."""

import calendar
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(x: float) -> Decimal:
    """Convert a float to a Decimal rounded to the paisa (0.01)."""
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)


def rupees(x: float) -> Decimal:
    """Round to a whole rupee, as salaries, rents and plans usually are."""
    return Decimal(round(x)).quantize(CENT)


def month_end(first: date) -> date:
    return date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])


def is_bank_holiday(d: date) -> bool:
    """Indian banks close on Sundays and on the 2nd and 4th Saturday of each month.

    Public holidays vary by state and year, so they're not modelled.
    """
    if d.weekday() == 6:
        return True
    return d.weekday() == 5 and (d.day - 1) // 7 + 1 in (2, 4)


def next_working_day(d: date) -> date:
    while is_bank_holiday(d):
        d += timedelta(days=1)
    return d


def prev_working_day(d: date) -> date:
    while is_bank_holiday(d):
        d -= timedelta(days=1)
    return d


def working_day_within(d: date, start: date, end: date) -> date:
    """The nearest working day to d that stays inside [start, end] (forward first)."""
    forward = next_working_day(d)
    if forward <= end:
        return forward
    back = prev_working_day(d)
    return back if back >= start else d
