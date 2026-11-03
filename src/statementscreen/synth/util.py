"""Small money and date helpers shared by the generator and the hard-case injectors."""

import calendar
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(x: float) -> Decimal:
    """Convert a float to a Decimal rounded to the cent."""
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)


def month_end(first: date) -> date:
    return date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])


def next_business_day(d: date) -> date:
    """d itself if it's a weekday, otherwise the following Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d
