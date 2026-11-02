"""Generate one month of realistic-looking, fully synthetic bank transactions.

Everything is driven by a single integer ``seed``: the same seed always gives the
same statement, which keeps tests and experiments reproducible.
"""

import calendar
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from faker import Faker

from statementscreen.synth.banks import FICTIONAL_BANKS
from statementscreen.synth.models import IncomeTruth, Statement, Transaction

CENT = Decimal("0.01")

# (description, category, typical amount, spread) for everyday card spending.
_SPENDING = [
    ("GROCERY MART", "groceries", 65.0, 25.0),
    ("CORNER CAFE", "dining", 12.0, 5.0),
    ("FUEL STOP", "transport", 45.0, 12.0),
    ("ONLINE SHOP", "shopping", 40.0, 30.0),
    ("PHARMACY PLUS", "health", 20.0, 10.0),
]


def _money(x: float) -> Decimal:
    """Convert a float to a Decimal rounded to the cent."""
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)


def generate_statement(seed: int, year: int = 2026, month: int = 1) -> Statement:
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])

    pay_frequency = str(rng.choice(["monthly", "biweekly"]))
    net_pay = _money(
        rng.uniform(2500, 7000) if pay_frequency == "monthly" else rng.uniform(1200, 3500)
    )
    income = IncomeTruth(
        employer=fake.company().upper(), pay_frequency=pay_frequency, net_pay=net_pay
    )

    # Build the un-ordered list of (date, description, amount, category).
    raw: list[tuple[date, str, Decimal, str]] = []

    # Paychecks
    if pay_frequency == "monthly":
        pay_dates = [end - timedelta(days=int(rng.integers(0, 3)))]
    else:
        first = start + timedelta(days=int(rng.integers(0, 14)))
        pay_dates = [
            d for d in (first, first + timedelta(days=14), first + timedelta(days=28)) if d <= end
        ]
    for d in pay_dates:
        raw.append((d, f"PAYROLL {income.employer}", net_pay, "salary"))

    # Fixed monthly bills
    rent = _money(rng.uniform(900, 2200))
    raw.append((start + timedelta(days=int(rng.integers(0, 4))), "RENT PAYMENT", -rent, "rent"))
    raw.append(
        (
            start + timedelta(days=int(rng.integers(5, 20))),
            "CITY UTILITIES",
            -_money(rng.uniform(60, 180)),
            "utilities",
        )
    )
    raw.append(
        (
            start + timedelta(days=int(rng.integers(5, 25))),
            "STREAMFLIX SUBSCRIPTION",
            -_money(15.99),
            "subscription",
        )
    )

    # Everyday card spending: a Poisson-distributed number of purchases
    n_purchases = int(rng.poisson(25))
    days_in_month = (end - start).days + 1
    for _ in range(n_purchases):
        desc, cat, mean, sd = _SPENDING[int(rng.integers(len(_SPENDING)))]
        amount = max(1.0, rng.normal(mean, sd))
        d = start + timedelta(days=int(rng.integers(0, days_in_month)))
        raw.append((d, f"CARD PURCHASE {desc}", -_money(amount), cat))

    # Sort by date (stable, so same-day order is deterministic) and compute running balance.
    raw.sort(key=lambda r: r[0])
    opening = _money(rng.uniform(500, 5000))
    balance = opening
    transactions = []
    for d, desc, amt, cat in raw:
        balance += amt
        transactions.append(Transaction(d, desc, amt, balance, cat))

    return Statement(
        bank_name=str(rng.choice(FICTIONAL_BANKS)),
        account_holder=fake.name(),
        account_number=f"****{int(rng.integers(1000, 10000))}",
        period_start=start,
        period_end=end,
        opening_balance=opening,
        income=income,
        transactions=transactions,
    )
