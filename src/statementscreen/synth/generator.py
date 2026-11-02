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
from statementscreen.synth.descriptions import CITIES, MERCHANTS, random_style
from statementscreen.synth.models import IncomeTruth, Statement, Transaction

CENT = Decimal("0.01")

# Category -> (typical amount, spread) for everyday card spending.
_SPENDING = {
    "groceries": (65.0, 25.0),
    "dining": (12.0, 5.0),
    "transport": (45.0, 12.0),
    "shopping": (40.0, 30.0),
    "health": (20.0, 10.0),
}

# Probabilities of optional, less common events in a month.
P_SIDE_TRANSFER = 0.35  # money from a friend/family: NOT salary, a classic income-verification trap
P_BONUS = 0.10
P_REFUND = 0.25
P_ATM = 0.50


def _money(x: float) -> Decimal:
    """Convert a float to a Decimal rounded to the cent."""
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)


def _business_day(d: date, start: date) -> date:
    """Move a weekend payday to the previous Friday (or next Monday if that leaves the month)."""
    if d.weekday() < 5:
        return d
    friday = d - timedelta(days=d.weekday() - 4)
    return friday if friday >= start else d + timedelta(days=7 - d.weekday())


def generate_statement(seed: int, year: int = 2026, month: int = 1) -> Statement:
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)
    style = random_style(rng)

    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    days_in_month = (end - start).days + 1
    home_city = str(rng.choice(CITIES))

    def day_in_range(lo: int, hi: int) -> date:
        """A random date between start+lo and start+hi-1 days."""
        return start + timedelta(days=int(rng.integers(lo, hi)))

    pay_frequency = str(rng.choice(["monthly", "biweekly"]))
    net_pay = _money(
        rng.uniform(2500, 7000) if pay_frequency == "monthly" else rng.uniform(1200, 3500)
    )
    income = IncomeTruth(
        employer=fake.company().upper(), pay_frequency=pay_frequency, net_pay=net_pay
    )

    # Un-ordered list of (date, description, amount, category).
    raw: list[tuple[date, str, Decimal, str]] = []

    # --- Paychecks (weekend paydays shift to a business day, like real payroll) ---
    if pay_frequency == "monthly":
        pay_dates = [end - timedelta(days=int(rng.integers(0, 3)))]
    else:
        first = day_in_range(0, 14)
        pay_dates = [first + timedelta(days=14 * k) for k in range(3)]
    pay_dates = [_business_day(d, start) for d in pay_dates]
    for d in pay_dates:
        if d <= end:
            raw.append((d, style.payroll(rng, income.employer), net_pay, "salary"))

    if rng.random() < P_BONUS:
        bonus = _money(rng.uniform(300, 2000))
        raw.append(
            (day_in_range(0, days_in_month), style.payroll(rng, income.employer), bonus, "bonus")
        )

    # --- Fixed monthly bills ---
    raw.append(
        (
            day_in_range(0, 4),
            style.bill(rng, "OAKRIDGE PROPERTY MGMT"),
            -_money(rng.uniform(900, 2200)),
            "rent",
        )
    )
    raw.append(
        (
            day_in_range(5, 20),
            style.bill(rng, "CITY UTILITIES"),
            -_money(rng.uniform(60, 180)),
            "utilities",
        )
    )
    raw.append((day_in_range(5, 25), style.bill(rng, "STREAMFLIX"), -_money(15.99), "subscription"))

    # --- Everyday card spending: a Poisson-distributed number of purchases ---
    categories = list(_SPENDING)
    purchases: list[str] = []
    for _ in range(int(rng.poisson(25))):
        cat = categories[int(rng.integers(len(categories)))]
        mean, sd = _SPENDING[cat]
        merchant = str(rng.choice(MERCHANTS[cat]))
        d = day_in_range(0, days_in_month)
        raw.append(
            (
                d,
                style.card(rng, merchant, home_city, d),
                -_money(max(1.0, rng.normal(mean, sd))),
                cat,
            )
        )
        purchases.append(merchant)

    # --- Occasional events ---
    if purchases and rng.random() < P_REFUND:
        merchant = str(rng.choice(purchases))
        d = day_in_range(5, days_in_month)
        raw.append((d, style.refund(rng, merchant, d), _money(rng.uniform(5, 80)), "refund"))

    if rng.random() < P_SIDE_TRANSFER:
        sender = fake.name().upper()
        raw.append(
            (
                day_in_range(0, days_in_month),
                style.transfer_in(rng, sender),
                _money(rng.uniform(20, 600)),
                "transfer_in",
            )
        )

    if rng.random() < P_ATM:
        amount = Decimal(int(rng.choice([20, 40, 60, 100, 200])))
        raw.append((day_in_range(0, days_in_month), style.atm(rng, home_city), -amount, "cash"))

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
