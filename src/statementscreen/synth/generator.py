"""Generate a multi-month, fully synthetic bank statement.

Everything is driven by a single integer ``seed``: the same seed always gives the
same statement, which keeps tests and experiments reproducible.

How a statement is built:
1. Pick who the customer is (salaried / gig / self-employed) and how they earn.
2. Lay down their income for the *whole* period. Pay cycles (biweekly, weekly) run
   continuously across months, so some months get an extra paycheck, as in real life.
3. For each month, add bills (same rent every month), card spending scaled to
   income, and occasional events (refunds, transfers, ATM withdrawals).
4. Sort everything by date and compute the running balance.
"""

import calendar
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from faker import Faker

from statementscreen.synth.banks import FICTIONAL_BANKS
from statementscreen.synth.descriptions import CITIES, MERCHANTS, DescriptionStyle, random_style
from statementscreen.synth.models import IncomeTruth, Statement, Transaction

CENT = Decimal("0.01")

CUSTOMER_TYPES = ("salaried", "gig", "self_employed")

# Categories that count as regular income in the ground truth.
INCOME_CATEGORIES = frozenset({"salary", "gig_income", "business_income"})

# Invented gig platforms (not real companies).
GIG_PLATFORMS = ("SWIFTDROP DELIVERY", "RIDEWAVE", "TASKHIVE", "QUICKCART COURIER")

# Category -> (typical amount, spread) for everyday card spending.
_SPENDING = {
    "groceries": (65.0, 25.0),
    "dining": (12.0, 5.0),
    "transport": (45.0, 12.0),
    "shopping": (40.0, 30.0),
    "health": (20.0, 10.0),
}
_MEAN_PURCHASE = sum(mean for mean, _ in _SPENDING.values()) / len(_SPENDING)

# Probabilities of optional, less common events in any given month.
P_SIDE_TRANSFER = 0.35  # money from a friend/family: NOT income, a classic verification trap
P_REFUND = 0.25
P_ATM = 0.50

# (date, description, amount, category) before balances are computed.
Raw = tuple[date, str, Decimal, str]


def _money(x: float) -> Decimal:
    """Convert a float to a Decimal rounded to the cent."""
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)


def _month_starts(start: date, months: int) -> list[date]:
    """First day of each month in the statement, e.g. Jan 1, Feb 1, Mar 1."""
    out, y, m = [], start.year, start.month
    for _ in range(months):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _month_end(first: date) -> date:
    return date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])


def _business_day(d: date, period_start: date) -> date:
    """Move a weekend date to the previous Friday (or next Monday if that leaves the period)."""
    if d.weekday() < 5:
        return d
    friday = d - timedelta(days=d.weekday() - 4)
    return friday if friday >= period_start else d + timedelta(days=7 - d.weekday())


def _random_day(rng: np.random.Generator, first: date, lo: int = 0, hi: int | None = None) -> date:
    """A random date in the month starting at ``first``, between day lo and day hi-1 (0-based)."""
    hi = hi if hi is not None else _month_end(first).day
    return first + timedelta(days=int(rng.integers(lo, hi)))


# ---------------------------------------------------------------- income, by customer type


def _salaried_income(rng, fake, style, months, start, end) -> tuple[list[Raw], IncomeTruth]:
    employer = fake.company().upper()
    frequency = str(rng.choice(["monthly", "biweekly"]))
    net_pay = _money(rng.uniform(2500, 7000) if frequency == "monthly" else rng.uniform(1200, 3500))

    if frequency == "monthly":
        days_before_end = int(rng.integers(0, 3))  # same payday every month, e.g. the 29th
        dates = [_month_end(m) - timedelta(days=days_before_end) for m in months]
    else:
        d, dates = start + timedelta(days=int(rng.integers(0, 14))), []
        while d <= end:
            dates.append(d)
            d += timedelta(days=14)

    raw = [
        (_business_day(d, start), style.payroll(rng, employer), net_pay, "salary") for d in dates
    ]
    return raw, IncomeTruth("salaried", (employer,), frequency, net_pay, Decimal(0))


def _gig_income(rng, fake, style, months, start, end) -> tuple[list[Raw], IncomeTruth]:
    n_platforms = int(rng.integers(1, 4))
    platforms = [str(p) for p in rng.choice(GIG_PLATFORMS, size=n_platforms, replace=False)]
    monthly_target = rng.uniform(1800, 4500)
    shares = rng.dirichlet(np.ones(n_platforms))  # how the income splits across platforms

    raw: list[Raw] = []
    for platform, share in zip(platforms, shares, strict=True):
        weekly_mean = monthly_target * share * 12 / 52
        payout_weekday = int(rng.integers(0, 5))  # each platform pays on its own weekday
        d = start + timedelta(days=(payout_weekday - start.weekday()) % 7)
        while d <= end:
            if rng.random() > 0.10:  # ~1 week in 10 they don't work that platform
                amount = _money(weekly_mean * rng.lognormal(0, 0.3))
                raw.append((d, style.platform_payout(rng, platform), amount, "gig_income"))
            d += timedelta(days=7)
    return raw, IncomeTruth("gig", tuple(platforms), "weekly", None, Decimal(0))


def _self_employed_income(rng, fake, style, months, start, end) -> tuple[list[Raw], IncomeTruth]:
    clients = [fake.company().upper() for _ in range(int(rng.integers(2, 6)))]
    monthly_target = rng.uniform(3000, 9000)

    raw: list[Raw] = []
    for first in months:
        n_payments = int(rng.integers(1, 5))
        month_total = monthly_target * rng.lognormal(0, 0.35)  # some months are much better
        for share in rng.dirichlet(np.ones(n_payments)):
            d = _business_day(_random_day(rng, first), start)
            client = str(rng.choice(clients))
            amount = _money(month_total * share)
            raw.append((d, style.client_payment(rng, client), amount, "business_income"))
    return raw, IncomeTruth("self_employed", tuple(clients), "irregular", None, Decimal(0))


_INCOME_BUILDERS = {
    "salaried": _salaried_income,
    "gig": _gig_income,
    "self_employed": _self_employed_income,
}


# ---------------------------------------------------------------- spending, per month


def _month_spending(
    rng: np.random.Generator,
    fake: Faker,
    style: DescriptionStyle,
    first: date,
    month_index: int,
    customer_type: str,
    monthly_income: float,
    rent: Decimal,
    spend_share: float,
    city: str,
) -> list[Raw]:
    raw: list[Raw] = []

    # Fixed bills: the same rent to the same landlord every month.
    landlord = style.bill(rng, "OAKRIDGE PROPERTY MGMT")
    raw.append((_random_day(rng, first, 0, 3), landlord, -rent, "rent"))
    utilities = -_money(rng.uniform(60, 180))
    raw.append(
        (_random_day(rng, first, 5, 20), style.bill(rng, "CITY UTILITIES"), utilities, "utilities")
    )
    raw.append(
        (
            _random_day(rng, first, 14, 16),
            style.bill(rng, "STREAMFLIX"),
            -_money(15.99),
            "subscription",
        )
    )

    # Self-employed people pay estimated tax themselves every quarter.
    if customer_type == "self_employed" and month_index % 3 == 2:
        tax = -_money(monthly_income * 3 * rng.uniform(0.08, 0.12))
        raw.append(
            (_random_day(rng, first, 10, 16), style.bill(rng, "ESTIMATED TAX PMT"), tax, "tax")
        )

    # Card spending: a budget that scales with income, split into Poisson-many purchases.
    budget = monthly_income * spend_share * rng.uniform(0.85, 1.15)
    categories = list(_SPENDING)
    purchases: list[str] = []
    for _ in range(int(rng.poisson(budget / _MEAN_PURCHASE))):
        cat = categories[int(rng.integers(len(categories)))]
        mean, sd = _SPENDING[cat]
        merchant = str(rng.choice(MERCHANTS[cat]))
        d = _random_day(rng, first)
        amount = -_money(max(1.0, rng.normal(mean, sd)))
        raw.append((d, style.card(rng, merchant, city, d), amount, cat))
        purchases.append(merchant)

    # Occasional events.
    if purchases and rng.random() < P_REFUND:
        merchant = str(rng.choice(purchases))
        d = _random_day(rng, first, 5)
        raw.append((d, style.refund(rng, merchant, d), _money(rng.uniform(5, 80)), "refund"))

    if rng.random() < P_SIDE_TRANSFER:
        sender = fake.name().upper()
        gift = _money(rng.uniform(20, 600))
        raw.append((_random_day(rng, first), style.transfer_in(rng, sender), gift, "transfer_in"))

    if rng.random() < P_ATM:
        cash = -_money(int(rng.choice([20, 40, 60, 100, 200])))
        raw.append((_random_day(rng, first), style.atm(rng, city), cash, "cash"))

    return raw


# ---------------------------------------------------------------- public API


def generate_statement(
    seed: int,
    customer_type: str = "salaried",
    months: int = 6,
    start: date = date(2025, 1, 1),
) -> Statement:
    """Build one statement covering ``months`` whole months, beginning with ``start``'s month."""
    if customer_type not in _INCOME_BUILDERS:
        raise ValueError(f"customer_type must be one of {CUSTOMER_TYPES}, got {customer_type!r}")

    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)
    style = random_style(rng)

    month_firsts = _month_starts(start.replace(day=1), months)
    period_start, period_end = month_firsts[0], _month_end(month_firsts[-1])
    city = str(rng.choice(CITIES))

    # 1. Income for the whole period, then the true average per month.
    build_income = _INCOME_BUILDERS[customer_type]
    raw, income = build_income(rng, fake, style, month_firsts, period_start, period_end)
    income_total = sum((amt for _, _, amt, cat in raw if cat in INCOME_CATEGORIES), Decimal(0))
    monthly_income = (income_total / months).quantize(CENT, rounding=ROUND_HALF_UP)
    income = IncomeTruth(
        income.customer_type, income.sources, income.pay_frequency, income.net_pay, monthly_income
    )

    # 2. Spending, month by month. Rent and spending habits are fixed per person.
    rent = _money(float(monthly_income) * rng.uniform(0.25, 0.40))
    spend_share = rng.uniform(0.20, 0.40)
    for i, first in enumerate(month_firsts):
        raw += _month_spending(
            rng,
            fake,
            style,
            first,
            i,
            customer_type,
            float(monthly_income),
            rent,
            spend_share,
            city,
        )

    # 3. Sort by date (stable, so same-day order is deterministic) and compute running balance.
    raw.sort(key=lambda r: r[0])
    opening = _money(float(monthly_income) * rng.uniform(0.6, 1.3))
    balance = opening
    transactions = []
    for d, desc, amt, cat in raw:
        balance += amt
        transactions.append(Transaction(d, desc, amt, balance, cat))

    return Statement(
        bank_name=str(rng.choice(FICTIONAL_BANKS)),
        account_holder=fake.name(),
        account_number=f"****{int(rng.integers(1000, 10000))}",
        period_start=period_start,
        period_end=period_end,
        opening_balance=opening,
        income=income,
        transactions=transactions,
    )
