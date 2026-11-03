"""Generate a multi-month, fully synthetic bank statement.

Everything is driven by a single integer ``seed``: the same seed always gives the
same statement, which keeps tests and experiments reproducible.

How a statement is built:
1. Pick who the customer is (salaried / gig / self-employed) and how they earn.
2. Lay down their income for the *whole* period. Pay cycles (biweekly, weekly) run
   continuously across months, so some months get an extra paycheck, as in real life.
3. Inject any requested income hard cases (see ``hard_cases.py``).
4. For each month, add bills (same rent every month), an optional loan instalment,
   card spending scaled to income, and occasional events (refunds, transfers, ATM).
5. Inject ``emi_bounce`` if requested, sort by date and compute the running balance.

Hard cases use their *own* random generator, so the same seed with and without hard
cases describes the same person: only the injected lines differ.
"""

from collections.abc import Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from faker import Faker

from statementscreen.synth import hard_cases as hc
from statementscreen.synth.banks import FICTIONAL_BANKS
from statementscreen.synth.descriptions import CITIES, MERCHANTS, DescriptionStyle, random_style
from statementscreen.synth.models import Draft, IncomeTruth, Statement, Transaction
from statementscreen.synth.util import CENT, month_end
from statementscreen.synth.util import money as _money

CUSTOMER_TYPES = ("salaried", "gig", "self_employed")

# Categories that count as regular income in the ground truth.
INCOME_CATEGORIES = frozenset({"salary", "gig_income", "business_income"})

# Invented names (not real companies).
GIG_PLATFORMS = ("SWIFTDROP DELIVERY", "RIDEWAVE", "TASKHIVE", "QUICKCART COURIER")
LENDERS = ("BRIGHTPATH AUTO FINANCE", "SUMMIT RIDGE LENDING", "NORTHSTAR CONSUMER LOANS")

# Category -> (typical amount, spread) for everyday card spending.
_SPENDING = {
    "groceries": (65.0, 25.0),
    "dining": (12.0, 5.0),
    "transport": (45.0, 12.0),
    "shopping": (40.0, 30.0),
    "health": (20.0, 10.0),
}
_MEAN_PURCHASE = sum(mean for mean, _ in _SPENDING.values()) / len(_SPENDING)

P_LOAN = 0.40  # share of customers repaying a loan in monthly instalments (EMIs)

# Probabilities of optional, less common events in any given month.
P_SIDE_TRANSFER = 0.35  # money from a friend/family: NOT income, a classic verification trap
P_REFUND = 0.25
P_ATM = 0.50


def _month_starts(start: date, months: int) -> list[date]:
    """First day of each month in the statement, e.g. Jan 1, Feb 1, Mar 1."""
    out, y, m = [], start.year, start.month
    for _ in range(months):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _business_day(d: date, period_start: date) -> date:
    """Move a weekend date to the previous Friday (or next Monday if that leaves the period)."""
    if d.weekday() < 5:
        return d
    friday = d - timedelta(days=d.weekday() - 4)
    return friday if friday >= period_start else d + timedelta(days=7 - d.weekday())


def _random_day(rng: np.random.Generator, first: date, lo: int = 0, hi: int | None = None) -> date:
    """A random date in the month starting at ``first``, between day lo and day hi-1 (0-based)."""
    hi = hi if hi is not None else month_end(first).day
    return first + timedelta(days=int(rng.integers(lo, hi)))


def _regular_monthly_income(drafts: list[Draft], months: int) -> Decimal:
    total = sum((d.amount for d in drafts if d.category in INCOME_CATEGORIES), Decimal(0))
    return (total / months).quantize(CENT, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------- income, by customer type


def _salaried_income(rng, fake, style, months, start, end) -> tuple[list[Draft], IncomeTruth]:
    employer = fake.company().upper()
    frequency = str(rng.choice(["monthly", "biweekly"]))
    net_pay = _money(rng.uniform(2500, 7000) if frequency == "monthly" else rng.uniform(1200, 3500))

    if frequency == "monthly":
        days_before_end = int(rng.integers(0, 3))  # same payday every month, e.g. the 29th
        dates = [month_end(m) - timedelta(days=days_before_end) for m in months]
    else:
        d, dates = start + timedelta(days=int(rng.integers(0, 14))), []
        while d <= end:
            dates.append(d)
            d += timedelta(days=14)

    drafts = [
        Draft(_business_day(d, start), style.payroll(rng, employer), net_pay, "salary")
        for d in dates
    ]
    return drafts, IncomeTruth("salaried", (employer,), frequency, net_pay, Decimal(0))


def _gig_income(rng, fake, style, months, start, end) -> tuple[list[Draft], IncomeTruth]:
    n_platforms = int(rng.integers(1, 4))
    platforms = [str(p) for p in rng.choice(GIG_PLATFORMS, size=n_platforms, replace=False)]
    monthly_target = rng.uniform(1800, 4500)
    shares = rng.dirichlet(np.ones(n_platforms))  # how the income splits across platforms

    drafts: list[Draft] = []
    for platform, share in zip(platforms, shares, strict=True):
        weekly_mean = monthly_target * share * 12 / 52
        payout_weekday = int(rng.integers(0, 5))  # each platform pays on its own weekday
        d = start + timedelta(days=(payout_weekday - start.weekday()) % 7)
        while d <= end:
            if rng.random() > 0.10:  # ~1 week in 10 they don't work that platform
                amount = _money(weekly_mean * rng.lognormal(0, 0.3))
                drafts.append(Draft(d, style.platform_payout(rng, platform), amount, "gig_income"))
            d += timedelta(days=7)
    return drafts, IncomeTruth("gig", tuple(platforms), "weekly", None, Decimal(0))


def _self_employed_income(rng, fake, style, months, start, end) -> tuple[list[Draft], IncomeTruth]:
    clients = [fake.company().upper() for _ in range(int(rng.integers(2, 6)))]
    monthly_target = rng.uniform(3000, 9000)

    drafts: list[Draft] = []
    for first in months:
        n_payments = int(rng.integers(1, 5))
        month_total = monthly_target * rng.lognormal(0, 0.35)  # some months are much better
        for share in rng.dirichlet(np.ones(n_payments)):
            d = _business_day(_random_day(rng, first), start)
            client = str(rng.choice(clients))
            amount = _money(month_total * share)
            drafts.append(Draft(d, style.client_payment(rng, client), amount, "business_income"))
    return drafts, IncomeTruth("self_employed", tuple(clients), "irregular", None, Decimal(0))


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
) -> list[Draft]:
    drafts: list[Draft] = []

    def add(posted: date, description: str, amount: Decimal, category: str) -> None:
        drafts.append(Draft(posted, description, amount, category))

    # Fixed bills: the same rent to the same landlord every month.
    add(_random_day(rng, first, 0, 3), style.bill(rng, "OAKRIDGE PROPERTY MGMT"), -rent, "rent")
    utilities = -_money(rng.uniform(60, 180))
    add(_random_day(rng, first, 5, 20), style.bill(rng, "CITY UTILITIES"), utilities, "utilities")
    add(
        _random_day(rng, first, 14, 16),
        style.bill(rng, "STREAMFLIX"),
        -_money(15.99),
        "subscription",
    )

    # Self-employed people pay estimated tax themselves every quarter.
    if customer_type == "self_employed" and month_index % 3 == 2:
        tax = -_money(monthly_income * 3 * rng.uniform(0.08, 0.12))
        add(_random_day(rng, first, 10, 16), style.bill(rng, "ESTIMATED TAX PMT"), tax, "tax")

    # Card spending: a budget that scales with income, split into Poisson-many purchases.
    budget = monthly_income * spend_share * rng.uniform(0.85, 1.15)
    categories = list(_SPENDING)
    purchases: list[str] = []
    for _ in range(int(rng.poisson(budget / _MEAN_PURCHASE))):
        cat = categories[int(rng.integers(len(categories)))]
        mean, sd = _SPENDING[cat]
        merchant = str(rng.choice(MERCHANTS[cat]))
        d = _random_day(rng, first)
        add(d, style.card(rng, merchant, city, d), -_money(max(1.0, rng.normal(mean, sd))), cat)
        purchases.append(merchant)

    # Occasional events.
    if purchases and rng.random() < P_REFUND:
        merchant = str(rng.choice(purchases))
        d = _random_day(rng, first, 5)
        add(d, style.refund(rng, merchant, d), _money(rng.uniform(5, 80)), "refund")

    if rng.random() < P_SIDE_TRANSFER:
        sender = fake.name().upper()
        gift = _money(rng.uniform(20, 600))
        add(_random_day(rng, first), style.transfer_in(rng, sender), gift, "transfer_in")

    if rng.random() < P_ATM:
        cash = -_money(int(rng.choice([20, 40, 60, 100, 200])))
        add(_random_day(rng, first), style.atm(rng, city), cash, "cash")

    return drafts


# ---------------------------------------------------------------- public API


def generate_statement(
    seed: int,
    customer_type: str = "salaried",
    months: int = 6,
    start: date = date(2025, 1, 1),
    hard_cases: Sequence[str] = (),
) -> Statement:
    """Build one statement covering ``months`` whole months, beginning with ``start``'s month.

    ``hard_cases`` lists which tricky situations to inject (see ``hard_cases.py``).
    """
    if customer_type not in _INCOME_BUILDERS:
        raise ValueError(f"customer_type must be one of {CUSTOMER_TYPES}, got {customer_type!r}")
    cases = tuple(c for c in hc.HARD_CASES if c in hard_cases)  # canonical order, no duplicates
    hc.validate(tuple(hard_cases), customer_type)

    rng = np.random.default_rng(seed)
    hc_rng = np.random.default_rng([seed, 1])  # separate stream, so the base person is unchanged
    fake = Faker()
    fake.seed_instance(seed)
    style = random_style(rng)

    month_firsts = _month_starts(start.replace(day=1), months)
    period_start, period_end = month_firsts[0], month_end(month_firsts[-1])
    city = str(rng.choice(CITIES))

    # 1. Income for the whole period. The person's lifestyle (rent, spending) is set by
    #    their *usual* income, measured before any hard cases are injected.
    build_income = _INCOME_BUILDERS[customer_type]
    drafts, income = build_income(rng, fake, style, month_firsts, period_start, period_end)
    usual_income = float(_regular_monthly_income(drafts, months))

    # 2. Income hard cases. Order matters: rename runs last so it also renames the bonus.
    details: dict[str, str] = {}
    sources = income.sources
    if "missed_salary" in cases:
        drafts = hc.missed_salary(drafts, hc_rng, details)
    if "salary_date_shift" in cases:
        drafts = hc.salary_date_shift(drafts, hc_rng, period_end, details)
    if "bonus_month" in cases:
        drafts = hc.bonus_month(drafts, hc_rng, style, income, month_firsts, details)
    if "employer_rename" in cases:
        drafts, new_name = hc.employer_rename(drafts, hc_rng, style, income.employer, details)
        sources = (*sources, new_name)

    income = IncomeTruth(
        customer_type=income.customer_type,
        sources=sources,
        pay_frequency=income.pay_frequency,
        net_pay=income.net_pay,
        monthly_income=_regular_monthly_income(drafts, months),  # what actually arrived
    )

    # 3. Spending, month by month. Rent, loan and spending habits are fixed per person.
    rent = _money(usual_income * rng.uniform(0.25, 0.40))
    spend_share = rng.uniform(0.20, 0.40)
    has_loan = rng.random() < P_LOAN or "emi_bounce" in cases
    lender = str(rng.choice(LENDERS))
    emi = _money(usual_income * rng.uniform(0.05, 0.12))
    emi_day = int(rng.integers(3, 10))
    for i, first in enumerate(month_firsts):
        drafts += _month_spending(
            rng, fake, style, first, i, customer_type, usual_income, rent, spend_share, city
        )
        if has_loan:
            posted = _business_day(first + timedelta(days=emi_day), period_start)
            drafts.append(Draft(posted, style.bill(rng, f"{lender} LOAN"), -emi, "loan_payment"))

    if "emi_bounce" in cases:
        drafts = hc.emi_bounce(drafts, hc_rng, style, lender, period_end, details)

    # 4. Sort by date (stable, so same-day order is deterministic) and compute running balance.
    drafts.sort(key=lambda d: d.posted)
    opening = _money(usual_income * rng.uniform(0.6, 1.3))
    balance = opening
    transactions = []
    for d in drafts:
        balance += d.amount
        transactions.append(
            Transaction(d.posted, d.description, d.amount, balance, d.category, d.hard_cases)
        )

    return Statement(
        bank_name=str(rng.choice(FICTIONAL_BANKS)),
        account_holder=fake.name(),
        account_number=f"****{int(rng.integers(1000, 10000))}",
        period_start=period_start,
        period_end=period_end,
        opening_balance=opening,
        income=income,
        transactions=transactions,
        hard_cases=cases,
        hard_case_details=details,
    )
