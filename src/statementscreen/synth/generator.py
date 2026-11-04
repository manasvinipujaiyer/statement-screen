"""Generate a multi-month, fully synthetic Indian savings-account statement (INR).

Everything is driven by a single integer ``seed``: the same seed always gives the
same statement, which keeps tests and experiments reproducible.

How a statement is built:
1. Pick who the customer is (salaried / gig / self-employed) and how they earn.
2. Lay down their income for the whole period, on bank working days.
3. Inject any requested income hard cases (see ``hard_cases.py``).
4. For each month: rent, EMI (NACH), bills (BBPS), subscriptions, UPI spends scaled to
   income, ATM withdrawals and occasional events (refunds, transfers from family).
   Self-employed customers also pay advance tax and, for some, GST.
5. Post everything in date order. A savings account can't go negative, so a debit that
   would overdraw it is declined (dropped and counted), like a failed UPI payment.
6. Inject ``emi_bounce`` if requested (it needs the balances from step 5).
7. Credit quarterly savings interest, computed from the daily balances.

Hard cases use their *own* random generator, so the same seed with and without hard
cases describes the same person: only the injected lines (and their knock-on effect on
balances, declines and interest) differ.
"""

import dataclasses
import math
from collections.abc import Sequence
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from faker import Faker

from statementscreen.synth import hard_cases as hc
from statementscreen.synth import names
from statementscreen.synth.banks import random_bank, random_ifsc
from statementscreen.synth.descriptions import Narrator, spelling_variants
from statementscreen.synth.models import Draft, IncomeTruth, Statement, Transaction
from statementscreen.synth.util import (
    CENT,
    money,
    month_end,
    prev_working_day,
    rupees,
    working_day_within,
)

CUSTOMER_TYPES = ("salaried", "gig", "self_employed")

# Categories that count as regular income in the ground truth.
INCOME_CATEGORIES = frozenset({"salary", "gig_income", "business_income"})

# Share of customers with each commitment.
P_RENT = 0.75
P_LOAN = 0.45
P_POSTPAID = 0.50
P_BROADBAND = 0.60
P_SUBSCRIPTION = 0.60
P_GST = 0.50  # share of self-employed customers registered for GST

# Probabilities of occasional events in any given month.
P_FAMILY_TRANSFER = 0.35  # money from family: NOT income, a classic verification trap
P_REFUND = 0.25

SPEND_MIN, SPEND_MAX = 20, 3000  # INR per UPI spend
SAVINGS_RATE = (0.025, 0.035)  # annual interest rate range on savings balances

# Indian advance-tax due dates and the share of annual tax paid on each.
ADVANCE_TAX = {(6, 15): 0.15, (9, 15): 0.30, (12, 15): 0.30, (3, 15): 0.25}


def _month_starts(start: date, months: int) -> list[date]:
    """First day of each month in the statement, e.g. Jan 1, Feb 1, Mar 1."""
    out, y, m = [], start.year, start.month
    for _ in range(months):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _day_in(first: date, day: int) -> date:
    """Day ``day`` (1-based) of the month starting at ``first``, capped at the month end."""
    return min(first + timedelta(days=day - 1), month_end(first))


def _random_day(rng: np.random.Generator, first: date) -> date:
    return first + timedelta(days=int(rng.integers(0, month_end(first).day)))


def _regular_monthly_income(drafts: list[Draft], months: int) -> Decimal:
    total = sum((d.amount for d in drafts if d.category in INCOME_CATEGORIES), Decimal(0))
    return (total / months).quantize(CENT, rounding=ROUND_HALF_UP)


def _lognormal_clip(rng: np.random.Generator, median: float, sigma: float, lo: float, hi: float):
    return float(np.clip(math.exp(rng.normal(math.log(median), sigma)), lo, hi))


# ---------------------------------------------------------------- income, by customer type


def _salaried_income(rng, fake, narrator, months, start, end):
    employer = names.employer_name(rng, fake)
    net_pay = rupees(_lognormal_clip(rng, 55_000, 0.6, 18_000, 250_000))
    variants = spelling_variants(employer)

    # Indian employers pay on the last working day, or on a fixed day early next month.
    if rng.random() < 0.5:
        dates = [prev_working_day(month_end(m)) for m in months]
    else:
        day = int(rng.integers(1, 8))
        dates = [working_day_within(_day_in(m, day), start, end) for m in months]

    drafts = []
    for d in dates:
        # A quarter of credits use another spelling of the employer's name.
        shown = employer if rng.random() >= 0.25 else str(rng.choice(variants))
        drafts.append(Draft(d, narrator.salary(shown, d), net_pay, "salary", (), employer))
    return drafts, IncomeTruth("salaried", (employer,), "monthly", net_pay, Decimal(0))


def _gig_income(rng, fake, narrator, months, start, end):
    n_platforms = int(rng.integers(1, 3))
    platforms = [str(p) for p in rng.choice(names.GIG_PLATFORMS, size=n_platforms, replace=False)]
    monthly_target = rng.uniform(12_000, 45_000)
    shares = rng.dirichlet(np.ones(n_platforms))

    drafts: list[Draft] = []
    for platform, share in zip(platforms, shares, strict=True):
        weekly_mean = monthly_target * share * 12 / 52
        weekday = int(rng.integers(0, 5))  # each platform settles on its own weekday
        d = start + timedelta(days=(weekday - start.weekday()) % 7)
        imps = rng.random() < 0.5  # platforms settle by IMPS or NEFT
        while d <= end:
            if rng.random() > 0.10:  # ~1 week in 10 they don't work that platform
                posted = working_day_within(d, start, end)
                amount = money(weekly_mean * rng.lognormal(0, 0.3))
                if imps:
                    text = narrator.imps_credit(platform, "PAYOUT")
                else:
                    text = narrator.neft_credit(platform, posted, "WEEKLY SETTLEMENT")
                drafts.append(Draft(posted, text, amount, "gig_income", (), platform))
            d += timedelta(days=7)
    return drafts, IncomeTruth("gig", tuple(platforms), "weekly", None, Decimal(0))


def _self_employed_income(rng, fake, narrator, months, start, end):
    clients = [names.business_client(rng) for _ in range(int(rng.integers(2, 6)))]
    monthly_target = _lognormal_clip(rng, 70_000, 0.7, 25_000, 300_000)

    drafts: list[Draft] = []
    for first in months:
        n_credits = int(rng.integers(2, 9))
        month_total = monthly_target * rng.lognormal(0, 0.4)  # some months are much better
        for share in rng.dirichlet(np.ones(n_credits)):
            posted = working_day_within(_random_day(rng, first), start, end)
            amount = money(month_total * share)
            channel = rng.random()
            if amount >= 200_000:  # large payments go by RTGS
                payer = str(rng.choice(clients))
                text = narrator.rtgs_credit(payer, posted)
            elif channel < 0.55:
                payer = str(rng.choice(clients))
                text = narrator.neft_credit(payer, posted, f"INV{int(rng.integers(100, 999))}")
            elif channel < 0.85:  # individual customers paying by UPI
                payer = fake.name().upper()
                text = narrator.upi_credit(payer, "PAYMENT")
            else:
                payer = fake.name().upper()
                text = narrator.imps_credit(payer, "PAYMENT")
            drafts.append(Draft(posted, text, amount, "business_income", (), payer))
    return drafts, IncomeTruth("self_employed", tuple(clients), "irregular", None, Decimal(0))


_INCOME_BUILDERS = {
    "salaried": _salaried_income,
    "gig": _gig_income,
    "self_employed": _self_employed_income,
}


# ---------------------------------------------------------------- spending


class _Household:
    """One person's fixed habits, drawn once: rent, loan, bills, favourite shops."""

    def __init__(
        self,
        rng: np.random.Generator,
        fake: Faker,
        usual: float,
        customer_type: str,
        force_loan: bool = False,  # emi_bounce needs a loan; budget for it from the start
    ):
        self.rent = rupees(usual * rng.uniform(0.15, 0.35)) if rng.random() < P_RENT else None
        self.landlord = fake.name().upper()
        self.rent_day = int(rng.integers(1, 8))
        self.rent_channel = str(rng.choice(["upi", "imps", "neft"], p=[0.6, 0.25, 0.15]))

        rent_share = float(self.rent or 0) / usual
        has_loan = rng.random() < P_LOAN or force_loan  # always draw, so the stream is stable
        emi_share = min(rng.uniform(0.10, 0.35), 0.55 - rent_share)  # lenders cap commitments
        self.emi = rupees(usual * max(emi_share, 0.05))
        self.has_loan = has_loan
        self.lender = str(rng.choice(names.LENDERS))
        self.emi_day = int(rng.choice([2, 5, 10]))
        self.umrn = f"{random_bank(rng).code}7{int(rng.integers(0, 10**15)):015d}"

        self.postpaid = rupees(rng.uniform(299, 999)) if rng.random() < P_POSTPAID else None
        self.prepaid = Decimal(int(rng.choice([199, 299, 399, 599, 799])))
        self.broadband = rupees(rng.uniform(499, 1499)) if rng.random() < P_BROADBAND else None
        subs = list(names.SUBSCRIPTIONS)
        n_subs = int(rng.integers(1, 3)) if rng.random() < P_SUBSCRIPTION else 0
        self.subscriptions = [
            (s, Decimal(int(rng.choice(names.SUBSCRIPTIONS[s]))), int(rng.integers(1, 29)))
            for s in rng.choice(subs, size=n_subs, replace=False)
        ]
        self.bill_day = int(rng.integers(5, 21))
        self.merchants = names.favourite_merchants(rng, fake)
        self.city = fake.city()

        bills = 2500 + float(self.postpaid or self.prepaid) + float(self.broadband or 0)
        commitments = rent_share + (emi_share if has_loan else 0) + bills / usual
        self.spend_share = float(np.clip(rng.uniform(0.15, 0.40), 0.08, 0.90 - commitments))
        # Busier spenders make more UPI payments: 30 a month at low incomes, up to 90.
        span = (math.log(usual) - math.log(12_000)) / (math.log(300_000) - math.log(12_000))
        self.upi_per_month = 30 + 60 * float(np.clip(span, 0, 1))

        self.tax_share = rng.uniform(0.05, 0.12) if customer_type == "self_employed" else 0.0
        self.pays_gst = customer_type == "self_employed" and rng.random() < P_GST


def _month_spending(
    rng: np.random.Generator,
    fake: Faker,
    narrator: Narrator,
    home: _Household,
    first: date,
    period: tuple[date, date],
    usual: float,
) -> list[Draft]:
    start, end = period
    drafts: list[Draft] = []

    def add(posted: date, text: str, amount: Decimal, category: str, who: str) -> None:
        drafts.append(Draft(posted, text, amount, category, (), who))

    # Rent to an individual landlord, by UPI, IMPS or NEFT.
    if home.rent is not None:
        d = _day_in(first, home.rent_day)
        if home.rent_channel == "upi":
            text = narrator.upi_debit(home.landlord, "RENT")
        elif home.rent_channel == "imps":
            text = narrator.imps_debit(home.landlord, "RENT")
        else:
            text = narrator.neft_debit(home.landlord, d, "RENT")
        add(d, text, -home.rent, "rent", home.landlord)

    # EMI collected by NACH on a working day.
    if home.has_loan:
        d = working_day_within(_day_in(first, home.emi_day), start, end)
        add(d, narrator.nach_debit(home.lender, home.umrn), -home.emi, "loan_payment", home.lender)

    # Bills via BBPS, prepaid recharge via UPI, subscriptions by card.
    d = _day_in(first, home.bill_day)
    add(d, narrator.bill(names.ELECTRICITY), -rupees(rng.uniform(600, 4500)), "utilities",
        names.ELECTRICITY)  # fmt: skip
    if home.postpaid is not None:
        add(d, narrator.bill(names.POSTPAID), -home.postpaid, "utilities", names.POSTPAID)
    else:
        add(d, narrator.upi_debit(names.PREPAID, "RECHARGE"), -home.prepaid, "utilities",
            names.PREPAID)  # fmt: skip
    if home.broadband is not None:
        add(d, narrator.bill(names.BROADBAND), -home.broadband, "utilities", names.BROADBAND)
    for sub, price, day in home.subscriptions:
        add(_day_in(first, day), narrator.card(sub), -price, "subscription", sub)

    # UPI spends: a budget that scales with income, split over many small payments.
    budget = usual * home.spend_share * rng.uniform(0.85, 1.15)
    cats = list(names.SPEND_CATEGORIES)
    weights = np.array([names.SPEND_CATEGORIES[c][0] for c in cats])
    n = int(rng.poisson(home.upi_per_month))
    picked = rng.choice(cats, size=n, p=weights / weights.sum())
    raw = np.array([names.SPEND_CATEGORIES[c][1] for c in picked]) * rng.lognormal(0, 0.6, n)
    scale = budget / max(float(raw.sum()), 1.0)
    for cat, base in zip(picked, raw, strict=True):
        merchant = str(rng.choice(home.merchants[str(cat)]))
        value = float(np.clip(base * scale, SPEND_MIN, SPEND_MAX))
        # Most UPI payments are whole rupees; a few (fuel, online orders) carry paise.
        amount = -(rupees(value) if rng.random() < 0.8 else money(value))
        add(_random_day(rng, first), narrator.upi_debit(merchant), amount, str(cat), merchant)

    # ATM cash: mostly multiples of 500.
    for _ in range(min(int(rng.poisson(1.0)), 3)):
        if rng.random() < 0.8:
            cash = 500 * int(rng.integers(1, 21))
        else:
            cash = 100 * int(rng.integers(5, 101))
        bank = narrator.own_bank.name.upper()
        add(_random_day(rng, first), narrator.atm(home.city), -Decimal(cash).quantize(CENT),
            "cash", bank)  # fmt: skip

    # Occasional events.
    if rng.random() < P_REFUND:
        merchant = str(rng.choice(home.merchants["shopping"]))
        add(_random_day(rng, first), narrator.refund(merchant), money(rng.uniform(50, 1500)),
            "refund", merchant)  # fmt: skip
    if rng.random() < P_FAMILY_TRANSFER:
        relative = fake.name().upper()
        gift = rupees(round(rng.uniform(500, 15_000), -2))
        text = (
            narrator.upi_credit(relative) if rng.random() < 0.7 else narrator.imps_credit(relative)
        )
        add(_random_day(rng, first), text, gift, "transfer_in", relative)

    # Self-employed: advance tax on the Indian due dates, and GST on the 20th if registered.
    if home.tax_share:
        for (month, day), share in ADVANCE_TAX.items():
            if first.month == month:
                tax = rupees(usual * 12 * home.tax_share * share)
                d = working_day_within(_day_in(first, day), start, end)
                add(d, narrator.advance_tax(), -tax, "tax", "INCOME TAX DEPARTMENT")
    if home.pays_gst:
        gst = rupees(usual * rng.uniform(0.04, 0.09))
        d = working_day_within(_day_in(first, 20), start, end)
        add(d, narrator.gst_payment(), -gst, "tax", "GST")

    return drafts


# ---------------------------------------------------------------- posting


# Fixed commitments get paid first; discretionary spends are what a short month cuts back.
MANDATORY = frozenset({"rent", "loan_payment", "utilities", "tax"})
DISCRETIONARY = frozenset({*names.SPEND_CATEGORIES, "cash", "subscription"})


def _balances_before(kept: list[Draft], opening: Decimal) -> list[Decimal]:
    out, balance = [], opening
    for d in kept:
        out.append(balance)
        balance += d.amount
    return out


def _post(drafts: list[Draft], opening: Decimal) -> tuple[list[Draft], list[Decimal], list[Draft]]:
    """Apply drafts in date order so the balance never goes below zero.

    - A discretionary debit that would overdraw is declined (dropped), like a failed UPI.
    - A mandatory debit that would overdraw first cancels the most recent discretionary
      spends (the person would have held back), and is declined only if that's not enough.

    Returns the kept drafts, the balance just before each one, and the declined drafts.
    """
    balance, kept, declined = opening, [], []
    for d in drafts:
        if balance + d.amount < 0 and d.category in MANDATORY and not d.hard_cases:
            j = len(kept) - 1
            # Walk back, but never past a hard-case line: what happened before a bounce
            # (e.g. the balance being too low) must stay true.
            while balance + d.amount < 0 and j >= 0 and not kept[j].hard_cases:
                if kept[j].category in DISCRETIONARY:
                    balance -= kept[j].amount  # undo that spend
                    declined.append(kept.pop(j))
                j -= 1
        if balance + d.amount < 0:
            if d.hard_cases:  # hard-case lines are placed so this can't happen
                raise RuntimeError(f"hard-case line would overdraw: {d}")
            declined.append(d)
            continue
        kept.append(d)
        balance += d.amount
    return kept, _balances_before(kept, opening), declined


def _quarter_ends(start: date, end: date) -> list[date]:
    return [
        date(y, m, 31 if m in (3, 12) else 30)
        for y in range(start.year, end.year + 1)
        for m in (3, 6, 9, 12)
        if start <= date(y, m, 31 if m in (3, 12) else 30) <= end
    ]


def _interest_credits(
    kept: list[Draft], opening: Decimal, period_start: date, rate: float, narrator: Narrator
) -> list[Draft]:
    """Quarterly interest on the average end-of-day balance, like an Indian savings account."""
    if not kept:
        return []
    end_of_day: dict[date, Decimal] = {}
    balance = opening
    for d in kept:
        balance += d.amount
        end_of_day[d.posted] = balance

    credits = []
    for q_end in _quarter_ends(period_start, kept[-1].posted):
        q_start = max(period_start, date(q_end.year, q_end.month - 2, 1))
        day, balance, total = q_start, opening, Decimal(0)
        for known in sorted(k for k in end_of_day if k < q_start):
            balance = end_of_day[known]
        while day <= q_end:
            balance = end_of_day.get(day, balance)
            total += balance
            day += timedelta(days=1)
        days = (q_end - q_start).days + 1
        amount = money(float(total) / days * rate * days / 365)
        if amount >= 1:  # banks don't credit interest below a rupee
            text = narrator.interest(q_start, q_end)
            credits.append(
                Draft(q_end, text, amount, "interest", (), narrator.own_bank.name.upper())
            )
    return credits


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
    fake = Faker("en_IN")
    fake.seed_instance(seed)

    bank = random_bank(rng)
    holder = fake.name().upper()
    narrator = Narrator.random(rng, bank, holder)
    # Same quirks and same counterparty accounts, but drawing from the hard-case stream.
    hc_narrator = dataclasses.replace(narrator, rng=hc_rng)

    month_firsts = _month_starts(start.replace(day=1), months)
    period_start, period_end = month_firsts[0], month_end(month_firsts[-1])

    # 1. Income for the whole period. The person's lifestyle (rent, spending) is set by
    #    their *usual* income, measured before any hard cases are injected.
    build_income = _INCOME_BUILDERS[customer_type]
    drafts, income = build_income(rng, fake, narrator, month_firsts, period_start, period_end)
    usual = float(_regular_monthly_income(drafts, months))

    # 2. Income hard cases. Order matters: rename runs last so it also renames the bonus.
    details: dict[str, str] = {}
    sources = income.sources
    if "missed_salary" in cases:
        drafts = hc.missed_salary(drafts, hc_rng, details)
    if "salary_date_shift" in cases:
        drafts = hc.salary_date_shift(drafts, hc_rng, period_end, details)
    if "bonus_month" in cases:
        drafts = hc.bonus_month(drafts, hc_rng, hc_narrator, income, month_firsts, details)
    if "employer_rename" in cases:
        drafts, new_name = hc.employer_rename(drafts, hc_rng, hc_narrator, income.employer, details)
        sources = (*sources, new_name)

    income = IncomeTruth(
        customer_type=income.customer_type,
        sources=sources,
        pay_frequency=income.pay_frequency,
        net_pay=income.net_pay,
        monthly_income=_regular_monthly_income(drafts, months),  # what actually arrived
    )

    # 3. Spending, month by month.
    home = _Household(rng, fake, usual, customer_type, force_loan="emi_bounce" in cases)
    for first in month_firsts:
        drafts += _month_spending(
            rng, fake, narrator, home, first, (period_start, period_end), usual
        )

    # 4. Post in date order (stable sort, so same-day order is deterministic).
    drafts.sort(key=lambda d: d.posted)
    # Last month's income is still in the account, so the opening balance at least covers
    # the first month's fixed commitments (rent, EMI, bills, tax).
    month1_end = month_end(period_start)
    month1_commitments = -sum(
        (d.amount for d in drafts if d.category in MANDATORY and d.posted <= month1_end),
        Decimal(0),
    )
    buffer = rupees(rng.uniform(500, 3000))
    opening = max(money(usual * rng.uniform(0.3, 1.5)), month1_commitments + buffer)
    kept, before, declined = _post(drafts, opening)

    # 5. EMI bounce needs balances, so it runs on the posted lines; then re-post.
    n_declined = len(declined)
    if "emi_bounce" in cases:
        kept = hc.emi_bounce(kept, before, declined, hc_rng, hc_narrator, home.umrn, details)
        kept, before, more = _post(kept, opening)
        n_declined += len(more)
        if "INSUFFICIENT BAL" in details["emi_bounce"]:
            n_declined -= 1  # that declined EMI is now the bounce, not a silent decline

    # 6. Quarterly interest (only adds money, so it can't cause new declines).
    rate = float(rng.uniform(*SAVINGS_RATE))
    kept += _interest_credits(kept, opening, period_start, rate, narrator)
    kept.sort(key=lambda d: d.posted)
    kept, _, more = _post(kept, opening)
    assert not more

    balance = opening
    transactions = []
    for d in kept:
        balance += d.amount
        transactions.append(
            Transaction(
                d.posted, d.description, d.amount, balance, d.category, d.hard_cases, d.counterparty
            )  # fmt: skip
        )

    return Statement(
        bank_name=bank.name,
        account_holder=holder,
        account_number=f"XXXXXXXXXX{int(rng.integers(1000, 10000))}",
        period_start=period_start,
        period_end=period_end,
        opening_balance=opening,
        income=income,
        transactions=transactions,
        hard_cases=cases,
        hard_case_details=details,
        ifsc=random_ifsc(rng, bank),
        micr=f"{int(rng.integers(100, 1000))}{int(rng.integers(0, 10**6)):06d}",
        branch=f"{fake.city()} Main Branch",
        customer_id=f"{int(rng.integers(10**8, 10**9))}",
        address=fake.address().replace("\n", ", "),
        declined_debits=n_declined,
    )
