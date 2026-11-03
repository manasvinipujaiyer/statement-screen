from datetime import date
from decimal import Decimal

import pytest

from statementscreen.synth import generate_statement
from statementscreen.synth.banks import FICTIONAL_BANKS
from statementscreen.synth.generator import CUSTOMER_TYPES, INCOME_CATEGORIES


def test_same_seed_gives_same_statement() -> None:
    assert generate_statement(seed=42) == generate_statement(seed=42)


def test_different_seeds_differ() -> None:
    assert generate_statement(seed=1) != generate_statement(seed=2)


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_balances_reconcile_to_the_cent(customer_type: str) -> None:
    s = generate_statement(seed=7, customer_type=customer_type, months=12)
    prev = s.opening_balance
    for t in s.transactions:
        assert t.balance == prev + t.amount
        prev = t.balance
    assert s.closing_balance == s.opening_balance + sum(t.amount for t in s.transactions)


def test_period_covers_whole_months_and_transactions_stay_inside() -> None:
    s = generate_statement(seed=3, months=7, start=date(2025, 11, 1))
    assert s.period_start == date(2025, 11, 1)
    assert s.period_end == date(2026, 5, 31)  # 7 months, crossing a year boundary
    dates = [t.posted for t in s.transactions]
    assert dates == sorted(dates)
    assert all(s.period_start <= d <= s.period_end for d in dates)


def test_rent_is_the_same_every_month() -> None:
    s = generate_statement(seed=11, months=9)
    rents = [t.amount for t in s.transactions if t.category == "rent"]
    assert len(rents) == 9
    assert len(set(rents)) == 1


def test_salary_matches_ground_truth() -> None:
    for seed in range(20):
        s = generate_statement(seed=seed, months=6)
        paychecks = [t for t in s.transactions if t.category == "salary"]
        assert all(t.amount == s.income.net_pay for t in paychecks)
        gaps = [(b.posted - a.posted).days for a, b in zip(paychecks, paychecks[1:], strict=False)]
        if s.income.pay_frequency == "monthly":
            assert len(paychecks) == 6
            assert all(26 <= g <= 35 for g in gaps)
        else:  # biweekly: 14 days apart, give or take a weekend shift
            assert len(paychecks) in (12, 13)  # Jan-Jun is 181 days; 181 / 14 = 12.9
            assert all(11 <= g <= 17 for g in gaps)


def test_gig_income_comes_weekly_from_listed_platforms() -> None:
    s = generate_statement(seed=5, customer_type="gig", months=6)
    payouts = [t for t in s.transactions if t.category == "gig_income"]
    assert len(payouts) >= 20  # ~26 weeks per platform, minus some skipped weeks
    assert all(any(p in t.description for p in s.income.sources) for t in payouts)
    assert s.income.net_pay is None  # amounts vary, there's no fixed paycheck


def test_self_employed_income_is_irregular() -> None:
    s = generate_statement(seed=5, customer_type="self_employed", months=6)
    payments = [t.amount for t in s.transactions if t.category == "business_income"]
    assert len(payments) >= 6
    assert len(set(payments)) > 1


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_monthly_income_truth_is_the_true_average(customer_type: str) -> None:
    s = generate_statement(seed=9, customer_type=customer_type, months=8)
    total = sum(t.amount for t in s.transactions if t.category in INCOME_CATEGORIES)
    assert abs(s.income.monthly_income - total / 8) < Decimal("0.01")


def test_unknown_customer_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_statement(seed=1, customer_type="astronaut")


def test_only_fictional_banks() -> None:
    assert all(generate_statement(seed=i).bank_name in FICTIONAL_BANKS for i in range(20))


def test_paydays_never_fall_on_weekends() -> None:
    for seed in range(30):
        for ct in CUSTOMER_TYPES:
            s = generate_statement(seed=seed, customer_type=ct)
            income = [t for t in s.transactions if t.category in INCOME_CATEGORIES]
            assert all(t.posted.weekday() < 5 for t in income)


def test_payroll_description_names_the_employer() -> None:
    for seed in range(20):
        s = generate_statement(seed=seed)
        for t in s.transactions:
            if t.category == "salary":
                assert s.income.employer in t.description


def test_description_styles_vary_across_statements() -> None:
    # The first word of a payroll line differs by style ("ACH", "DIRECT", or the employer).
    first_words = set()
    for seed in range(30):
        s = generate_statement(seed=seed)
        salary = next(t for t in s.transactions if t.category == "salary")
        first_words.add(salary.description.split()[0] in ("ACH", "DIRECT"))
    assert first_words == {True, False}


def test_optional_events_appear_somewhere_in_a_batch() -> None:
    seen = {t.category for seed in range(20) for t in generate_statement(seed=seed).transactions}
    assert {"refund", "transfer_in", "cash"} <= seen


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_overdrafts_are_rare(customer_type: str) -> None:
    # Real people occasionally overdraw, so we allow a few, but most statements stay positive.
    n = 100
    negative = sum(
        1
        for seed in range(n)
        if min(t.balance for t in generate_statement(seed, customer_type).transactions) < 0
    )
    assert negative / n < 0.08
