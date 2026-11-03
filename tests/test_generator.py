import re
from datetime import date
from decimal import Decimal

import pytest

from statementscreen.synth import generate_statement
from statementscreen.synth.banks import BANK_CODES, BANK_NAMES, UPI_HANDLES
from statementscreen.synth.descriptions import spelling_variants
from statementscreen.synth.generator import CUSTOMER_TYPES, INCOME_CATEGORIES
from statementscreen.synth.util import is_bank_holiday

IFSC_LIKE = re.compile(r"\b[A-Z]{4}0\d{6}\b")


def _batch(n: int = 30, months: int = 6):
    for seed in range(n):
        yield generate_statement(seed, CUSTOMER_TYPES[seed % 3], months=months)


def test_same_seed_gives_same_statement() -> None:
    assert generate_statement(seed=42) == generate_statement(seed=42)


def test_different_seeds_differ() -> None:
    assert generate_statement(seed=1) != generate_statement(seed=2)


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_balances_reconcile_to_the_paisa(customer_type: str) -> None:
    s = generate_statement(seed=7, customer_type=customer_type, months=12)
    prev = s.opening_balance
    for t in s.transactions:
        assert t.balance == prev + t.amount
        prev = t.balance
    assert s.closing_balance == s.opening_balance + sum(t.amount for t in s.transactions)


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_balance_never_goes_negative(customer_type: str) -> None:
    # Stricter than the old "overdrafts are rare": a savings account can't overdraw.
    for seed in range(60):
        s = generate_statement(seed, customer_type, months=6 + seed % 7)
        assert s.opening_balance >= 0
        assert all(t.balance >= 0 for t in s.transactions), f"seed {seed}"


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_emi_is_paid_every_month(customer_type: str) -> None:
    # Fixed commitments are protected: an EMI never silently disappears.
    for seed in range(60):
        months = 6 + seed % 7
        s = generate_statement(seed, customer_type, months=months)
        emi_months = {(t.posted.year, t.posted.month) for t in s.transactions
                      if t.category == "loan_payment"}  # fmt: skip
        assert len(emi_months) in (0, months), f"seed {seed}"


def test_period_covers_whole_months_and_transactions_stay_inside() -> None:
    s = generate_statement(seed=3, months=7, start=date(2025, 11, 1))
    assert s.period_start == date(2025, 11, 1)
    assert s.period_end == date(2026, 5, 31)  # 7 months, crossing a year boundary
    dates = [t.posted for t in s.transactions]
    assert dates == sorted(dates)
    assert all(s.period_start <= d <= s.period_end for d in dates)


def test_rent_is_the_same_every_month() -> None:
    renters = 0
    for seed in range(20):
        s = generate_statement(seed, months=9)
        rents = [t.amount for t in s.transactions if t.category == "rent"]
        if rents:
            renters += 1
            assert len(rents) == 9
            assert len(set(rents)) == 1
    assert renters >= 10  # ~75% of customers pay rent


def test_salary_matches_ground_truth() -> None:
    for seed in range(20):
        s = generate_statement(seed=seed, months=6)
        salary = [t for t in s.transactions if t.category == "salary"]
        assert s.income.pay_frequency == "monthly"
        assert len(salary) == 6
        assert all(t.amount == s.income.net_pay for t in salary)
        gaps = [(b.posted - a.posted).days for a, b in zip(salary, salary[1:], strict=False)]
        assert all(25 <= g <= 37 for g in gaps)
        assert Decimal(18_000) <= s.income.net_pay <= Decimal(250_000)


def test_gig_income_comes_weekly_from_listed_platforms() -> None:
    s = generate_statement(seed=5, customer_type="gig", months=6)
    payouts = [t for t in s.transactions if t.category == "gig_income"]
    assert len(payouts) >= 20  # ~26 weeks per platform, minus some skipped weeks
    assert all(t.counterparty in s.income.sources for t in payouts)
    assert s.income.net_pay is None  # amounts vary, there's no fixed paycheck


def test_self_employed_income_is_irregular() -> None:
    s = generate_statement(seed=5, customer_type="self_employed", months=6)
    payments = [t.amount for t in s.transactions if t.category == "business_income"]
    assert len(payments) >= 12  # 2-8 credits a month
    assert len(set(payments)) > 1


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_monthly_income_truth_is_the_true_average(customer_type: str) -> None:
    s = generate_statement(seed=9, customer_type=customer_type, months=8)
    total = sum(t.amount for t in s.transactions if t.category in INCOME_CATEGORIES)
    assert abs(s.income.monthly_income - total / 8) < Decimal("0.01")


def test_income_ranges_follow_the_agreed_inr_bands() -> None:
    for seed in range(30):
        gig = generate_statement(seed, "gig", months=6).income.monthly_income
        self_emp = generate_statement(seed, "self_employed", months=6).income.monthly_income
        assert Decimal(5_000) <= gig <= Decimal(60_000)  # 12k-45k target, weekly noise
        assert Decimal(8_000) <= self_emp <= Decimal(900_000)  # 25k-3L target, big swings


def test_unknown_customer_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_statement(seed=1, customer_type="astronaut")


def test_only_fictional_banks_ifsc_codes_and_upi_handles() -> None:
    for s in _batch():
        assert s.bank_name in BANK_NAMES
        assert s.ifsc[:4] in BANK_CODES and IFSC_LIKE.fullmatch(s.ifsc)
        for t in s.transactions:
            for code in IFSC_LIKE.findall(t.description.upper()):
                assert code[:4] in BANK_CODES, t.description
            text = t.description.lower()
            for m in re.finditer(r"@([a-z]+)", text):
                handle = m.group(1)
                if m.end() == len(text):  # cut off by truncation: must start a real handle
                    assert any(h.startswith(handle) for h in UPI_HANDLES), t.description
                else:
                    assert handle in UPI_HANDLES, t.description


def test_salary_and_nach_post_on_bank_working_days() -> None:
    # Stricter than the old weekday rule: also excludes the 2nd and 4th Saturday.
    for s in _batch(45):
        for t in s.transactions:
            if t.category in INCOME_CATEGORIES or t.category == "loan_payment":
                assert not is_bank_holiday(t.posted), (t.posted, t.description)


def test_salary_counterparty_is_the_employer_and_narration_names_it() -> None:
    named = total = 0
    for seed in range(20):
        s = generate_statement(seed=seed)
        variants = [v.upper() for v in spelling_variants(s.income.employer)]
        for t in s.transactions:
            if t.category == "salary":
                assert t.counterparty == s.income.employer
                total += 1
                text = t.description.upper()
                named += any(v.split()[0] in text for v in variants)
    # Typos and truncation hide the name on a few lines, never on most.
    assert named / total >= 0.85


_MONTHS = r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b"


def _without_references(text: str) -> str:
    """Drop the parts that change every month anyway: numbers and the pay-month label."""
    return re.sub(_MONTHS, "", re.sub(r"\d+", "", text.upper()))


def test_employer_spelling_varies_within_a_statement() -> None:
    varied = 0
    for seed in range(20):
        s = generate_statement(seed=seed, months=12)
        forms = {_without_references(t.description) for t in s.transactions
                 if t.category == "salary"}  # fmt: skip
        varied += len(forms) > 1  # same employer, same account, different spelling
    assert varied >= 10


def test_india_channels_appear_and_us_formats_are_gone() -> None:
    text = " ".join(t.description.upper() for s in _batch(30) for t in s.transactions)
    for channel in ("NEFT", "IMPS", "UPI", "ATM", "BBPS"):
        assert channel in text, channel
    assert "NACH" in text or "ACH D-" in text
    for us in ("ACH PPD", "VISA DDA", "POS DEBIT", "CHECKING ACCOUNT", "DIR DEP"):
        assert us not in text, us


def test_narration_mess_is_present_across_a_batch() -> None:
    statements = list(_batch(40))
    descriptions = [t.description for s in statements for t in s.transactions]
    assert any(d != d.upper() for d in descriptions)  # some Title Case names
    assert any("PRIVATE LIMITED" in d.upper() or " P LTD" in d.upper() for d in descriptions)
    lengths = {max(len(t.description) for t in s.transactions) for s in statements}
    assert min(lengths) <= 60 < max(lengths)  # some statements truncate, some don't


def test_occasional_events_appear_somewhere_in_a_batch() -> None:
    seen = {t.category for s in _batch(30) for t in s.transactions}
    assert {"refund", "transfer_in", "cash", "interest", "subscription", "rent"} <= seen


def test_interest_is_credited_quarterly_and_is_not_income() -> None:
    s = generate_statement(seed=2, months=12, start=date(2025, 1, 1))
    interest = [t for t in s.transactions if t.category == "interest"]
    assert [(t.posted.month, t.posted.day) for t in interest] == [
        (3, 31),
        (6, 30),
        (9, 30),
        (12, 31),
    ]
    assert all(0 < t.amount < s.income.monthly_income for t in interest)
    assert "interest" not in INCOME_CATEGORIES


def test_upi_payee_keeps_one_upi_id() -> None:
    s = generate_statement(seed=4, months=6)
    ids: dict[str, set[str]] = {}
    for t in s.transactions:
        for vpa in re.findall(r"[a-z]+\d\d@[a-z]+", t.description.lower()):
            ids.setdefault(t.counterparty, set()).add(vpa)
    assert ids and all(len(v) == 1 for v in ids.values())
