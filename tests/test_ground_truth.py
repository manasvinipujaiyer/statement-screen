import json
import re
from decimal import Decimal

import pytest

from statementscreen.synth import generate_statement
from statementscreen.synth.generator import CUSTOMER_TYPES, INCOME_CATEGORIES
from statementscreen.synth.hard_cases import HARD_CASES, applicable
from statementscreen.synth.labels import paise, statement_truth, transaction_rows


def _all_cases(customer_type: str) -> list[str]:
    return [c for c in HARD_CASES if applicable(c, customer_type)]


def test_paise_is_exact_and_rejects_fractions() -> None:
    assert paise(Decimal("63343.00")) == 6_334_300
    assert paise(Decimal("-0.01")) == -1
    with pytest.raises(ValueError):
        paise(Decimal("1.005"))


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_flags_match_the_transactions(customer_type: str) -> None:
    for seed in range(10):
        s = generate_statement(seed, customer_type, hard_cases=_all_cases(customer_type))
        for t in s.transactions:
            text = t.description.upper()
            if t.is_salary:
                assert t.category == "salary" and t.amount > 0
                assert t.canonical_counterparty == s.income.employer
            if t.is_emi:
                assert t.amount < 0 and ("NACH" in text or "ACH D-" in text)
            if t.is_bounce:
                assert "emi_bounce" in t.hard_cases
                if t.category == "loan_reversal":
                    assert "RTN" in text and t.amount > 0
                if t.category in ("bounce_charge", "bounce_gst"):
                    assert t.amount in (Decimal("-500.00"), Decimal("-90.00"))
            # A line is never both salary and EMI, and only EMI-related lines can bounce.
            assert not (t.is_salary and t.is_emi)
            if t.is_bounce and not t.is_emi:
                assert t.category in ("loan_reversal", "bounce_charge", "bounce_gst")


def test_bonus_is_not_salary_and_shortfall_transfer_is_not_a_bounce() -> None:
    seen_bonus = seen_transfer = False
    for seed in range(20):
        s = generate_statement(seed, "salaried", hard_cases=["bonus_month", "emi_bounce"])
        for t in s.transactions:
            if t.category == "bonus":
                seen_bonus = True
                assert not t.is_salary
            if t.category == "transfer_out":
                seen_transfer = True
                assert "emi_bounce" in t.hard_cases and not t.is_bounce
    assert seen_bonus and seen_transfer


def test_renamed_employer_keeps_one_canonical_counterparty() -> None:
    for seed in range(15):
        s = generate_statement(seed, "salaried", months=9, hard_cases=["employer_rename"])
        payroll = {t.canonical_counterparty for t in s.transactions if t.is_salary}
        assert payroll == {s.income.employer}
        # ...even though the narrations show at least two different names.
        old, new = s.income.sources
        assert any(new.split()[1] in t.description.upper() for t in s.transactions if t.is_salary)


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_statement_truth_is_complete_and_json_safe(customer_type: str) -> None:
    s = generate_statement(3, customer_type, hard_cases=_all_cases(customer_type))
    truth = statement_truth(s, "stmt_0001")
    text = json.dumps(truth, sort_keys=True)  # raises if anything isn't JSON-safe
    assert "." not in re.sub(r'"[^"]*"', "", text)  # no float numbers outside strings
    assert truth["customer_type"] == customer_type
    assert truth["true_monthly_income_paise"] == paise(s.income.monthly_income)
    assert truth["hard_case_tags"] == list(s.hard_cases)
    assert set(truth["hard_case_details"]) == set(s.hard_cases)
    assert truth["transaction_count"] == len(s.transactions)
    if customer_type == "salaried":
        assert truth["true_employer"] == s.income.sources[0]
    else:
        assert truth["true_employer"] is None


def test_true_monthly_income_matches_the_income_rows() -> None:
    for ct in CUSTOMER_TYPES:
        s = generate_statement(11, ct, months=8)
        rows = transaction_rows(s)
        income = sum(r["amount_paise"] for r in rows if r["category"] in INCOME_CATEGORIES)
        assert abs(statement_truth(s)["true_monthly_income_paise"] - income / 8) <= 1


def test_transaction_rows_reconcile_in_paise() -> None:
    s = generate_statement(5, "self_employed", months=12, hard_cases=["emi_bounce"])
    rows = transaction_rows(s, "stmt_0042")
    balance = paise(s.opening_balance)
    for i, r in enumerate(rows):
        assert r["statement_id"] == "stmt_0042" and r["row"] == i
        assert isinstance(r["amount_paise"], int)
        balance += r["amount_paise"]
        assert r["balance_paise"] == balance
    assert balance == paise(s.closing_balance)
