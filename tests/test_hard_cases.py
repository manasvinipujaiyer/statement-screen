from datetime import date, timedelta
from decimal import Decimal

import pytest

from statementscreen.synth import generate_statement
from statementscreen.synth.generator import CUSTOMER_TYPES
from statementscreen.synth.hard_cases import (
    BOUNCE_CHARGE,
    BOUNCE_GST,
    HARD_CASES,
    SALARIED_ONLY,
    applicable,
)

TAGGABLE = [c for c in HARD_CASES if c != "missed_salary"]  # missed_salary has no line to tag


def _balances_reconcile(s) -> bool:
    prev = s.opening_balance
    for t in s.transactions:
        if t.balance != prev + t.amount or t.balance < 0:
            return False
        prev = t.balance
    return True


@pytest.mark.parametrize("case", TAGGABLE)
def test_tagged_hard_case_appears_in_transactions(case: str) -> None:
    for seed in range(15):
        s = generate_statement(seed, "salaried", months=6, hard_cases=[case])
        assert s.hard_cases == (case,)
        assert any(case in t.hard_cases for t in s.transactions), f"seed {seed}"
        assert _balances_reconcile(s)


def test_missed_salary_leaves_a_gap_on_the_recorded_payday() -> None:
    for seed in range(15):
        s = generate_statement(seed, "salaried", months=6, hard_cases=["missed_salary"])
        missed = date.fromisoformat(s.hard_case_details["missed_salary"])
        paydays = [t.posted for t in s.transactions if t.category == "salary"]
        assert len(paydays) == 5
        assert all(abs((d - missed).days) > 5 for d in paydays), f"seed {seed}"


def test_all_hard_cases_together() -> None:
    for seed in range(15):
        s = generate_statement(seed, "salaried", months=9, hard_cases=list(HARD_CASES))
        assert s.hard_cases == HARD_CASES
        for case in TAGGABLE:
            assert any(case in t.hard_cases for t in s.transactions), f"{case}, seed {seed}"
        assert set(s.hard_case_details) == set(HARD_CASES)
        assert _balances_reconcile(s)
        assert s.transactions == sorted(s.transactions, key=lambda t: t.posted)


def test_salary_date_shift_delays_by_3_to_10_days() -> None:
    s = generate_statement(3, "salaried", hard_cases=["salary_date_shift"])
    old, new = (
        date.fromisoformat(x) for x in s.hard_case_details["salary_date_shift"].split(" -> ")
    )
    assert timedelta(days=3) <= new - old <= timedelta(days=12)  # +2 past a bank holiday


def test_employer_rename_keeps_one_canonical_counterparty() -> None:
    for seed in range(15):
        s = generate_statement(seed, "salaried", hard_cases=["employer_rename"])
        old, new = s.income.sources
        assert old != new
        renamed = [t for t in s.transactions if "employer_rename" in t.hard_cases]
        assert renamed and all(t.canonical_counterparty == old for t in renamed)
        # The new name shows in the narration (allowing for a typo or truncation now and then).
        mark = new.split()[1]  # the rename always changes the second word
        assert mark not in old.split()
        shown = sum(mark in t.description.upper() for t in renamed)
        assert shown / len(renamed) >= 0.6, (seed, mark, [t.description for t in renamed])


def test_bonus_is_not_counted_as_regular_income() -> None:
    plain = generate_statement(8, "salaried")
    bonus = generate_statement(8, "salaried", hard_cases=["bonus_month"])
    lines = [t for t in bonus.transactions if t.category == "bonus"]
    assert len(lines) == 1 and lines[0].canonical_counterparty == bonus.income.employer
    assert bonus.income.monthly_income == plain.income.monthly_income


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_emi_bounce_has_charges_and_a_consistent_story(customer_type: str) -> None:
    forms = set()
    for seed in range(30):
        s = generate_statement(seed, customer_type, hard_cases=["emi_bounce"])
        tagged = [t for t in s.transactions if "emi_bounce" in t.hard_cases]
        by_cat = {t.category: t for t in tagged}
        assert by_cat["bounce_charge"].amount == -BOUNCE_CHARGE
        assert by_cat["bounce_gst"].amount == -BOUNCE_GST
        emi = Decimal(s.hard_case_details["emi_bounce"].split()[1])

        if "loan_reversal" in by_cat:  # technical: debit posts, then is reversed
            forms.add("technical")
            assert set(by_cat) == {"loan_payment", "loan_reversal", "bounce_charge", "bounce_gst"}
            assert by_cat["loan_reversal"].amount == -by_cat["loan_payment"].amount
        else:  # insufficient funds: the EMI never posts that day, the balance was too low
            forms.add("insufficient")
            day = by_cat["bounce_charge"].posted
            assert not any(t.category == "loan_payment" and t.posted == day for t in s.transactions)
            assert "INSUFFICIENT" in by_cat["bounce_charge"].description.upper() or (
                len(by_cat["bounce_charge"].description) >= 38  # may be cut off by truncation
            )
            if "transfer_out" in by_cat:
                assert by_cat["transfer_out"].balance < emi
        assert _balances_reconcile(s)
    assert forms == {"technical", "insufficient"}


def test_hard_cases_only_change_what_they_touch() -> None:
    # Same seed with and without a hard case = same person. Compare a statement where
    # nothing was declined in either version; interest depends on balances, so skip it.
    cases = ["bonus_month", "salary_date_shift"]
    for seed in range(12, 80):
        plain = generate_statement(seed, "salaried", hard_cases=[])
        tricky = generate_statement(seed, "salaried", hard_cases=cases)
        if plain.declined_debits == tricky.declined_debits == 0:
            break
    else:
        pytest.fail("no seed without declines")

    def lines(s):
        return [(t.posted, t.description, t.amount) for t in s.transactions
                if not t.hard_cases and t.category != "interest"]  # fmt: skip

    assert set(lines(tricky)) <= set(lines(plain))
    assert len(lines(tricky)) == len(lines(plain)) - 1  # one salary moved (and tagged)


@pytest.mark.parametrize("case", sorted(SALARIED_ONLY))
def test_salary_cases_are_rejected_for_non_salaried(case: str) -> None:
    for customer_type in ("gig", "self_employed"):
        assert not applicable(case, customer_type)
        with pytest.raises(ValueError):
            generate_statement(1, customer_type, hard_cases=[case])


def test_unknown_hard_case_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_statement(1, hard_cases=["alien_invasion"])


@pytest.mark.parametrize("customer_type", CUSTOMER_TYPES)
def test_emi_bounce_does_not_starve_the_customer(customer_type: str) -> None:
    # Regression: a bounce once wiped out the buffer and caused dozens of extra declined
    # payments, an artificial "spending stops" pattern. For the same person (who has a
    # loan anyway), a bounce should add only a handful of declines on average.
    extra = []
    for seed in range(40):
        plain = generate_statement(seed, customer_type)
        if not any(t.is_emi for t in plain.transactions):
            continue
        bounced = generate_statement(seed, customer_type, hard_cases=["emi_bounce"])
        extra.append(bounced.declined_debits - plain.declined_debits)
    assert len(extra) >= 10
    assert sum(extra) / len(extra) <= 3
