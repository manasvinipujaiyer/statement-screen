from datetime import date, timedelta

import pytest

from statementscreen.synth import generate_statement
from statementscreen.synth.hard_cases import HARD_CASES, SALARIED_ONLY

TAGGABLE = [c for c in HARD_CASES if c != "missed_salary"]  # missed_salary has no line to tag


def _balances_reconcile(s) -> bool:
    prev = s.opening_balance
    for t in s.transactions:
        if t.balance != prev + t.amount:
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
    assert timedelta(days=3) <= new - old <= timedelta(days=12)  # +2 if it lands on a weekend


def test_employer_rename_adds_the_new_name_to_sources() -> None:
    s = generate_statement(4, "salaried", hard_cases=["employer_rename"])
    old, new = s.income.sources
    assert old != new
    renamed = [t for t in s.transactions if "employer_rename" in t.hard_cases]
    assert renamed and all(new in t.description for t in renamed)


def test_bonus_is_not_counted_as_regular_income() -> None:
    plain = generate_statement(8, "salaried")
    bonus = generate_statement(8, "salaried", hard_cases=["bonus_month"])
    assert any(t.category == "bonus" for t in bonus.transactions)
    assert bonus.income.monthly_income == plain.income.monthly_income


def test_emi_bounce_reverses_the_payment_and_charges_a_fee() -> None:
    for ct in ("salaried", "gig", "self_employed"):
        s = generate_statement(6, ct, hard_cases=["emi_bounce"])
        tagged = [t for t in s.transactions if "emi_bounce" in t.hard_cases]
        by_cat = {t.category: t for t in tagged}
        assert set(by_cat) == {"loan_payment", "loan_reversal", "fee"}
        assert by_cat["loan_reversal"].amount == -by_cat["loan_payment"].amount


def test_hard_cases_only_change_what_they_touch() -> None:
    # Same seed with and without a hard case = same person; only tagged lines may differ.
    plain = generate_statement(12, "salaried", hard_cases=[])
    tricky = generate_statement(12, "salaried", hard_cases=["bonus_month", "salary_date_shift"])
    untouched = [
        (t.posted, t.description, t.amount) for t in tricky.transactions if not t.hard_cases
    ]
    original = {(t.posted, t.description, t.amount) for t in plain.transactions}
    assert set(untouched) <= original
    assert len(untouched) == len(plain.transactions) - 1  # one paycheck moved, one bonus added


@pytest.mark.parametrize("case", sorted(SALARIED_ONLY))
def test_salary_cases_are_rejected_for_non_salaried(case: str) -> None:
    with pytest.raises(ValueError):
        generate_statement(1, "gig", hard_cases=[case])


def test_unknown_hard_case_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_statement(1, hard_cases=["alien_invasion"])
