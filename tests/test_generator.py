from statementscreen.synth import generate_statement
from statementscreen.synth.banks import FICTIONAL_BANKS


def test_same_seed_gives_same_statement() -> None:
    assert generate_statement(seed=42) == generate_statement(seed=42)


def test_different_seeds_differ() -> None:
    assert generate_statement(seed=1) != generate_statement(seed=2)


def test_balances_reconcile_to_the_cent() -> None:
    s = generate_statement(seed=7)
    expected = s.opening_balance + sum(t.amount for t in s.transactions)
    assert s.closing_balance == expected
    # Each running balance must equal the previous balance plus this amount.
    prev = s.opening_balance
    for t in s.transactions:
        assert t.balance == prev + t.amount
        prev = t.balance


def test_transactions_are_in_date_order_and_inside_the_period() -> None:
    s = generate_statement(seed=3, year=2026, month=2)
    dates = [t.posted for t in s.transactions]
    assert dates == sorted(dates)
    assert all(s.period_start <= d <= s.period_end for d in dates)


def test_salary_matches_ground_truth() -> None:
    for seed in range(20):
        s = generate_statement(seed=seed)
        paychecks = [t for t in s.transactions if t.category == "salary"]
        assert paychecks, "every statement should contain at least one paycheck"
        assert all(t.amount == s.income.net_pay for t in paychecks)
        expected_count = (1,) if s.income.pay_frequency == "monthly" else (2, 3)
        assert len(paychecks) in expected_count


def test_only_fictional_banks() -> None:
    assert all(generate_statement(seed=i).bank_name in FICTIONAL_BANKS for i in range(20))
