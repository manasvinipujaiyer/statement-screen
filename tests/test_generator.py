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


def test_paydays_never_fall_on_weekends() -> None:
    for seed in range(50):
        s = generate_statement(seed=seed)
        assert all(t.posted.weekday() < 5 for t in s.transactions if t.category == "salary")


def test_payroll_description_names_the_employer() -> None:
    for seed in range(20):
        s = generate_statement(seed=seed)
        for t in s.transactions:
            if t.category in ("salary", "bonus"):
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
    seen = {t.category for seed in range(50) for t in generate_statement(seed=seed).transactions}
    assert {"bonus", "refund", "transfer_in", "cash"} <= seen


def test_overdrafts_are_rare() -> None:
    # Real people occasionally overdraw, so we allow a few, but most months stay positive.
    n = 200
    negative = sum(
        1
        for seed in range(n)
        if min(t.balance for t in generate_statement(seed=seed).transactions) < 0
    )
    assert negative / n < 0.05
