"""Ground truth for a statement, as plain JSON-ready data.

Two shapes:

- ``statement_truth``: one dict per statement (written to labels/stmt_NNNN.json).
- ``transaction_rows``: one dict per transaction (written to transactions.parquet).

Money is integer **paise** (INR 1 = 100 paise) so it survives JSON and Parquet exactly:
no floats anywhere. Dates are ISO strings (YYYY-MM-DD). Nothing here depends on the
time of the run or the machine, so the same statement always gives the same truth.
"""

from decimal import Decimal

from statementscreen.synth.models import Statement


def paise(amount: Decimal) -> int:
    """Decimal rupees -> exact integer paise. Refuses anything finer than a paisa."""
    value = amount * 100
    if value != value.to_integral_value():
        raise ValueError(f"{amount} has fractions of a paisa")
    return int(value)


def statement_truth(s: Statement, statement_id: str = "") -> dict:
    inc = s.income
    return {
        "statement_id": statement_id,
        "customer_type": inc.customer_type,
        "true_monthly_income_paise": paise(inc.monthly_income),
        "true_employer": inc.employer,  # None (JSON null) for gig and self-employed
        "income_sources": list(inc.sources),
        "pay_frequency": inc.pay_frequency,
        "net_pay_paise": paise(inc.net_pay) if inc.net_pay is not None else None,
        "hard_case_tags": list(s.hard_cases),
        "hard_case_details": dict(s.hard_case_details),
        "bank_name": s.bank_name,
        "ifsc": s.ifsc,
        "account_number": s.account_number,
        "period_start": s.period_start.isoformat(),
        "period_end": s.period_end.isoformat(),
        "opening_balance_paise": paise(s.opening_balance),
        "closing_balance_paise": paise(s.closing_balance),
        "transaction_count": len(s.transactions),
        "declined_debits": s.declined_debits,
    }


def transaction_rows(s: Statement, statement_id: str = "") -> list[dict]:
    return [
        {
            "statement_id": statement_id,
            "row": i,  # position on the statement, 0-based
            "posted": t.posted.isoformat(),
            "description": t.description,
            "amount_paise": paise(t.amount),
            "balance_paise": paise(t.balance),
            "category": t.category,
            "canonical_counterparty": t.canonical_counterparty,
            "is_salary": t.is_salary,
            "is_emi": t.is_emi,
            "is_bounce": t.is_bounce,
            "hard_case_tags": "|".join(t.hard_cases),  # "" when untouched
        }
        for i, t in enumerate(s.transactions)
    ]
