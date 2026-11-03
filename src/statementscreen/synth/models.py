"""Data model for a synthetic bank statement.

Money is stored as ``Decimal`` (not ``float``) because floats can't represent
cents exactly: 0.1 + 0.2 != 0.3. Balances must reconcile to the cent, so we
use exact decimal arithmetic throughout.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Transaction:
    """One line on the statement. Positive amount = money in, negative = money out."""

    posted: date
    description: str
    amount: Decimal
    balance: Decimal  # running balance *after* this transaction
    category: str  # ground-truth label, e.g. "salary", "rent" (never printed on the PDF)


@dataclass(frozen=True)
class IncomeTruth:
    """Ground truth about the account holder's income, used later to score the model.

    ``monthly_income`` is the true average of *regular* income per month (salary,
    gig payouts or client payments). Bonuses, transfers and refunds are excluded,
    because a lender shouldn't count them as stable income.
    """

    customer_type: str  # "salaried" | "gig" | "self_employed"
    sources: tuple[str, ...]  # who pays them: the employer, gig platforms, or clients
    pay_frequency: str  # "monthly" | "biweekly" | "weekly" | "irregular"
    net_pay: Decimal | None  # fixed amount per paycheck (salaried); None when amounts vary
    monthly_income: Decimal

    @property
    def employer(self) -> str:
        """The main income source (for salaried customers, the employer)."""
        return self.sources[0]


@dataclass
class Statement:
    bank_name: str
    account_holder: str
    account_number: str  # masked, e.g. "****1234"
    period_start: date
    period_end: date
    opening_balance: Decimal
    income: IncomeTruth
    transactions: list[Transaction] = field(default_factory=list)

    @property
    def closing_balance(self) -> Decimal:
        if not self.transactions:
            return self.opening_balance
        return self.transactions[-1].balance
