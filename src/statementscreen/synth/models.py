"""Data model for a synthetic bank statement.

Money is stored as ``Decimal`` (not ``float``) because floats can't represent
paise exactly: 0.1 + 0.2 != 0.3. Balances must reconcile to the paisa, so we
use exact decimal arithmetic throughout. Amounts are in INR.
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import NamedTuple


class Draft(NamedTuple):
    """A transaction before its running balance is known (used while building a statement)."""

    posted: date
    description: str
    amount: Decimal
    category: str
    hard_cases: tuple[str, ...] = ()
    counterparty: str = ""  # canonical name of who paid / was paid, whatever the narration says


BOUNCE_CATEGORIES = frozenset({"loan_reversal", "bounce_charge", "bounce_gst"})


@dataclass(frozen=True)
class Transaction:
    """One line on the statement. Positive amount = money in, negative = money out.

    The ``is_*`` flags are derived from ``category`` (and the bounce tag), so they can
    never disagree with it.
    """

    posted: date
    description: str
    amount: Decimal
    balance: Decimal  # running balance *after* this transaction
    category: str  # ground-truth label, e.g. "salary", "rent" (never printed on the PDF)
    hard_cases: tuple[str, ...] = ()  # which injected hard cases touched this line, if any
    canonical_counterparty: str = ""  # one name per counterparty: variants and renames collapse

    @property
    def is_salary(self) -> bool:
        return self.category == "salary"  # bonuses are deliberately not salary

    @property
    def is_emi(self) -> bool:
        return self.category == "loan_payment"

    @property
    def is_bounce(self) -> bool:
        """Lines that exist because an EMI bounced: the reversal and the charges, plus the
        EMI debit itself when it was posted and then reversed. Not the self-transfer that
        caused a shortfall: that's the cause, not part of the bounce."""
        if self.category in BOUNCE_CATEGORIES:
            return True
        return self.is_emi and "emi_bounce" in self.hard_cases


@dataclass(frozen=True)
class IncomeTruth:
    """Ground truth about the account holder's income, used later to score the model.

    ``monthly_income`` is the true average of *regular* income per month (salary,
    gig payouts or client payments) that actually arrived. Bonuses, transfers, refunds
    and interest are excluded, because a lender shouldn't count them as stable income.
    """

    customer_type: str  # "salaried" | "gig" | "self_employed"
    sources: tuple[str, ...]  # names income arrived under (incl. an employer's new name)
    pay_frequency: str  # "monthly" | "weekly" | "irregular"
    net_pay: Decimal | None  # fixed monthly salary (salaried); None when amounts vary
    monthly_income: Decimal

    @property
    def employer(self) -> str | None:
        """The employer's canonical name for salaried customers; None for gig/self-employed."""
        return self.sources[0] if self.customer_type == "salaried" else None


@dataclass
class Statement:
    bank_name: str
    account_holder: str
    account_number: str  # masked, e.g. "XXXXXXXXXX4821"
    period_start: date
    period_end: date
    opening_balance: Decimal
    income: IncomeTruth
    transactions: list[Transaction] = field(default_factory=list)
    hard_cases: tuple[str, ...] = ()  # hard cases injected into this statement
    hard_case_details: dict[str, str] = field(default_factory=dict)  # e.g. which payday was missed
    ifsc: str = ""
    micr: str = ""
    branch: str = ""
    customer_id: str = ""
    address: str = ""
    account_type: str = "Savings Account"
    declined_debits: int = 0  # debits dropped because they would have overdrawn the account

    @property
    def closing_balance(self) -> Decimal:
        if not self.transactions:
            return self.opening_balance
        return self.transactions[-1].balance
