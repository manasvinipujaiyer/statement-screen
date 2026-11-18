"""Every setting of the income profile, in one place (hashed for the pre-registration)."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProfileConfig:
    # Warnings and confidence
    min_history_months: int = 6  # shorter history: a warning, and low-confidence income
    # Other regular income: a credit group present in at least this many months
    recurring_min_months: int = 3
    # Channels that are never income (interest, refunds and reversals, charges)
    not_income_channels: tuple[str, ...] = ("INTEREST", "REVERSAL", "CHARGE", "NACH")
    # A regular-income group that pays about weekly is a gig payout
    weekly_min_per_month: float = 3.0
    weekly_gap_days: tuple[float, float] = (5.0, 9.0)
    # Variant (b), chosen on dev: for a statement estimated as self-employed, also count
    # one-off credits from counterparties that are not regular (customers paying once).
    count_one_off_credits_for_self_employed: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        text = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


DEFAULT = ProfileConfig()
