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
    # A regular-income group that pays about weekly is a gig payout. The upper gap allows a
    # skipped week now and then (dev, step 4: a platform with 3.4 payouts a month averaged
    # 9.15-day gaps and was called self-employed under the first limit of 9.0).
    weekly_min_per_month: float = 3.0
    weekly_gap_days: tuple[float, float] = (5.0, 10.5)
    # Variant (b), chosen on dev: for a statement estimated as self-employed, also count
    # one-off credits from counterparties that are not regular (customers paying once).
    # Dev (300 statements): income within 10% for 98.67% with it, 82.00% without
    # (reports/week5_profile_dev_variant_b.json vs week5_profile_dev.json, step 4).
    count_one_off_credits_for_self_employed: bool = True

    def as_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        text = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


DEFAULT = ProfileConfig()
