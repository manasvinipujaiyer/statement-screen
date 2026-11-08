"""Every threshold pattern detection uses, in one place, so each run can log all of them."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PatternConfig:
    # --- cleaning: the account holder's own name (step 2)
    holder_match: float = 90.0  # RapidFuzz token_sort_ratio for "same name as the holder"
    min_prefix: int = 4  # shortest cut-off text allowed to match a name by prefix

    # --- grouping counterparties within a statement (step 3)
    # Picked on dev by statementscreen.sweep_grouping (reports/week3_grouping_sweep.json):
    # best pairwise F1 0.9952 at 0.45; 0.40-0.50 are equal to 4 decimals.
    group_threshold: float = 0.45
    group_grid: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
    fuzz_weight: float = 0.5  # similarity = w * RapidFuzz + (1 - w) * TF-IDF cosine
    ngram_min: int = 2  # character n-grams for TF-IDF
    ngram_max: int = 4
    linkage: str = "average"
    prefix_join_min: int = 6  # a name this long or longer may join its longer form by prefix

    # --- group features (step 4)
    day_tolerance: int = 3  # days (circular) from the usual day-of-month that count as "on time"
    amount_tolerance: float = 0.05  # share of the median that counts as "the same amount"

    # --- rules (step 5). Starting values; changed only from dev results, and logged.
    # salary: credit groups; "core" credits are within salary_core_band of the median
    salary_channels: tuple[str, ...] = ("NEFT", "RTGS", "IMPS", "OTHER")
    salary_core_band: float = 0.10
    salary_min_months: int = 3
    # Of the statement's months. Was 0.75; dev sweep (reports/week3_rules_sweep.json):
    # 0.55-0.65 tie at the best salary F1 (15 -> 4 misses, no false salary); middle taken.
    salary_min_coverage: float = 0.60
    salary_max_per_month: float = 1.5
    salary_min_on_time: float = 0.60  # share of core credits within day_tolerance
    # Added on dev: a core that is a small slice of a payer's credits is not a salary
    # (weekly gig payouts had a few near the median that looked monthly).
    salary_min_core_share: float = 0.50
    rename_median_band: float = 0.10  # two names with medians this close may be one employer
    # EMI: NACH debit groups
    emi_channels: tuple[str, ...] = ("NACH",)
    emi_core_band: float = 0.02
    emi_min_months: int = 3
    emi_max_per_month: float = 1.2
    emi_min_on_time: float = 0.60
    # bounce: returned EMIs and their charges, by narration style
    bounce_return_styles: tuple[str, ...] = ("nach_rtn", "ach_rtn")
    bounce_charge_styles: tuple[str, ...] = ("ach_rtn_chrg", "nach_rtn_chrgs")
    bounce_tax_style: str = "gst_on_charge"  # counted when straight after a return charge
    lender_match: float = 85.0  # RapidFuzz token_sort_ratio linking a return to an EMI group
    bounce_link_days: int = 7  # a return links to a same-amount EMI debit at most this early
    # rent: debit transfer groups to one person
    rent_channels: tuple[str, ...] = ("NEFT", "IMPS", "UPI", "OTHER")
    rent_min_months: int = 3
    # Kept at 0.60: the dev sweep gave identical results for 0.40-0.60 (no evidence to move).
    rent_min_coverage: float = 0.60
    rent_max_per_month: float = 1.2
    rent_min_near_median: float = 0.80  # share within amount_tolerance of the median
    rent_min_on_time: float = 0.60
    # INR 1,500. Dev sweep: INR 1,000-2,000 tie (all 1,189 false flags, monthly phone
    # recharges of INR 199-799, gone; no rent lost); middle taken. INR 2,500 loses rent.
    rent_min_amount_paise: int = 150_000

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = PatternConfig()
