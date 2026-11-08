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

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = PatternConfig()
