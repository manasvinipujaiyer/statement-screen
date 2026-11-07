"""Every threshold pattern detection uses, in one place, so each run can log all of them."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PatternConfig:
    # --- cleaning: the account holder's own name (step 2)
    holder_match: float = 90.0  # RapidFuzz token_sort_ratio for "same name as the holder"
    min_prefix: int = 4  # shortest cut-off text allowed to match a name by prefix

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT = PatternConfig()
