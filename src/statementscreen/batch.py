"""Plan a batch of synthetic statements.

Planning decides, for statement number i, everything about it: its seed, customer type,
layout, length, start month and hard cases. It's pure arithmetic (no PDFs, no files), so
it is fast and easy to test.

Each statement's plan depends only on ``(main_seed, i)``, through
``numpy.random.SeedSequence(main_seed, spawn_key=(i,))``. So statement 7 is identical
whether you generate 10 statements or 2,000: adding more never changes earlier ones.

Nothing here reads the clock, so the same command always plans the same batch.
"""

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from statementscreen.synth.generator import CUSTOMER_TYPES
from statementscreen.synth.hard_cases import HARD_CASES, applicable
from statementscreen.synth.render import LAYOUT_NAMES

GENERATOR_VERSION = "1.0.0"  # bump when the generator's output changes on purpose

DEFAULT_MIX = {"salaried": 0.6, "gig": 0.2, "self_employed": 0.2}
DEFAULT_HARD_CASE_RATE = 0.2
MIN_MONTHS, MAX_MONTHS = 6, 12
# Statements start between Jan 2024 and whatever month still lets them end by Dec 2025.
FIRST_START = date(2024, 1, 1)
LAST_END = date(2025, 12, 31)


def parse_mix(text: str) -> dict[str, float]:
    """'salaried=0.6,gig=0.2,self_employed=0.2' -> dict. Must name known types and sum to 1."""
    mix: dict[str, float] = {}
    for part in text.split(","):
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or name not in CUSTOMER_TYPES:
            example = "salaried=0.6,gig=0.2,self_employed=0.2"
            raise ValueError(f"bad mix entry {part!r}; use e.g. {example}")
        if name in mix:
            raise ValueError(f"{name!r} appears twice in the mix")
        try:
            share = float(value)
        except ValueError:
            raise ValueError(f"{value!r} is not a number in mix entry {part!r}") from None
        if not 0 <= share <= 1:
            raise ValueError(f"share for {name!r} must be between 0 and 1, got {share}")
        mix[name] = share
    return validate_mix(mix)


def validate_mix(mix: dict[str, float]) -> dict[str, float]:
    if not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"mix shares must sum to 1, got {sum(mix.values()):g}")
    return {t: mix.get(t, 0.0) for t in CUSTOMER_TYPES}  # fixed order, missing types = 0


@dataclass(frozen=True)
class BatchConfig:
    seed: int = 42
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    hard_case_rate: float = DEFAULT_HARD_CASE_RATE

    def __post_init__(self) -> None:
        validate_mix(self.mix)
        if not 0 <= self.hard_case_rate <= 1:
            raise ValueError(f"hard_case_rate must be between 0 and 1, got {self.hard_case_rate}")


@dataclass(frozen=True)
class StatementPlan:
    number: int  # 1-based
    statement_id: str  # "stmt_0001"
    seed: int  # passed to generate_statement
    customer_type: str
    layout: str
    months: int
    start: date
    hard_cases: tuple[str, ...]


def _month_offset(first: date, k: int) -> date:
    y, m = divmod(first.month - 1 + k, 12)
    return date(first.year + y, m + 1, 1)


def plan_statement(config: BatchConfig, number: int) -> StatementPlan:
    """Everything about statement ``number`` (1-based), from (config.seed, number) alone."""
    if number < 1:
        raise ValueError("statement numbers start at 1")
    rng = np.random.default_rng(np.random.SeedSequence(config.seed, spawn_key=(number,)))

    mix = validate_mix(config.mix)
    customer_type = str(rng.choice(CUSTOMER_TYPES, p=[mix[t] for t in CUSTOMER_TYPES]))
    layout = str(rng.choice(LAYOUT_NAMES))  # even split
    months = int(rng.integers(MIN_MONTHS, MAX_MONTHS + 1))

    # Latest start that still ends by LAST_END: count months from FIRST_START.
    span = (LAST_END.year - FIRST_START.year) * 12 + LAST_END.month - FIRST_START.month + 1
    start = _month_offset(FIRST_START, int(rng.integers(0, span - months + 1)))

    # Each applicable hard case independently, at the configured rate. Always draw for
    # every case (even inapplicable ones), so the stream doesn't depend on the type.
    draws = rng.random(len(HARD_CASES))
    hard_cases = tuple(
        case
        for case, u in zip(HARD_CASES, draws, strict=True)
        if applicable(case, customer_type) and u < config.hard_case_rate
    )
    seed = int(rng.integers(0, 2**63 - 1))
    return StatementPlan(
        number=number,
        statement_id=f"stmt_{number:04d}",
        seed=seed,
        customer_type=customer_type,
        layout=layout,
        months=months,
        start=start,
        hard_cases=hard_cases,
    )


def plan_batch(config: BatchConfig, n: int) -> list[StatementPlan]:
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > 9999:
        raise ValueError("n is limited to 9999 (statement IDs have four digits)")
    return [plan_statement(config, i) for i in range(1, n + 1)]
