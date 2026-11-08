"""Scoring for Week 3 pattern detection. Reads the ground truth; pattern detection never
imports this module.

Pairwise grouping metrics: every pair of rows in the same statement is either grouped
together or not, by us and by the truth (canonical_counterparty). Pairs are counted from
group sizes, so no pair list or matrix is ever built:

    precision = pairs together in both / pairs we put together
    recall    = pairs together in both / pairs the truth puts together

Rows whose truth canonical_counterparty is null are left out of grouping scores.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


def _pairs(sizes: Counter) -> int:
    return sum(n * (n - 1) // 2 for n in sizes.values())


@dataclass
class PairCounts:
    both: int = 0  # pairs grouped together by us and by the truth
    predicted: int = 0  # pairs we grouped together
    truth: int = 0  # pairs the truth groups together

    def add(self, predicted: Sequence[str], truth: Sequence[str]) -> None:
        """One statement's group ids (ours and the truth's), row by row."""
        self.both += _pairs(Counter(zip(predicted, truth, strict=True)))
        self.predicted += _pairs(Counter(predicted))
        self.truth += _pairs(Counter(truth))

    def scores(self) -> dict:
        p = self.both / self.predicted if self.predicted else None
        r = self.both / self.truth if self.truth else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
        return {"precision": p, "recall": r, "f1": f1, "pairs_both": self.both,
                "pairs_predicted": self.predicted, "pairs_truth": self.truth}  # fmt: skip
