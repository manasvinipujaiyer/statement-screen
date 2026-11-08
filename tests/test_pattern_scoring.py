"""Scoring for pattern detection."""

from statementscreen.pattern_scoring import PairCounts


def test_pairwise_counts_by_hand() -> None:
    # truth: {0,1,2} {3}; ours: {0,1} {2,3}
    c = PairCounts()
    c.add(["a", "a", "b", "b"], ["x", "x", "x", "y"])
    assert (c.both, c.predicted, c.truth) == (1, 2, 3)  # both: (0,1)
    s = c.scores()
    assert s["precision"] == 0.5 and s["recall"] == 1 / 3
    assert abs(s["f1"] - 0.4) < 1e-12


def test_pairwise_counts_add_up_over_statements() -> None:
    c = PairCounts()
    c.add(["a", "a"], ["x", "x"])
    c.add(["a", "b"], ["x", "x"])
    assert (c.both, c.predicted, c.truth) == (1, 1, 2)


def test_no_pairs_gives_none_not_a_crash() -> None:
    c = PairCounts()
    c.add(["a"], ["x"])
    assert c.scores()["precision"] is None and c.scores()["f1"] is None
