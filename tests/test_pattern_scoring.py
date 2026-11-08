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


def test_prf_by_hand() -> None:
    from statementscreen.pattern_scoring import prf

    s = prf(tp=8, fp=2, fn=8)
    assert s["precision"] == 0.8 and s["recall"] == 0.5
    assert prf(0, 0, 0)["precision"] is None


def test_employer_normalisation_is_independent_of_the_detector() -> None:
    from statementscreen.pattern_scoring import normalise_employer

    assert normalise_employer("Kesalu Solutions Pvt. Ltd.") == "kesalu solutions"
    assert normalise_employer("SAGAR & SONS PRIVATE LIMITED") == "sagar sons"
    assert normalise_employer(None) == ""


def test_rule_sweep_picks_the_middle_of_a_tie() -> None:
    from statementscreen.sweep_rules import pick

    assert pick([0, 1, 2, 3, 4], [0.5, 0.9, 0.9, 0.9, 0.7]) == 2
    assert pick([0, 1, 2, 3], [0.9, 0.9, 0.1, 0.1]) == 1  # even run: the higher middle
    assert pick([0, 1, 2], [None, 0.2, 0.1]) == 1
