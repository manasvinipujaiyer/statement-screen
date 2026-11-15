"""The comparison: known answers on tiny hand-built examples, the bootstrap, the
plain-word explanations and the scored-once guard."""

import json
import math

import numpy as np
import polars as pl
import pytest

from statementscreen import compare as cp
from statementscreen.train import LABELS


def _frame(t: list[bool], r: list[bool], m: list[bool], statement: list[str] | None = None,
           layout: str = "columns", tags: str = "") -> pl.DataFrame:  # fmt: skip
    """Rows with the same truth / rules / model for every label (scores follow m)."""
    n = len(t)
    cols = {"statement_id": statement or ["stmt_0001"] * n, "row": list(range(n)),
            "layout": [layout] * n, "tags": [tags] * n, "narration": ["X"] * n,
            "category": ["c"] * n}  # fmt: skip
    for label in LABELS:
        cols |= {f"t_{label}": t, f"r_{label}": r, f"m_{label}": m,
                 f"s_{label}": [0.9 if x else 0.1 for x in m]}  # fmt: skip
    return pl.DataFrame(cols)


def test_precision_recall_f1_on_a_hand_example() -> None:
    #        tp     fp     fn     tn     tp
    t = [True, False, True, False, True]
    r = [True, True, False, False, True]  # rules: tp 2, fp 1, fn 1
    m = [True, False, True, False, False]  # model: tp 2, fp 0, fn 1
    table = cp.label_table(_frame(t, r, m))["is_salary"]
    assert (table["rules"]["tp"], table["rules"]["fp"], table["rules"]["fn"]) == (2, 1, 1)
    assert table["rules"]["precision"] == pytest.approx(2 / 3)
    assert table["rules"]["recall"] == pytest.approx(2 / 3)
    assert table["rules"]["f1"] == pytest.approx(2 / 3)
    assert table["model"]["precision"] == 1.0 and table["model"]["recall"] == pytest.approx(2 / 3)
    assert table["model"]["f1"] == pytest.approx(0.8)


def test_per_statement_counts_follow_the_given_order() -> None:
    frame = _frame([True, True, False], [True, False, True], [True, True, True],
                   statement=["stmt_0002", "stmt_0001", "stmt_0002"])  # fmt: skip
    got = cp.per_statement(frame, "is_emi", "rules", ["stmt_0001", "stmt_0002", "stmt_0003"])
    assert got.tolist() == [[0, 0, 1], [1, 1, 0], [0, 0, 0]]


def test_bootstrap_is_fixed_by_its_seed() -> None:
    rng = np.random.default_rng(0)
    rules = rng.integers(0, 5, size=(50, 3))
    model = rng.integers(0, 5, size=(50, 3))
    a = cp.bootstrap(rules, model, n=200, seed=7)
    assert a == cp.bootstrap(rules, model, n=200, seed=7)
    assert a != cp.bootstrap(rules, model, n=200, seed=8)
    assert a["rules"]["low"] <= a["rules"]["high"]


def test_identical_predictions_give_a_difference_of_exactly_zero() -> None:
    rng = np.random.default_rng(1)
    same = rng.integers(0, 5, size=(40, 3))
    diff = cp.bootstrap(same, same.copy(), n=300, seed=3)["model_minus_rules"]
    assert diff["low"] == 0.0 and diff["high"] == 0.0 and diff["includes_zero"] is True


def test_score_set_has_salary_and_false_salary() -> None:
    frame = pl.concat([
        _frame([True, False], [True, False], [True, False], ["stmt_0001"] * 2),
        _frame([False, False], [False, True], [False, False], ["stmt_0002"] * 2),
        _frame([False, False], [False, False], [False, True], ["stmt_0003"] * 2),
    ])  # fmt: skip
    statements = pl.DataFrame({
        "statement_id": ["stmt_0001", "stmt_0002", "stmt_0003"],
        "customer_type": ["salaried", "gig", "self_employed"],
        "t_has_salary": [True, False, False],
        "r_has_salary": [True, True, False],   # rules: a false alarm on the gig statement
        "m_has_salary": [True, False, True],   # model: any row predicted as salary
    })  # fmt: skip
    s = cp.score_set(frame, statements, by_format_group=False, n=50)
    assert s["has_salary"]["rules"]["precision"] == 0.5
    assert s["has_salary"]["model"]["precision"] == 0.5
    assert s["false_salary"]["gig"] == {"statements": 1, "rules": 1, "model": 0}
    assert s["false_salary"]["self_employed"] == {"statements": 1, "rules": 0, "model": 1}
    # one true salary row at 0.9, one other row also at 0.9: average precision 0.5
    assert s["labels"]["is_salary"]["model"]["pr_auc"] == pytest.approx(0.5)
    assert set(s["breakdown"]) == {"by_layout", "by_hard_case"}
    salary = s["labels"]["is_salary"]
    expected = salary["model"]["f1"] - salary["rules"]["f1"]
    assert salary["model_minus_rules"]["f1"] == pytest.approx(expected)


def test_breakdowns_by_hard_case_and_format_group() -> None:
    a = _frame([True], [True], [False], ["stmt_0001"], tags="bonus_month|emi_bounce")
    b = _frame([True], [False], [True], ["stmt_0002"], tags="")
    groups = pl.Series("format_group", ["new_opening", "known_opening"])
    frame = pl.concat([a, b]).with_columns(groups)
    out = cp.breakdowns(frame, by_format_group=True)
    assert set(out["by_hard_case"]) == {"none", "bonus_month", "emi_bounce"}
    assert out["by_hard_case"]["emi_bounce"]["is_bounce"]["rules"]["tp"] == 1
    assert out["by_format_group"]["new_opening"]["rows"] == 1
    assert out["by_format_group"]["known_opening"]["is_rent"]["model"]["tp"] == 1


def test_synthetic_to_shift_drop() -> None:
    def s(f_rules, f_model):
        return {"labels": {label: {"rules": {"f1": f_rules}, "model": {"f1": f_model}}
                           for label in LABELS}}  # fmt: skip

    d = cp.drops(s(1.0, 0.9), s(0.5, 0.85))
    assert d["is_bounce"]["rules"] == pytest.approx(0.5)
    assert d["is_bounce"]["model"] == pytest.approx(0.05)


def test_plain_words() -> None:
    ngrams = ["rt", "nach"]
    assert cp.describe("tfidf_0001", 0.3, ngrams) == "narration contains 'nach'"
    assert cp.describe("tfidf_0000", 0.0, ngrams) == "narration does not contain 'rt'"
    assert cp.describe("log_amount", math.log1p(500), ngrams) == "amount is INR 500"
    assert cp.describe("is_credit", 1, ngrams) == "money in (a credit)"
    assert cp.describe("channel_nach", 0, ngrams) == "channel is not NACH"
    coverage = cp.describe("grp_month_coverage", 0.75, ngrams)
    assert coverage == "its group covers 75% of the statement's months"
    assert cp.describe("grp_gap_mean_days", float("nan"), ngrams).endswith("is missing")


def test_top_contributions_point_toward_the_wrong_answer() -> None:
    names = ["is_credit", "log_amount", "tfidf_0000", "day_of_month"]
    contrib = np.array([0.5, -2.0, 1.5, -0.1])
    values = np.array([1.0, math.log1p(590), 0.2, 3.0])
    up = cp.top_contributions(contrib, values, names, ["rtn"], toward=1)  # a false flag
    assert [c["feature"] for c in up] == ["tfidf_0000", "is_credit"]  # only positive pushes
    down = cp.top_contributions(contrib, values, names, ["rtn"], toward=-1)  # a miss
    assert [c["feature"] for c in down] == ["log_amount", "day_of_month"]
    assert down[0]["in_words"] == "amount is INR 590"


def test_errors_are_spread_over_labels_and_kinds() -> None:
    frame = _frame([True] * 10 + [False] * 10, [False] * 20, [False] * 10 + [True] * 10)
    picked = cp.pick_errors(frame, limit=8)
    assert len(picked) == 8
    assert {(e["label"], e["kind"]) for e in picked} == {
        (label, kind) for label in LABELS for kind in ("miss", "false flag")}  # fmt: skip


def test_the_shift_result_is_written_once(tmp_path) -> None:
    path = tmp_path / "report.json"
    first = {"shift_first_result": True, "sets": {"shift": {"f1": 0.5}}}
    cp.write_report(path, first, post_hoc=False)
    with pytest.raises(ValueError, match="already holds the one-time shift result"):
        cp.write_report(path, {"shift_first_result": True, "sets": {}}, post_hoc=False)
    cp.write_report(path, {"shift_first_result": True, "sets": {"shift": {"f1": 0.9}}},
                    post_hoc=True)  # fmt: skip
    saved = json.loads(path.read_text())
    assert saved["sets"]["shift"]["f1"] == 0.5  # the first result is kept as it was
    assert saved["post_hoc_runs"][0]["label"] == "post-hoc"
    assert saved["post_hoc_runs"][0]["sets"]["shift"]["f1"] == 0.9
    assert cp.main(["--out", str(path), "--models", str(tmp_path / "none")]) == 2  # refused
