"""Anomaly features and finding (tamper/detect/anomaly.py) on hand-built statements and a
tiny model fitted here (never the real models/ file)."""

from datetime import date

import numpy as np
from sklearn.ensemble import IsolationForest

from statementscreen.parsing.models import ParsedRow, ParsedStatement, StatementHeader
from statementscreen.tamper.detect import anomaly
from statementscreen.tamper.detect.anomaly import FEATURES, anomaly_findings, features


def _statement(amounts: list[int], opening: int = 100_000) -> ParsedStatement:
    rows, balance = [], opening
    for i, a in enumerate(amounts):
        balance += a
        rows.append(ParsedRow("s", i, date(2025, 1, 1 + i % 28), None, None, "x", a, balance,
                              1, "columns"))  # fmt: skip
    h = StatementHeader(period_start=date(2025, 1, 1), period_end=date(2025, 2, 28))
    return ParsedStatement("s", "columns", 1, "ok", header=h, rows=rows)


def _profile(income: int | None, salary: int | None, months: int = 2) -> dict:
    return {"statement_months": months,
            "findings": {"estimated_monthly_income_paise": {"value": income},
                         "typical_monthly_salary_paise": {"value": salary}}}  # fmt: skip


def test_features_are_computed_from_rows_and_profile():
    # credits 50,000.00 and 1,234.50; debits 10,000.00 and 2,000.00 (paise below)
    s = _statement([5_000_000, -1_000_000, 123_450, -200_000])
    f = features(s, _profile(income=2_500_000, salary=2_500_000))
    assert set(f) == set(FEATURES)
    assert f["credits_to_debits"] == (5_000_000 + 123_450) / 1_200_000
    assert f["income_to_debits"] == 2_500_000 * 2 / 1_200_000
    assert f["salary_to_spending"] == 2_500_000 / (1_200_000 / 2)
    assert f["round_100_share"] == 3 / 4  # 1,234.50 is not a whole INR 100
    assert f["round_1000_share"] == 3 / 4  # 50,000, 10,000 and 2,000 are
    assert f["credits_per_month"] == 1.0
    assert f["largest_to_median_credit"] == 5_000_000 / ((5_000_000 + 123_450) / 2)


def test_no_salary_or_income_gives_zero_not_an_error():
    f = features(_statement([10_000, -5_000]), _profile(income=None, salary=None))
    assert f["income_to_debits"] == 0 and f["salary_to_spending"] == 0


def test_no_rows_or_parse_error_gives_no_features():
    assert features(_statement([]), _profile(None, None)) is None
    err = ParsedStatement("s", None, 0, "error", "unreadable")
    assert features(err, _profile(None, None)) is None


def test_features_never_describe_how_the_file_was_made():
    for name in FEATURES:
        for word in ("producer", "creator", "font", "layout", "date", "metadata"):
            assert word not in name


def _ordinary(rng: np.random.Generator) -> ParsedStatement:
    """Salary 50,000 twice, 30 random debits of INR 100 to 3,000."""
    debits = [-int(x) for x in rng.integers(10_000, 300_000, 30)]
    return _statement([5_000_000, *debits, 5_000_000])


def _tiny_model() -> dict:
    """Fitted on 200 ordinary statements."""
    rng = np.random.default_rng(0)
    vectors = []
    for _ in range(200):
        f = features(_ordinary(rng), _profile(5_000_000, 5_000_000))
        vectors.append(anomaly.vector(f)[0])
    model = IsolationForest(n_estimators=100, random_state=0).fit(np.array(vectors))
    scores = -model.score_samples(np.array(vectors))
    return {"model": model, "threshold": float(np.quantile(scores, 0.98)),
            "features": list(FEATURES)}  # fmt: skip


def test_ordinary_statement_gives_nothing_and_unusual_one_is_weak():
    model = _tiny_model()
    rng = np.random.default_rng(1)  # new statements from the same distribution
    flagged = sum(bool(anomaly_findings(_ordinary(rng), _profile(5_000_000, 5_000_000), model))
                  for _ in range(50))  # fmt: skip
    assert flagged <= 5  # about 2% expected; 10% is a generous ceiling
    unusual = _statement([95_000_000, -100_000, 95_000_000])
    (f,) = anomaly_findings(unusual, _profile(95_000_000, 95_000_000), model)
    assert (f.code, f.group, f.strength) == ("ANOMALY_SCORE", "anomaly", "weak")
    assert "threshold" in f.reason


def test_no_model_or_mismatched_model_is_not_run(monkeypatch):
    s = _statement([5_000_000, -100_000])
    p = _profile(5_000_000, 5_000_000)
    monkeypatch.setattr(anomaly, "load_model", lambda: None)
    (f,) = anomaly_findings(s, p)
    assert (f.code, f.strength) == ("CHECK_NOT_RUN", "info") and "no model" in f.reason
    wrong = _tiny_model() | {"features": ["something_else"]}
    (f,) = anomaly_findings(s, p, wrong)
    assert (f.code, f.strength) == ("CHECK_NOT_RUN", "info") and "differ" in f.reason
