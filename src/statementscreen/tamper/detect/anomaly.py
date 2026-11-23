"""Anomaly finding: an IsolationForest on per-statement money features. Weak.

    ANOMALY_SCORE  weak  the statement's money pattern is unusual compared with genuine
                         train statements (score above the threshold set on genuine dev
                         statements for a 2% false-alarm rate)

Features come from the parsed rows and the income profile only: never from metadata,
fonts, layout name or anything about how the file was made (Week 4's lesson: a model
must not learn "made by our generator"). The profile is built from the already parsed
statement with the same steps as profile_pdf (rows_frame -> process_statement ->
profile_statement), so the PDF is parsed once; the profile code is unchanged.

The model (models/week6_iforest.pkl, a pickle written by scripts/train_anomaly.py and
loaded only from this repository) is trained on genuine train statements only; its
threshold is set on genuine dev statements.
"""

import math
import pickle
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from statementscreen.parsing.models import ParsedStatement
from statementscreen.parsing.tables import rows_frame
from statementscreen.patterns.cli import process_statement
from statementscreen.profile.build import profile_statement
from statementscreen.profile.cli import GROUP_COLUMNS, ROW_COLUMNS, RULE_COLUMNS
from statementscreen.tamper.detect.findings import Finding

MODEL_PATH = Path(__file__).resolve().parents[4] / "models" / "week6_iforest.pkl"
FEATURES = (
    "income_to_debits",  # estimated income x months / total debits
    "salary_to_spending",  # typical monthly salary / average monthly debits (0: no salary)
    "credits_to_debits",  # total credits / total debits
    "round_100_share",  # share of amounts that are whole multiples of INR 100
    "round_1000_share",  # share of amounts that are whole multiples of INR 1,000
    "last_digit_chi2",  # how uneven the last rupee digits are (chi-square / n)
    "balance_volatility",  # std of balances / mean absolute balance
    "largest_to_median_credit",  # largest credit / median credit
    "credits_per_month",  # number of credits / statement months
)


def _empty(columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict.fromkeys(columns, pl.Null))


def profile_of(parsed: ParsedStatement) -> dict:
    """The income profile of an already parsed statement (profile_pdf's steps, unchanged)."""
    sid, start, end = parsed.statement_id, parsed.header.period_start, parsed.header.period_end
    rows = rows_frame(parsed)
    if parsed.status != "ok" or rows.height == 0:
        empty = (_empty(ROW_COLUMNS), _empty(GROUP_COLUMNS), _empty(RULE_COLUMNS))
        return profile_statement(sid, *empty, None, start, end)
    stmt = rows.select("statement_id", "row", "posted", "narration", "amount_paise")
    t = process_statement(stmt, start, end, parsed.layout)
    result = t["statement_results"].row(0, named=True)
    return profile_statement(sid, t["rows"], t["group_features"], t["group_rules"], result,
                             start, end)  # fmt: skip


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def features(parsed: ParsedStatement, profile: dict) -> dict[str, float] | None:
    """The FEATURES of one statement, or None when there are no rows to measure."""
    rows = parsed.rows
    if parsed.status != "ok" or not rows:
        return None
    amounts = [abs(r.amount_paise) for r in rows]
    credits = [r.amount_paise for r in rows if r.amount_paise > 0]
    debits = -sum(r.amount_paise for r in rows if r.amount_paise < 0)
    months = max(int(profile["statement_months"] or 0), 1)
    f = profile["findings"]
    income = f["estimated_monthly_income_paise"]["value"] or 0
    salary = f["typical_monthly_salary_paise"]["value"] or 0
    digits = [(a // 100) % 10 for a in amounts]  # last whole-rupee digit
    expected = len(digits) / 10
    chi2 = sum((digits.count(d) - expected) ** 2 / expected for d in range(10))
    balances = [r.balance_paise for r in rows]
    mean_abs = statistics.fmean(abs(b) for b in balances)
    spread = _ratio(max(credits), statistics.median(credits)) if credits else 0.0
    return {
        "income_to_debits": _ratio(income * months, debits),
        "salary_to_spending": _ratio(salary, debits / months),
        "credits_to_debits": _ratio(sum(credits), debits),
        "round_100_share": sum(a % 10_000 == 0 for a in amounts) / len(amounts),
        "round_1000_share": sum(a % 100_000 == 0 for a in amounts) / len(amounts),
        "last_digit_chi2": chi2 / len(digits),
        "balance_volatility": _ratio(statistics.pstdev(balances), mean_abs),
        "largest_to_median_credit": spread,
        "credits_per_month": len(credits) / months,
    }


def vector(feats: dict[str, float]) -> np.ndarray:
    """Features in FEATURES order; log1p of each (they are non-negative ratios and counts),
    so a few very large ratios don't dominate."""
    return np.array([[math.log1p(max(feats[k], 0.0)) for k in FEATURES]])


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict | None:
    """{"model", "threshold", "features", ...} as saved by train_anomaly.py, or None.
    Only ever loads this repository's own file (a pickle can run code when loaded)."""
    if not Path(path).is_file():
        return None
    with Path(path).open("rb") as f:
        return pickle.load(f)


def score(model: dict, feats: dict[str, float]) -> float:
    """Higher = more unusual (IsolationForest's score_samples, negated)."""
    return float(-model["model"].score_samples(vector(feats))[0])


def anomaly_findings(parsed: ParsedStatement, profile: dict | None = None,
                     model: dict | None = None) -> list[Finding]:  # fmt: skip
    model = model if model is not None else load_model()
    feats = None
    if model is None:
        not_run = f"no model at {MODEL_PATH.name}"
    elif tuple(model["features"]) != FEATURES:
        not_run = "the saved model's features differ from this code's"
    else:
        feats = features(parsed, profile if profile is not None else profile_of(parsed))
        not_run = "no parsed transactions to measure"
    if model is None or feats is None:
        return [Finding("CHECK_NOT_RUN", "anomaly", None, None, "info",
                        f"ANOMALY_SCORE: {not_run}")]  # fmt: skip
    s = score(model, feats)
    if s <= model["threshold"]:
        return []
    return [Finding("ANOMALY_SCORE", "anomaly", None, None, "weak",
                    f"the statement's money pattern is unusual (score {s:.4f} above the "
                    f"threshold {model['threshold']:.4f}, set for 2% false alarms on genuine "
                    "dev statements)")]  # fmt: skip
