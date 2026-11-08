"""Recurring-payment features for each counterparty group of ONE statement.

Input: one statement's grouped rows (group.group_statement) and how many calendar months
the statement covers. Output: one row per group. Money stays integer paise; ratios and
day statistics are floats.

Day of month is compared on a 31-day circle, so the 31st and the 1st are 1 day apart:
    circular_day_distance(a, b) = min(|a - b|, 31 - |a - b|)
The group's usual day is the medoid (the observed day with the smallest total circular
distance to all the others), and its "arc" is the shortest stretch of the circle that
covers every observed day, e.g. 30 -> 1.
"""

import statistics
from collections import Counter
from datetime import date

import polars as pl

from statementscreen.patterns.config import DEFAULT, PatternConfig

DAYS = 31


def circular_day_distance(a: int, b: int) -> int:
    d = abs(a - b) % DAYS
    return min(d, DAYS - d)


def day_medoid(days: list[int]) -> int:
    """The observed day closest (circularly) to all the others; ties go to the earliest."""
    return min(sorted(set(days)), key=lambda c: sum(circular_day_distance(c, d) for d in days))


def day_arc(days: list[int]) -> tuple[int, int, int]:
    """(start, end, span): the shortest circular stretch covering every day.

    Days 30, 31 and 1 give (30, 1, 2): from the 30th round to the 1st, 2 days long."""
    unique = sorted(set(days))
    if len(unique) == 1:
        return unique[0], unique[0], 0
    gaps = [((unique[(i + 1) % len(unique)] - d) % DAYS, i) for i, d in enumerate(unique)]
    widest, i = max(gaps)  # the biggest empty stretch is what the arc leaves out
    return unique[(i + 1) % len(unique)], unique[i], DAYS - widest


def months_between(start: date, end: date) -> int:
    """Calendar months from start's month to end's month, inclusive."""
    return (end.year - start.year) * 12 + end.month - start.month + 1


def statement_months(period_start: date | None, period_end: date | None,
                     posted: list[date]) -> int:  # fmt: skip
    """Months the statement covers: its printed period if valid, else first to last row."""
    if period_start and period_end and period_start <= period_end:
        return months_between(period_start, period_end)
    return months_between(min(posted), max(posted)) if posted else 0


FEATURE_SCHEMA = {
    "statement_id": pl.String, "layout": pl.String, "group_id": pl.String,
    "group_name": pl.String, "direction": pl.String, "main_channel": pl.String,
    "n_channels": pl.Int32, "is_self": pl.Boolean, "n_txn": pl.Int32, "n_cut": pl.Int32,
    "total_paise": pl.Int64, "median_amount_paise": pl.Int64,
    "first_posted": pl.Date, "last_posted": pl.Date,
    "statement_months": pl.Int32, "months_present": pl.Int32, "month_coverage": pl.Float64,
    "months_missing_inside": pl.Int32, "txn_per_month": pl.Float64,
    "max_txn_in_a_month": pl.Int32,
    "gap_mean_days": pl.Float64, "gap_std_days": pl.Float64,
    "dom_medoid": pl.Int32, "dom_mean_deviation": pl.Float64, "dom_share_on_time": pl.Float64,
    "dom_arc_start": pl.Int32, "dom_arc_end": pl.Int32, "dom_arc_span": pl.Int32,
    "amount_cv": pl.Float64, "amount_share_near_median": pl.Float64,
}  # fmt: skip


def _one_group(rows: list[dict], months_in_statement: int, cfg: PatternConfig) -> dict:
    posted = sorted(r["posted"] for r in rows)
    amounts = [abs(r["amount_paise"]) for r in rows]
    per_month = Counter((d.year, d.month) for d in posted)
    first_m, last_m = min(per_month), max(per_month)
    span = (last_m[0] - first_m[0]) * 12 + last_m[1] - first_m[1] + 1
    gaps = [(b - a).days for a, b in zip(posted, posted[1:], strict=False)]
    days = [d.day for d in posted]
    medoid = day_medoid(days)
    deviations = [circular_day_distance(d, medoid) for d in days]
    arc_start, arc_end, arc_span = day_arc(days)
    median = statistics.median_low(amounts)
    mean = statistics.fmean(amounts)
    channels = Counter(r["channel"] for r in rows)
    return {
        "group_id": rows[0]["group_id"],
        "group_name": rows[0]["group_name"],
        "direction": rows[0]["direction"],
        "main_channel": min(channels, key=lambda c: (-channels[c], c)),
        "n_channels": len(channels),
        "is_self": all(r["is_self"] for r in rows),
        "n_txn": len(rows),
        "n_cut": sum(bool(r["counterparty_cut"]) for r in rows),
        "total_paise": sum(amounts),
        "median_amount_paise": median,
        "first_posted": posted[0],
        "last_posted": posted[-1],
        "statement_months": months_in_statement,
        "months_present": len(per_month),
        "month_coverage": len(per_month) / months_in_statement if months_in_statement else None,
        "months_missing_inside": span - len(per_month),
        "txn_per_month": len(rows) / len(per_month),
        "max_txn_in_a_month": max(per_month.values()),
        "gap_mean_days": statistics.fmean(gaps) if gaps else None,
        "gap_std_days": statistics.pstdev(gaps) if gaps else None,
        "dom_medoid": medoid,
        "dom_mean_deviation": statistics.fmean(deviations),
        "dom_share_on_time": sum(d <= cfg.day_tolerance for d in deviations) / len(days),
        "dom_arc_start": arc_start,
        "dom_arc_end": arc_end,
        "dom_arc_span": arc_span,
        "amount_cv": statistics.pstdev(amounts) / mean if mean else None,
        "amount_share_near_median": sum(
            abs(a - median) <= cfg.amount_tolerance * median for a in amounts
        )
        / len(amounts),  # fmt: skip
    }


def group_features(grouped: pl.DataFrame, months_in_statement: int, layout: str | None,
                   cfg: PatternConfig = DEFAULT) -> pl.DataFrame:  # fmt: skip
    """One row of features per group of one statement's grouped rows."""
    by_group: dict[str, list[dict]] = {}
    for r in grouped.sort("row").iter_rows(named=True):
        by_group.setdefault(r["group_id"], []).append(r)
    sid = grouped["statement_id"][0] if grouped.height else None
    records = [{"statement_id": sid, "layout": layout, **_one_group(rows, months_in_statement, cfg)}
               for _, rows in sorted(by_group.items())]  # fmt: skip
    return pl.DataFrame(records, schema=FEATURE_SCHEMA)
