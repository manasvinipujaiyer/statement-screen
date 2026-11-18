"""Reliability and stress tests of the income profile, on DEV only.

    uv run python -m statementscreen.stress data/parsed_2000/ data/week1_n2000/

Each dev statement's PARSED rows go through patterns (Week 3 rules) and the profile, once
as they are (the baseline) and once after each perturbation, and every run is scored
against the truth with score_profile (this module is on the scoring side). The truth is
never perturbed: a 3-month view is scored against the full statement's true income.

Perturbations (perturb_*), applied to one statement's parsed rows and period, each with
its own random stream seeded from (SEED, perturbation number, statement number):

    drop_month     the rows of one random month are removed (period unchanged)
    cut_3_months   only the first 3 calendar months are kept (period shortened)
    typos          in TYPO_RATE of narrations, one letter of one name-like word (letters
                   only, 4+ long, not a channel or format word) swapped, dropped or doubled
    lower_case     every narration in lower case
    cut_20_chars   every narration cut to its first 20 characters
    shift_dates    every row moved by -3..+3 days, kept inside the period
    scale_amounts  every amount multiplied by 1.1 (rounded to the paisa)

Rows are renumbered 0..n-1 after a drop, in their original order. For each perturbation
the report gives every profile metric, the change against the baseline, and whether the
expected sign appears (e.g. a 3-month history must warn and must not be "normal"
confidence). Determinism: the baseline is built twice and its JSON must hash the same.
Robustness: an empty, a debits-only and a single-month statement (made from real dev
statements) must each return a profile with warnings. Writes reports/week5_stress.json
(the layout hold-out of step 5b adds its own section to the same file).
"""

import argparse
import calendar
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from statementscreen.patterns.cli import process_statement
from statementscreen.profile.build import profile_statement, to_json
from statementscreen.profile.cli import GROUP_COLUMNS, ROW_COLUMNS, RULE_COLUMNS
from statementscreen.profile.config import DEFAULT
from statementscreen.score_profile import metrics, statement_table, truth_table
from statementscreen.splits import SPLITS_DIR, SplitError, load_split
from statementscreen.summary import peak_memory_bytes

SEED = 42
CHUNK = 100
TYPO_RATE = 0.3
DEFAULT_REPORT = Path("reports/week5_stress.json")
PARSED_COLUMNS = ["statement_id", "row", "posted", "narration", "amount_paise"]
FORMAT_WORDS = {
    "NEFT", "RTGS", "IMPS", "NACH", "SALARY", "PAYOUT", "PAYMENT", "TRANSFER", "WEEKLY",
    "SETTLEMENT", "RENT", "HOUSE", "RECHARGE", "REFUND", "REVERSAL", "SELF", "NETBANK",
    "CHRG", "CHRGS", "CHGS", "INSUFF", "FUNDS", "INSUFFICIENT", "MANDATE", "BOUNCE",
    "CHARGES", "INCL", "CREDIT", "INTEREST", "CAPITALISED", "ADVANCE", "CHALLAN", "BILL",
    "BILLPAY", "LOAN", "RETURN", "ONLINE", "STOPPED", "BLOCKED", "REGISTERED",
}  # fmt: skip
NAME_LIKE = re.compile(r"(?<![A-Za-z0-9@])[A-Za-z]{4,}(?![A-Za-z0-9@])")


# ---------------------------------------------------------------- perturbations


Period = tuple[date | None, date | None]


def _month_end(d: date) -> date:
    return d.replace(day=calendar.monthrange(d.year, d.month)[1])


def _renumber(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.sort("row").with_columns(pl.int_range(pl.len()).cast(pl.Int32).alias("row"))


def perturb_drop_month(rows: pl.DataFrame, period: Period, rng) -> tuple[pl.DataFrame, Period]:
    months = sorted({(d.year, d.month) for d in rows["posted"]})
    if not months:
        return rows, period
    y, m = months[int(rng.integers(len(months)))]
    keep = ~((pl.col("posted").dt.year() == y) & (pl.col("posted").dt.month() == m))
    return _renumber(rows.filter(keep)), period


def perturb_cut_3_months(rows: pl.DataFrame, period: Period, rng) -> tuple[pl.DataFrame, Period]:
    start = period[0] or (min(rows["posted"]) if rows.height else None)
    if start is None:
        return rows, period
    y, m = start.year, start.month + 2
    y, m = (y + (m - 1) // 12, (m - 1) % 12 + 1)
    end = _month_end(date(y, m, 1))
    return _renumber(rows.filter(pl.col("posted") <= end)), (start, end)


def _typo(word: str, rng) -> str:
    j = int(rng.integers(1, len(word) - 1))
    op = int(rng.integers(3))
    if op == 0:
        return word[:j] + word[j + 1] + word[j] + word[j + 2 :]
    if op == 1:
        return word[:j] + word[j + 1 :]
    return word[:j] + word[j] + word[j:]


def misspell_a_name(narration: str, rng) -> str:
    """One letter of one name-like word: a whole word of 4+ letters, not touching a digit
    or '@' (so never part of a code, reference or UPI handle), and not a format word."""
    words = [m for m in NAME_LIKE.finditer(narration) if m.group().upper() not in FORMAT_WORDS]
    if not words:
        return narration
    m = words[int(rng.integers(len(words)))]
    return narration[: m.start()] + _typo(m.group(), rng) + narration[m.end() :]


def perturb_typos(rows: pl.DataFrame, period: Period, rng) -> tuple[pl.DataFrame, Period]:
    texts = [misspell_a_name(t, rng) if rng.random() < TYPO_RATE else t
             for t in rows["narration"]]  # fmt: skip
    return rows.with_columns(pl.Series("narration", texts, dtype=pl.String)), period


def perturb_lower_case(rows: pl.DataFrame, period: Period, rng) -> tuple[pl.DataFrame, Period]:
    return rows.with_columns(pl.col("narration").str.to_lowercase()), period


def perturb_cut_20_chars(rows: pl.DataFrame, period: Period, rng) -> tuple[pl.DataFrame, Period]:
    return rows.with_columns(pl.col("narration").str.slice(0, 20)), period


def perturb_shift_dates(rows: pl.DataFrame, period: Period, rng) -> tuple[pl.DataFrame, Period]:
    lo, hi = period
    shifted = []
    for d in rows["posted"]:
        new = d + timedelta(days=int(rng.integers(-3, 4)))
        if lo is not None:
            new = max(new, lo)
        if hi is not None:
            new = min(new, hi)
        shifted.append(new)
    return rows.with_columns(pl.Series("posted", shifted, dtype=pl.Date)), period


def perturb_scale_amounts(rows: pl.DataFrame, period: Period, rng) -> tuple[pl.DataFrame, Period]:
    scaled = (pl.col("amount_paise") * 11 / 10).round().cast(pl.Int64)
    return rows.with_columns(scaled.alias("amount_paise")), period


PERTURBATIONS: dict[str, Callable] = {
    "drop_month": perturb_drop_month,
    "cut_3_months": perturb_cut_3_months,
    "typos": perturb_typos,
    "lower_case": perturb_lower_case,
    "cut_20_chars": perturb_cut_20_chars,
    "shift_dates": perturb_shift_dates,
    "scale_amounts": perturb_scale_amounts,
}


def rng_for(perturbation: str, statement_id: str, seed: int = SEED) -> np.random.Generator:
    k = list(PERTURBATIONS).index(perturbation) + 1
    return np.random.default_rng([seed, k, int(statement_id.removeprefix("stmt_"))])


# ---------------------------------------------------------------- one statement


def _empty(columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict.fromkeys(columns, pl.Null))


def profile_rows(sid: str, rows: pl.DataFrame, period: Period, layout: str | None) -> dict:
    """Parsed rows of one statement -> patterns (Week 3 rules) -> profile."""
    start, end = period
    if rows.height == 0:
        return profile_statement(sid, _empty(ROW_COLUMNS), _empty(GROUP_COLUMNS),
                                 _empty(RULE_COLUMNS), None, start, end, DEFAULT)  # fmt: skip
    t = process_statement(rows.select(PARSED_COLUMNS), start, end, layout)
    result = t["statement_results"].row(0, named=True)
    return profile_statement(sid, t["rows"], t["group_features"], t["group_rules"], result,
                             start, end, DEFAULT)  # fmt: skip


def run_all(parsed: Path, ids: list[str], perturbation: str | None) -> dict[str, dict]:
    """Profiles of every statement, as parsed (None) or after one perturbation."""
    columns = ["statement_id", "layout", "period_start", "period_end"]
    statements = pl.read_parquet(parsed / "statements.parquet", columns=columns)
    in_ids = pl.col("statement_id").is_in(ids)
    meta = {r["statement_id"]: r for r in statements.filter(in_ids).iter_rows(named=True)}
    profiles = {}
    for k in range(0, len(ids), CHUNK):
        chunk = ids[k : k + CHUNK]
        in_chunk = pl.col("statement_id").is_in(chunk)
        rows = pl.scan_parquet(parsed / "rows.parquet").filter(in_chunk).select(PARSED_COLUMNS)
        rows = rows.collect()
        by_id = {key[0]: v for key, v in rows.partition_by("statement_id", as_dict=True).items()}
        for sid in chunk:
            m = meta[sid]
            stmt = by_id.get(sid, rows.clear())
            period = (m["period_start"], m["period_end"])
            if perturbation is not None:
                perturb = PERTURBATIONS[perturbation]
                stmt, period = perturb(stmt, period, rng_for(perturbation, sid))
            profiles[sid] = profile_rows(sid, stmt, period, m["layout"])
    return profiles


def fingerprint(profiles: dict[str, dict]) -> str:
    text = "\n".join(to_json(profiles[sid]) for sid in sorted(profiles))
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- expected signs


def expected_signs(name: str, profiles: dict[str, dict], base: dict[str, dict]) -> dict:
    """Did the expected warning or change appear? Counts over the dev statements."""

    def value(p: dict, finding: str):
        return p["findings"][finding]["value"]

    if name == "cut_3_months":
        warned = sum(any(w.startswith("history shorter than 6 months") for w in p["warnings"])
                     for p in profiles.values())  # fmt: skip
        normal = sum(value(p, "income_confidence") == "normal" for p in profiles.values())
        met = warned == len(profiles) and normal == 0
        return {
            "expected": "every statement warns of a short history; none is 'normal' confidence",
            "warned": warned,
            "normal_confidence": normal,
            "statements": len(profiles),
            "met": met,
        }
    if name == "drop_month":
        salaried = [s for s, p in base.items() if value(p, "has_salary")]
        shown = sum(bool(value(profiles[s], "missed_salary_months")) for s in salaried
                    if value(profiles[s], "has_salary"))  # fmt: skip
        still = sum(bool(value(profiles[s], "has_salary")) for s in salaried)
        return {"expected": "a salaried statement still has a salary and shows a missed month",
                "salaried_at_baseline": len(salaried), "still_has_salary": still,
                "shows_missed_month": shown}  # fmt: skip
    if name == "scale_amounts":
        both = [s for s in base if value(base[s], "estimated_monthly_income_paise")
                and value(profiles[s], "estimated_monthly_income_paise")]  # fmt: skip
        tracks = sum(abs(value(profiles[s], "estimated_monthly_income_paise")
                         / value(base[s], "estimated_monthly_income_paise") - 1.1) <= 0.011
                     for s in both)  # fmt: skip
        return {
            "expected": "every income estimate is 1.1 times the baseline's (within 1%)",
            "estimated_both": len(both),
            "tracks_1_1": tracks,
            "met": tracks == len(both),
        }
    return {"expected": "no particular warning; see the change in the metrics"}


# ---------------------------------------------------------------- robustness


def robustness(parsed: Path, ids: list[str]) -> dict:
    """An empty, a debits-only and a single-month statement, made from a real dev one."""
    sid = ids[0]
    m = (pl.read_parquet(parsed / "statements.parquet")
         .filter(pl.col("statement_id") == sid).row(0, named=True))  # fmt: skip
    rows = (pl.scan_parquet(parsed / "rows.parquet").filter(pl.col("statement_id") == sid)
            .select(PARSED_COLUMNS).collect())  # fmt: skip
    first = min(rows["posted"])
    one_month = rows.filter((pl.col("posted").dt.year() == first.year)
                            & (pl.col("posted").dt.month() == first.month))  # fmt: skip
    cases = {
        "empty": (rows.clear(), (m["period_start"], m["period_end"])),
        "debits_only": (_renumber(rows.filter(pl.col("amount_paise") < 0)),
                        (m["period_start"], m["period_end"])),
        "single_month": (_renumber(one_month), (first.replace(day=1), _month_end(first))),
    }  # fmt: skip
    out = {}
    for name, (stmt, period) in cases.items():
        try:
            p = profile_rows(sid, stmt, period, m["layout"])
            out[name] = {"ok": bool(p["warnings"]), "warnings": p["warnings"],
                         "rows": stmt.height}  # fmt: skip
        except Exception as e:  # reported, never hidden
            out[name] = {"ok": False, "error": repr(e), "rows": stmt.height}
    return {"made_from": sid, "cases": out, "all_ok": all(c["ok"] for c in out.values())}


# ---------------------------------------------------------------- the report


def _headline(m: dict) -> dict:
    inc = m["income"]["missing_as_100pct"]
    return {"income_median_ape": inc["median_ape"], "income_within_5pct": inc["within_5pct"],
            "income_within_10pct": inc["within_10pct"],
            "without_estimate": m["income"]["share_without_estimate"],
            "has_salary_precision": m["has_salary"]["precision"],
            "has_salary_recall": m["has_salary"]["recall"],
            "employer_exact": m["employer"]["exact_match_rate"],
            "employer_fuzzy": m["employer"]["fuzzy_match_rate"],
            "customer_type_accuracy": m["customer_type"]["accuracy"],
            "emi_count_exact": m["emi"]["count_exact_rate"],
            "bounce_count_exact": m["bounces"]["count_exact_rate"]}  # fmt: skip


def _change(now: dict, base: dict) -> dict:
    return {k: (now[k] - base[k] if now[k] is not None and base[k] is not None else None)
            for k in now}  # fmt: skip


def run(parsed: Path, batch: Path, splits_dir: Path = SPLITS_DIR) -> dict:
    started = time.perf_counter()
    ids = load_split("dev", batch / "manifest.csv", splits_dir)
    truth = truth_table(batch, ids)
    base = run_all(parsed, ids, None)
    again = run_all(parsed, ids, None)
    base_metrics = metrics(statement_table(base, truth))
    base_head = _headline(base_metrics)
    section = {}
    for name in PERTURBATIONS:
        t0 = time.perf_counter()
        profiles = run_all(parsed, ids, name)
        head = _headline(metrics(statement_table(profiles, truth)))
        section[name] = {"metrics": head, "change_vs_baseline": _change(head, base_head),
                         "expected": expected_signs(name, profiles, base),
                         "deterministic": fingerprint(profiles)
                         == fingerprint(run_all(parsed, ids, name)),
                         "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
        print(f"  {name}: done in {section[name]['seconds']} s", flush=True)
    return {
        "split": "dev",
        "statements": len(ids),
        "seed": SEED,
        "typo_rate": TYPO_RATE,
        "baseline": {"metrics": base_head,
                     "profile_json_sha256": fingerprint(base),
                     "byte_identical_twice": fingerprint(base) == fingerprint(again)},
        "perturbations": section,
        "robustness": robustness(parsed, ids),
        "profile_config_sha256": DEFAULT.sha256(),
        "seconds_total": round(time.perf_counter() - started, 1),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
    }  # fmt: skip


def _pct(x: float | None) -> str:
    return "  n/a  " if x is None else f"{x:7.2%}"


def _signed(x: float | None) -> str:
    return "   n/a  " if x is None else f"{x:+8.2%}"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.stress",
        description="Stress-test the income profile on dev.",
    )
    p.add_argument("parsed", type=Path, help="the parse of the reference batch")
    p.add_argument("batch", type=Path, help="the reference batch (truth, manifest)")
    p.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    args = p.parse_args(argv)
    try:
        report = run(args.parsed, args.batch, args.splits_dir)
    except (SplitError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    old = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    if "layout_holdout" in old:
        report["layout_holdout"] = old["layout_holdout"]  # step 5b's section is kept
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    b = report["baseline"]
    print(f"Wrote {args.out}: {report['statements']} dev statements in "
          f"{report['seconds_total']} s, peak memory {report['peak_memory_mb']} MB")  # fmt: skip
    print(f"  baseline profile JSON byte-identical twice: {b['byte_identical_twice']}")
    head = ("income<=10%", "has_sal R", "has_sal P", "employer", "type acc")
    keys = ("income_within_10pct", "has_salary_recall", "has_salary_precision",
            "employer_exact", "customer_type_accuracy")  # fmt: skip
    print(f"  {'':14}" + "".join(f"{h:>18}" for h in head))
    print(f"  {'baseline':14}" + "".join(f"{_pct(b['metrics'][k]):>18}" for k in keys))
    for name, s in report["perturbations"].items():
        cells = "".join(f"{_pct(s['metrics'][k])} {_signed(s['change_vs_baseline'][k])}"
                        for k in keys)  # fmt: skip
        print(f"  {name:14}{cells}  deterministic {s['deterministic']}")
        e = s["expected"]
        extra = {k: v for k, v in e.items() if k != "expected"}
        if extra:
            print(f"  {'':14}expected: {e['expected']} -> {extra}")
    r = report["robustness"]
    print(f"  robustness (from {r['made_from']}): " + "; ".join(
        f"{k} {'ok' if c['ok'] else 'FAILED'} {c.get('warnings', c.get('error'))}"
        for k, c in r["cases"].items()))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
