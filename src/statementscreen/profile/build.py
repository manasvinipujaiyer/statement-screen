"""One statement's income profile, from its pattern outputs alone.

Inputs (never transactions.parquet, labels, the manifest or generator code):
  rows         the patterns rows: cleaned rows with group, rule flags and style
  groups       the patterns group_features
  group_rules  the patterns group_rules (which groups the rules called salary/EMI/rent)
  result       the patterns statement_results row (has_salary, employer, salary_reason)
  period       the statement period the parser read (statements.parquet)

The income definition (README, "Income profile"): regular income that actually arrived
(salary, gig payouts, client and customer payments), divided by the number of calendar
months the statement covers. Bonuses, family money, self-transfers, refunds and interest
never count. Estimated as:
  salary         the credits the Week 3 rules flag as salary (which already leave out
                 bonuses and extra credits);
  other regular  credit groups that are not salary, not the holder's own transfers, not
  income         interest/refunds/charges (by channel), present in at least
                 recurring_min_months months (gig platforms, repeat clients);
  one-offs       only with count_one_off_credits_for_self_employed (variant b), for a
                 statement estimated as self-employed.
No estimate when there is neither salary nor other regular income.

Every finding is {"value": ..., "reason": "..."}, and the reason is never empty.
"""

import json
import statistics
from datetime import date

import polars as pl

from statementscreen.patterns.config import DEFAULT as PATTERNS_DEFAULT
from statementscreen.patterns.config import PatternConfig
from statementscreen.patterns.features import day_arc, day_medoid, statement_months
from statementscreen.patterns.rules import inr
from statementscreen.profile.config import DEFAULT, ProfileConfig

PROFILE_VERSION = 1
RENAMED = "salary (one employer under"


def _per_month(total: int, months: int) -> int:
    """Whole paise per month, rounded half up."""
    return (2 * total + months) // (2 * months) if months else 0


def _months_in(start: date, end: date) -> list[tuple[int, int]]:
    out, (y, m) = [], (start.year, start.month)
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _amounts(frame: pl.DataFrame) -> list[int]:
    return [abs(int(a)) for a in frame["amount_paise"]]


def _group_ids(group_rules: pl.DataFrame, flag: str) -> list[str]:
    """Sorted ids of the groups a rule fired on (is_salary, is_emi or is_rent)."""
    if group_rules.height == 0:
        return []
    return sorted(group_rules.filter(pl.col(flag))["group_id"])


def profile_statement(statement_id: str, rows: pl.DataFrame, groups: pl.DataFrame,
                      group_rules: pl.DataFrame, result: dict | None,
                      period_start: date | None, period_end: date | None,
                      cfg: ProfileConfig = DEFAULT,
                      patterns_cfg: PatternConfig = PATTERNS_DEFAULT) -> dict:  # fmt: skip
    findings: dict[str, dict] = {}
    warnings: list[str] = []

    def put(name: str, value, reason: str) -> None:
        findings[name] = {"value": value, "reason": reason}

    posted = rows["posted"].to_list() if rows.height else []
    months = statement_months(period_start, period_end, posted)
    if period_start and period_end and period_start <= period_end:
        month_keys = _months_in(period_start, period_end)
    elif posted:
        month_keys = _months_in(min(posted), max(posted))
    else:
        month_keys = []
    credits = rows.filter(pl.col("direction") == "credit") if rows.height else rows
    if rows.height == 0:
        warnings.append("no transactions in the statement")
    elif credits.height == 0:
        warnings.append("only debits: no money came into the account")
    if months < cfg.min_history_months:
        warnings.append(f"history shorter than {cfg.min_history_months} months "
                        f"({months} month{'s' if months != 1 else ''})")  # fmt: skip

    # ---------------------------------------------------------------- salary
    has_salary = bool(result and result.get("has_salary"))
    salary_reason = (result or {}).get("salary_reason") or "no salary: no transactions"
    put("has_salary", has_salary, salary_reason)
    employer = (result or {}).get("employer") if has_salary else None
    put("employer", employer, f"employer of record of the salary credits: {salary_reason}"
        if employer else "no salary detected, so no employer")  # fmt: skip
    salary = rows.filter(pl.col("is_salary")) if rows.height else rows
    renamed = group_rules.height > 0 and any(
        (note or "").startswith(RENAMED) for note in group_rules["salary_note"]
    )
    if renamed:
        warnings.append("employer name changed during the statement")
    if has_salary and salary.height:
        amounts = _amounts(salary)
        put("typical_monthly_salary_paise", int(statistics.median_low(amounts)),
            f"median of {len(amounts)} salary credits ({inr(min(amounts))} to "
            f"{inr(max(amounts))})")  # fmt: skip
        paid = {(d.year, d.month) for d in salary["posted"]}
        missed = [f"{y}-{m:02d}" for y, m in month_keys if (y, m) not in paid]
        put("salary_months_paid", {"paid": len(paid & set(month_keys)), "of": months},
            f"salary credited in {len(paid & set(month_keys))} of the statement's {months} "
            "months")  # fmt: skip
        put("missed_salary_months", missed,
            f"no salary credit in {', '.join(missed)}" if missed
            else "a salary credit in every month of the statement")  # fmt: skip
        days = [d.day for d in salary["posted"]]
        start, end, span = day_arc(days)
        put("payday_spread", {"usual_day": day_medoid(days), "from": start, "to": end,
                              "span_days": span},
            f"paid between day {start} and day {end} of the month ({span} days apart on a "
            "31-day circle)")  # fmt: skip
    else:
        for name in ("typical_monthly_salary_paise", "salary_months_paid",
                     "missed_salary_months", "payday_spread"):  # fmt: skip
            put(name, None, "no salary detected")

    # ---------------------------------------------------------------- other regular income
    salary_groups = set(_group_ids(group_rules, "is_salary"))
    regular = []
    credit_groups = groups.filter(pl.col("direction") == "credit") if groups.height else groups
    for g in credit_groups.sort("group_id").iter_rows(named=True):
        if (g["group_id"] in salary_groups or g["is_self"]
                or g["main_channel"] in cfg.not_income_channels
                or g["months_present"] < cfg.recurring_min_months):  # fmt: skip
            continue
        gap = g["gap_mean_days"]
        lo, hi = cfg.weekly_gap_days
        weekly = (g["txn_per_month"] >= cfg.weekly_min_per_month and gap is not None
                  and lo <= gap <= hi)  # fmt: skip
        kind = "weekly payouts" if weekly else "regular receipts"
        monthly = _per_month(int(g["total_paise"]), months)
        regular.append({
            "group_id": g["group_id"], "counterparty": g["group_name"], "kind": kind,
            "monthly_average_paise": monthly, "months_present": g["months_present"],
            "transactions": g["n_txn"], "channel": g["main_channel"],
            "reason": (f"{g['n_txn']} {g['main_channel']} credits from '{g['group_name']}' in "
                       f"{g['months_present']} of {months} months"
                       + (f", about every {gap:.0f} days" if weekly else "")
                       + f": {inr(monthly)} a month on average"),
        })  # fmt: skip
    regular.sort(key=lambda x: (-x["monthly_average_paise"], x["group_id"]))
    put("other_regular_income", regular,
        f"{len(regular)} recurring credit group(s) besides salary, present in at least "
        f"{cfg.recurring_min_months} months" if regular
        else "no recurring credit group besides salary")  # fmt: skip

    # ---------------------------------------------------------------- customer type
    if has_salary:
        ctype, why = "salaried", f"a monthly salary from '{employer}'"
    elif regular and regular[0]["kind"] == "weekly payouts":
        ctype, why = "gig", f"largest regular income is weekly payouts: {regular[0]['reason']}"
    elif regular:
        ctype, why = "self_employed", ("no salary; regular receipts from "
                                       f"{len(regular)} counterparty group(s), largest: "
                                       f"{regular[0]['reason']}")  # fmt: skip
    else:
        ctype, why = "unknown", "no salary and no recurring income to tell the type from"
    put("customer_type", ctype, why)

    # ---------------------------------------------------------------- income
    salary_total = sum(_amounts(salary)) if has_salary and salary.height else 0
    other_total = sum(int(g["total_paise"]) for g in credit_groups.iter_rows(named=True)
                      if g["group_id"] in {r["group_id"] for r in regular})  # fmt: skip
    parts = []
    if salary_total:
        parts.append(f"salary {inr(_per_month(salary_total, months))} a month "
                     f"({salary.height} credits over {months} months)")  # fmt: skip
    if other_total:
        parts.append(f"other regular income {inr(_per_month(other_total, months))} a month "
                     f"({len(regular)} group(s))")  # fmt: skip
    one_off_total = 0
    if cfg.count_one_off_credits_for_self_employed and ctype == "self_employed":
        known = list(salary_groups | {r["group_id"] for r in regular})
        never = list(cfg.not_income_channels)
        not_regular = ~pl.col("group_id").is_in(known)
        one_offs = credits.filter(
            not_regular & ~pl.col("is_self") & ~pl.col("channel").is_in(never)
        )
        one_off_total = sum(_amounts(one_offs))
        if one_off_total:
            parts.append(f"one-off customer credits {inr(_per_month(one_off_total, months))} "
                         f"a month ({one_offs.height} credits)")  # fmt: skip
    total = salary_total + other_total + one_off_total
    if months == 0 or total == 0:
        income = None
        warnings.append("no recurring income found")
        put("estimated_monthly_income_paise", None,
            "not estimated: no salary and no recurring income in the statement")  # fmt: skip
    else:
        income = _per_month(total, months)
        put("estimated_monthly_income_paise", income,
            f"{inr(income)} a month: " + " + ".join(parts) + f", over {months} months")  # fmt: skip
    if income is None:
        conf, conf_why = None, "no income estimate"
    elif months < cfg.min_history_months:
        conf, conf_why = "low", f"only {months} months of history"
    elif ctype == "unknown":
        conf, conf_why = "low", "customer type unknown"
    else:
        conf, conf_why = "normal", f"{months} months of history and a known income source"
    put("income_confidence", conf, conf_why)

    # ---------------------------------------------------------------- EMI, bounces, rent
    emi_groups = _group_ids(group_rules, "is_emi")
    emis = []
    for gid in emi_groups:
        paid = rows.filter((pl.col("group_id") == gid) & pl.col("is_emi"))
        emis.append({"lender": paid["group_name"][0],
                     "monthly_paise": int(statistics.median_low(_amounts(paid)))})  # fmt: skip
    emi_total = sum(e["monthly_paise"] for e in emis)
    emi_words = "; ".join(f"{e['lender']} {inr(e['monthly_paise'])}" for e in emis)
    put("emi_count", len(emis), f"{len(emis)} regular NACH debit group(s): {emi_words}"
        if emis else "no regular NACH debit found")  # fmt: skip
    put("emi_total_monthly_paise", emi_total,
        f"sum of each EMI's usual amount: {emi_words}" if emis else "no EMI")  # fmt: skip
    put("emi_lenders", [e["lender"] for e in emis],
        "names of the EMI groups" if emis else "no EMI")  # fmt: skip

    bounced = charges = returns = taxes = rows  # an empty statement has none of them
    if rows.height:
        bounced = rows.filter(pl.col("is_bounce"))
        charges = bounced.filter(pl.col("style").is_in(list(patterns_cfg.bounce_charge_styles)))
        returns = bounced.filter(pl.col("style").is_in(list(patterns_cfg.bounce_return_styles)))
        taxes = bounced.filter(pl.col("style") == patterns_cfg.bounce_tax_style)
    n_bounced = charges.height or returns.height
    penalty = sum(_amounts(charges)) + sum(_amounts(taxes))
    found = (f"{charges.height} return charge(s) and {returns.height} returned EMI(s) found; "
             "each bounce is counted once")  # fmt: skip
    put("bounced_emis", n_bounced, found if bounced.height else "no returned EMI or charge")
    put("bounce_charges_paise", penalty,
        f"{inr(penalty)}: {charges.height} return charge(s) plus {taxes.height} GST line(s)"
        if penalty else "no bounce charges")  # fmt: skip
    if n_bounced:
        warnings.append(f"{n_bounced} bounced EMI{'s' if n_bounced != 1 else ''}")

    rent_groups = _group_ids(group_rules, "is_rent")
    rents = []
    for gid in rent_groups:
        paid = rows.filter((pl.col("group_id") == gid) & pl.col("is_rent"))
        rents.append({"payee": paid["group_name"][0],
                      "monthly_paise": int(statistics.median_low(_amounts(paid)))})  # fmt: skip
    put("rent", rents or None,
        "; ".join(f"rent to '{r['payee']}' {inr(r['monthly_paise'])} a month" for r in rents)
        if rents else "no regular rent payment found")  # fmt: skip

    # ---------------------------------------------------------------- EMI to income
    if not income:
        put("emi_to_income_ratio", None, "not computed: no income estimate (zero or unknown)")
    else:
        ratio = round(emi_total / income, 4)
        why = f"{inr(emi_total)} of EMIs a month / {inr(income)} of income = {ratio:.0%}"
        put("emi_to_income_ratio", ratio, why)

    return {
        "statement_id": statement_id,
        "profile_version": PROFILE_VERSION,
        "statement_months": months,
        "period": [d.isoformat() if d else None for d in (period_start, period_end)],
        "findings": findings,
        "warnings": warnings,
    }


def to_json(profile: dict) -> str:
    """One line of JSON, keys sorted: byte-identical for the same profile."""
    return json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def flat(profile: dict) -> dict:
    """One flat row for profiles.parquet: scalar values as columns, the rest as JSON text."""
    out = {"statement_id": profile["statement_id"],
           "statement_months": profile["statement_months"],
           "warnings": " | ".join(profile["warnings"])}  # fmt: skip
    for name, f in profile["findings"].items():
        v = f["value"]
        out[name] = v if v is None or isinstance(v, bool | int | float | str) else json.dumps(
            v, sort_keys=True, ensure_ascii=False)  # fmt: skip
    return out
