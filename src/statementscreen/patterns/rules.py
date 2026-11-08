"""Explainable rules on ONE statement's grouped rows: salary, EMI, bounce, rent, employer.

Every flag carries a plain-language reason, and every group a rule considered carries a
note saying why the rule did or didn't fire (the first check that failed). All thresholds
come from PatternConfig.

A group's "core" is its rows within a band of the group's median amount, so a bonus or
another one-off amount never counts towards (or is flagged as) salary, EMI or rent.

salary  credit groups paid by NEFT/RTGS/IMPS/OTHER whose core is monthly and on time.
        Groups with no months in common, similar medians and the same usual day are tried
        together as one stream (an employer rename). Employer = the latest name of the
        stream with the largest core total.
EMI     NACH debit groups whose core (+-2%) is monthly and on time. A debit later returned
        is still an EMI.
bounce  returned EMIs (NACH/ACH return styles), their return charges, and the GST line
        straight after a return charge. UPI refunds are not bounces.
rent    debit transfers to one counterparty, monthly, the same amount, on time.
"""

import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date

import polars as pl
from rapidfuzz import fuzz

from statementscreen.patterns.config import DEFAULT, PatternConfig
from statementscreen.patterns.features import circular_day_distance, day_arc, day_medoid


def inr(paise: int) -> str:
    """Whole rupees with Indian digit grouping: 8002800 paise -> 'INR 80,028'."""
    digits = str(abs(paise) // 100)
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "INR " + ",".join(([head] if head else []) + groups + [tail])


@dataclass(frozen=True)
class Core:
    """A set of rows' core: those within `band` of their median amount."""

    rows: list[dict]
    n_outside: int  # rows of the group left out of the core
    median: int
    months: frozenset[tuple[int, int]]
    statement_months: int
    medoid: int
    on_time: int  # core rows within day_tolerance of the usual day
    arc: tuple[int, int, int]
    max_deviation: float  # largest |amount - median| / median in the core
    channel: str
    total: int
    last_posted: date

    @property
    def coverage(self) -> float:
        return len(self.months) / self.statement_months if self.statement_months else 0.0

    @property
    def per_month(self) -> float:
        return len(self.rows) / len(self.months)

    @property
    def on_time_share(self) -> float:
        return self.on_time / len(self.rows)


def core_of(rows: list[dict], band: float, statement_months: int, cfg: PatternConfig) -> Core:
    amounts = [abs(r["amount_paise"]) for r in rows]
    median = statistics.median_low(amounts)
    core = [r for r, a in zip(rows, amounts, strict=True) if abs(a - median) <= band * median]
    days = [r["posted"].day for r in core]
    medoid = day_medoid(days)
    channels = Counter(r["channel"] for r in core)
    return Core(
        rows=core,
        n_outside=len(rows) - len(core),
        median=median,
        months=frozenset((r["posted"].year, r["posted"].month) for r in core),
        statement_months=statement_months,
        medoid=medoid,
        on_time=sum(circular_day_distance(d, medoid) <= cfg.day_tolerance for d in days),
        arc=day_arc(days),
        max_deviation=(
            max(abs(abs(r["amount_paise"]) - median) for r in core) / median if median else 0.0
        ),  # fmt: skip
        channel=min(channels, key=lambda c: (-channels[c], c)),
        total=sum(abs(r["amount_paise"]) for r in core),
        last_posted=max(r["posted"] for r in core),
    )


def _failure(core: Core, min_months: int, min_coverage: float, max_per_month: float,
             min_on_time: float, cfg: PatternConfig) -> str | None:  # fmt: skip
    """The first check the core fails, in words, or None if it passes them all."""
    m, total = len(core.months), core.statement_months
    if m < min_months:
        return f"core in {m} of {total} months (needs {min_months})"
    if core.coverage < min_coverage:
        return f"core in {m} of {total} months ({core.coverage:.0%}, needs {min_coverage:.0%})"
    if core.per_month > max_per_month:
        return f"{core.per_month:.1f} per month (at most {max_per_month:g})"
    if core.on_time_share < min_on_time:
        return (f"only {core.on_time} of {len(core.rows)} within {cfg.day_tolerance} days of "
                f"day {core.medoid} (needs {min_on_time:.0%})")  # fmt: skip
    return None


def _days(core: Core, cfg: PatternConfig) -> str:
    start, end, _ = core.arc
    if core.on_time == len(core.rows):
        return f"paid on day {start}" if start == end else f"paid on day {start}-{end}"
    return (f"usually on day {core.medoid} ({core.on_time} of {len(core.rows)} within "
            f"{cfg.day_tolerance} days)")  # fmt: skip


def _describe(what: str, core: Core, cfg: PatternConfig) -> str:
    within = max(1, round(core.max_deviation * 100 + 0.4999)) if core.max_deviation else 0
    amount = (f"amount within {within}% of median {inr(core.median)}" if within
              else f"amount always {inr(core.median)}")  # fmt: skip
    return (f"monthly {core.channel} {what} in {len(core.months)} of {core.statement_months} "
            f"months, {amount}, {_days(core, cfg)}")  # fmt: skip


# ---------------------------------------------------------------- salary


@dataclass
class Stream:
    groups: list[str]
    core: Core
    name: str  # the latest group name in the stream
    reason: str


def _could_be_one_employer(a: Core, b: Core, cfg: PatternConfig) -> bool:
    return (not a.months & b.months
            and abs(a.median - b.median) <= cfg.rename_median_band * max(a.median, b.median)
            and circular_day_distance(a.medoid, b.medoid) <= cfg.day_tolerance)  # fmt: skip


def salary_streams(groups: dict[str, list[dict]], names: dict[str, str], months: int,
                   cfg: PatternConfig) -> tuple[list[Stream], dict[str, str]]:  # fmt: skip
    """Salary streams, and a note per credit group saying why it is or isn't salary."""
    notes: dict[str, str] = {}
    cores: dict[str, Core] = {}
    for gid, rows in groups.items():
        if rows[0]["direction"] != "credit":
            continue
        if all(r["is_self"] for r in rows):
            notes[gid] = "not salary: transfers from the account holder (self)"
            continue
        core = core_of(rows, cfg.salary_core_band, months, cfg)
        if core.channel not in cfg.salary_channels:
            notes[gid] = f"not salary: paid by {core.channel}"
            continue
        cores[gid] = core

    def check(core: Core) -> str | None:
        return _failure(core, cfg.salary_min_months, cfg.salary_min_coverage,
                        cfg.salary_max_per_month, cfg.salary_min_on_time, cfg)  # fmt: skip

    # Regular-looking groups that could be one employer under two names: link them.
    regular = sorted(g for g, c in cores.items()
                     if c.per_month <= cfg.salary_max_per_month
                     and c.on_time_share >= cfg.salary_min_on_time)  # fmt: skip
    parent = {g: g for g in regular}

    def root(g: str) -> str:
        while parent[g] != g:
            g = parent[g]
        return g

    for i, a in enumerate(regular):
        for b in regular[i + 1 :]:
            if _could_be_one_employer(cores[a], cores[b], cfg):
                parent[root(b)] = root(a)
    components: dict[str, list[str]] = {}
    for g in regular:
        components.setdefault(root(g), []).append(g)

    streams: list[Stream] = []
    in_stream: set[str] = set()
    for members in components.values():
        if len(members) < 2:
            continue
        union = core_of([r for g in members for r in groups[g]], cfg.salary_core_band, months,
                        cfg)  # fmt: skip
        if check(union) is None:
            by_time = sorted(members, key=lambda g: cores[g].last_posted)
            latest = names[by_time[-1]]
            reason = (_describe("credit", union, cfg) + "; name changed: "
                      + " -> ".join(f"'{names[g]}'" for g in by_time))  # fmt: skip
            streams.append(Stream(by_time, union, latest, reason))
            in_stream.update(members)
            for g in members:
                notes[g] = f"salary (one employer under {len(members)} names)"

    for gid, core in sorted(cores.items()):
        if gid in in_stream:
            continue
        failed = check(core)
        if failed:
            notes[gid] = f"not salary: {failed}"
            continue
        reason = _describe("credit", core, cfg)
        if core.n_outside:
            reason += (f"; {core.n_outside} other credit(s) more than "
                       f"{cfg.salary_core_band:.0%} from the median not counted")  # fmt: skip
        streams.append(Stream([gid], core, names[gid], reason))
        notes[gid] = "salary"
    return streams, notes


# ---------------------------------------------------------------- EMI and rent


def regular_debits(groups: dict[str, list[dict]], names: dict[str, str], months: int,
                   cfg: PatternConfig) -> tuple[dict, dict, dict, dict]:  # fmt: skip
    """(emi cores, emi notes, rent cores, rent notes) over the debit groups."""
    emi: dict[str, Core] = {}
    rent: dict[str, Core] = {}
    emi_notes: dict[str, str] = {}
    rent_notes: dict[str, str] = {}
    for gid, rows in groups.items():
        if rows[0]["direction"] != "debit":
            continue
        channel = Counter(r["channel"] for r in rows).most_common(1)[0][0]
        if channel in cfg.emi_channels:
            core = core_of(rows, cfg.emi_core_band, months, cfg)
            failed = _failure(core, cfg.emi_min_months, 0.0, cfg.emi_max_per_month,
                              cfg.emi_min_on_time, cfg)  # fmt: skip
            emi_notes[gid] = f"not EMI: {failed}" if failed else "EMI"
            if not failed:
                emi[gid] = core
        else:
            emi_notes[gid] = f"not EMI: paid by {channel}"
        if all(r["is_self"] for r in rows):
            rent_notes[gid] = "not rent: transfers to the account holder (self)"
            continue
        if channel not in cfg.rent_channels:
            rent_notes[gid] = f"not rent: paid by {channel}"
            continue
        core = core_of(rows, cfg.amount_tolerance, months, cfg)
        failed = _failure(core, cfg.rent_min_months, cfg.rent_min_coverage,
                          cfg.rent_max_per_month, cfg.rent_min_on_time, cfg)  # fmt: skip
        near = len(core.rows) / len(rows)
        if not failed and near < cfg.rent_min_near_median:
            failed = (f"only {len(core.rows)} of {len(rows)} payments within "
                      f"{cfg.amount_tolerance:.0%} of the median (needs "
                      f"{cfg.rent_min_near_median:.0%})")  # fmt: skip
        rent_notes[gid] = f"not rent: {failed}" if failed else "rent"
        if not failed:
            rent[gid] = core
    return emi, emi_notes, rent, rent_notes


# ---------------------------------------------------------------- one statement

FLAGS = ("is_salary", "is_emi", "is_bounce", "is_rent")
GROUP_RULES_SCHEMA = {
    "statement_id": pl.String, "group_id": pl.String, "group_name": pl.String,
    "direction": pl.String, "is_salary": pl.Boolean, "is_emi": pl.Boolean,
    "is_rent": pl.Boolean, "salary_note": pl.String, "emi_note": pl.String,
    "rent_note": pl.String,
}  # fmt: skip
STATEMENT_SCHEMA = {
    "statement_id": pl.String, "layout": pl.String, "has_salary": pl.Boolean,
    "employer": pl.String, "salary_groups": pl.String, "salary_median_paise": pl.Int64,
    "salary_reason": pl.String, "emi_groups": pl.Int32, "bounces": pl.Int32,
    "rent_groups": pl.Int32,
}  # fmt: skip


def apply_rules(
    grouped: pl.DataFrame, months: int, layout: str | None, cfg: PatternConfig = DEFAULT
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """(rows with flags and reasons, one row of notes per group, the statement's result)."""
    rows = grouped.sort("row").to_dicts()
    sid = rows[0]["statement_id"] if rows else None
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    names = {g: rs[0]["group_name"] for g, rs in groups.items()}

    streams, salary_notes = salary_streams(groups, names, months, cfg)
    emi, emi_notes, rent, rent_notes = regular_debits(groups, names, months, cfg)

    flags = {r["row"]: dict.fromkeys(FLAGS, False) for r in rows}
    reasons: dict[int, list[str]] = {r["row"]: [] for r in rows}

    def flag(row: int, name: str, reason: str) -> None:
        flags[row][name] = True
        reasons[row].append(reason)

    for s in streams:
        for r in s.core.rows:
            flag(r["row"], "is_salary", f"salary: {s.reason}")
    for gid, core in emi.items():
        reason = f"EMI to {names[gid]}: " + _describe("debit", core, cfg)
        for r in core.rows:
            flag(r["row"], "is_emi", reason)
    for gid, core in rent.items():
        reason = f"rent to {names[gid]}: " + _describe("transfer", core, cfg)
        if any("rent" in (r["purpose"] or "") for r in groups[gid]):
            reason += "; narration says rent"
        for r in core.rows:
            flag(r["row"], "is_rent", reason)

    bounce_rows = 0
    by_row = {r["row"]: r for r in rows}
    for r in rows:
        style = r["style"]
        if style in cfg.bounce_return_styles or style in cfg.bounce_charge_styles:
            what = "EMI returned" if style in cfg.bounce_return_styles else "return charge"
            linked = [g for g in emi if fuzz.token_sort_ratio(r["counterparty"], names[g])
                      >= cfg.lender_match]  # fmt: skip
            link = f" (EMI group {linked[0]})" if linked else ""
            note = f", {r['purpose']}" if r["purpose"] else ""
            flag(r["row"], "is_bounce", f"bounce: {what} by {r['counterparty']}{link}{note}")
            bounce_rows += 1
        elif style == cfg.bounce_tax_style:
            before = by_row.get(r["row"] - 1)
            if before is not None and before["style"] in cfg.bounce_charge_styles:
                reason = f"bounce: GST on the return charge in row {before['row']}"
                flag(r["row"], "is_bounce", reason)
                bounce_rows += 1

    out = grouped.sort("row").with_columns(
        *[pl.Series(f, [flags[r["row"]][f] for r in rows], dtype=pl.Boolean) for f in FLAGS],
        pl.Series("flag_reason", [" | ".join(reasons[r["row"]]) for r in rows], dtype=pl.String),
    )
    salary_groups = {g for s in streams for g in s.groups}
    group_rules = pl.DataFrame([{
        "statement_id": sid, "group_id": g, "group_name": names[g],
        "direction": groups[g][0]["direction"], "is_salary": g in salary_groups,
        "is_emi": g in emi, "is_rent": g in rent, "salary_note": salary_notes.get(g),
        "emi_note": emi_notes.get(g), "rent_note": rent_notes.get(g),
    } for g in sorted(groups)], schema=GROUP_RULES_SCHEMA)  # fmt: skip

    main = max(streams, key=lambda s: s.core.total) if streams else None
    why = None
    if main is None:
        # Explain the miss with the biggest credit group the salary rule actually checked.
        checked = [g for g, n in salary_notes.items() if n.startswith("not salary: core")
                   or n.startswith("not salary: only") or "per month" in n]  # fmt: skip
        if checked:
            best = max(checked, key=lambda g: sum(abs(r["amount_paise"]) for r in groups[g]))
            why = f"no salary; largest checked credit group '{names[best]}': {salary_notes[best]}"
        else:
            why = "no salary; no bank-transfer credit groups"
    result = {
        "statement_id": sid,
        "layout": layout,
        "has_salary": main is not None,
        "employer": main.name if main else None,
        "salary_groups": ",".join(sorted(salary_groups)) or None,
        "salary_median_paise": main.core.median if main else None,
        "salary_reason": main.reason if main else why,
        "emi_groups": len(emi),
        "bounces": bounce_rows,
        "rent_groups": len(rent),
    }
    return out, group_rules, result
