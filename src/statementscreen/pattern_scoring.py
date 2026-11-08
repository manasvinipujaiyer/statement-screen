"""Scoring for Week 3 pattern detection. Reads the ground truth; pattern detection never
imports this module.

Pairwise grouping metrics: every pair of rows in the same statement is either grouped
together or not, by us and by the truth (canonical_counterparty). Pairs are counted from
group sizes, so no pair list or matrix is ever built:

    precision = pairs together in both / pairs we put together
    recall    = pairs together in both / pairs the truth puts together

Rows whose truth canonical_counterparty is null are left out of grouping scores.

Everything else is counted once per statement (score_statement) and then summed over any
set of statements (aggregate), so the overall numbers and every breakdown (layout,
customer type, hard-case tag) are computed the same way.
"""

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

FLAGS = ("is_salary", "is_emi", "is_bounce", "is_rent")
EMPLOYER_FUZZY = 90.0  # RapidFuzz token_sort_ratio for a fuzzy employer match
_LEGAL = {"pvt", "private", "ltd", "limited", "llp", "p", "co", "company", "and"}


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


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def normalise_employer(name: str | None) -> str:
    """For comparing employers only: lowercase words without punctuation or legal form.
    Deliberately separate from the detector's own cleaning."""
    if not name:
        return ""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower().replace("&", " and ")).split()
    return " ".join(w for w in words if w not in _LEGAL)


# ---------------------------------------------------------------- per statement


@dataclass
class StatementScore:
    statement_id: str
    layout: str
    customer_type: str
    tags: list[str]
    pairs: PairCounts = field(default_factory=PairCounts)
    salary_rows: int = 0  # true salary rows
    salary_rows_in_main_group: int = 0  # ... in the predicted group holding most of them
    tx: dict[str, list[int]] = field(default_factory=lambda: {f: [0, 0, 0] for f in FLAGS})
    true_salary: bool = False
    predicted_salary: bool = False
    true_employer: str | None = None
    predicted_employer: str | None = None
    employer_exact: bool = False
    employer_fuzzy: bool = False


def score_statement(
    rows: pl.DataFrame, result: dict, label: dict, manifest: dict
) -> StatementScore:
    """rows: our rows joined to the truth's (columns t_<flag>, canonical_counterparty)."""
    s = StatementScore(result["statement_id"], manifest["layout"], manifest["customer_type"],
                       [t for t in (manifest["hard_case_tags"] or "").split("|") if t])  # fmt: skip
    known = rows.filter(pl.col("canonical_counterparty").is_not_null())
    s.pairs.add(known["group_id"].to_list(), known["canonical_counterparty"].to_list())
    salary_groups = rows.filter(pl.col("t_is_salary"))["group_id"].to_list()
    s.salary_rows = len(salary_groups)
    if salary_groups:
        s.salary_rows_in_main_group = Counter(salary_groups).most_common(1)[0][1]
    for f in FLAGS:
        ours, truth = rows[f], rows[f"t_{f}"]
        s.tx[f] = [int((ours & truth).sum()), int((ours & ~truth).sum()),
                   int((~ours & truth).sum())]  # fmt: skip
    s.true_employer = label.get("true_employer")
    s.true_salary = s.true_employer is not None
    s.predicted_salary = bool(result["has_salary"])
    s.predicted_employer = result["employer"]
    if s.true_salary and s.predicted_employer:
        a, b = normalise_employer(s.true_employer), normalise_employer(s.predicted_employer)
        s.employer_exact = a == b
        s.employer_fuzzy = fuzz.token_sort_ratio(a, b) >= EMPLOYER_FUZZY
    return s


def aggregate(scores: Iterable[StatementScore]) -> dict:
    scores = list(scores)
    pairs = PairCounts()
    tx = {f: [0, 0, 0] for f in FLAGS}
    for s in scores:
        pairs.both += s.pairs.both
        pairs.predicted += s.pairs.predicted
        pairs.truth += s.pairs.truth
        for f in FLAGS:
            tx[f] = [a + b for a, b in zip(tx[f], s.tx[f], strict=True)]
    salary_rows = sum(s.salary_rows for s in scores)
    with_employer = [s for s in scores if s.true_salary]
    exact = sum(s.employer_exact for s in with_employer)
    fuzzy = sum(s.employer_fuzzy for s in with_employer)
    hs = prf(sum(s.true_salary and s.predicted_salary for s in scores),
             sum(not s.true_salary and s.predicted_salary for s in scores),
             sum(s.true_salary and not s.predicted_salary for s in scores))  # fmt: skip

    def rate(n: int, d: int) -> float | None:
        return n / d if d else None

    return {
        "statements": len(scores),
        "grouping": {
            **pairs.scores(),
            "employer_recall": rate(sum(s.salary_rows_in_main_group for s in scores), salary_rows),
            "true_salary_rows": salary_rows,
        },
        "transactions": {f: prf(*tx[f]) for f in FLAGS},
        "has_salary": hs,
        "employer": {
            "statements_with_true_employer": len(with_employer),
            "exact_match_rate": rate(exact, len(with_employer)),
            "fuzzy_match_rate": rate(fuzzy, len(with_employer)),
            "fuzzy_threshold": EMPLOYER_FUZZY,
        },
    }


def false_salary(scores: Sequence[StatementScore]) -> dict:
    out = {}
    for kind in ("gig", "self_employed"):
        group = [s for s in scores if s.customer_type == kind]
        flagged = [s for s in group if s.predicted_salary]
        out[kind] = {
            "statements": len(group),
            "flagged_has_salary": len(flagged),
            "rate": len(flagged) / len(group) if group else None,
            "salary_rows_flagged": sum(s.tx["is_salary"][1] for s in group),
            "statement_ids": [s.statement_id for s in flagged],
        }
    return out


def breakdowns(scores: Sequence[StatementScore]) -> dict:
    def by(key) -> dict:
        keys = sorted({k for s in scores for k in key(s)})
        return {k: aggregate(s for s in scores if k in key(s)) for k in keys}

    return {
        "by_layout": by(lambda s: [s.layout]),
        "by_customer_type": by(lambda s: [s.customer_type]),
        "by_hard_case": by(lambda s: s.tags or ["none"]),
    }


# ---------------------------------------------------------------- examples

NOTE_FOR = {"is_salary": "salary_note", "is_emi": "emi_note", "is_rent": "rent_note"}


def _spread(items: list[dict], limit: int) -> list[dict]:
    """Up to `limit` items, taken in turn from each flag so no flag crowds out the rest."""
    by_flag: dict[str, list[dict]] = {}
    for item in items:
        by_flag.setdefault(item["flag"], []).append(item)
    out: list[dict] = []
    while len(out) < limit and any(by_flag.values()):
        for flag in FLAGS:
            if by_flag.get(flag) and len(out) < limit:
                out.append(by_flag[flag].pop(0))
    return sorted(out, key=lambda i: (i["flag"], i["statement_id"], i["row"]))


def examples(joined: pl.DataFrame, limit: int = 50) -> tuple[list[dict], list[dict], dict]:
    """(misses, false flags, counts) with the rule's reason or why it didn't fire."""
    misses, false_flags = [], []
    counts = {}
    cols = ["statement_id", "row", "narration", "amount_paise", "group_id", "group_name",
            "style", "category", "canonical_counterparty", "flag_reason",
            "salary_note", "emi_note", "rent_note"]  # fmt: skip
    for f in FLAGS:
        miss = joined.filter(pl.col(f"t_{f}") & ~pl.col(f)).select(cols)
        false = joined.filter(pl.col(f) & ~pl.col(f"t_{f}")).select(cols)
        amounts = false["amount_paise"].abs()
        true_amounts = joined.filter(pl.col(f"t_{f}"))["amount_paise"].abs()
        counts[f] = {
            "true_amount_paise": {
                "min": true_amounts.min(),
                "p01": true_amounts.quantile(0.01),
                "p10": true_amounts.quantile(0.1),
                "median": true_amounts.median(),
            }
            if true_amounts.len()
            else None,  # fmt: skip
            "misses": miss.height,
            "false_flags": false.height,
            "misses_by_true_category": dict(Counter(miss["category"].to_list()).most_common()),
            "false_flags_by_true_category": dict(
                Counter(false["category"].to_list()).most_common()
            ),  # fmt: skip
            "false_flag_amount_paise": {
                "min": amounts.min(),
                "median": amounts.median(),
                "p90": amounts.quantile(0.9),
                "max": amounts.max(),
            }
            if false.height
            else None,  # fmt: skip
        }
        for r in miss.iter_rows(named=True):
            if f == "is_salary" and "not counted as salary" in (r["flag_reason"] or ""):
                why = r["flag_reason"]  # in a salary group, but this credit was left out
            elif f in NOTE_FOR:
                why = r[NOTE_FOR[f]]
            else:
                why = f"style '{r['style']}' is not a bounce style"
            misses.append({"flag": f, "statement_id": r["statement_id"], "row": r["row"],
                           "narration": r["narration"], "amount_paise": r["amount_paise"],
                           "group": f"{r['group_id']} '{r['group_name']}'",
                           "why_not": why or "rule did not consider this group",
                           "truth": {"category": r["category"],
                                     "counterparty": r["canonical_counterparty"]}})  # fmt: skip
        for r in false.iter_rows(named=True):
            truth = {"category": r["category"], "counterparty": r["canonical_counterparty"]}
            false_flags.append({"flag": f, "statement_id": r["statement_id"], "row": r["row"],
                                "narration": r["narration"], "amount_paise": r["amount_paise"],
                                "reason": r["flag_reason"], "truth": truth})  # fmt: skip
    return _spread(misses, limit), _spread(false_flags, limit), counts


def statement_errors(scores: Sequence[StatementScore], results: dict[str, dict],
                     limit: int = 50) -> list[dict]:  # fmt: skip
    """has_salary misses and false alarms, and employer mismatches, with the rule's reason."""
    out = []
    for s in scores:
        if s.true_salary == s.predicted_salary and (not s.true_salary or s.employer_fuzzy):
            continue
        kind = ("has_salary missed" if s.true_salary and not s.predicted_salary
                else "has_salary false alarm" if s.predicted_salary and not s.true_salary
                else "employer mismatch")  # fmt: skip
        out.append({"kind": kind, "statement_id": s.statement_id,
                    "customer_type": s.customer_type, "hard_cases": s.tags,
                    "true_employer": s.true_employer, "predicted_employer": s.predicted_employer,
                    "salary_reason": results[s.statement_id]["salary_reason"]})  # fmt: skip
    return out[:limit]


# ---------------------------------------------------------------- the report


@dataclass
class Truth:
    """The ground truth for a set of statements, loaded once."""

    rows: pl.DataFrame  # statement_id, row, canonical_counterparty, category, t_<flag>
    labels: dict[str, dict]
    manifest: dict[str, dict]


def load_truth(batch: Path, ids: list[str]) -> Truth:
    rows = (pl.scan_parquet(batch / "transactions.parquet")
            .filter(pl.col("statement_id").is_in(ids))
            .select("statement_id", pl.col("row").cast(pl.Int64), "canonical_counterparty",
                    "category", *[pl.col(f).alias(f"t_{f}") for f in FLAGS[:3]],
                    (pl.col("category") == "rent").alias("t_is_rent"))
            .collect())  # fmt: skip
    labels = {sid: json.loads((batch / "labels" / f"{sid}.json").read_text(encoding="utf-8"))
              for sid in ids}  # fmt: skip
    manifest = {r["statement_id"]: r for r in pl.read_csv(
        batch / "manifest.csv", infer_schema=False).iter_rows(named=True) if r["statement_id"]
        in labels}  # fmt: skip
    return Truth(rows, labels, manifest)


def score_frames(
    rows: pl.DataFrame, group_rules: pl.DataFrame, results_df: pl.DataFrame, truth: Truth
) -> tuple[pl.DataFrame, list[StatementScore], dict[str, dict]]:
    """Join the pattern output to the truth and score every statement."""
    notes = group_rules.select("statement_id", "group_id", "salary_note", "emi_note",
                               "rent_note")  # fmt: skip
    joined = (rows.join(truth.rows, on=["statement_id", "row"], how="left")
              .join(notes, on=["statement_id", "group_id"], how="left")
              .with_columns([pl.col(f"t_{f}").fill_null(False) for f in FLAGS]))  # fmt: skip
    results = {r["statement_id"]: r for r in results_df.iter_rows(named=True)}
    scores = [score_statement(stmt, results[sid], truth.labels[sid], truth.manifest[sid])
              for (sid,), stmt in joined.group_by("statement_id", maintain_order=True)]  # fmt: skip
    scores.sort(key=lambda s: s.statement_id)
    return joined, scores, results


def build_report(patterns_dir: Path, batch: Path, ids: list[str]) -> dict:
    rows = pl.read_parquet(patterns_dir / "rows.parquet")
    group_rules = pl.read_parquet(patterns_dir / "group_rules.parquet")
    results_df = pl.read_parquet(patterns_dir / "statement_results.parquet")
    found = set(results_df["statement_id"])
    if found != set(ids):
        raise ValueError(f"{patterns_dir} holds {len(found)} statements but the split has "
                         f"{len(ids)}; run patterns on this split first")  # fmt: skip
    joined, scores, results = score_frames(rows, group_rules, results_df, load_truth(batch, ids))
    unmatched = joined.filter(pl.col("category").is_null()).height
    misses, false_flags, counts = examples(joined)
    return {
        "statements": len(ids),
        "rows": rows.height,
        "rows_without_truth": unmatched,
        "overall": aggregate(scores),
        "false_salary_on_non_salaried": false_salary(scores),
        "breakdown": breakdowns(scores),
        "example_counts": counts,
        "misses": misses,
        "false_flags": false_flags,
        "statement_errors": statement_errors(scores, results),
    }
