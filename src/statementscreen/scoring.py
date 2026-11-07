"""Compare parsed rows with the generator's ground truth.

This is the ONLY place that reads the truth (transactions.parquet, manifest.csv). The
parser never does.

Rows are aligned by (statement_id, row). For each aligned row:

- posted, amount_paise, balance_paise: exact equality.
- narration, three ways:
    exact       parsed == truth
    no_space    equal after removing all whitespace (wrapped lines lose or gain spaces)
    prefix      parsed (no whitespace) is the start of the truth (no whitespace): the
                layout cut the narration short. Counted as "cut short", not as an error.
- exact row: date, amount and balance exact AND narration passes no_space or prefix.
"""

from collections import Counter
from pathlib import Path

import polars as pl

TRUTH_COLUMNS = ["statement_id", "row", "posted", "description", "amount_paise", "balance_paise"]
PARSED_COLUMNS = ["statement_id", "row", "posted", "narration", "amount_paise", "balance_paise"]


def _squash(col: str) -> pl.Expr:
    return pl.col(col).str.replace_all(r"\s+", "")


def compare_rows(parsed: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """One row per (statement_id, row) present on either side, with a flag per check.

    A truth row with no parsed partner (or the reverse) gets all checks False.
    """
    t = truth.select(TRUTH_COLUMNS)
    p = parsed.select(PARSED_COLUMNS).rename(
        {"posted": "p_posted", "amount_paise": "p_amount", "balance_paise": "p_balance"}
    )
    joined = t.join(p, on=["statement_id", "row"], how="full", coalesce=True)
    both = pl.col("description").is_not_null() & pl.col("narration").is_not_null()
    no_space = _squash("narration") == _squash("description")
    prefix = (
        ~no_space
        & (pl.col("narration").str.len_chars() > 0)
        & _squash("description").str.starts_with(_squash("narration"))
    )
    out = joined.with_columns(
        in_truth=pl.col("description").is_not_null(),
        in_parsed=pl.col("narration").is_not_null(),
        posted_ok=both & (pl.col("posted") == pl.col("p_posted")),
        amount_ok=both & (pl.col("amount_paise") == pl.col("p_amount")),
        balance_ok=both & (pl.col("balance_paise") == pl.col("p_balance")),
        narration_exact=both & (pl.col("narration") == pl.col("description")),
        narration_no_space=both & no_space,
        narration_prefix=both & prefix,
    ).with_columns(
        exact_row=pl.col("posted_ok") & pl.col("amount_ok") & pl.col("balance_ok")
        & (pl.col("narration_no_space") | pl.col("narration_prefix"))
    )  # fmt: skip
    return out.with_columns(
        [pl.col(c).fill_null(False) for c in ("posted_ok", "amount_ok", "balance_ok",
         "narration_exact", "narration_no_space", "narration_prefix", "exact_row")]
    ).sort("statement_id", "row")  # fmt: skip


def per_statement(rows: pl.DataFrame) -> pl.DataFrame:
    """Row counts and exact rows per statement."""
    return (
        rows.group_by("statement_id")
        .agg(
            truth_rows=pl.col("in_truth").sum(),
            parsed_rows=pl.col("in_parsed").sum(),
            exact_rows=pl.col("exact_row").sum(),
        )
        .with_columns(row_count_match=pl.col("truth_rows") == pl.col("parsed_rows"))
        .sort("statement_id")
    )


# ---------------------------------------------------------------------- reading inputs

PARSED_FILES = {"rows": "rows.parquet", "statements": "statements.parquet",
                "issues": "issues.parquet"}  # fmt: skip


def load_parsed(parsed_dir: Path) -> dict[str, pl.DataFrame]:
    """The parse command's output: rows, statements (one per PDF tried) and issues."""
    return {name: pl.read_parquet(parsed_dir / file) for name, file in PARSED_FILES.items()}


def load_truth(batch_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(transactions, manifest) from a generated batch folder."""
    truth = pl.read_parquet(batch_dir / "transactions.parquet")
    manifest = pl.read_csv(batch_dir / "manifest.csv", schema_overrides={"seed": pl.String},
                           infer_schema=False)  # fmt: skip
    return truth, manifest


# ---------------------------------------------------------------------- the report

FIELDS = ("posted", "amount_paise", "balance_paise")
OK_COLUMN = {"posted": "posted_ok", "amount_paise": "amount_ok", "balance_paise": "balance_ok"}
PARSED_NAME = {"posted": "p_posted", "amount_paise": "p_amount", "balance_paise": "p_balance"}


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 6) if whole else None


def _failures(compared: pl.DataFrame, limit: int) -> tuple[int, list[dict]]:
    """(total, first ``limit``) field-level failures, in statement and row order."""
    found: list[dict] = []
    total = 0
    bad = compared.filter(~pl.col("exact_row"))
    for r in bad.iter_rows(named=True):
        if not r["in_parsed"] or not r["in_truth"]:
            problems = [("row", "present" if r["in_truth"] else "absent",
                         "present" if r["in_parsed"] else "missing")]  # fmt: skip
        else:
            problems = [(f, r[f], r[PARSED_NAME[f]]) for f in FIELDS if not r[OK_COLUMN[f]]]
            if not (r["narration_no_space"] or r["narration_prefix"]):
                problems.append(("narration", r["description"], r["narration"]))
        total += len(problems)
        for field, expected, got in problems:
            if len(found) < limit:
                found.append({"statement_id": r["statement_id"], "row": r["row"], "field": field,
                              "expected": _show(expected), "got": _show(got)})  # fmt: skip
    return total, found


def _show(value: object) -> str | int | None:
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)  # dates


def _row_stats(compared: pl.DataFrame) -> dict:
    truth_rows = int(compared["in_truth"].sum())
    aligned = int((compared["in_truth"] & compared["in_parsed"]).sum())
    no_space = int(compared["narration_no_space"].sum())
    prefix = int(compared["narration_prefix"].sum())
    return {
        "truth_rows": truth_rows,
        "parsed_rows": int(compared["in_parsed"].sum()),
        "aligned_rows": aligned,
        "exact_rows": int(compared["exact_row"].sum()),
        "exact_row_rate": _rate(int(compared["exact_row"].sum()), truth_rows),
        "field_accuracy": {f: _rate(int(compared[OK_COLUMN[f]].sum()), truth_rows) for f in FIELDS},
        "narration": {
            "exact": int(compared["narration_exact"].sum()),
            "equal_ignoring_whitespace": no_space,
            "cut_short_by_layout": prefix,
            "wrong": aligned - no_space - prefix,
            "exact_rate": _rate(int(compared["narration_exact"].sum()), truth_rows),
            "equal_ignoring_whitespace_rate": _rate(no_space, truth_rows),
            "cut_short_rate": _rate(prefix, truth_rows),
        },
    }


def build_report(
    parsed: dict[str, pl.DataFrame],
    truth: pl.DataFrame,
    manifest: pl.DataFrame,
    *,
    first_contact: dict | None = None,
    target: float = 0.995,
    max_failures: int = 50,
) -> dict:
    """Score the parsed statements against the truth for the same statements.

    Only statements the parse command tried (one entry each in ``statements``, even if
    unreadable) are scored. Manifest statements that weren't tried are counted, not
    hidden. Layout breakdowns use the manifest's (true) layout.
    """
    statements, rows, issues = parsed["statements"], parsed["rows"], parsed["issues"]
    tried = statements["statement_id"].to_list()
    layout_of = manifest.select("statement_id", pl.col("layout").alias("true_layout"))
    truth = truth.filter(pl.col("statement_id").is_in(tried))
    compared = compare_rows(rows, truth).join(layout_of, on="statement_id", how="left")

    issue_counts = issues.group_by("statement_id").len("n_issues")
    per = (
        per_statement(compared)
        .join(statements.select("statement_id", "status", pl.col("layout").alias("detected")),
              on="statement_id", how="full", coalesce=True)
        .join(layout_of, on="statement_id", how="left")
        .join(issue_counts, on="statement_id", how="left")
        .with_columns(pl.col(c).fill_null(0)
                      for c in ("n_issues", "truth_rows", "parsed_rows", "exact_rows"))
        .with_columns(
            row_count_match=pl.col("truth_rows") == pl.col("parsed_rows"),
            fully_clean=(pl.col("status") == "ok")
            & (pl.col("detected") == pl.col("true_layout"))
            & (pl.col("truth_rows") == pl.col("parsed_rows"))
            & (pl.col("exact_rows") == pl.col("truth_rows"))
            & (pl.col("n_issues") == 0),
        )
    )  # fmt: skip

    def statement_stats(p: pl.DataFrame) -> dict:
        n = p.height
        return {
            "statements": n,
            "row_count_match_rate": _rate(int(p["row_count_match"].sum()), n),
            "fully_clean_statements": int(p["fully_clean"].sum()),
            "fully_clean_rate": _rate(int(p["fully_clean"].sum()), n),
            "layout_detected_correctly": int((p["detected"] == p["true_layout"]).sum()),
        }

    by_layout = {}
    for layout in sorted(per["true_layout"].drop_nulls().unique().to_list()):
        stats = statement_stats(per.filter(pl.col("true_layout") == layout))
        stats |= _row_stats(compared.filter(pl.col("true_layout") == layout))
        stats["meets_target"] = (stats["exact_row_rate"] or 0.0) >= target
        by_layout[layout] = stats

    checks = issues.filter(pl.col("kind") == "check").sort("statement_id", "row", "code")
    parse_issues = issues.filter(pl.col("kind") == "parse").sort("statement_id", "row", "code")
    failure_total, failures = _failures(compared, max_failures)
    mismatched_counts = per.filter(~pl.col("row_count_match")).sort("statement_id")

    report = {
        "target_exact_row_rate": target,
        "overall": statement_stats(per) | _row_stats(compared),
        "by_layout": by_layout,
        "first_contact": first_contact,
        "parse_errors": statements.filter(pl.col("status") != "ok")
        .select("statement_id", "error").sort("statement_id").to_dicts(),
        "manifest_statements_not_parsed": manifest.filter(
            ~pl.col("statement_id").is_in(tried)).height,
        "statements_not_in_manifest": sorted(set(tried) - set(manifest["statement_id"])),
        "row_count_mismatches": mismatched_counts.select(
            "statement_id", "truth_rows", "parsed_rows").to_dicts(),
        # Clean synthetic PDFs always reconcile, so every self-check flag is a parser bug:
        # all of them are listed, never sampled.
        "self_check_flags": {
            "count": checks.height,
            "by_code": dict(sorted(Counter(checks["code"].to_list()).items())),
            "all": checks.drop("kind").to_dicts(),
        },
        "parse_issues": {
            "count": parse_issues.height,
            "by_code": dict(sorted(Counter(parse_issues["code"].to_list()).items())),
            "first_50": parse_issues.drop("kind").head(50).to_dicts(),
        },
        "failures": {"count": failure_total, "first": failures},
    }  # fmt: skip
    return report
