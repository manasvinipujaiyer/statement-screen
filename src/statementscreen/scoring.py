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
