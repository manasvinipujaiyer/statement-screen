"""Show what narrations look like in the TRAIN split of a parse, grouped by their opening.

Usage:
    uv run python scripts/train_narrations.py data/parsed_2000/
    uv run python scripts/train_narrations.py data/parsed_2000/ --shapes 50 --transfers 40

Reads only the parser output (rows.parquet, statements.parquet) and splits/split_v1.*,
never the ground truth. Loads train only; it can't open dev or test. For each layout it
prints the most common narration "shapes" (first word plus a short second word, e.g.
"UPI/DR" or "NEFT CR"), with credit/debit counts and examples, then every bank-transfer
narration of a couple of statements so we can see where the holder's own name appears.

This is an inspection tool only: it doesn't clean or detect anything.
"""

import argparse
import re
from pathlib import Path

import polars as pl

from statementscreen.splits import load_split_for_parsed

TRANSFER = ("NEFT", "IMPS", "MMT", "RTGS", "TRF", "TRANSFER", "IB", "INB")


def shape(narration: str) -> str:
    tokens = [t for t in re.split(r"[/\-: ]+", narration.strip()) if t]
    if not tokens:
        return "(empty)"
    first = tokens[0].upper()
    sep = re.match(r"[A-Za-z0-9]+([/\-: ])?", narration.strip())
    sep_char = sep.group(1) if sep and sep.group(1) else ""
    second = tokens[1].upper() if len(tokens) > 1 else ""
    if first == "UPI" and second not in ("DR", "CR", "REV"):
        return f"{first}{sep_char}"  # "UPI-<name>-...": the name isn't a shape
    if second.isalpha() and len(second) <= 4:
        return f"{first}{sep_char}{second}"
    return f"{first}{sep_char}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parsed", type=Path)
    p.add_argument("--shapes", type=int, default=30, help="shapes to list per layout")
    p.add_argument("--examples", type=int, default=2, help="examples per shape")
    p.add_argument("--transfers", type=int, default=25, help="transfer rows per statement")
    p.add_argument("--seed", type=int, default=0, help="which examples to show")
    p.add_argument("--grep", help="only print train rows matching this regex (case-insensitive)")
    p.add_argument("--limit", type=int, default=60, help="--grep rows per layout")
    args = p.parse_args()

    parsed_ids = pl.read_parquet(args.parsed / "statements.parquet")["statement_id"]
    train = load_split_for_parsed("train", parsed_ids.to_list())
    rows = (
        pl.scan_parquet(args.parsed / "rows.parquet")
        .filter(pl.col("statement_id").is_in(train))
        .select("statement_id", "row", "layout", "narration", "amount_paise")
        .collect()
        .with_columns(
            pl.col("narration").map_elements(shape, return_dtype=pl.String).alias("shape"),
            pl.when(pl.col("amount_paise") > 0).then(pl.lit("CR")).otherwise(pl.lit("DR"))
            .alias("dir"),
        )
    )  # fmt: skip
    print(f"train statements {len(train)}, rows {rows.height}")

    if args.grep:
        hits = rows.filter(pl.col("narration").str.contains(f"(?i){args.grep}"))
        print(f"rows matching /{args.grep}/: {hits.height}")
        for layout in sorted(hits["layout"].unique().to_list()):
            lh = hits.filter(pl.col("layout") == layout)
            print(f"\n===== {layout}: {lh.height} matches, by shape =====")
            for s in lh.group_by("shape").len().sort("len", descending=True).iter_rows():
                print(f"   {s[1]:>6}  {s[0]}")
            print(f"\n--- {layout}: first {args.limit} matches, with the row before and after ---")
            for ex in lh.sample(fraction=1.0, shuffle=True, seed=args.seed).head(
                args.limit
            ).iter_rows(named=True):  # fmt: skip
                around = rows.filter((pl.col("statement_id") == ex["statement_id"])
                                     & pl.col("row").is_between(ex["row"] - 1, ex["row"] + 1)
                                     ).sort("row")  # fmt: skip
                for r in around.iter_rows(named=True):
                    mark = ">" if r["row"] == ex["row"] else " "
                    print(f" {mark} {r['statement_id']} {r['row']:>4} {r['dir']} "
                          f"{r['amount_paise'] / 100:>12,.2f}  {r['narration']}")  # fmt: skip
                print()
        return

    for layout in sorted(rows["layout"].unique().to_list()):
        lr = rows.filter(pl.col("layout") == layout).sample(fraction=1.0, shuffle=True,
                                                             seed=args.seed)  # fmt: skip
        counts = (
            lr.group_by("shape")
            .agg(pl.len().alias("n"), (pl.col("dir") == "CR").sum().alias("cr"))
            .sort("n", descending=True)
        )
        print(f"\n===== {layout}: {lr.height} rows, {counts.height} shapes =====")
        for s in counts.head(args.shapes).iter_rows(named=True):
            print(f"\n[{s['shape']}]  n={s['n']}  CR={s['cr']}  DR={s['n'] - s['cr']}")
            for ex in lr.filter(pl.col("shape") == s["shape"]).head(args.examples).iter_rows(
                named=True
            ):  # fmt: skip
                print(f"   {ex['dir']} {ex['amount_paise'] / 100:>12,.2f}  {ex['narration']}")
        rest = counts.slice(args.shapes)
        if rest.height:
            print(f"\n(+{rest.height} rarer shapes, {rest['n'].sum()} rows): "
                  f"{', '.join(rest['shape'].head(40).to_list())}")  # fmt: skip

        for sid in sorted(lr["statement_id"].unique().to_list())[:2]:
            tr = (
                lr.filter((pl.col("statement_id") == sid)
                          & pl.col("shape").str.contains("^(" + "|".join(TRANSFER) + ")"))
                .sort("row")
            )  # fmt: skip
            print(f"\n--- {layout} {sid}: {tr.height} transfer rows (first {args.transfers}) ---")
            for ex in tr.head(args.transfers).iter_rows(named=True):
                print(f"   {ex['row']:>4} {ex['dir']} {ex['amount_paise'] / 100:>12,.2f}  "
                      f"{ex['narration']}")  # fmt: skip


if __name__ == "__main__":
    main()
