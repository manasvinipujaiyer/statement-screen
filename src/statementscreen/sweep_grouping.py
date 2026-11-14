"""Pick the grouping threshold by sweeping a grid on the DEV split.

    uv run python -m statementscreen.sweep_grouping data/parsed_2000/ data/week1_n2000/

For each dev statement, one at a time: clean the parsed rows, group them at every
threshold in PatternConfig.group_grid (distances computed once per statement), and count
pairwise agreement with the truth's canonical_counterparty. The threshold with the best
pairwise F1 wins (ties: the smaller threshold). Every value tried is written to
reports/week3_grouping_sweep.json.

Scoring side: reads transactions.parquet (canonical_counterparty only) and the manifest's
hash via load_split. Refuses the test split.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.pattern_scoring import PairCounts
from statementscreen.patterns.clean import clean_statement
from statementscreen.patterns.config import DEFAULT
from statementscreen.patterns.group import group_ids
from statementscreen.splits import SplitError, load_split, load_split_for_parsed

DEFAULT_OUT = Path("reports/week3_grouping_sweep.json")


def sweep(parsed: Path, batch: Path, split: str, cfg=DEFAULT) -> dict:
    parsed_ids = pl.read_parquet(parsed / "statements.parquet", columns=["statement_id"])
    ids = load_split(split, batch / "manifest.csv")  # checks the manifest hash
    load_split_for_parsed(split, parsed_ids["statement_id"].to_list())  # same batch parsed

    rows = (pl.scan_parquet(parsed / "rows.parquet")
            .filter(pl.col("statement_id").is_in(ids))
            .select("statement_id", "row", "posted", "narration", "amount_paise", "layout")
            .collect())  # fmt: skip
    truth = (pl.scan_parquet(batch / "transactions.parquet")
             .filter(pl.col("statement_id").is_in(ids))
             .select("statement_id", pl.col("row").cast(pl.Int64), "canonical_counterparty")
             .collect())  # fmt: skip

    grid = list(cfg.group_grid)
    overall = {t: PairCounts() for t in grid}
    by_layout: dict[str, dict[float, PairCounts]] = {}
    scored = without_truth = 0
    start = time.perf_counter()
    statements = rows.sort("statement_id", "row").group_by("statement_id", maintain_order=True)
    for (sid,), stmt in statements:
        layout = stmt["layout"][0]
        cleaned = clean_statement(stmt.drop("layout"), cfg)
        ids_at = group_ids(cleaned, grid, cfg)
        stmt_truth = truth.filter(pl.col("statement_id") == sid)
        labels = cleaned.select("row").with_row_index("i").join(stmt_truth, on="row", how="left")
        keep = labels.filter(pl.col("canonical_counterparty").is_not_null())
        idx, true = keep["i"].to_list(), keep["canonical_counterparty"].to_list()
        scored += len(idx)
        without_truth += labels.height - len(idx)
        layout_counts = by_layout.setdefault(layout, {t: PairCounts() for t in grid})
        for t in grid:
            predicted = [ids_at[t][i] for i in idx]
            overall[t].add(predicted, true)
            layout_counts[t].add(predicted, true)
    seconds = time.perf_counter() - start

    results = [{"threshold": t, **overall[t].scores(),
                "by_layout": {lay: c[t].scores() for lay, c in sorted(by_layout.items())}}
               for t in grid]  # fmt: skip
    best = max(results, key=lambda r: (r["f1"] or -1.0, -r["threshold"]))
    return {
        "split": split,
        "statements": len(ids),
        "rows_scored": scored,
        "rows_without_truth": without_truth,
        "seconds": round(seconds, 1),
        "seconds_per_statement": round(seconds / max(len(ids), 1), 4),
        "config": cfg.as_dict(),
        "results": results,
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
        "selection": "max pairwise F1 on this split; ties go to the smaller threshold",
    }


def _fmt(x: float | None) -> str:
    return "  n/a " if x is None else f"{x:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m statementscreen.sweep_grouping",
                                description="Sweep the grouping threshold on a split.")  # fmt: skip
    p.add_argument("parsed", type=Path, help="folder written by statementscreen.parse")
    p.add_argument("batch", type=Path, help="generated batch folder (truth + manifest)")
    p.add_argument("--split", default="dev", help="split to sweep on (default: dev)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        report = sweep(args.parsed, args.batch, args.split)
    except SplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}: {report['statements']} {report['split']} statements, "
          f"{report['rows_scored']} rows scored ({report['rows_without_truth']} without truth), "
          f"{report['seconds']} s")  # fmt: skip
    print("  threshold  precision  recall     F1")
    for r in report["results"]:
        mark = "  <- best" if r["threshold"] == report["best_threshold"] else ""
        print(f"  {r['threshold']:9.2f}  {_fmt(r['precision'])}     {_fmt(r['recall'])}"
              f"     {_fmt(r['f1'])}{mark}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
