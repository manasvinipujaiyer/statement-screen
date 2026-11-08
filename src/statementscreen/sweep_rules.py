"""Tune rule thresholds on the DEV split, one at a time, and log every value tried.

    uv run python -m statementscreen.sweep_rules data/parsed_2000/ data/week1_n2000/

For each threshold in SWEEPS (in order), every value in its grid is tried with the other
thresholds at their current values: pattern detection runs over each dev statement, one
at a time, and the result is scored against the truth. The value with the best F1 on the
threshold's flag wins; ties go to the middle of the tied run (the most margin either
side). The winner carries into the following sweeps. False salary flags on gig and
self-employed statements are reported for every value.

Scoring side: reads the truth via load_split / load_truth. Refuses the test split. Does
NOT change PatternConfig; the chosen values are copied into it by hand, with a comment.
Writes reports/week3_rules_sweep.json.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import polars as pl

from statementscreen.pattern_scoring import aggregate, false_salary, load_truth, score_frames
from statementscreen.patterns.cli import process_statement
from statementscreen.patterns.config import DEFAULT, PatternConfig
from statementscreen.splits import SPLITS_DIR, SplitError, load_split, load_split_for_parsed

DEFAULT_OUT = Path("reports/week3_rules_sweep.json")
SWEEPS: list[tuple[str, str, list]] = [
    ("rent_min_amount_paise", "is_rent", [0, 50_000, 100_000, 150_000, 200_000, 250_000]),
    ("rent_min_coverage", "is_rent", [0.40, 0.45, 0.50, 0.55, 0.60]),
    ("salary_min_coverage", "is_salary", [0.55, 0.60, 0.65, 0.70, 0.75]),
]


def evaluate(statements: list[tuple[pl.DataFrame, dict]], truth, cfg: PatternConfig) -> dict:
    tables: dict[str, list[pl.DataFrame]] = {"rows": [], "group_rules": [],
                                             "statement_results": []}  # fmt: skip
    for stmt, meta in statements:
        out = process_statement(stmt, meta["period_start"], meta["period_end"], meta["layout"],
                                cfg)  # fmt: skip
        for name in tables:
            tables[name].append(out[name])
    _, scores, _ = score_frames(*(pl.concat(tables[n]) for n in tables), truth)
    overall = aggregate(scores)
    return {"transactions": overall["transactions"], "has_salary": overall["has_salary"],
            "false_salary": {k: {"statements": v["flagged_has_salary"],
                                 "rows": v["salary_rows_flagged"]}
                             for k, v in false_salary(scores).items()}}  # fmt: skip


def pick(values: list, f1s: list[float | None]) -> object:
    """The best F1; ties go to the middle of the tied run (the higher middle if even)."""
    scored = [-1.0 if f is None else f for f in f1s]
    best = max(scored)
    tied = [v for v, f in zip(values, scored, strict=True) if abs(f - best) < 1e-12]
    return tied[len(tied) // 2]


def sweep(parsed: Path, batch: Path, split: str, splits_dir: Path = SPLITS_DIR,
          cfg: PatternConfig = DEFAULT) -> dict:  # fmt: skip
    columns = ["statement_id", "layout", "period_start", "period_end"]
    meta_df = pl.read_parquet(parsed / "statements.parquet", columns=columns)
    ids = load_split(split, batch / "manifest.csv", splits_dir)
    load_split_for_parsed(split, meta_df["statement_id"].to_list(), splits_dir)
    meta = {r["statement_id"]: r for r in meta_df.iter_rows(named=True)}
    rows = (pl.scan_parquet(parsed / "rows.parquet")
            .filter(pl.col("statement_id").is_in(ids))
            .select("statement_id", "row", "posted", "narration", "amount_paise")
            .collect())  # fmt: skip
    by_statement = rows.sort("statement_id", "row").group_by("statement_id", maintain_order=True)
    statements = [(stmt, meta[sid]) for (sid,), stmt in by_statement]
    truth = load_truth(batch, ids)

    results = []
    for name, flag, grid in SWEEPS:
        started = time.perf_counter()
        tried = []
        for value in grid:
            scores = evaluate(statements, truth, replace(cfg, **{name: value}))
            tried.append({"value": value, **scores})
            t = scores["transactions"][flag]
            p, r, f1 = (f"{x:.4f}" if x is not None else "n/a"
                        for x in (t["precision"], t["recall"], t["f1"]))  # fmt: skip
            fs = scores["false_salary"]
            print(f"  {name} = {value}: {flag} P {p} R {r} F1 {f1}; false salary statements "
                  f"gig {fs['gig']['statements']}, self-employed "
                  f"{fs['self_employed']['statements']}", flush=True)  # fmt: skip
        chosen = pick(grid, [r["transactions"][flag]["f1"] for r in tried])
        results.append({"threshold": name, "flag": flag, "was": getattr(cfg, name),
                        "grid": grid, "tried": tried, "chosen": chosen,
                        "seconds": round(time.perf_counter() - started, 1)})  # fmt: skip
        cfg = replace(cfg, **{name: chosen})
    return {"split": split, "statements": len(ids),
            "selection": "best F1 on the threshold's flag; ties go to the middle of the tied "
                         "run; each winner carries into the next sweep",
            "sweeps": results, "final_config": cfg.as_dict()}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m statementscreen.sweep_rules",
                                description="Tune rule thresholds on a split.")  # fmt: skip
    p.add_argument("parsed", type=Path, help="folder written by statementscreen.parse")
    p.add_argument("batch", type=Path, help="generated batch folder (truth + manifest)")
    p.add_argument("--split", default="dev")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        report = sweep(args.parsed, args.batch, args.split, args.splits_dir)
    except SplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")  # fmt: skip
    print(f"Wrote {args.out}")
    for s in report["sweeps"]:
        print(f"  {s['threshold']}: was {s['was']}, chosen {s['chosen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
