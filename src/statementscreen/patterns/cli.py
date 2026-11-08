"""Run pattern detection over one split of a parse.

    uv run python -m statementscreen.patterns data/parsed_2000/ --split dev --out data/patterns/

Reads ONLY the parser output (rows.parquet, statements.parquet) and splits/split_v1.*:
never transactions.parquet, labels, the manifest or generator code. The split must match
the parsed folder's statements (load_split_for_parsed); "test" is refused unless
STATEMENTSCREEN_OPEN_TEST=1.

Statements are processed one at a time (rows are loaded in chunks of CHUNK statements, not
the whole split). Outputs in <out>/<split>/:

    rows.parquet            every parsed row, cleaned, with its group
    group_features.parquet  recurring-payment features per group (for Week 4)
    patterns_run.json       counts, seconds per statement, peak memory, the config, and
                            how many rows had an unknown or variant narration style
                            (with examples for the train split only)
"""

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.generate import OutputFolderError, prepare_output_folder
from statementscreen.patterns.clean import SCHEMA as CLEAN_SCHEMA
from statementscreen.patterns.clean import clean_statement
from statementscreen.patterns.config import DEFAULT, PatternConfig
from statementscreen.patterns.features import FEATURE_SCHEMA, group_features, statement_months
from statementscreen.patterns.group import group_statement
from statementscreen.splits import SPLITS_DIR, SplitError, load_split_for_parsed
from statementscreen.summary import peak_memory_bytes

CHUNK = 100
MAX_EXAMPLES = 20
OUTPUT_NAMES = ("_parts", "rows.parquet", "group_features.parquet", "patterns_run.json")
ROWS_SCHEMA = {**CLEAN_SCHEMA, "group_id": pl.String, "group_name": pl.String,
               "narration": pl.String, "layout": pl.String}  # fmt: skip
TABLES = {"rows": ROWS_SCHEMA, "group_features": FEATURE_SCHEMA}


def style_kind(style: str) -> str:
    if style == "unknown":
        return "unknown"
    return "variant" if style.endswith("?") else "known"


def process_statement(
    stmt: pl.DataFrame, period_start, period_end, layout: str | None, cfg: PatternConfig = DEFAULT
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One statement's parsed rows -> (grouped rows, group features)."""
    stmt = stmt.sort("row")
    grouped = group_statement(clean_statement(stmt, cfg), cfg).with_columns(
        stmt["narration"], pl.lit(layout, dtype=pl.String).alias("layout")
    )
    months = statement_months(period_start, period_end, stmt["posted"].to_list())
    return grouped.select(ROWS_SCHEMA.keys()), group_features(grouped, months, layout, cfg)


def merge_parts(out: Path) -> None:
    for name, schema in TABLES.items():
        parts = sorted((out / "_parts" / name).glob("*.parquet"))
        pl.scan_parquet(parts, schema=schema).sink_parquet(out / f"{name}.parquet")
    shutil.rmtree(out / "_parts")


def _seconds(values: list[float]) -> dict:
    ordered = sorted(values) or [0.0]
    return {"mean": round(statistics.fmean(ordered), 4),
            "median": round(statistics.median(ordered), 4),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
            "max": round(ordered[-1], 4)}  # fmt: skip


def run(parsed: Path, split: str, out_root: Path, splits_dir: Path = SPLITS_DIR,
        overwrite: bool = False, cfg: PatternConfig = DEFAULT) -> dict:  # fmt: skip
    columns = ["statement_id", "layout", "period_start", "period_end"]
    statements = pl.read_parquet(parsed / "statements.parquet", columns=columns)
    ids = load_split_for_parsed(split, statements["statement_id"].to_list(), splits_dir)
    meta = {r["statement_id"]: r for r in statements.iter_rows(named=True)}
    out = out_root / split
    prepare_output_folder(out, overwrite, allowed=OUTPUT_NAMES)
    for name in TABLES:
        (out / "_parts" / name).mkdir(parents=True)

    seconds: list[float] = []
    styles: dict[str, Counter] = {}
    examples: list[dict] = []
    without_rows: list[str] = []
    n_rows = n_groups = 0
    started = time.perf_counter()
    for k in range(0, len(ids), CHUNK):
        chunk = ids[k : k + CHUNK]
        rows = (pl.scan_parquet(parsed / "rows.parquet")
                .filter(pl.col("statement_id").is_in(chunk))
                .select("statement_id", "row", "posted", "narration", "amount_paise")
                .collect())  # fmt: skip
        by_id = rows.partition_by("statement_id", as_dict=True)
        frames: dict[str, list[pl.DataFrame]] = {name: [] for name in TABLES}
        for sid in chunk:
            stmt = by_id.get((sid,))
            if stmt is None or stmt.height == 0:
                without_rows.append(sid)  # listed, never skipped silently
                continue
            m = meta[sid]
            t0 = time.perf_counter()
            grouped, feats = process_statement(stmt, m["period_start"], m["period_end"],
                                               m["layout"], cfg)  # fmt: skip
            seconds.append(time.perf_counter() - t0)
            frames["rows"].append(grouped)
            frames["group_features"].append(feats)
            n_rows += grouped.height
            n_groups += feats.height
            counts = styles.setdefault(m["layout"] or "none", Counter())
            for r in grouped.select("row", "style", "narration").iter_rows(named=True):
                kind = style_kind(r["style"])
                counts[kind] += 1
                if kind != "known" and split == "train" and len(examples) < MAX_EXAMPLES:
                    examples.append({"statement_id": sid, **r})
        for name, parts in frames.items():
            if parts:
                pl.concat(parts).write_parquet(out / "_parts" / name / f"{k:05d}.parquet")
        print(f"  {split}: {min(k + CHUNK, len(ids))}/{len(ids)} statements", flush=True)
    wall = time.perf_counter() - started
    merge_parts(out)

    summary = {
        "split": split,
        "statements": len(ids),
        "statements_without_rows": without_rows,
        "rows": n_rows,
        "groups": n_groups,
        "styles": {lay: dict(sorted(c.items())) for lay, c in sorted(styles.items())},
        "style_examples": examples if split == "train" else "not shown outside train",
        "seconds_total": round(wall, 1),
        "seconds_per_statement": _seconds(seconds),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
        "config": cfg.as_dict(),
        "python": platform.python_version(),
        "platform": sys.platform,
    }
    text = json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    (out / "patterns_run.json").write_text(text, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.patterns",
        description="Clean, group and describe one split's statements.",
    )
    p.add_argument("parsed", type=Path, help="folder written by statementscreen.parse")
    p.add_argument("--split", required=True, help="train or dev (test is closed)")
    p.add_argument("--out", type=Path, default=Path("data/patterns"), help="output root")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--overwrite", action="store_true", help="replace a previous run's outputs")
    args = p.parse_args(argv)
    try:
        s = run(args.parsed, args.split, args.out, args.splits_dir, args.overwrite)
    except (SplitError, OutputFolderError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    sps = s["seconds_per_statement"]
    print(f"Wrote {args.out / args.split}: {s['statements']} statements, {s['rows']} rows, "
          f"{s['groups']} groups in {s['seconds_total']} s ({sps['mean']} s per statement, "
          f"peak memory {s['peak_memory_mb']} MB)")  # fmt: skip
    print(f"  narration styles: {s['styles']}")
    if s["statements_without_rows"]:
        print(f"  statements without rows: {', '.join(s['statements_without_rows'])}")
    return 0
