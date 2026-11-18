"""Build income profiles for one set of pattern outputs, one statement at a time.

    uv run python -m statementscreen.profile data/patterns/ --set dev --parsed data/parsed_2000/
    uv run python -m statementscreen.profile data/patterns/ --set shift_karan_v1 \\
        --parsed data/parsed_shift_n300/

Reads ONLY pattern outputs (data/patterns/<set>/rows, group_features, group_rules,
statement_results, patterns_run.json) and the parser's statements.parquet (for each
statement's period). Never transactions.parquet, labels, the manifest or generator code.
"test" is refused unless STATEMENTSCREEN_OPEN_TEST=1.

Writes data/profiles/<set>/:
    profiles.jsonl      one profile per line (JSON, keys sorted: byte-identical per input)
    profiles.parquet    one flat row per statement
    profiles_run.json   counts, warnings, seconds per statement, peak memory (ru_maxrss via
                        summary.peak_memory_bytes: bytes on macOS, KB on Linux), config
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.generate import OutputFolderError, prepare_output_folder
from statementscreen.profile.build import flat, profile_statement, to_json
from statementscreen.profile.config import DEFAULT, ProfileConfig
from statementscreen.splits import OPEN_TEST_ENV
from statementscreen.summary import peak_memory_bytes

CHUNK = 100
ROW_COLUMNS = ["statement_id", "row", "posted", "amount_paise", "direction", "channel", "style",
               "counterparty", "purpose", "is_self", "group_id", "group_name", "is_salary",
               "is_emi", "is_bounce", "is_rent"]  # fmt: skip
GROUP_COLUMNS = ["statement_id", "group_id", "group_name", "direction", "main_channel",
                 "is_self", "n_txn", "total_paise", "months_present", "txn_per_month",
                 "gap_mean_days"]  # fmt: skip
RULE_COLUMNS = ["statement_id", "group_id", "is_salary", "is_emi", "is_rent", "salary_note"]
RESULT_COLUMNS = ["statement_id", "has_salary", "employer", "salary_reason"]
OUTPUTS = ("profiles.jsonl", "profiles.parquet", "profiles_run.json")


def check_set(name: str) -> None:
    if name == "test" and os.environ.get(OPEN_TEST_ENV) != "1":
        raise ValueError(f"the test split is closed; set {OPEN_TEST_ENV}=1 to open it")


def _chunk(path: Path, columns: list[str], ids: list[str]) -> dict:
    frame = (pl.scan_parquet(path).filter(pl.col("statement_id").is_in(ids)).select(columns)
             .collect())  # fmt: skip
    return {k[0]: v for k, v in frame.partition_by("statement_id", as_dict=True).items()}


def run(patterns_root: Path, name: str, parsed: Path, out_root: Path,
        cfg: ProfileConfig = DEFAULT, overwrite: bool = False) -> dict:  # fmt: skip
    check_set(name)
    src = patterns_root / name
    run_info = json.loads((src / "patterns_run.json").read_text(encoding="utf-8"))
    if run_info["split"] != name:
        raise ValueError(f"{src} holds a patterns run of {run_info['split']!r}, not {name!r}")
    results = pl.read_parquet(src / "statement_results.parquet", columns=RESULT_COLUMNS)
    ids = sorted(set(results["statement_id"]) | set(run_info["statements_without_rows"]))
    periods = pl.read_parquet(parsed / "statements.parquet",
                              columns=["statement_id", "period_start", "period_end"])  # fmt: skip
    period = {r["statement_id"]: r for r in periods.iter_rows(named=True)}
    missing = [sid for sid in ids if sid not in period]
    if missing:
        raise ValueError(f"{parsed} has no statement {missing[0]}; parsed from another batch?")
    result_of = {r["statement_id"]: r for r in results.iter_rows(named=True)}
    out = out_root / name
    prepare_output_folder(out, overwrite, allowed=OUTPUTS)

    seconds: list[float] = []
    warnings: Counter = Counter()
    flats = []
    started = time.perf_counter()
    with (out / "profiles.jsonl").open("w", encoding="utf-8") as f:
        for k in range(0, len(ids), CHUNK):
            chunk = ids[k : k + CHUNK]
            rows = _chunk(src / "rows.parquet", ROW_COLUMNS, chunk)
            groups = _chunk(src / "group_features.parquet", GROUP_COLUMNS, chunk)
            rules = _chunk(src / "group_rules.parquet", RULE_COLUMNS, chunk)
            for sid in chunk:
                t0 = time.perf_counter()
                p = profile_statement(
                    sid, rows.get(sid, pl.DataFrame(schema={c: pl.Null for c in ROW_COLUMNS})),
                    groups.get(sid, pl.DataFrame(schema={c: pl.Null for c in GROUP_COLUMNS})),
                    rules.get(sid, pl.DataFrame(schema={c: pl.Null for c in RULE_COLUMNS})),
                    result_of.get(sid), period[sid]["period_start"], period[sid]["period_end"],
                    cfg)  # fmt: skip
                seconds.append(time.perf_counter() - t0)
                f.write(to_json(p) + "\n")
                flats.append(flat(p))
                warnings.update(w.split(" (")[0] for w in p["warnings"])
            print(f"  {name}: {min(k + CHUNK, len(ids))}/{len(ids)} statements", flush=True)
    pl.DataFrame(flats, infer_schema_length=None).write_parquet(out / "profiles.parquet")
    summary = {
        "set": name,
        "statements": len(ids),
        "warnings": dict(sorted(warnings.items())),
        "with_income_estimate": sum(r["estimated_monthly_income_paise"] is not None
                                    for r in flats),
        "customer_types": dict(sorted(Counter(r["customer_type"] for r in flats).items())),
        "seconds_total": round(time.perf_counter() - started, 1),
        "seconds_per_statement": round(sum(seconds) / max(len(seconds), 1), 4),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
        "config": cfg.as_dict(),
        "config_sha256": cfg.sha256(),
    }  # fmt: skip
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (out / "profiles_run.json").write_text(text, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.profile",
        description="Build income profiles from pattern outputs.",
    )
    p.add_argument("patterns", type=Path, help="root folder written by statementscreen.patterns")
    p.add_argument("--set", required=True, help="dev, train, or an evaluation batch name")
    p.add_argument("--parsed", type=Path, required=True, help="the parse the patterns came from")
    p.add_argument("--out", type=Path, default=Path("data/profiles"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    try:
        s = run(args.patterns, args.set, args.parsed, args.out, overwrite=args.overwrite)
    except (ValueError, OutputFolderError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"Wrote {args.out / args.set}: {s['statements']} profiles, "
          f"{s['with_income_estimate']} with an income estimate, in {s['seconds_total']} s "
          f"({s['seconds_per_statement']} s per statement, peak memory "
          f"{s['peak_memory_mb']} MB)")  # fmt: skip
    print(f"  customer types: {s['customer_types']}")
    print(f"  warnings: {s['warnings']}")
    return 0
