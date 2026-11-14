"""Score pattern detection on a split against the ground truth.

    uv run python -m statementscreen.score_patterns data/patterns/ data/week1_n2000/ --split dev

Reads the patterns output for the split (data/patterns/<split>/) plus the batch's
transactions.parquet, labels and manifest. Joins on statement_id and row. The split is
loaded with load_split, which checks the manifest's hash and refuses "test" unless
STATEMENTSCREEN_OPEN_TEST=1. Writes reports/week3_patterns_<split>.json, then logs a local
MLflow run to mlruns/ (see statementscreen.tracking; --tracking-dir, --no-mlflow).
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from statementscreen.pattern_scoring import FLAGS, build_report
from statementscreen.splits import SPLITS_DIR, SplitError, load_split
from statementscreen.summary import peak_memory_bytes
from statementscreen.tracking import DEFAULT_DIR, EXPERIMENT, TrackingRefused, log_run


def _pct(x: float | None) -> str:
    return "  n/a  " if x is None else f"{x:7.2%}"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.score_patterns",
        description="Score pattern detection against the truth.",
    )
    p.add_argument("patterns", type=Path, help="root folder written by statementscreen.patterns")
    p.add_argument("batch", type=Path, help="generated batch folder (truth, labels, manifest)")
    p.add_argument("--split", required=True, help="train or dev (test is closed)")
    p.add_argument("--out", type=Path, help="default: reports/week3_patterns_<split>.json")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--tracking-dir", type=Path, default=DEFAULT_DIR,
                   help="local MLflow folder (default mlruns/, git-ignored)")  # fmt: skip
    p.add_argument("--no-mlflow", action="store_true", help="don't log an MLflow run")
    args = p.parse_args(argv)
    out = args.out or Path(f"reports/week3_patterns_{args.split}.json")
    patterns_dir = args.patterns / args.split
    try:
        started = time.perf_counter()
        ids = load_split(args.split, args.batch / "manifest.csv", args.splits_dir)
        report = build_report(patterns_dir, args.batch, ids)
        seconds = time.perf_counter() - started
    except (SplitError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    run = json.loads((patterns_dir / "patterns_run.json").read_text(encoding="utf-8"))
    scoring_run = {
        "seconds_total": round(seconds, 1),
        "seconds_per_statement": round(seconds / max(len(ids), 1), 4),
        # peak_memory_bytes handles ru_maxrss: bytes on macOS, kilobytes on Linux
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
    }
    report = {"split": args.split, "batch": args.batch.name, "patterns_run": run,
              "scoring_run": scoring_run, "config": run["config"], **report}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")  # fmt: skip

    o = report["overall"]
    g = o["grouping"]
    print(f"Wrote {out}: {report['statements']} {args.split} statements, {report['rows']} rows "
          f"({report['rows_without_truth']} without truth) in {scoring_run['seconds_total']} s "
          f"({scoring_run['seconds_per_statement']} s per statement, peak memory "
          f"{scoring_run['peak_memory_mb']} MB)")  # fmt: skip
    print(f"  grouping   pairwise P {_pct(g['precision'])}  R {_pct(g['recall'])}  "
          f"F1 {_pct(g['f1'])}  employer recall {_pct(g['employer_recall'])}")  # fmt: skip
    for f in FLAGS:
        t = o["transactions"][f]
        print(f"  {f:10} P {_pct(t['precision'])}  R {_pct(t['recall'])}  "
              f"(tp {t['tp']}, fp {t['fp']}, fn {t['fn']})")  # fmt: skip
    h, e = o["has_salary"], o["employer"]
    print(f"  has_salary P {_pct(h['precision'])}  R {_pct(h['recall'])}  "
          f"(tp {h['tp']}, fp {h['fp']}, fn {h['fn']})")  # fmt: skip
    print(f"  employer   exact {_pct(e['exact_match_rate'])}  fuzzy {_pct(e['fuzzy_match_rate'])}"
          f"  of {e['statements_with_true_employer']} salaried")  # fmt: skip
    for kind, fs in report["false_salary_on_non_salaried"].items():
        print(f"  false salary on {kind}: {fs['flagged_has_salary']} of {fs['statements']} "
              f"statements, {fs['salary_rows_flagged']} rows")  # fmt: skip

    if args.no_mlflow:
        return 0
    sweeps = [out.parent / "week3_grouping_sweep.json", out.parent / "week3_rules_sweep.json"]
    try:
        run_id, other_warnings = log_run(report, out, args.tracking_dir, sweeps)
    except TrackingRefused as e:
        print(f"MLflow: {e}", file=sys.stderr)
        return 3  # the report is written; only tracking was refused
    print(f"  MLflow run {run_id} logged to {args.tracking_dir}/ "
          f"(experiment '{EXPERIMENT}')")  # fmt: skip
    for w in other_warnings:
        print(f"  MLflow setup warning (not about the store, logged as a tag): {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
