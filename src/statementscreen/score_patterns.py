"""Score pattern detection on a split against the ground truth.

    uv run python -m statementscreen.score_patterns data/patterns/ data/week1_n2000/ --split dev

Reads the patterns output for the split (data/patterns/<split>/) plus the batch's
transactions.parquet, labels and manifest. Joins on statement_id and row. The split is
loaded with load_split, which checks the manifest's hash and refuses "test" unless
STATEMENTSCREEN_OPEN_TEST=1. Writes reports/week3_patterns_<split>.json.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.pattern_scoring import FLAGS, build_report
from statementscreen.splits import SPLITS_DIR, SplitError, load_split


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
    args = p.parse_args(argv)
    out = args.out or Path(f"reports/week3_patterns_{args.split}.json")
    patterns_dir = args.patterns / args.split
    try:
        ids = load_split(args.split, args.batch / "manifest.csv", args.splits_dir)
        report = build_report(patterns_dir, args.batch, ids)
    except (SplitError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    run = json.loads((patterns_dir / "patterns_run.json").read_text(encoding="utf-8"))
    report = {"split": args.split, "batch": args.batch.name, "patterns_run": run,
              "config": run["config"], **report}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")  # fmt: skip

    o = report["overall"]
    g = o["grouping"]
    print(f"Wrote {out}: {report['statements']} {args.split} statements, {report['rows']} rows "
          f"({report['rows_without_truth']} without truth)")  # fmt: skip
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
