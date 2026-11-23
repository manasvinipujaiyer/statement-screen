"""Log the one-time Week 6 scoring run to the local MLflow store (the last registered step).

    uv run python -m statementscreen.tamper.log_run \\
        --report reports/week6_tamper_test.json \\
        --csv reports/week6_tamper_test_copies.csv \\
        --preregistration reports/week6_preregistration.json

Runs AFTER score_tamper and only reads its report: it never recomputes, rewrites or filters
a number, so it cannot change any result. It refuses (exit 2) if the pre-registration does
not match the current code, and exits 3 if MLflow warns about the local store (the report
is already written; only the logging was refused), the same guard Week 5 used.

Logged to the local SQLite store (mlruns/, git-ignored; nothing leaves the machine):
    params     the pre-registration's commit hash, the rule version, the four parts of the
               detector fingerprint and the SHA-256 of the report and the per-copy CSV
    metrics    every number in the report's detection, pooled, overlay and false-alarm
               sections (counts, totals, rates, interval ends), one metric per number
    artifacts  the report, the per-copy CSV and the pre-registration
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.tamper.fingerprint import StaleFindings
from statementscreen.tamper.preregister import PreregError, load_and_verify
from statementscreen.tracking import DEFAULT_DIR, TrackingRefused, _open_store

EXPERIMENT = "week6_tamper"
SECTIONS = ("detection", "pooled_detection", "overlay_holdout", "false_alarms")
CHUNK = 500  # metrics per MLflow call: the store limits how many go in one batch
_BAD_KEY = re.compile(r"[^A-Za-z0-9_\-./]")


def flatten(value: object, prefix: str) -> dict[str, float]:
    """Every number under `value` as {dotted.path: float}; an interval [lo, hi] under a key
    ending in ci95 becomes <key>_lo and <key>_hi. Strings, booleans and None are skipped."""
    if isinstance(value, bool) or value is None:
        return {}
    if isinstance(value, int | float):
        return {_BAD_KEY.sub("_", prefix): float(value)}
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, inner in value.items():
            out |= flatten(inner, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and prefix.endswith("ci95") and len(value) == 2:
        for suffix, end in zip(("_lo", "_hi"), value, strict=True):
            out |= flatten(end, prefix + suffix)
    return out


def report_metrics(report: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for section in SECTIONS:
        metrics |= flatten(report.get(section), section)
    return metrics


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_params(prereg: dict, report: dict, report_path: Path, csv_path: Path) -> dict[str, str]:
    params = {
        "frozen_commit": prereg["frozen_commit"],
        "rule_version": str(report.get("rule_version", prereg["rule_version"])),
        "report_sha256": sha256(report_path),
        "copies_csv_sha256": sha256(csv_path),
    }
    params |= {f"fingerprint.{k}": v for k, v in sorted(prereg["detector_fingerprint"].items())}
    return params


def log(
    params: dict[str, str],
    metrics: dict[str, float],
    artifacts: Sequence[Path],
    tracking_dir: Path = DEFAULT_DIR,
) -> tuple[str, list[str]]:
    """One MLflow run: (run id, other warnings seen). Raises TrackingRefused, before anything
    is logged, if MLflow warns about the local store."""
    mlflow, messages = _open_store(tracking_dir, EXPERIMENT)
    items = sorted(metrics.items())
    with mlflow.start_run(run_name="tamper_test_once") as run:
        mlflow.log_params(params)
        for start in range(0, len(items), CHUNK):
            mlflow.log_metrics(dict(items[start : start + CHUNK]))
        for path in artifacts:
            mlflow.log_artifact(str(path))
        if messages:
            mlflow.set_tag("setup_warnings", " | ".join(messages)[:5000])
        return run.info.run_id, messages


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.log_run",
        description="Log the one-time Week 6 scoring run to the local MLflow store.",
    )
    p.add_argument("--report", type=Path, default=Path("reports/week6_tamper_test.json"))
    p.add_argument("--csv", type=Path, default=Path("reports/week6_tamper_test_copies.csv"))
    p.add_argument(
        "--preregistration", type=Path, default=Path("reports/week6_preregistration.json")
    )
    p.add_argument("--tracking-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args(argv)
    try:
        for path in (args.report, args.csv):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found: score first")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        prereg = load_and_verify(args.preregistration, report.get("rule_version", ""))
        params = run_params(prereg, report, args.report, args.csv)
        metrics = report_metrics(report)
        if not metrics:
            raise ValueError(f"no metrics found in {args.report}")
    except (FileNotFoundError, ValueError, PreregError, StaleFindings) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        run_id, seen = log(
            params, metrics, [args.report, args.csv, args.preregistration], args.tracking_dir
        )
    except TrackingRefused as e:
        print(f"MLflow: {e}", file=sys.stderr)
        return 3
    print(f"MLflow run {run_id}: {len(metrics)} metrics, {len(params)} params logged to "
          f"{args.tracking_dir}/ (experiment '{EXPERIMENT}')")  # fmt: skip
    for w in seen:
        print(f"  MLflow setup warning (not about the store, logged as a tag): {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
