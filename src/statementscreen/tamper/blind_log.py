"""Log the Week 7 blind scoring run to the local MLflow store (after the report is written).

    uv run python -m statementscreen.tamper.blind_log \\
        --report reports/week7_blind_report.json \\
        --preregistration reports/week7_blind_preregistration.json

Only reads the report: it never recomputes, rewrites or filters a number, so it cannot change a
result. It refuses (exit 2) if the pre-registration does not verify against the current code, or
if the report and the verdict file are not the ones the pre-registration and the report name. It
exits 3 if MLflow warns about the local store (the report is already written; only the logging
was refused), the same guard Week 5 and Week 6 used.

Logged to the local store (mlruns/, git-ignored; nothing leaves the machine):
    params     the pre-registration's commit, the rule version, the detector fingerprint, and the
               SHA-256 of the key, the verdict file, the PDF set and the report
    metrics    every number in the report's mix, detection, pooled, false-alarm, precision,
               recall and confusion sections, one metric per number
    artifacts  the report, the verdict file and the pre-registration

Known limit: nothing here stops a second MLflow entry for the same report; the committed report
is the record of truth.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.tamper import blind_prereg as bpr
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import BlindError, sha256_bytes
from statementscreen.tamper.log_run import CHUNK, _merge, flatten
from statementscreen.tracking import DEFAULT_DIR, TrackingRefused, _open_store

EXPERIMENT = "week7_blind"
SECTIONS = (
    "mix",
    "detection",
    "pooled_detection",
    "false_alarms",
    "precision_of_tampered",
    "recall_of_tampered",
    "confusion_verdict_by_truth",
)


def report_metrics(report: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for section in SECTIONS:
        _merge(metrics, flatten(report.get(section), section))
    return metrics


def run_params(prereg: dict, report: dict, report_path: Path) -> dict[str, str]:
    inputs = report["inputs"]
    params = {
        "frozen_commit": prereg["frozen_commit"],
        "rule_version": str(report.get("rule_version", prereg["rule_version"])),
        "post_hoc": str(bool(report.get("post_hoc"))),
        "key_sha256": inputs["key_sha256"],
        "verdicts_sha256": inputs["verdicts_sha256"],
        "pdf_set_sha256": inputs["pdf_set_sha256"],
        "report_sha256": sha256_bytes(Path(report_path).read_bytes()),
    }
    params |= {f"fingerprint.{k}": v for k, v in sorted(prereg["detector_fingerprint"].items())}
    return params


def check_report(prereg: dict, report: dict, verdicts: Path) -> None:
    """The report must be the one scored against this pre-registration and this verdict file."""
    sealed, inputs = prereg["sealed"], report.get("inputs", {})
    if report.get("preregistration", {}).get("frozen_commit") != prereg["frozen_commit"]:
        raise BlindError("the report was scored against a different pre-registration commit")
    for name in ("key_sha256", "pdf_set_sha256"):
        if inputs.get(name) != sealed[name]:
            raise BlindError(f"the report's {name} is not the pre-registered one")
    if inputs.get("verdicts_sha256") != sha256_bytes(Path(verdicts).read_bytes()):
        raise BlindError(f"{verdicts} is not the verdict file the report was scored from")


def log(
    params: dict[str, str],
    metrics: dict[str, float],
    artifacts: Sequence[Path],
    tracking_dir: Path = DEFAULT_DIR,
    run_name: str = "blind_once",
) -> tuple[str, list[str]]:
    """One MLflow run: (run id, other warnings seen). Raises TrackingRefused, before anything
    is logged, if MLflow warns about the local store."""
    mlflow, messages = _open_store(tracking_dir, EXPERIMENT)
    items = sorted(metrics.items())
    with mlflow.start_run(run_name=run_name) as run:
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
        prog="python -m statementscreen.tamper.blind_log",
        description="Log the Week 7 blind scoring run to the local MLflow store.",
    )
    p.add_argument("--report", type=Path, default=Path(bpr.REPORT))
    p.add_argument("--verdicts", type=Path, default=Path(bpr.VERDICTS))
    p.add_argument("--preregistration", type=Path, default=bpr.DEFAULT_OUT)
    p.add_argument("--tracking-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args(argv)
    try:
        for path in (args.report, args.verdicts):
            if not path.is_file():
                raise FileNotFoundError(f"{path} not found: score first")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        prereg = bpr.load_and_verify(args.preregistration)
        check_report(prereg, report, args.verdicts)
        params = run_params(prereg, report, args.report)
        metrics = report_metrics(report)
        if not metrics:
            raise ValueError(f"no metrics found in {args.report}")
    except (
        FileNotFoundError,
        BlindError,
        fp.StaleFindings,
        OSError,
        KeyError,
        ValueError,
        subprocess.CalledProcessError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    name = "blind_post_hoc" if report.get("post_hoc") else "blind_once"
    try:
        run_id, seen = log(
            params,
            metrics,
            [args.report, args.verdicts, args.preregistration],
            args.tracking_dir,
            run_name=name,
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
