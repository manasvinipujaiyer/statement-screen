"""Score the detector's cached findings against the answer keys, and write the report.

    uv run python -m statementscreen.tamper.score_tamper --copies data/tamper/dev \\
        --genuine dev=data/tamper/genuine_dev/detect \\
        --manifest data/week1_n2000/manifest.csv \\
        --out reports/week6_tamper_dev.json \\
        --table data/tamper/dev/copies_table.csv --table-extended

Reads, and only here: the answer keys (<copies>/keys), the manifest (each genuine
statement's layout) and the findings that tamper/run.py cached. It never opens a PDF.

Refuses findings made by a different detector (the fingerprint in each detect_run.json must
match the current code, config, model file and uv.lock; there is no override), and refuses a
run that doesn't cover exactly the copies in the keys. Each --genuine population is its own
row and is never combined with another. A report is not overwritten without --overwrite.

If any genuine PDF has a strong finding, the report is still written but the command prints
a STOP message and exits with code 3: nothing should be changed before that is looked at.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.verdict import RULE_VERSION
from statementscreen.tamper.fingerprint import StaleFindings, check
from statementscreen.tamper.scoring import (
    Item,
    build_report,
    copy_rows,
    genuine_items,
    layouts_from_manifest,
    load_copies,
    load_findings,
    write_copy_table,
)

RUN_FILES = ("findings.parquet", "items.parquet", "detect_run.json")
RUN_INFO = (
    "source",
    "split",
    "pdfs",
    "findings",
    "items_with_findings",
    "crashed",
    "seconds_total",
    "seconds_per_pdf",
    "peak_memory_mb",
    "peak_memory_note",
    "fingerprint",
)


class ScoreError(RuntimeError):
    """The findings or keys can't be scored safely."""


def read_run(run_dir: Path) -> tuple[dict, dict[str, list[Finding]], list[str]]:
    """(run summary, findings by item, every item the run covered). Checks the fingerprint."""
    run_dir = Path(run_dir)
    missing = [n for n in RUN_FILES if not (run_dir / n).is_file()]
    if missing:
        raise ScoreError(
            f"no finished detector run in {run_dir} (missing {', '.join(missing)}); "
            "run python -m statementscreen.tamper.run first"
        )
    summary = json.loads((run_dir / "detect_run.json").read_text(encoding="utf-8"))
    check(summary.get("fingerprint"))  # raises StaleFindings, naming the part that differs
    findings = load_findings(run_dir / "findings.parquet")
    items = pl.read_parquet(run_dir / "items.parquet")["item"].to_list()
    return summary, findings, items


def _cover(what: str, expected: set[str], ran: list[str]) -> None:
    missing, extra = sorted(expected - set(ran)), sorted(set(ran) - expected)
    if missing or extra:
        raise ScoreError(
            f"{what}: the detector run doesn't match: {len(missing)} not run "
            f"(e.g. {missing[:3]}), {len(extra)} not expected (e.g. {extra[:3]})"
        )


def _info(summary: dict) -> dict:
    return {k: summary.get(k) for k in RUN_INFO}


def load_inputs(
    copies_dir: Path | None,
    copies_run: Path | None,
    genuine_specs: list[str],
    manifest: Path,
) -> tuple[list[Item], dict[str, list[Item]], dict[str, dict]]:
    """(copies, genuine populations, run info) from the keys, the manifest and the cached
    detector runs. Refuses a stale or incomplete run (ScoreError, StaleFindings)."""
    inputs: dict[str, dict] = {}
    copies: list[Item] = []
    if copies_dir:
        summary, findings, ran = read_run(copies_run or Path(copies_dir) / "detect")
        copies = load_copies(Path(copies_dir), findings)
        _cover("copies", {c.item for c in copies}, ran)
        inputs["copies"] = _info(summary)
    layouts = layouts_from_manifest(manifest)
    genuine: dict[str, list[Item]] = {}
    for spec in genuine_specs:
        name, _, folder = spec.partition("=")
        if not name or not folder:
            raise ScoreError(f"--genuine wants NAME=RUN_DIR, got {spec!r}")
        summary, findings, ran = read_run(Path(folder))
        unknown = sorted(set(ran) - set(layouts))
        if unknown:
            raise ScoreError(f"genuine {name}: not in the manifest, e.g. {unknown[:3]}")
        genuine[name] = genuine_items(ran, layouts, findings)
        inputs[f"genuine_{name}"] = _info(summary)
    if not copies and not genuine:
        raise ScoreError("nothing to score: give --copies and/or --genuine")
    return copies, genuine, inputs


def _fmt(metric: dict) -> str:
    ci = metric.get("ci95")
    return metric["text"] + (f" [{ci[0]:.0%}-{ci[1]:.0%}]" if ci else "")


def _row(name: str, block: dict, extra_key: str, extra_label: str) -> str:
    w, wo = block["with_history"], block[extra_key]
    return (
        f"  {name:<26} tampered {_fmt(w['tampered'])}  flagged {_fmt(w['tampered_or_suspicious'])}"
        f"\n  {'':<26} {extra_label}: tampered {wo['tampered']['text']}, "
        f"flagged {wo['tampered_or_suspicious']['text']}"
    )


def summary_text(report: dict) -> str:
    """The report's headline numbers, for the terminal."""
    lines = [
        f"Rule version {report['rule_version']}; intervals: 95% bootstrap over source "
        "statements. 'flagged' = tampered or suspicious.",
        "",
        "Detection per tamper type:",
    ]
    for t, rep in report["detection"].items():
        lines.append(_row(t, rep, "without_history", "without file history"))
    if report["pooled_detection"]:
        lines.append(_row("ALL SIX TYPES", report["pooled_detection"], "without_history",
                          "without file history"))  # fmt: skip
    if report["overlay_holdout"]:
        lines += ["", "Overlay hold-out:"]
        lines.append(_row("overlay", report["overlay_holdout"], "without_history",
                          "without file history"))  # fmt: skip
    lines += ["", "False alarms:"]
    resaved = report["false_alarms"]["resaved_genuine"]
    if resaved:
        lines.append(_row("resaved_genuine", resaved, "without_history", "without file history"))
    for name, rep in report["false_alarms"]["genuine"].items():
        lines.append(_row(f"genuine {name}", rep, "without_anomaly", "without anomaly"))
        lines.append(f"  {'':<26} ({rep['population']})")
    for title, notes in (
        ("File-history dependence (detection)", report["history_dependence"]["detection"]),
        ("File-history dependence (false alarms)", report["history_dependence"]["false_alarms"]),
        ("File-history dependence (overlay)", report["history_dependence"]["overlay"]),
        ("Layout gaps over 15 points", report["layout_notes"]),
    ):
        if notes:
            lines += ["", f"{title}:", *[f"  - {n}" for n in notes]]
    lines += [
        "",
        f"Misses: {len(report['misses'])} tamper copies not called 'tampered'; "
        f"false-alarm cases: {len(report['false_alarm_cases'])}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.score_tamper",
        description="Score cached detector findings against the answer keys.",
    )
    p.add_argument("--copies", type=Path, help="a tamper set folder (its keys/ and detect/)")
    p.add_argument(
        "--copies-run", type=Path, help="the detector run for the copies (default: <copies>/detect)"
    )
    p.add_argument(
        "--genuine",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="a genuine population (dev, train, test, shift) and its detector run",
    )
    p.add_argument("--manifest", type=Path, required=True, help="manifest.csv, for layouts")
    p.add_argument("--rule-version", default=RULE_VERSION)
    p.add_argument("--out", type=Path, required=True, help="the report (JSON)")
    p.add_argument("--table", type=Path, help="per-copy table (CSV)")
    p.add_argument("--table-extended", action="store_true", help="add the no-history verdict")
    p.add_argument("--overwrite", action="store_true", help="replace an existing report")
    args = p.parse_args(argv)

    try:
        if args.out.exists() and not args.overwrite:
            raise ScoreError(f"{args.out} exists; pass --overwrite to replace it")
        copies, genuine, inputs = load_inputs(
            args.copies, args.copies_run, args.genuine, args.manifest
        )
        report = build_report(copies, genuine, args.rule_version)
        report["inputs"] = inputs
    except (ScoreError, StaleFindings) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.table:
        rows = copy_rows(copies, args.rule_version, args.table_extended)
        write_copy_table(args.table, rows)
    print(summary_text(report))
    print(f"\nWrote {args.out}" + (f" and {args.table}" if args.table else ""))
    strong = report["genuine_with_strong_findings"]
    if strong:
        print(
            f"\nSTOP: {len(strong)} genuine PDF(s) have a strong finding. Change nothing; "
            "look at these first:",
            file=sys.stderr,
        )
        for case in strong[:10]:
            print(f"  {case['population']} {case['item']}: {case['codes']}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
