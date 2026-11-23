"""Score the detector's cached findings against the answer keys, and write the report.

    uv run python -m statementscreen.tamper.score_tamper --copies data/tamper/dev \\
        --genuine dev=data/tamper/genuine_dev/detect \\
        --manifest data/week1_n2000/manifest.csv \\
        --out reports/week6_tamper_dev.json \\
        --table data/tamper/dev/copies_table.csv --table-extended

Reads, and only here: the answer keys (<copies>/keys), the manifests (each genuine
statement's layout) and the findings that tamper/run.py cached. It never opens a PDF.
--manifest is the default manifest for every genuine population; --genuine-manifest NAME=PATH
gives one population its own (the shift batch has its own manifest). The manifest used for
each population is recorded in the report.

Refuses findings made by a different detector (the fingerprint in each detect_run.json must
match the current code, config, model file and uv.lock; there is no override), and refuses a
run that doesn't cover exactly the copies in the keys. Each --genuine population is its own
row and is never combined with another. A report is not overwritten without --overwrite.

The test and shift populations are recognised from each run's own record AND its contents
(tamper/provenance.py), not from the name typed here: the recorded population must match the
--genuine name, the statement ids must be exactly the test (or shift) statements, no other
population may contain them, a --limit trial run is never scored, and 300 genuine test, 300
shift and 800 copies (100 of each of eight types, seed 6, built without a shortfall) are
required, with a SHA-256 for every PDF. They need --preregistration (compared with the pushed
version) and, for the copies, --table; they are scored once into the registered file (never
overwritten, not even after deleting a committed one, --overwrite refused); a later run is
--post-hoc and goes to a file of its own. If a check crashed on a PDF, a verdict that would
read "no evidence of tampering" reads "could not check" instead (strong findings and weak
verdicts stand); those PDFs are counted in the totals, reported in their own column, and the
crashes are counted per check.

If any genuine PDF has a strong finding, the report is still written but the command prints
a STOP message and exits with code 3: nothing should be changed before that is looked at.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.tamper import preregister as pre
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.verdict import RULE_VERSION
from statementscreen.tamper.fingerprint import StaleFindings, check
from statementscreen.tamper.preregister import PreregError, load_and_verify
from statementscreen.tamper.provenance import (
    ProvenanceError,
    check_populations,
    copies_hashes,
    file_sha256,
    run_hashes,
)
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
    "population",
    "limit",
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
    manifest: Path | None,
    genuine_manifests: Sequence[str] = (),
) -> tuple[list[Item], dict[str, list[Item]], dict[str, dict]]:
    """(copies, genuine populations, run info) from the keys, the manifests and the cached
    detector runs. Each genuine population is looked up in its own manifest
    (genuine_manifests, NAME=PATH) or else the default `manifest`; a population with neither
    is refused. Refuses a stale or incomplete run (ScoreError, StaleFindings)."""
    inputs: dict[str, dict] = {}
    copies: list[Item] = []
    if copies_dir:
        run = Path(copies_run) if copies_run else Path(copies_dir) / "detect"
        summary, findings, ran = read_run(run)
        crashed = frozenset(summary.get("crashed", []))
        copies = load_copies(Path(copies_dir), findings, crashed)
        _cover("copies", {c.item for c in copies}, ran)
        hashes = {**copies_hashes(Path(copies_dir)), "detect": run_hashes(run)}
        inputs["copies"] = {**_info(summary), "sha256": hashes}
    own: dict[str, Path] = {}
    for spec in genuine_manifests:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise ScoreError(f"--genuine-manifest wants NAME=PATH, got {spec!r}")
        if name in own:
            raise ScoreError(f"--genuine-manifest given twice for {name!r}")
        own[name] = Path(path)
    named = [spec.partition("=")[0] for spec in genuine_specs]
    stray = sorted(set(own) - set(named))
    if stray:
        raise ScoreError(f"--genuine-manifest for a population not given with --genuine: {stray}")
    genuine: dict[str, list[Item]] = {}
    for spec in genuine_specs:
        name, _, folder = spec.partition("=")
        if not name or not folder:
            raise ScoreError(f"--genuine wants NAME=RUN_DIR, got {spec!r}")
        if name in genuine:
            raise ScoreError(f"--genuine given twice for {name!r}")
        used = own.get(name, manifest)
        if used is None:
            raise ScoreError(f"genuine {name}: no manifest; give --manifest or --genuine-manifest")
        layouts = layouts_from_manifest(used)
        summary, findings, ran = read_run(Path(folder))
        unknown = sorted(set(ran) - set(layouts))
        if unknown:
            raise ScoreError(f"genuine {name}: not in the manifest {used}, e.g. {unknown[:3]}")
        crashed = frozenset(summary.get("crashed", []))
        genuine[name] = genuine_items(ran, layouts, findings, crashed)
        hashes = {"manifest": file_sha256(used), "detect": run_hashes(Path(folder))}
        inputs[f"genuine_{name}"] = {**_info(summary), "manifest": str(used), "sha256": hashes}
    if not copies and not genuine:
        raise ScoreError("nothing to score: give --copies and/or --genuine")
    return copies, genuine, inputs


def guard_one_time(args: argparse.Namespace, registered: list[str]) -> None:
    """The test and shift populations are scored once. The first result goes to the registered
    file and is never overwritten; any later run is --post-hoc, in a file of its own."""
    if not registered:
        if args.post_hoc:
            raise ScoreError("--post-hoc applies only to the test and shift populations")
        return
    if args.overwrite:
        raise ScoreError("--overwrite is not allowed for the test and shift populations")
    if "copies" in registered and not args.table:
        raise ScoreError("the test copies are scored with their per-copy CSV: pass --table")
    first, first_table = Path(pre.FIRST_REPORT), Path(pre.FIRST_TABLE)
    outputs = [(args.out, first)] + ([(args.table, first_table)] if args.table else [])
    for path, registered_path in outputs:
        if path.exists():
            raise ScoreError(f"{path} exists; the test and shift results are never overwritten")
        if pre.has_git_history(path):
            raise ScoreError(
                f"{path} is in git's history: that result was already written (and committed), "
                "so deleting the file does not make a new first result"
            )
        if args.post_hoc:
            if "post_hoc" not in path.name:
                raise ScoreError(f"a post-hoc result goes to its own file: {path} needs post_hoc")
        elif path.resolve() != registered_path.resolve():
            raise ScoreError(f"the one-time scoring writes {registered_path}, not {path}")
    if args.post_hoc and not first.exists():
        raise ScoreError(f"--post-hoc needs the first result ({first}) to exist")
    if not args.post_hoc and first.exists():
        raise ScoreError(
            f"{first} exists: the test and shift populations were already scored once; "
            "a later run needs --post-hoc and its own file"
        )


def _fmt(metric: dict) -> str:
    ci = metric.get("ci95")
    return metric["text"] + (f" [{ci[0]:.0%}-{ci[1]:.0%}]" if ci else "")


def _row(name: str, block: dict, extra_key: str, extra_label: str) -> str:
    w, wo = block["with_history"], block[extra_key]
    return (
        f"  {name:<26} tampered {_fmt(w['tampered'])}  flagged {_fmt(w['tampered_or_suspicious'])}"
        f"  could not check {block['could_not_check']['text']}"
        f"\n  {'':<26} {extra_label}: tampered {wo['tampered']['text']}, "
        f"flagged {wo['tampered_or_suspicious']['text']}"
    )


def crashed_checks_lines(report: dict) -> list[str]:
    """Per group, the checks that crashed and on how many PDFs (whatever the PDF's verdict)."""
    groups = {
        **report["detection"],
        "overlay": report["overlay_holdout"],
        "resaved_genuine": report["false_alarms"]["resaved_genuine"],
        **{f"genuine {n}": r for n, r in report["false_alarms"]["genuine"].items()},
    }
    rows = [
        f"  {name:<26} " + ", ".join(f"{check} {n}" for check, n in rep["crashed_checks"].items())
        for name, rep in groups.items()
        if rep and rep["crashed_checks"]
    ]
    return ["Crashed checks (PDFs on which each check crashed):", *(rows or ["  none"])]


def summary_text(report: dict) -> str:
    """The report's headline numbers, for the terminal."""
    lines = [
        ("POST-HOC RE-RUN (not the one-time result). " if report.get("post_hoc") else "")
        + f"Rule version {report['rule_version']}; intervals: 95% bootstrap over source "
        "statements. 'flagged' = tampered or suspicious; 'could not check' = the detector "
        "crashed on the PDF (counted in the totals, never as 'no evidence').",
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
    lines += ["", *crashed_checks_lines(report)]
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
    p.add_argument(
        "--manifest",
        type=Path,
        help="manifest.csv for layouts: the default for every genuine population",
    )
    p.add_argument(
        "--genuine-manifest",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="this population's own manifest.csv (e.g. the shift batch), instead of --manifest",
    )
    p.add_argument(
        "--preregistration",
        type=Path,
        help="reports/week6_preregistration.json: required for the test and shift "
        "populations; scoring refuses if the detector or anything frozen differs from it",
    )
    p.add_argument("--rule-version", default=RULE_VERSION)
    p.add_argument("--out", type=Path, required=True, help="the report (JSON)")
    p.add_argument("--table", type=Path, help="per-copy table (CSV)")
    p.add_argument("--table-extended", action="store_true", help="add the no-history verdict")
    p.add_argument("--overwrite", action="store_true", help="replace an existing report")
    p.add_argument(
        "--post-hoc",
        action="store_true",
        help="a re-run of the test and shift scoring: written to its own file (its name must "
        "contain post_hoc) and labelled post-hoc; the first result is never overwritten",
    )
    args = p.parse_args(argv)

    try:
        registered = check_populations(args.copies, args.copies_run, args.genuine)
        if args.preregistration:
            load_and_verify(args.preregistration, args.rule_version)
        elif registered:
            raise ScoreError(
                f"{', '.join(registered)}: the test and shift populations are scored once, "
                "against the pre-registration: pass --preregistration "
                "reports/week6_preregistration.json"
            )
        guard_one_time(args, registered)
        if not registered and args.out.exists() and not args.overwrite:
            raise ScoreError(f"{args.out} exists; pass --overwrite to replace it")
        copies, genuine, inputs = load_inputs(
            args.copies, args.copies_run, args.genuine, args.manifest, args.genuine_manifest
        )
        report = build_report(copies, genuine, args.rule_version)
        if args.preregistration:
            inputs["preregistration"] = {
                "file": str(args.preregistration),
                "sha256": file_sha256(args.preregistration),
                "upstream_commit": pre.upstream_sha(),
            }
        report["inputs"] = inputs
        if args.post_hoc:
            report["post_hoc"] = True
    except (ScoreError, StaleFindings, PreregError, ProvenanceError) as e:
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
