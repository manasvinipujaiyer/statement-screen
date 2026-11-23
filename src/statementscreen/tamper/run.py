"""Run the detector over a folder of PDFs, one at a time, and cache its findings.

    uv run python -m statementscreen.tamper.run data/tamper/dev/copies --out data/tamper/dev/detect
    uv run python -m statementscreen.tamper.run data/week1_n2000/pdfs --split dev \\
        --manifest data/week1_n2000/manifest.csv --out data/tamper/genuine_dev/detect

A driver that sits outside tamper/detect/: it reads PDFs (and hashes the manifest when it
needs a split), never answer keys or labels.
With --split it takes the statement ids from the split loader (which refuses "test" unless
STATEMENTSCREEN_OPEN_TEST=1); the detector package itself never imports the loader.
Without --split, a folder containing test statements is refused unless it is run as
--population test with STATEMENTSCREEN_OPEN_TEST=1, so the test PDFs cannot be read under
another label.
PDFs are taken in sorted order and processed one at a time (no multiprocessing).

Writes, in --out:
    findings.parquet  item, group, code, page, row, strength, reason: findings only, no
                      verdicts, so verdict rules can change without re-reading any PDF
    items.parquet     one row per PDF: item, seconds, peak_memory_mb (the whole process's
                      peak so far, not this PDF's own), n_findings, crashed, sha256 (of the
                      PDF's bytes, so a report can show exactly which PDFs were detected)
    detect_run.json   counts, seconds per PDF, peak memory and the detector fingerprint
                      (fingerprint.py); scoring refuses findings from a different detector

Safe for long runs: PDFs go through in chunks, each written as its own part file. If a run
is interrupted, --resume skips the finished chunks (only if the detector is unchanged).
A finished run is never overwritten without --overwrite. A PDF that crashes the detector
becomes a CHECK_NOT_RUN finding and is listed in detect_run.json, never skipped silently.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.splits import OPEN_TEST_ENV, SPLITS_DIR, SplitError, load_split
from statementscreen.summary import peak_memory_bytes
from statementscreen.tamper import provenance
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.pdf import collect_findings
from statementscreen.tamper.fingerprint import differences, fingerprint

CHUNK = 50
OUTPUTS = ("findings.parquet", "items.parquet", "detect_run.json")
FINDING_SCHEMA = {
    "item": pl.String,
    "group": pl.String,
    "code": pl.String,
    "page": pl.Int64,
    "row": pl.Int64,
    "strength": pl.String,
    "reason": pl.String,
}
ITEM_SCHEMA = {
    "item": pl.String,
    "seconds": pl.Float64,
    "peak_memory_mb": pl.Float64,
    "n_findings": pl.Int64,
    "crashed": pl.Boolean,
    "sha256": pl.String,
}


class RunError(RuntimeError):
    """The run can't start, or can't continue safely."""


def pdf_list(
    folder: Path, split: str | None, manifest: Path | None, splits_dir: Path, limit: int | None
) -> list[Path]:
    """The PDFs to run, sorted: all of the folder's, or only the split's statements."""
    folder = Path(folder)
    if not folder.is_dir():
        raise RunError(f"{folder} is not a folder")
    if split:
        if manifest is None:
            raise RunError("--split needs --manifest (the batch the split was built from)")
        pdfs = [folder / f"{sid}.pdf" for sid in load_split(split, Path(manifest), splits_dir)]
        missing = [p.name for p in pdfs if not p.is_file()]
        if missing:
            raise RunError(
                f"{len(missing)} PDFs of split {split!r} are not in {folder}, e.g. {missing[:3]}"
            )
    else:
        pdfs = sorted(folder.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    if not pdfs:
        raise RunError(f"no PDFs to run in {folder}")
    return pdfs


def open_gate_is_set() -> bool:
    """True when STATEMENTSCREEN_OPEN_TEST=1. The tests patch this function: no test may set
    the variable itself (tests/test_stress.py enforces that)."""
    return os.environ.get(OPEN_TEST_ENV) == "1"


def refuse_test_pdfs(
    pdfs: list[Path],
    split: str | None,
    population: str | None,
    splits_dir: Path,
    folder: Path,
) -> None:
    """A run without --split must not read test statements, whatever the folder is called:
    only --population test, with the open-test variable set, may. The registered shift batch
    is exempt: its ids (stmt_0001..) reuse the main batch's, but it is another batch."""
    if split:
        return  # the split loader guards "test" itself
    if population == "shift" and provenance.is_shift_folder(folder):
        return
    try:
        held_out = provenance.split_test_ids(splits_dir)
    except provenance.ProvenanceError as e:
        raise RunError(f"{e}; cannot check that no test PDF is read") from e
    leaked = sorted(p.stem for p in pdfs if p.stem in held_out)
    if leaked and population != "test":
        raise RunError(
            f"{len(leaked)} of these PDFs are test statements (e.g. {leaked[:3]}): the test "
            f"split is opened only by --split test or --population test with {OPEN_TEST_ENV}=1"
        )
    if leaked and not open_gate_is_set():
        raise RunError(f"these PDFs include test statements: set {OPEN_TEST_ENV}=1 to open them")


def registered_run(folder: Path, split: str | None, population: str | None) -> bool:
    """True for the runs whose output is scored once in step 6: the test split, the shift
    batch, and the PDFs of a test copies set (a build_run.json beside the folder says test)."""
    if split == "test" or population in provenance.REGISTERED:
        return True
    build = Path(folder).parent / "build_run.json"
    try:
        return json.loads(build.read_text(encoding="utf-8")).get("set") == "test"
    except (OSError, ValueError):
        return False


def detect_one(pdf: Path) -> tuple[list[Finding], float, float, bool]:
    """(findings, seconds, peak memory MB so far, crashed) for one PDF."""
    started = time.perf_counter()
    crashed = False
    try:
        found = collect_findings(pdf)
    except Exception as e:  # a bug on a malformed PDF: say so, keep going
        crashed = True
        reason = f"the detector crashed on this PDF: {type(e).__name__}: {e}"
        found = [Finding("CHECK_NOT_RUN", "parse", None, None, "info", reason)]
    seconds = round(time.perf_counter() - started, 3)
    return found, seconds, round(peak_memory_bytes() / 1024**2, 1), crashed


def _atomic_write(frame: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    os.replace(tmp, path)


def _write_chunk(parts: Path, k: int, pdfs: list[Path]) -> None:
    """Run one chunk and write its findings, then its items (the items file marks it done)."""
    findings: list[dict] = []
    items: list[dict] = []
    for pdf in pdfs:
        found, seconds, peak, crashed = detect_one(pdf)
        row = {
            "item": pdf.stem,
            "seconds": seconds,
            "peak_memory_mb": peak,
            "n_findings": len(found),
            "crashed": crashed,
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        }
        items.append(row)
        findings += [{"item": pdf.stem, **f.to_dict()} for f in found]
    findings_part = parts / f"findings_{k:05d}.parquet"
    _atomic_write(pl.DataFrame(findings, schema=FINDING_SCHEMA), findings_part)
    _atomic_write(pl.DataFrame(items, schema=ITEM_SCHEMA), parts / f"items_{k:05d}.parquet")


def _prepare(out: Path, resume: bool, overwrite: bool) -> Path:
    """Check the output folder; returns the parts folder, ready to use."""
    parts = out / "_parts"
    if overwrite:
        for name in OUTPUTS:
            (out / name).unlink(missing_ok=True)
        shutil.rmtree(parts, ignore_errors=True)
    if any((out / name).exists() for name in OUTPUTS):
        raise RunError(f"{out} already holds a finished run; use --overwrite to replace it")
    if parts.exists() and not resume:
        raise RunError(
            f"{out} holds an interrupted run; use --resume to continue it, "
            "or --overwrite to start over"
        )
    current = fingerprint()
    stamp = parts / "fingerprint.json"
    if parts.exists():
        if not stamp.is_file():
            raise RunError(f"{parts} has no fingerprint.json; use --overwrite to start over")
        changed = differences(json.loads(stamp.read_text(encoding="utf-8")), current)
        if changed:
            raise RunError(
                f"the detector changed since the interrupted run ({', '.join(changed)} differ); "
                "use --overwrite to start over"
            )
    else:
        parts.mkdir(parents=True)
        stamp.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return parts


def run(
    pdfs: list[Path],
    out: Path,
    source: Path,
    split: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
    population: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run the detector on every PDF; write the three outputs; return the run summary. The
    summary records the population (--population, else the split) and any --limit, so scoring
    can check a run against the name it is scored under."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parts = _prepare(out, resume, overwrite)
    started = time.perf_counter()
    resumed = 0
    for k, start in enumerate(range(0, len(pdfs), CHUNK)):
        chunk = pdfs[start : start + CHUNK]
        done = parts / f"items_{k:05d}.parquet"
        if done.exists():
            if pl.read_parquet(done)["item"].to_list() != [p.stem for p in chunk]:
                raise RunError(f"chunk {k} of the interrupted run holds different PDFs")
            resumed += 1
            continue
        _write_chunk(parts, k, chunk)
        print(
            f"  {start + len(chunk)}/{len(pdfs)} PDFs, {time.perf_counter() - started:.0f} s",
            flush=True,
        )

    items = pl.concat([pl.read_parquet(p) for p in sorted(parts.glob("items_*.parquet"))])
    findings = pl.concat([pl.read_parquet(p) for p in sorted(parts.glob("findings_*.parquet"))])
    items.write_parquet(out / "items.parquet")
    findings.write_parquet(out / "findings.parquet")
    seconds = sorted(items["seconds"].to_list())
    summary = {
        "what": "Week 6 detector findings, one PDF at a time (verdicts: see scoring.py)",
        "source": str(source),
        "split": split,
        "population": population or split,
        "limit": limit,
        "pdfs": len(pdfs),
        "findings": findings.height,
        "items_with_findings": findings["item"].n_unique(),
        "crashed": items.filter(pl.col("crashed"))["item"].to_list(),
        "chunk_size": CHUNK,
        "resumed_chunks": resumed,
        "seconds_total": round(sum(seconds), 1),
        "seconds_per_pdf": {
            "mean": round(sum(seconds) / len(seconds), 3),
            "p95": seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))],
            "max": seconds[-1],
        },
        "peak_memory_mb": items["peak_memory_mb"].max(),
        "peak_memory_note": "the process's peak so far when each PDF finished, not its own",
        "fingerprint": json.loads((parts / "fingerprint.json").read_text(encoding="utf-8")),
    }
    (out / "detect_run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(parts)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.run",
        description="Run the tamper detector over a folder of PDFs and cache its findings.",
    )
    p.add_argument("pdfs", type=Path, help="folder of PDFs")
    p.add_argument("--out", type=Path, required=True, help="folder for the outputs")
    p.add_argument("--split", help="only this split's statements (needs --manifest)")
    p.add_argument("--manifest", type=Path, help="the batch's manifest.csv (for --split)")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument(
        "--population",
        help="what this run covers (e.g. shift), recorded so scoring can check it; a run with "
        "--split records the split",
    )
    p.add_argument("--limit", type=int, help="first N PDFs only (a trial: never scored)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--overwrite", action="store_true", help="replace a finished run")
    args = p.parse_args(argv)
    try:
        if args.split and args.population and args.population != args.split:
            raise RunError(f"--population {args.population!r} contradicts --split {args.split!r}")
        pdfs = pdf_list(args.pdfs, args.split, args.manifest, args.splits_dir, args.limit)
        refuse_test_pdfs(pdfs, args.split, args.population, args.splits_dir, args.pdfs)
        if registered_run(args.pdfs, args.split, args.population):
            if args.overwrite:
                raise RunError(
                    "a registered run (test copies, genuine test, shift) is never overwritten: "
                    "the report records its hashes. If it failed before writing its output, "
                    "delete its partial folder by hand and re-run the identical command once; "
                    "a second failure means STOP and report"
                )
            if any((args.out / name).exists() for name in OUTPUTS):
                raise RunError(
                    f"{args.out} already holds a finished run; a registered run's output is "
                    "kept as it is (an interrupted one is continued with --resume)"
                )
        summary = run(
            pdfs,
            args.out,
            args.pdfs,
            args.split,
            args.resume,
            args.overwrite,
            args.population,
            args.limit,
        )
    except (RunError, SplitError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    per = summary["seconds_per_pdf"]
    print(
        f"Ran the detector on {summary['pdfs']} PDFs in {summary['seconds_total']} s "
        f"({per['mean']} s each, slowest {per['max']} s; peak memory "
        f"{summary['peak_memory_mb']} MB): {summary['findings']} findings on "
        f"{summary['items_with_findings']} PDFs; crashed: {summary['crashed'] or 'none'}"
    )
    print(f"Wrote {args.out}/" + ", ".join(OUTPUTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
