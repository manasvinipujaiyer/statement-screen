"""Where each scored population really comes from, read from the runs' own records.

The one-time test and shift scoring must not depend on what a population is CALLED on the
command line. Every detector run records what it covered (detect_run.json: the population or
split, the source folder, the number of PDFs, any --limit) and every build records what it
made (build_run.json: the set, the counts, any shortfall). These checks read those records:

  * a genuine population's recorded identity must equal its --genuine name, so a test run
    cannot be scored as "dev" (no pre-registration needed) or a dev run as "test";
  * a run with no recorded identity is refused (re-run with --split or --population);
  * a --limit trial run is never scored;
  * the registered populations must be complete and come from the registered folders:
    300 genuine test, 300 shift, and 800 copies (100 for each of the eight types), and a
    build that came up short is refused;
  * a PDF folder that is the shift batch can only be scored as "shift".

Also here: the SHA-256 of every input that cannot be hashed before step 6 (the test keys,
the build summary, the detector runs, the manifests), so the report records them.

Lives outside tamper/detect/, so it can't change a finding or the detector fingerprint.
"""

import hashlib
import json
import os
from pathlib import Path

import polars as pl

from statementscreen.tamper.copy import TYPES

REGISTERED = ("test", "shift")  # the genuine populations that are scored once, in step 6
COPIES_PER_TYPE = 100
EXPECTED_COUNTS = {"test": 300, "shift": 300, "copies": COPIES_PER_TYPE * len(TYPES)}
EXPECTED_SOURCES = {"test": "data/week1_n2000/pdfs", "shift": "data/week4_shift_n300/pdfs"}
RUN_FILES = ("findings.parquet", "items.parquet", "detect_run.json")


class ProvenanceError(RuntimeError):
    """A population is not what its name says, or is not complete."""


def _norm(path: object) -> str:
    return os.path.normpath(str(path)).replace(os.sep, "/")


def run_summary(run_dir: Path) -> dict:
    """detect_run.json of a finished detector run (the three output files must exist)."""
    run_dir = Path(run_dir)
    missing = [n for n in RUN_FILES if not (run_dir / n).is_file()]
    if missing:
        raise ProvenanceError(
            f"no finished detector run in {run_dir} (missing {', '.join(missing)}); "
            "run python -m statementscreen.tamper.run first"
        )
    return json.loads((run_dir / "detect_run.json").read_text(encoding="utf-8"))


def _items_in(run_dir: Path) -> int:
    return pl.read_parquet(Path(run_dir) / "items.parquet")["item"].n_unique()


def _complete(what: str, run_dir: Path, summary: dict, expected: int) -> None:
    found = _items_in(run_dir)
    if summary.get("pdfs") != expected or found != expected:
        raise ProvenanceError(
            f"{what} must cover {expected} PDFs; the run in {run_dir} covers {found} "
            f"(it records {summary.get('pdfs')}): an incomplete run is never scored"
        )


def check_genuine(name: str, run_dir: Path) -> bool:
    """Check one --genuine population against its run's own record; True if it is one of the
    registered populations (test, shift)."""
    summary = run_summary(run_dir)
    recorded = summary.get("population") or summary.get("split")
    if recorded is None:
        raise ProvenanceError(
            f"the run in {run_dir} does not record which population it covers; run "
            "tamper.run again with --split or --population"
        )
    if recorded != name:
        raise ProvenanceError(
            f"--genuine {name}= points at a detector run recorded as {recorded!r} ({run_dir}): "
            "a population is scored under the name it was run as"
        )
    if summary.get("limit"):
        raise ProvenanceError(f"the run in {run_dir} used --limit: a trial run is never scored")
    source = _norm(summary.get("source", ""))
    if source == _norm(EXPECTED_SOURCES["shift"]) and name != "shift":
        raise ProvenanceError(f"the PDFs in {source} are the shift batch: score them as 'shift'")
    if name in REGISTERED:
        if source != _norm(EXPECTED_SOURCES[name]):
            raise ProvenanceError(
                f"population {name!r} must come from {EXPECTED_SOURCES[name]}, "
                f"not {summary.get('source')!r}"
            )
        _complete(f"population {name!r}", run_dir, summary, EXPECTED_COUNTS[name])
    return name in REGISTERED


def check_copies(copies_dir: Path, copies_run: Path | None = None) -> bool:
    """Check a copies set against its build record and its detector run; True if it is the
    test set (scored once, in step 6)."""
    copies_dir = Path(copies_dir)
    run_dir = Path(copies_run) if copies_run else copies_dir / "detect"
    summary = run_summary(run_dir)
    build_path = copies_dir / "build_run.json"
    if not build_path.is_file():
        raise ProvenanceError(f"{build_path} not found: scoring needs the build's own record")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    is_test = build.get("set") == "test"
    if copies_dir.name == "test" and not is_test:
        raise ProvenanceError(
            f"{copies_dir} is named test but its build record says set={build.get('set')!r}"
        )
    if summary.get("limit"):
        raise ProvenanceError(f"the run in {run_dir} used --limit: a trial run is never scored")
    if build.get("short") or build.get("complete") is False:
        raise ProvenanceError(
            f"the build in {copies_dir} is incomplete (short: {build.get('short')}): STOP and "
            "report the shortfall; an incomplete build is never scored"
        )
    if is_test or copies_dir.name == "test":
        expected = {t: COPIES_PER_TYPE for t in TYPES}
        if build.get("per_type") != COPIES_PER_TYPE or build.get("made") != expected:
            raise ProvenanceError(
                f"the test set must hold {COPIES_PER_TYPE} copies of each of {len(TYPES)} "
                f"types; the build made {build.get('made')}"
            )
        _complete("the test copies", run_dir, summary, EXPECTED_COUNTS["copies"])
        if _norm(summary.get("source", "")) != _norm(copies_dir / "copies"):
            raise ProvenanceError(
                f"the detector run in {run_dir} was not made on {copies_dir / 'copies'} "
                f"(it records {summary.get('source')!r})"
            )
    return is_test or copies_dir.name == "test"


def check_populations(
    copies_dir: Path | None, copies_run: Path | None, genuine_specs: list[str]
) -> list[str]:
    """Check every population this scoring touches; return the registered ones among them
    ("copies", "test", "shift"). A malformed --genuine spec is left for the caller to report."""
    registered = []
    if copies_dir and check_copies(copies_dir, copies_run):
        registered.append("copies")
    for spec in genuine_specs:
        name, _, folder = spec.partition("=")
        if name and folder and check_genuine(name, Path(folder)) and name not in registered:
            registered.append(name)
    return registered


# ---------------------------------------------------------------- hashes of the inputs


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder_sha256(folder: Path, pattern: str) -> str:
    """SHA-256 over each matching file's name and bytes, in sorted order."""
    h = hashlib.sha256()
    for path in sorted(Path(folder).glob(pattern)):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def run_hashes(run_dir: Path) -> dict[str, str]:
    return {name: file_sha256(Path(run_dir) / name) for name in RUN_FILES}


def copies_hashes(copies_dir: Path) -> dict[str, str]:
    copies_dir = Path(copies_dir)
    return {
        "keys": folder_sha256(copies_dir / "keys", "copy_*.json"),
        "build_run.json": file_sha256(copies_dir / "build_run.json"),
    }
