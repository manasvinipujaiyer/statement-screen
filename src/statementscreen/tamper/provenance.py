"""Where each scored population really comes from, read from the runs' own records AND from
what they actually contain.

The one-time test and shift scoring must not depend on what a population is CALLED on the
command line, or on what a run says about itself. Two kinds of check, both outside
tamper/detect/ (so they can't change a finding or the detector fingerprint):

Records. Every detector run records what it covered (detect_run.json: the population or
split, the source folder, the number of PDFs, any --limit) and every build records what it
made (build_run.json: the set, the seed, the counts, any shortfall):
  * a genuine population's recorded identity must equal its --genuine name;
  * a run with no recorded identity is refused (re-run with --split or --population);
  * a --limit trial run is never scored;
  * the registered populations must be complete and come from the registered folders:
    300 genuine test, 300 shift, and 800 copies (100 for each of the eight types, built with
    seed 6 and without a shortfall);
  * a PDF folder that is the shift batch can only be scored as "shift";
  * a build in which more than 5 sources failed (build_errors) is a STOP: it is refused.

Content. The statement ids a run covers are compared with the split and the shift manifest.
Statement ids are unique only WITHIN a batch: the shift batch is stmt_0001..stmt_0300, the
same naming as the main batch, so its ids overlap the main batch's. Identity is therefore
decided per batch:
  * "test" must be exactly the test statements of the main batch, and "shift" exactly the
    shift batch's manifest ids and come from the shift folder;
  * any other population (dev, train, ...) may contain no test statement, so test PDFs cannot
    be scored under another label, whatever the run says about itself. (Shift PDFs copied to
    another folder are caught by that folder rule and by their overlap with the test ids, not
    by their own ids.)
  * a test copies set must be built only from test statements, any other set from none;
  * the registered runs must record a SHA-256 for every PDF (items.parquet), and the report
    records one combined hash, so it shows which PDFs were detected.

The split file is read only for its statement ids (never data), and only so that anything
that touches those statements can be refused.

Also here: the SHA-256 of every input that cannot be hashed before step 6 (the test keys, the
build summary, the detector runs, the manifests).
"""

import csv
import hashlib
import json
import os
from pathlib import Path

import polars as pl

from statementscreen.splits import SPLITS_DIR
from statementscreen.tamper.copy import TYPES

REGISTERED = ("test", "shift")  # the genuine populations that are scored once, in step 6
COPIES_PER_TYPE = 100
EXPECTED_BUILD_SEED = 6
MAX_BUILD_ERRORS = 5  # sources the builder may replace after a build error; more is a STOP
EXPECTED_COUNTS = {"test": 300, "shift": 300, "copies": COPIES_PER_TYPE * len(TYPES)}
EXPECTED_SOURCES = {"test": "data/week1_n2000/pdfs", "shift": "data/week4_shift_n300/pdfs"}
SPLIT_FILE = "split_v1.csv"
SHIFT_MANIFEST = "data/week4_shift_n300/manifest.csv"
RUN_FILES = ("findings.parquet", "items.parquet", "detect_run.json")


class ProvenanceError(RuntimeError):
    """A population is not what its name says, or is not complete."""


def _norm(path: object) -> str:
    return os.path.normpath(str(path)).replace(os.sep, "/")


def _statement_ids(path: Path, keep) -> set[str]:
    try:
        with Path(path).open(newline="", encoding="utf-8") as f:
            return {r["statement_id"] for r in csv.DictReader(f) if keep(r)}
    except FileNotFoundError as e:
        raise ProvenanceError(f"{path} not found: it is needed to check what a run covers") from e


def split_test_ids(splits_dir: Path = SPLITS_DIR) -> set[str]:
    """The test split's statement ids, read straight from the split file (ids only)."""
    return _statement_ids(Path(splits_dir) / SPLIT_FILE, lambda r: r["split"] == "test")


def shift_batch_ids(manifest: Path = Path(SHIFT_MANIFEST)) -> set[str]:
    """The shift batch's statement ids, from its manifest."""
    return _statement_ids(manifest, lambda r: True)


def is_shift_folder(folder: Path) -> bool:
    """True if this is the registered shift batch's PDF folder."""
    return _norm(folder) == _norm(EXPECTED_SOURCES["shift"])


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


def _items(run_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(Path(run_dir) / "items.parquet")


def _complete(what: str, run_dir: Path, summary: dict, expected: int) -> None:
    found = _items(run_dir)["item"].n_unique()
    if summary.get("pdfs") != expected or found != expected:
        raise ProvenanceError(
            f"{what} must cover {expected} PDFs; the run in {run_dir} covers {found} "
            f"(it records {summary.get('pdfs')}): an incomplete run is never scored"
        )


def _hashed(what: str, run_dir: Path) -> None:
    items = _items(run_dir)
    if "sha256" not in items.columns or items["sha256"].null_count() or items.height == 0:
        raise ProvenanceError(
            f"{what}: the run in {run_dir} records no SHA-256 for its PDFs; run tamper.run again"
        )


def pdf_hashes(run_dir: Path) -> str | None:
    """One SHA-256 over every (item, PDF hash) of a run, sorted; None for a run that predates
    the per-PDF hashes."""
    items = _items(run_dir)
    if "sha256" not in items.columns:
        return None
    h = hashlib.sha256()
    pairs = zip(items["item"].to_list(), items["sha256"].to_list(), strict=True)
    for item, digest in sorted(pairs):
        h.update(f"{item}\0{digest}\n".encode())
    return h.hexdigest()


def check_genuine(name: str, run_dir: Path) -> bool:
    """Check one --genuine population against its run's own record and its contents; True if
    it is one of the registered populations (test, shift)."""
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
    if is_shift_folder(source) and name != "shift":
        raise ProvenanceError(f"the PDFs in {source} are the shift batch: score them as 'shift'")
    ids = set(_items(run_dir)["item"].to_list())
    if name in REGISTERED:
        if source != _norm(EXPECTED_SOURCES[name]):
            raise ProvenanceError(
                f"population {name!r} must come from {EXPECTED_SOURCES[name]}, "
                f"not {summary.get('source')!r}"
            )
        _complete(f"population {name!r}", run_dir, summary, EXPECTED_COUNTS[name])
        expected = {"test": split_test_ids, "shift": shift_batch_ids}[name]()
        if ids != expected:
            raise ProvenanceError(
                f"population {name!r} must be exactly the {name} statements; the run covers "
                f"{len(ids - expected)} that are not and misses {len(expected - ids)}"
            )
        _hashed(f"population {name!r}", run_dir)
    else:
        leaked = sorted(ids & split_test_ids())
        if leaked:
            raise ProvenanceError(
                f"the run in {run_dir} covers test statements (e.g. {leaked[:3]}) but is "
                f"labelled {name!r}: the test statements are scored once, as 'test'"
            )
    return name in REGISTERED


def _key_sources(copies_dir: Path) -> set[str]:
    keys = sorted((Path(copies_dir) / "keys").glob("copy_*.json"))
    return {json.loads(p.read_text(encoding="utf-8"))["source"] for p in keys}


def check_copies(copies_dir: Path, copies_run: Path | None = None) -> bool:
    """Check a copies set against its build record, its answer keys and its detector run; True
    if it is the test set (scored once, in step 6)."""
    copies_dir = Path(copies_dir)
    run_dir = Path(copies_run) if copies_run else copies_dir / "detect"
    summary = run_summary(run_dir)
    build_path = copies_dir / "build_run.json"
    if not build_path.is_file():
        raise ProvenanceError(f"{build_path} not found: scoring needs the build's own record")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    is_test = build.get("set") == "test"
    named_test = copies_dir.name == "test"
    if named_test and not is_test:
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
    errors = build.get("build_errors") or []
    if build.get("stopped") or len(errors) > MAX_BUILD_ERRORS:
        raise ProvenanceError(
            f"{len(errors)} sources failed to build in {copies_dir} (more than "
            f"{MAX_BUILD_ERRORS} is a STOP): report it; the set is never scored"
        )
    recorded = summary.get("population") or summary.get("split")
    if recorded is not None and (recorded == "test") != (is_test or named_test):
        raise ProvenanceError(
            f"the detector run in {run_dir} is labelled {recorded!r} but the copies in "
            f"{copies_dir} are a {'test' if is_test or named_test else 'non-test'} set"
        )
    sources = _key_sources(copies_dir)
    if is_test or named_test:
        expected = {t: COPIES_PER_TYPE for t in TYPES}
        if build.get("per_type") != COPIES_PER_TYPE or build.get("made") != expected:
            raise ProvenanceError(
                f"the test set must hold {COPIES_PER_TYPE} copies of each of {len(TYPES)} "
                f"types; the build made {build.get('made')}"
            )
        if build.get("seed") != EXPECTED_BUILD_SEED:
            raise ProvenanceError(
                f"the test set must be built with seed {EXPECTED_BUILD_SEED}, not "
                f"{build.get('seed')!r}"
            )
        outside = sorted(sources - split_test_ids())
        if outside:
            raise ProvenanceError(
                f"the copies in {copies_dir} are built from statements that are not test "
                f"statements (e.g. {outside[:3]})"
            )
        _complete("the test copies", run_dir, summary, EXPECTED_COUNTS["copies"])
        _hashed("the test copies", run_dir)
        if _norm(summary.get("source", "")) != _norm(copies_dir / "copies"):
            raise ProvenanceError(
                f"the detector run in {run_dir} was not made on {copies_dir / 'copies'} "
                f"(it records {summary.get('source')!r})"
            )
    else:
        leaked = sorted(sources & split_test_ids())
        if leaked:
            raise ProvenanceError(
                f"the copies in {copies_dir} are built from test statements (e.g. {leaked[:3]}) "
                f"but its build record says set={build.get('set')!r}"
            )
    return is_test or named_test


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


def run_hashes(run_dir: Path) -> dict[str, str | None]:
    """The run's three files, and one combined hash of its PDFs (None for an older run)."""
    hashes: dict[str, str | None] = {name: file_sha256(Path(run_dir) / name) for name in RUN_FILES}
    hashes["pdfs"] = pdf_hashes(run_dir)
    return hashes


def copies_hashes(copies_dir: Path) -> dict[str, str]:
    copies_dir = Path(copies_dir)
    return {
        "keys": folder_sha256(copies_dir / "keys", "copy_*.json"),
        "build_run.json": file_sha256(copies_dir / "build_run.json"),
    }
