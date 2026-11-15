"""Evaluation batches: whole batches outside split_v1, such as the Week 4 shift batch.

    uv run python -m statementscreen.eval_batches register data/week4_shift_n300/ \\
        data/parsed_shift_n300/ --name shift_karan_v1 --reference data/week1_n2000/manifest.csv

An evaluation batch is used whole (no train/dev/test inside it). It must never let a
statement of the reference batch (the one split_v1 was built from) be evaluated outside
its split, so the split guard can't be bypassed. Statement ids don't prove that: every
batch numbers its statements stmt_0001, stmt_0002, ... So:

Registering (this command, the scoring side: it reads manifests) refuses
  - the reference batch itself (its manifest SHA-256 is the one in split_v1.json);
  - a batch sharing any statement with the reference batch (same per-statement seed);
  - a parsed folder whose statements are not exactly the batch's.
It checks that --reference really is split_v1's manifest, then writes
splits/eval_<name>.json (never overwritten): the manifest's SHA-256, the SHA-256 of the
sorted statement ids, and a fingerprint of the parsed rows (date, amount and balance of
every row).

Loading:
  - load_eval_for_parsed(name, parsed): for pattern detection, which never reads a
    manifest. Refuses a parse of the reference batch, and any parse whose ids or row
    fingerprint differ from the registered ones (e.g. reference statements re-parsed
    under the same ids).
  - load_eval(name, manifest): for scoring. Refuses a manifest whose SHA-256 differs.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.splits import SPLITS_DIR, VERSION, SplitError, ids_sha256, paths, sha256_file

FINGERPRINT_COLUMNS = ["statement_id", "row", "posted", "amount_paise", "balance_paise"]


def eval_path(name: str, splits_dir: Path = SPLITS_DIR) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise SplitError(f"evaluation batch names use a-z, 0-9 and _ only, got {name!r}")
    return splits_dir / f"eval_{name}.json"


def parsed_ids(parsed: Path) -> list[str]:
    statements = pl.read_parquet(parsed / "statements.parquet", columns=["statement_id"])
    return statements["statement_id"].to_list()


def rows_fingerprint(parsed: Path) -> str:
    """SHA-256 of every parsed row's statement, row number, date, amount and balance."""
    rows = pl.read_parquet(parsed / "rows.parquet", columns=FINGERPRINT_COLUMNS)
    return hashlib.sha256(rows.sort("statement_id", "row").write_csv().encode()).hexdigest()


def _split_meta(splits_dir: Path) -> dict:
    return json.loads(paths(splits_dir, VERSION)[1].read_text(encoding="utf-8"))


def _manifest(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- registering


def register(name: str, batch: Path, parsed: Path, reference: Path,
             splits_dir: Path = SPLITS_DIR) -> dict:  # fmt: skip
    out = eval_path(name, splits_dir)
    if out.exists():
        raise SplitError(f"refusing to overwrite registered evaluation batch {out}")
    meta = _split_meta(splits_dir)
    if sha256_file(reference) != meta["manifest_sha256"]:
        raise SplitError(f"{reference} is not the manifest split_{VERSION} was built from")
    manifest = batch / "manifest.csv"
    sha = sha256_file(manifest)
    if sha == meta["manifest_sha256"]:
        raise SplitError(f"{batch} is the reference batch of split_{VERSION}; use its splits "
                         "(train/dev), not an evaluation batch")  # fmt: skip
    rows = _manifest(manifest)
    shared = {r["seed"] for r in rows} & {r["seed"] for r in _manifest(reference)}
    if shared:
        raise SplitError(f"{batch} shares {len(shared)} statements (same seed) with the "
                         f"reference batch of split_{VERSION}; refusing")  # fmt: skip
    ids = [r["statement_id"] for r in rows]
    if ids_sha256(parsed_ids(parsed)) != ids_sha256(ids):
        raise SplitError(f"{parsed} does not hold exactly the statements of {manifest}")
    entry = {
        "name": name,
        "batch": batch.name,
        "manifest_sha256": sha,
        "statements": len(ids),
        "statement_ids_sha256": ids_sha256(ids),
        "parsed_rows_sha256": rows_fingerprint(parsed),
        "reference_manifest_sha256": meta["manifest_sha256"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry


# ---------------------------------------------------------------- loading


def _open(name: str, splits_dir: Path) -> dict:
    path = eval_path(name, splits_dir)
    if not path.exists():
        raise SplitError(f"no registered evaluation batch {name!r} ({path} not found)")
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_for_parsed(name: str, parsed: Path, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Sorted statement ids of evaluation batch `name`, after checking that the parsed
    folder is that batch. Never reads a manifest."""
    entry = _open(name, splits_dir)
    ids = parsed_ids(parsed)
    if ids_sha256(ids) == _split_meta(splits_dir)["statement_ids_sha256"]:
        raise SplitError(f"{parsed} is a parse of the reference batch of split_{VERSION}; "
                         "run it with --split, which keeps the test split closed")  # fmt: skip
    if ids_sha256(ids) != entry["statement_ids_sha256"]:
        raise SplitError(f"{parsed} does not hold the statements of evaluation batch {name!r}")
    if rows_fingerprint(parsed) != entry["parsed_rows_sha256"]:
        raise SplitError(f"{parsed} has the ids of evaluation batch {name!r} but different "
                         "rows; parsed from another batch?")  # fmt: skip
    return sorted(ids)


def load_eval(name: str, manifest: Path, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """Sorted statement ids of evaluation batch `name`, after checking the manifest."""
    entry = _open(name, splits_dir)
    actual = sha256_file(manifest)
    if actual != entry["manifest_sha256"]:
        raise SplitError(f"{manifest} SHA-256 {actual[:12]}... is not evaluation batch "
                         f"{name!r} ({entry['manifest_sha256'][:12]}...)")  # fmt: skip
    return sorted(r["statement_id"] for r in _manifest(manifest))


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.eval_batches",
        description="Register a whole batch for evaluation outside split_v1.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register", help="register a generated and parsed batch")
    r.add_argument("batch", type=Path, help="generated batch folder (with manifest.csv)")
    r.add_argument("parsed", type=Path, help="its parse (statementscreen.parse output)")
    r.add_argument("--name", required=True, help="e.g. shift_karan_v1")
    r.add_argument("--reference", type=Path, required=True,
                   help="the manifest split_v1 was built from")  # fmt: skip
    r.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    args = p.parse_args(argv)
    try:
        e = register(args.name, args.batch, args.parsed, args.reference, args.splits_dir)
    except (SplitError, FileNotFoundError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"Registered evaluation batch {e['name']}: {e['statements']} statements from "
          f"{e['batch']} -> {eval_path(e['name'], args.splits_dir)}")  # fmt: skip
    print(f"  manifest {e['manifest_sha256'][:12]}..., parsed rows "
          f"{e['parsed_rows_sha256'][:12]}...")  # fmt: skip
    print("  no statement shared with the reference batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
