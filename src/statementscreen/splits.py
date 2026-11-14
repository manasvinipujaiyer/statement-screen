"""Lock a train/dev/test split of a batch, and load it safely.

    uv run python -m statementscreen.splits data/week1_n2000/manifest.csv

Splits by statement (never by row), 70/15/15, stratified by layout x customer type, with
a fixed seed. This module is the only one besides scoring that reads manifest.csv, and
only for statement_id, layout and customer_type. Writes, and refuses to overwrite:

    splits/split_v1.csv   statement_id, split
    splits/split_v1.json  seed, fractions, counts, SHA-256 of the manifest and of the
                          sorted statement-id list

Loaders take a split name and refuse "test" unless STATEMENTSCREEN_OPEN_TEST=1 is set.
They check this before reading anything.

- load_split(name, manifest) is for scoring: it refuses a manifest whose SHA-256
  differs from the one the split was built from.
- load_split_for_parsed(name, parsed_ids) is for pattern detection, which must never read
  the manifest: it refuses unless the parsed folder's statement ids are exactly the
  split's statements (checked by the SHA-256 of the sorted id list).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

SPLITS = ("train", "dev", "test")
FRACTIONS = {"train": 0.70, "dev": 0.15, "test": 0.15}
DEFAULT_SEED = 42
VERSION = "v1"
SPLITS_DIR = Path("splits")
OPEN_TEST_ENV = "STATEMENTSCREEN_OPEN_TEST"


class SplitError(RuntimeError):
    """A split can't be built or loaded safely."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha256(ids: Iterable[str]) -> str:
    """SHA-256 of the sorted statement ids, one per line."""
    return hashlib.sha256("".join(f"{i}\n" for i in sorted(ids)).encode()).hexdigest()


def read_manifest(manifest: Path) -> list[dict[str, str]]:
    """statement_id, layout and customer_type from manifest.csv; nothing else."""
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = [{k: r[k] for k in ("statement_id", "layout", "customer_type")}
                for r in csv.DictReader(f)]  # fmt: skip
    ids = [r["statement_id"] for r in rows]
    if not rows:
        raise SplitError(f"{manifest} has no statements")
    if len(set(ids)) != len(ids):
        raise SplitError(f"{manifest} repeats statement ids")
    return rows


def _allocate(n: int) -> dict[str, int]:
    """Counts per split for one stratum; dev and test get at least 1 when n >= 3."""
    dev = round(n * FRACTIONS["dev"])
    test = round(n * FRACTIONS["test"])
    if n >= 3:
        dev, test = max(dev, 1), max(test, 1)
    return {"train": n - dev - test, "dev": dev, "test": test}


def assign(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    """statement_id -> split. Strata and ids are sorted first, so only the seed matters."""
    strata: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        strata.setdefault((r["layout"], r["customer_type"]), []).append(r["statement_id"])
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for key in sorted(strata):
        ids = sorted(strata[key])
        shuffled = [ids[i] for i in rng.permutation(len(ids))]
        start = 0
        for name, count in _allocate(len(ids)).items():
            for sid in shuffled[start : start + count]:
                out[sid] = name
            start += count
    return out


def summarise(rows: list[dict[str, str]], assigned: dict[str, str]) -> dict:
    counts: dict[str, dict] = {}
    for name in SPLITS:
        members = [r for r in rows if assigned[r["statement_id"]] == name]
        counts[name] = {
            "statements": len(members),
            "by_layout": dict(sorted(Counter(r["layout"] for r in members).items())),
            "by_customer_type": dict(sorted(Counter(r["customer_type"] for r in members).items())),
        }
    return counts


def build(manifest: Path, seed: int = DEFAULT_SEED) -> tuple[dict[str, str], dict]:
    """Return (assignment, metadata) without writing anything."""
    rows = read_manifest(manifest)
    assigned = assign(rows, seed)
    meta = {
        "version": VERSION,
        "seed": seed,
        "fractions": FRACTIONS,
        "stratified_by": ["layout", "customer_type"],
        "manifest": f"{manifest.parent.name}/{manifest.name}",
        "manifest_sha256": sha256_file(manifest),
        "statement_ids_sha256": ids_sha256(assigned),
        "statements": len(assigned),
        "counts": summarise(rows, assigned),
    }
    return assigned, meta


def paths(splits_dir: Path = SPLITS_DIR, version: str = VERSION) -> tuple[Path, Path]:
    return splits_dir / f"split_{version}.csv", splits_dir / f"split_{version}.json"


def write(assigned: dict[str, str], meta: dict, splits_dir: Path = SPLITS_DIR) -> None:
    csv_path, json_path = paths(splits_dir, meta["version"])
    existing = [p for p in (csv_path, json_path) if p.exists()]
    if existing:
        raise SplitError(f"refusing to overwrite locked split: {', '.join(map(str, existing))}")
    splits_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["statement_id", "split"])
        w.writerows(sorted(assigned.items()))
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- loading


def _open(name: str, splits_dir: Path, version: str) -> tuple[dict[str, str], dict]:
    if name not in SPLITS:
        raise SplitError(f"unknown split {name!r}; expected one of {', '.join(SPLITS)}")
    if name == "test" and os.environ.get(OPEN_TEST_ENV) != "1":
        raise SplitError(f"the test split is closed; set {OPEN_TEST_ENV}=1 to open it")
    csv_path, json_path = paths(splits_dir, version)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as f:
        assigned = {r["statement_id"]: r["split"] for r in csv.DictReader(f)}
    if ids_sha256(assigned) != meta["statement_ids_sha256"]:
        raise SplitError(f"{csv_path} does not match {json_path}")
    return assigned, meta


def _members(assigned: dict[str, str], name: str) -> list[str]:
    return sorted(sid for sid, s in assigned.items() if s == name)


def load_split(name: str, manifest: Path, splits_dir: Path = SPLITS_DIR,
               version: str = VERSION) -> list[str]:  # fmt: skip
    """Sorted statement ids in split `name`, after checking the manifest's SHA-256."""
    assigned, meta = _open(name, splits_dir, version)
    actual = sha256_file(manifest)
    if actual != meta["manifest_sha256"]:
        raise SplitError(f"{manifest} SHA-256 {actual[:12]}... does not match the split's "
                         f"{meta['manifest_sha256'][:12]}...; this split was built from a "
                         f"different batch")  # fmt: skip
    return _members(assigned, name)


def load_split_for_parsed(name: str, parsed_ids: Iterable[str], splits_dir: Path = SPLITS_DIR,
                          version: str = VERSION) -> list[str]:  # fmt: skip
    """Sorted statement ids in split `name`, after checking the parsed folder holds exactly
    the statements the split was built from. Never reads the manifest."""
    assigned, meta = _open(name, splits_dir, version)
    parsed = set(parsed_ids)
    if ids_sha256(parsed) != meta["statement_ids_sha256"]:
        missing, extra = len(set(assigned) - parsed), len(parsed - set(assigned))
        raise SplitError(f"parsed statements don't match split_{version} ({missing} missing, "
                         f"{extra} not in the split); parsed from another batch?")  # fmt: skip
    return _members(assigned, name)


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m statementscreen.splits",
                                description="Lock a train/dev/test split of a batch.")  # fmt: skip
    p.add_argument("manifest", type=Path, help="the batch's manifest.csv")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=SPLITS_DIR, help="folder for split files")
    args = p.parse_args(argv)
    try:
        assigned, meta = build(args.manifest, args.seed)
        write(assigned, meta, args.out)
    except (SplitError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    csv_path, json_path = paths(args.out, meta["version"])
    print(f"Wrote {csv_path} and {json_path} (seed {meta['seed']}, "
          f"manifest {meta['manifest_sha256'][:12]}...)")  # fmt: skip
    for name in SPLITS:
        c = meta["counts"][name]
        print(f"  {name:5} {c['statements']:5}  layouts {c['by_layout']}  "
              f"types {c['by_customer_type']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
