"""Fingerprint a small generated batch: the SHA-256 of every file it writes.

Used to show that a change to the generator leaves today's output byte-identical. The
fingerprint was recorded ONCE, with the generator as it was before narration dialects
were added (Week 4):

    uv run python -m statementscreen.batch_hash --record tests/data/default_batch_sha256.json

tests/test_dialect.py regenerates the same batch and compares every file with it.
run_summary.json is not part of a batch written by write_batch, so it is never hashed
(it records timings and differs on purpose).
"""

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from statementscreen.batch import BatchConfig, plan_batch

# Small but varied: every applicable hard case in every statement, so renames, bonuses
# and EMI bounces get a chance to appear. The default mix of seed 42's first 6 statements
# has no self-employed customer, so a second batch covers business income and tax.
BATCHES = {
    "mixed": {"seed": 42, "n": 6, "hard_case_rate": 1.0, "mix": None},
    "self_employed": {"seed": 42, "n": 3, "hard_case_rate": 1.0,
                      "mix": {"salaried": 0.0, "gig": 0.0, "self_employed": 1.0}},
}  # fmt: skip


def file_hashes(folder: Path) -> dict[str, str]:
    """{relative path: sha256} for every file under folder, in sorted order."""
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def fingerprint_one(name: str, folder: Path, **write_options) -> dict:
    """Write batch BATCHES[name] into folder (which must be empty) and hash it."""
    from statementscreen.writer import write_batch

    s = BATCHES[name]
    mix = {"mix": s["mix"]} if s["mix"] is not None else {}
    plans = plan_batch(BatchConfig(seed=s["seed"], hard_case_rate=s["hard_case_rate"], **mix),
                       s["n"])  # fmt: skip
    write_batch(plans, folder, **write_options)
    return {
        "settings": s,
        "plans": [f"{p.statement_id} {p.customer_type} {p.layout} {'|'.join(p.hard_cases)}"
                  for p in plans],
        "sha256": file_hashes(folder),
    }  # fmt: skip


def fingerprint(folder: Path, **write_options) -> dict:
    """Every batch in BATCHES, each in its own subfolder of folder."""
    return {name: fingerprint_one(name, folder / name, **write_options) for name in BATCHES}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.batch_hash",
        description="Record the fingerprint of a small default batch.",
    )
    p.add_argument("--record", type=Path, required=True, help="JSON file to write")
    args = p.parse_args(argv)
    if args.record.exists():
        print(f"error: {args.record} exists; it is recorded once and never overwritten",
              file=sys.stderr)  # fmt: skip
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        fp = fingerprint(Path(tmp))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, batch in fp.items():
        print(f"Recorded {len(batch['sha256'])} file hashes for batch '{name}' to {args.record}")
        for line in batch["plans"]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
