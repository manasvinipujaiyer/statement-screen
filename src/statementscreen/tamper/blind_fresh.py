"""Are the Week 7 blind source statements really new? A read-only check, before anything is sealed.

    uv run python -m statementscreen.tamper.blind_fresh \\
        data/blind_src/default_v1 data/blind_src/karan_v1

Compares every statement of the new batches with every statement of every earlier batch found
under --data (default data/): any folder one or two levels down that holds a manifest.csv and a
transactions.parquet. The new batches are left out of the earlier set, and are also compared
with each other. Three things are compared, and any match fails the check:

  seed     the statement's own seed from manifest.csv (a different main seed gives different
           statement seeds; the same one gives the same statement);
  content  SHA-256 of the statement's rows (date, narration, amount, balance), in row order;
  money    the same without the narration: the same statement written in another narration
           dialect has the same dates, amounts and balances.

Statement ids prove nothing here: every batch numbers its statements stmt_0001, stmt_0002, ...
It prints counts and statement ids only, never a narration or an amount. A manifest with no
transactions.parquet beside it cannot be compared; it is listed as skipped, not hidden.
Exit 0: nothing matches; 1: something matches; 2: the check could not be made.

A pre-seal audit, not part of the blind run: unlike the detector, verdict and scoring steps it
reads the earlier batches under data/ (manifests and transactions, never a PDF or the sealed
key, which does not exist yet). The run itself never calls it.
"""

import argparse
import csv
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

COLUMNS = ["statement_id", "row", "posted", "description", "amount_paise", "balance_paise"]
PATTERNS = ("*/manifest.csv", "*/*/manifest.csv")
KINDS = ("seed", "content", "money")


class FreshnessError(Exception):
    pass


@dataclass
class Result:
    new_statements: int = 0
    compared: list[tuple[str, int]] = field(default_factory=list)  # earlier batch, statements
    skipped: list[str] = field(default_factory=list)
    matches: list[tuple[str, str, str]] = field(default_factory=list)  # kind, new, other

    def count(self, kind: str) -> int:
        return sum(1 for k, _, _ in self.matches if k == kind)

    @property
    def ok(self) -> bool:
        return not self.matches


def statements(batch: Path) -> list[tuple[str, str, str, str]]:
    """(statement_id, seed, content hash, money hash) for each statement of one batch."""
    with (batch / "manifest.csv").open(newline="", encoding="utf-8") as f:
        seeds = {r["statement_id"]: r["seed"] for r in csv.DictReader(f)}
    frame = pl.read_parquet(batch / "transactions.parquet", columns=COLUMNS)
    frame = frame.sort("statement_id", "row")
    full: dict = {}
    money: dict = {}
    for sid, _, posted, description, amount, balance in frame.iter_rows():
        full.setdefault(sid, hashlib.sha256()).update(
            f"{posted}|{description}|{amount}|{balance}\n".encode()
        )
        money.setdefault(sid, hashlib.sha256()).update(f"{posted}|{amount}|{balance}\n".encode())
    if set(seeds) != set(full):
        raise FreshnessError(
            f"{batch}: manifest.csv and transactions.parquet name different statements"
        )
    return [
        (sid, seeds[sid], full[sid].hexdigest(), money[sid].hexdigest()) for sid in sorted(seeds)
    ]


def find_batches(data: Path, exclude: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """Earlier batches under `data`, and the ones that could not be compared (no transactions)."""
    left_out = {Path(p).resolve() for p in exclude}
    found: set[Path] = set()
    skipped: set[Path] = set()
    for pattern in PATTERNS:
        for manifest in data.glob(pattern):
            folder = manifest.parent.resolve()
            if folder in left_out:
                continue
            (found if (folder / "transactions.parquet").is_file() else skipped).add(folder)
    return sorted(found), sorted(skipped)


def check(new: Sequence[Path], earlier: Sequence[Path]) -> Result:
    result = Result()
    seen: dict[tuple[str, str], str] = {}
    for batch in earlier:
        rows = statements(batch)
        result.compared.append((batch.name, len(rows)))
        for sid, *values in rows:
            for kind, value in zip(KINDS, values, strict=True):
                seen.setdefault((kind, value), f"{batch.name}/{sid}")
    for batch in new:
        rows = statements(batch)
        if not rows:
            raise FreshnessError(f"{batch} holds no statements")
        result.new_statements += len(rows)
        for sid, *values in rows:
            label = f"{batch.name}/{sid}"
            for kind, value in zip(KINDS, values, strict=True):
                other = seen.get((kind, value))
                if other is not None:
                    result.matches.append((kind, label, other))
                else:
                    seen[(kind, value)] = label
    return result


def report(result: Result, batches: int) -> str:
    lines = [f"Fresh-source check: {result.new_statements} new statements in {batches} batches"]
    total = sum(n for _, n in result.compared)
    names = ", ".join(f"{name} ({n})" for name, n in result.compared)
    lines.append(
        f"Compared with {len(result.compared)} earlier batches, {total} statements: {names}"
    )
    if result.skipped:
        lines.append("Skipped, no transactions.parquet: " + ", ".join(result.skipped))
    lines.append(
        f"Matches: seed {result.count('seed')}, content {result.count('content')}, "
        f"money (same dates, amounts and balances) {result.count('money')}"
    )
    lines += [f"  {kind}: {new} = {other}" for kind, new, other in result.matches[:20]]
    if len(result.matches) > 20:
        lines.append(f"  ... and {len(result.matches) - 20} more")
    lines.append("FRESH: no new statement matches any earlier one" if result.ok else "NOT FRESH")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.blind_fresh",
        description="Check that new source batches share no statement with any earlier batch.",
    )
    p.add_argument("batches", nargs="+", type=Path, help="the new batch folders")
    p.add_argument("--data", type=Path, default=Path("data"), help="where earlier batches live")
    args = p.parse_args(argv)
    try:
        for batch in args.batches:
            for name in ("manifest.csv", "transactions.parquet"):
                if not (batch / name).is_file():
                    raise FreshnessError(f"{batch / name} not found")
        earlier, skipped = find_batches(args.data, args.batches)
        if not earlier:
            raise FreshnessError(
                f"no earlier batch (manifest.csv + transactions.parquet) found under {args.data}: "
                "a check against nothing proves nothing"
            )
        result = check(args.batches, earlier)
        result.skipped = [s.name for s in skipped]
    except (FreshnessError, OSError, pl.exceptions.PolarsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(report(result, len(args.batches)))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
