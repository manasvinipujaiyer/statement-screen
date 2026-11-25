"""Turn the blind run's cached findings into one verdict per PDF, before anything is unsealed.

    uv run python -m statementscreen.tamper.blind_verdicts --detect data/blind/detect \\
        --pdfs data/blind/pdfs --preregistration reports/week7_blind_preregistration.json \\
        --out reports/week7_blind_verdicts.csv

Writes the verdict file and, beside it, its SHA-256 (reports/week7_blind_verdicts.sha256). The
user commits and pushes both BEFORE the unseal step; scoring refuses a verdict file that differs.

One row per opaque name: verdict, verdict without file-history findings, the codes that fired
(not info), the checks that did not run, the detector's reasons, and the PDF's SHA-256. Nothing
in it depends on the answers.

It refuses (no override): a pre-registration that is not the pushed, verified one; PDFs that are
not the sealed set; detector output from another fingerprint, another population, a --limit trial,
a different PDF count, or hashes that differ from the PDFs on disk. It never opens the sealed
key (a test makes every way of reading that path fail and expects this step to pass), and the
verdict is a pure function of the findings under the pre-registered rule version.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.tamper import blind_prereg as bpr
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import BlindError, sha256_bytes
from statementscreen.tamper.scoring import Item, load_findings, score_item

COLUMNS = (
    "name",
    "verdict",
    "verdict_without_history",
    "finding_codes",
    "checks_not_run",
    "reasons",
    "pdf_sha256",
)
POPULATION = "blind"


def _check_run(prereg: dict, detect: Path, hashes: dict[str, str]) -> dict:
    """The detector output must be a complete, untrial run of the frozen detector on these PDFs."""
    info = json.loads((detect / "detect_run.json").read_text(encoding="utf-8"))
    fp.check(info.get("fingerprint"), prereg["detector_fingerprint"])  # raises StaleFindings
    if info.get("population") != POPULATION or info.get("limit") is not None:
        raise BlindError(
            f"{detect} is not a full blind run (population {info.get('population')!r}, "
            f"limit {info.get('limit')!r})"
        )
    items = pl.read_parquet(detect / "items.parquet")
    if sorted(items["item"].to_list()) != sorted(hashes) or info.get("pdfs") != len(hashes):
        raise BlindError(f"the detector output in {detect} does not cover exactly the sealed PDFs")
    rows = items.iter_rows(named=True)
    wrong = sorted(r["item"] for r in rows if r["sha256"] != hashes[r["item"]])
    if wrong:
        raise BlindError(
            f"{len(wrong)} PDFs changed since the detector read them, e.g. {wrong[:3]}"
        )
    return info


def make_verdicts(prereg: dict, detect: Path, pdfs: Path) -> list[dict]:
    """One row per sealed PDF, sorted by name. Reads PDFs' bytes (hashing), the cached findings
    and nothing else."""
    hashes = bpr.check_pdf_set(prereg, pdfs)
    info = _check_run(prereg, Path(detect), hashes)
    findings = load_findings(Path(detect) / "findings.parquet")
    crashed = frozenset(info.get("crashed", ()))
    rows = []
    for name in sorted(hashes):
        item = Item(name, name, "blind", "", findings=tuple(findings.get(name, ())),
                    crashed=name in crashed)  # fmt: skip
        s = score_item(item, prereg["rule_version"])
        counted = [r for r in s.verdict["reasons"] if r["strength"] != "info"]
        rows.append(
            {
                "name": name,
                "verdict": s.label,
                "verdict_without_history": s.label_without_history,
                "finding_codes": ";".join(
                    sorted({f.code for f in item.findings if f.strength != "info"})
                ),  # fmt: skip
                "checks_not_run": ";".join(s.unrun_checks),
                "reasons": " | ".join(r["reason"] for r in counted),
                "pdf_sha256": hashes[name],
            }
        )
    return rows


def write_verdicts(rows: list[dict], out: Path) -> str:
    """Write the CSV and its .sha256 beside it; never overwrite. Returns the CSV's SHA-256."""
    out = Path(out)
    sha_path = out.with_suffix(".sha256")
    if out.exists() or sha_path.exists():
        raise BlindError(f"{out} or {sha_path} exists: a verdict file is never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = sha256_bytes(out.read_bytes())
    sha_path.write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.blind_verdicts",
        description="One verdict per blind PDF, from the cached findings. Never opens the key.",
    )
    p.add_argument("--detect", type=Path, default=Path("data/blind/detect"))
    p.add_argument("--pdfs", type=Path, default=Path("data/blind/pdfs"))
    p.add_argument("--preregistration", type=Path, default=bpr.DEFAULT_OUT)
    p.add_argument("--out", type=Path, default=Path(bpr.VERDICTS))
    args = p.parse_args(argv)
    try:
        prereg = bpr.load_and_verify(args.preregistration)
        rows = make_verdicts(prereg, args.detect, args.pdfs)
        digest = write_verdicts(rows, args.out)
    except (
        BlindError,
        fp.StaleFindings,
        OSError,
        KeyError,
        ValueError,
        subprocess.CalledProcessError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    count = Counter(r["verdict"] for r in rows)
    print(f"Wrote {args.out}: {len(rows)} verdicts "
          + ", ".join(f"{k}: {v}" for k, v in sorted(count.items())))  # fmt: skip
    print(f"  SHA-256 {digest}  ({args.out.with_suffix('.sha256')})")
    print("  commit and push both files BEFORE the unseal step; nothing has been unsealed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
