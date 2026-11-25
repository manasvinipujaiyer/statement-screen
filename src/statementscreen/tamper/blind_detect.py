"""Run the frozen Week 6 detector over the sealed Week 7 blind PDFs, once.

    uv run python -m statementscreen.tamper.blind_detect --pdfs data/blind/pdfs \\
        --out data/blind/detect --preregistration reports/week7_blind_preregistration.json

Before reading a single PDF it checks, with no override:
  - the pre-registration is the pushed one, its commit is pushed and in this history, every source
    file and frozen artefact hashes as recorded, the rule version matches, and the detector
    fingerprint equals both the recorded one and the one Week 6 pre-registered;
  - the folder holds exactly the sealed PDFs (names and bytes: the PDF-set hash).
Then it runs tamper/run.py's driver one PDF at a time (population "blind") and writes
findings.parquet, items.parquet and detect_run.json in --out. A finished run is never
overwritten. An interrupted one is continued with --resume, which re-checks the fingerprint and
skips finished chunks; that is not a re-run.

This command never opens the sealed key, and nothing here knows where it is: the detector sees
the opaque PDFs and nothing else.
"""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.tamper import blind_prereg as bpr
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import run
from statementscreen.tamper.blind_common import BlindError

POPULATION = "blind"


def detect(prereg: dict, pdfs: Path, out: Path, resume: bool = False) -> dict:
    """Run the detector over the sealed PDFs (refusing any other set). Returns the run summary."""
    hashes = bpr.check_pdf_set(prereg, pdfs)
    paths = [Path(pdfs) / f"{name}.pdf" for name in sorted(hashes)]
    return run.run(paths, Path(out), Path(pdfs), None, resume, False, POPULATION)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.blind_detect",
        description="Run the frozen detector over the sealed blind PDFs. Never opens the key.",
    )
    p.add_argument("--pdfs", type=Path, default=Path("data/blind/pdfs"))
    p.add_argument("--out", type=Path, default=Path("data/blind/detect"))
    p.add_argument("--preregistration", type=Path, default=bpr.DEFAULT_OUT)
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = p.parse_args(argv)
    try:
        prereg = bpr.load_and_verify(args.preregistration)
        summary = detect(prereg, args.pdfs, args.out, args.resume)
    except (
        BlindError,
        run.RunError,
        fp.StaleFindings,
        OSError,
        KeyError,
        ValueError,
        subprocess.CalledProcessError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    per = summary["seconds_per_pdf"]
    print(
        f"Ran the detector on {summary['pdfs']} PDFs in {summary['seconds_total']} s "
        f"({per['mean']} s each, slowest {per['max']} s; peak memory "
        f"{summary['peak_memory_mb']} MB): {summary['findings']} findings on "
        f"{summary['items_with_findings']} PDFs; crashed: {summary['crashed'] or 'none'}"
    )
    print(f"Wrote {args.out}/" + ", ".join(run.OUTPUTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
