"""Gate before any tampered copy: rebuild genuine PDFs with NO edit; all must be identical.

    uv run python -m statementscreen.tamper.rebuild_check data/week1_n2000/ --split dev

Each statement of the split is read back (rebuild.read_back) and drawn again by the
project's renderer with its printed text. The result must be byte-identical to the
original. Every statement that isn't is listed with the cause: the first drawn string
that differs (page, position, original vs rebuilt), a different page count, or a read-back
error. Nothing falls back to a looser comparison and no statement is skipped. Writes
reports/week6_rebuild_check_<split>.json; exits 1 unless every statement is identical.
"""

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import pdfplumber

from statementscreen.splits import SPLITS_DIR, SplitError, load_split
from statementscreen.synth.render import render_statement
from statementscreen.tamper.rebuild import RebuildError, read_back, runs


def rebuild_bytes(pdf: Path) -> bytes:
    rb = read_back(pdf)
    with tempfile.TemporaryDirectory() as tmp:
        out = render_statement(rb.statement, Path(tmp) / "rebuilt.pdf", rb.layout, rb.printed)
        return out.read_bytes()


def first_difference(original: Path, rebuilt: bytes) -> str:
    """Where the rebuilt PDF first differs from the original, in words."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rebuilt.pdf"
        path.write_bytes(rebuilt)
        with pdfplumber.open(original) as a, pdfplumber.open(path) as b:
            if len(a.pages) != len(b.pages):
                return f"page count {len(a.pages)} vs {len(b.pages)} rebuilt"
            for n, (pa, pb) in enumerate(zip(a.pages, b.pages, strict=True), start=1):
                ra, rb = runs(pa), runs(pb)
                for x, y in zip(ra, rb, strict=False):
                    if x != y:
                        return (f"page {n}: original {x.text!r} at ({x.x:.2f}, {x.y:.2f}) "
                                f"vs rebuilt {y.text!r} at ({y.x:.2f}, {y.y:.2f})")  # fmt: skip
                if len(ra) != len(rb):
                    return f"page {n}: {len(ra)} strings vs {len(rb)} rebuilt"
            meta_a, meta_b = a.metadata or {}, b.metadata or {}
            if meta_a != meta_b:
                return f"metadata differs: {meta_a} vs {meta_b}"
    return "same strings, pages and metadata, but the bytes differ"


def check(pdfs: list[Path]) -> dict:
    identical, differ = [], []
    for k, pdf in enumerate(pdfs, start=1):
        try:
            rebuilt = rebuild_bytes(pdf)
            if rebuilt == pdf.read_bytes():
                identical.append(pdf.stem)
            else:
                differ.append({"statement": pdf.stem, "cause": first_difference(pdf, rebuilt)})
        except RebuildError as e:
            differ.append({"statement": pdf.stem, "cause": f"read-back error: {e}"})
        if k % 50 == 0 or k == len(pdfs):
            print(f"  {k}/{len(pdfs)} checked", flush=True)
    return {"statements": len(pdfs), "byte_identical": len(identical),
            "share_identical": len(identical) / len(pdfs) if pdfs else None,
            "differ": differ}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.rebuild_check",
        description="Rebuild genuine PDFs with no edit and compare bytes.",
    )
    p.add_argument("batch", type=Path, help="the reference batch (pdfs/ and manifest.csv)")
    p.add_argument("--split", required=True, help="train or dev (test is closed)")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--out", type=Path, help="default: reports/week6_rebuild_check_<split>.json")
    args = p.parse_args(argv)
    try:
        ids = load_split(args.split, args.batch / "manifest.csv", args.splits_dir)
    except (SplitError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    result = check([args.batch / "pdfs" / f"{sid}.pdf" for sid in ids])
    result |= {"split": args.split, "seconds_total": round(time.perf_counter() - started, 1)}
    out = args.out or Path(f"reports/week6_rebuild_check_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}: {result['byte_identical']} of {result['statements']} "
          f"{args.split} PDFs rebuild byte-identical ({result['share_identical']:.2%}) in "
          f"{result['seconds_total']} s")  # fmt: skip
    for d in result["differ"]:
        print(f"  DIFFERS {d['statement']}: {d['cause']}")
    return 0 if not result["differ"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
