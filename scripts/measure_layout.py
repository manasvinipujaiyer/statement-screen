"""Measure the font and layout values on genuine train PDFs, to set the Week 6 tolerances.

    uv run python scripts/measure_layout.py

For every genuine train statement (one PDF at a time) this reads the money cells and
overlapping characters (tamper/detect/layout.py) and records, per layout, the largest
value each layout check would test: cells off the column's usual font, how far a cell's
right edge is from the column's usual edge, the spread of gaps inside a number, and the
largest overlap between two characters. Writes reports/week6_layout_baseline.json.

Genuine train only: no tampered copies, no dev or test statements.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from statementscreen.splits import load_split
from statementscreen.summary import peak_memory_bytes
from statementscreen.tamper.detect.layout import measure, read_layout

METRICS = ("cells_off_usual_font", "max_alignment_off", "max_spacing_spread",
           "max_overlap_share")  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=Path, default=Path("data/week1_n2000"))
    p.add_argument("--out", type=Path, default=Path("reports/week6_layout_baseline.json"))
    p.add_argument("--limit", type=int, default=None, help="first N statements only (a trial)")
    args = p.parse_args(argv)

    ids = load_split("train", args.batch / "manifest.csv")[: args.limit]
    per_layout: dict[str, dict] = {}
    problems: list[dict] = []
    start = time.perf_counter()
    for n, sid in enumerate(ids, start=1):
        m = measure(read_layout(args.batch / "pdfs" / f"{sid}.pdf"))
        if m["problem"] or m["layout"] is None:
            problems.append({"statement_id": sid, "problem": m["problem"]})
            continue
        agg = per_layout.setdefault(m["layout"], {"statements": 0, "cells": 0,
                                                  **{k: 0.0 for k in METRICS},
                                                  "worst": {k: None for k in METRICS}})  # fmt: skip
        agg["statements"] += 1
        agg["cells"] += m["cells"]
        for k in METRICS:  # keep the largest value and the statement it came from
            if agg["worst"][k] is None or m[k] > agg[k]:
                agg[k], agg["worst"][k] = m[k], sid
        if n % 100 == 0:
            print(f"  {n}/{len(ids)} statements, {time.perf_counter() - start:.0f} s", flush=True)

    seconds = time.perf_counter() - start
    peak_mb = peak_memory_bytes() / 1024**2  # bytes on macOS, kilobytes on Linux: the helper knows
    by_layout = {}
    for name, agg in sorted(per_layout.items()):
        by_layout[name] = agg | {m: round(agg[m], 4) for m in METRICS}
    report = {
        "what": "largest font/layout values on genuine train PDFs (sets Week 6 tolerances)",
        "split": "train",
        "statements": len(ids),
        "by_layout": by_layout,
        "problems": problems,
        "seconds": round(seconds, 1),
        "seconds_per_pdf": round(seconds / max(len(ids), 1), 4),
        "peak_mb": round(peak_mb, 1),
    }
    if args.limit is None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
