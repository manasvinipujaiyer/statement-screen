"""Count the fonts genuine train PDFs use, per layout, to set config.MAX_FONTS.

    uv run python scripts/measure_fonts.py

For every genuine train statement (one PDF at a time): its layout, from page 1's table
header (the parser's detect_layout), and its distinct BaseFont names (pypdf). Writes
reports/week6_font_baseline.json: per layout, how many statements use each number of
fonts, every font name seen, and the largest count.

Genuine train only: no tampered copies, no dev or test statements.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from statementscreen.parsing.layouts import UnknownLayoutError, detect_layout
from statementscreen.parsing.words import group_lines, to_words
from statementscreen.splits import load_split
from statementscreen.tamper.detect.history import font_names


def page1_layout(pdf_path: Path) -> str | None:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        try:
            return detect_layout(group_lines(to_words(page.extract_words())))
        except UnknownLayoutError:
            return None
        finally:
            page.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--batch", type=Path, default=Path("data/week1_n2000"))
    p.add_argument("--out", type=Path, default=Path("reports/week6_font_baseline.json"))
    p.add_argument("--limit", type=int, default=None, help="first N statements only (a trial)")
    args = p.parse_args(argv)

    ids = load_split("train", args.batch / "manifest.csv")[: args.limit]
    counts: dict[str, Counter] = {}
    names: dict[str, Counter] = {}
    unknown: list[str] = []
    start = time.perf_counter()
    for sid in ids:
        pdf = args.batch / "pdfs" / f"{sid}.pdf"
        layout = page1_layout(pdf)
        if layout is None:
            unknown.append(sid)
            continue
        fonts = font_names(PdfReader(pdf))
        counts.setdefault(layout, Counter())[len(fonts)] += 1
        names.setdefault(layout, Counter()).update(fonts)
    report = {
        "what": "distinct fonts per genuine train PDF, per layout (sets config.MAX_FONTS)",
        "split": "train",
        "statements": len(ids),
        "by_layout": {
            layout: {
                "statements": sum(counts[layout].values()),
                "font_count_histogram": {str(k): v for k, v in sorted(counts[layout].items())},
                "max_fonts": max(counts[layout]),
                "font_names": dict(sorted(names[layout].items())),
            }
            for layout in sorted(counts)
        },
        "unknown_layout": unknown,
        "seconds": round(time.perf_counter() - start, 1),
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
