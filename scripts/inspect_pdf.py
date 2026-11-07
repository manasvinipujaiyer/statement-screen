"""Show what pdfplumber sees on one page of a PDF: its text lines and its words with positions.

Usage:
    uv run python scripts/inspect_pdf.py data/w2_inspect/pdfs/stmt_0001.pdf            # page 1
    uv run python scripts/inspect_pdf.py data/w2_inspect/pdfs/stmt_0001.pdf --page 2
    uv run python scripts/inspect_pdf.py FILE --page 1 --words 120   # show more words

pdfplumber measures positions in points (1/72 inch) from the TOP-LEFT of the page:
x0/x1 are a word's left/right edges, top/bottom its vertical extent (top grows downward).
That's the opposite vertical direction to ReportLab, which draws from the bottom-left.

This is an inspection tool only: it doesn't parse anything.
"""

import argparse

import pdfplumber


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--page", type=int, default=1, help="1-based page number")
    p.add_argument("--words", type=int, default=60, help="how many words to list")
    args = p.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        print(f"{args.pdf}: {len(pdf.pages)} pages")
        page = pdf.pages[args.page - 1]
        try:
            print(f"page {args.page}: {page.width:.1f} x {page.height:.1f} pt\n")

            print("=== text lines (top = distance from the top of the page) ===")
            for line in page.extract_text_lines():
                print(f"top={line['top']:6.1f}  {line['text']}")

            words = page.extract_words()
            print(f"\n=== first {min(args.words, len(words))} of {len(words)} words ===")
            print(f"{'x0':>6} {'x1':>6} {'top':>6} {'bottom':>6}  text")
            for w in words[: args.words]:
                box = f"{w['x0']:6.1f} {w['x1']:6.1f} {w['top']:6.1f} {w['bottom']:6.1f}"
                print(f"{box}  {w['text']}")
        finally:
            page.close()  # free the page's cached objects as soon as we're done


if __name__ == "__main__":
    main()
