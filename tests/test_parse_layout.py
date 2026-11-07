from pathlib import Path

import pdfplumber
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from statementscreen.parsing.layouts import LAYOUTS, UnknownLayoutError, detect_layout
from statementscreen.parsing.words import Word, group_lines, is_footer, to_words
from statementscreen.synth import generate_statement
from statementscreen.synth.render import LAYOUT_NAMES, render_statement


def _page_lines(path: Path, page: int = 0):
    with pdfplumber.open(path) as pdf:
        p = pdf.pages[page]
        try:
            return group_lines(to_words(p.extract_words()))
        finally:
            p.close()


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_both_layouts_are_detected_from_header_text(tmp_path, layout: str) -> None:
    for seed in (1, 2, 3):
        path = render_statement(generate_statement(seed), tmp_path / f"{seed}.pdf", layout)
        assert detect_layout(_page_lines(path)) == layout
        assert detect_layout(_page_lines(path, page=1)) == layout  # headers repeat on page 2


def test_unrelated_reportlab_pdf_is_an_unknown_layout(tmp_path) -> None:
    path = tmp_path / "letter.pdf"
    c = Canvas(str(path), pagesize=A4)
    c.drawString(72, 750, "Dear customer, your new debit card is on its way.")
    c.drawString(72, 730, "Date Narration Balance")  # a few header-ish words, not a layout
    c.showPage()
    c.save()
    with pytest.raises(UnknownLayoutError, match="unknown layout"):
        detect_layout(_page_lines(path))


def test_a_line_matching_two_layouts_is_ambiguous_not_guessed() -> None:
    both = [Word(t, i * 50.0, i * 50.0 + 40, 100.0) for i, t in enumerate(
        ("Date Narration Chq./Ref.No. Value Dt Withdrawal Amt. Deposit Amt. Closing Balance "
         "Txn Date Description Amount (INR) Balance (INR)").split())]  # fmt: skip
    with pytest.raises(UnknownLayoutError, match="columns, drcr"):
        detect_layout([both])


def test_layout_specs_are_text_only() -> None:
    # Positions come from the page; a spec holds only header text and alignment.
    for spec in LAYOUTS.values():
        for col in spec.columns:
            assert col.align in ("left", "right") and col.header.strip()


def test_lines_group_by_top_and_footer_is_recognised() -> None:
    words = [Word("b", 50, 60, 100.5), Word("a", 10, 20, 100.0), Word("c", 10, 20, 112.0)]
    lines = group_lines(words)
    assert [[w.text for w in line] for line in lines] == [["a", "b"], ["c"]]
    footer = [Word("SYNTHETIC", 36, 70, 808), Word("DOCUMENT", 72, 110, 808)]
    assert is_footer(footer)
    page_only = [Word("Page", 500, 520, 808), Word("3", 522, 526, 808),
                 Word("of", 528, 536, 808), Word("11", 538, 546, 808)]  # fmt: skip
    assert is_footer(page_only)
    assert not is_footer([Word("Pages", 0, 1, 0)])
