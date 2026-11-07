import re
from decimal import Decimal

import pytest
from pypdf import PdfReader

from statementscreen.synth import generate_statement
from statementscreen.synth.render import FOOTER_TEXT, LAYOUT_NAMES, inr, render_statement

A4_W, A4_H = 595.28, 841.89


def _render(tmp_path, layout: str, seed: int = 5, months: int = 6, **kw):
    s = generate_statement(seed, months=months, **kw)
    return s, render_statement(s, tmp_path / f"{layout}_{seed}_{months}.pdf", layout)


def _text(path) -> list[str]:
    return [page.extract_text() for page in PdfReader(path).pages]


def test_indian_digit_grouping() -> None:
    cases = {"0": "0.00", "999.5": "999.50", "1000": "1,000.00", "123456.78": "1,23,456.78",
             "1234567.8": "12,34,567.80", "-25000000": "-2,50,00,000.00"}  # fmt: skip
    for raw, expected in cases.items():
        assert inr(Decimal(raw)) == expected


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_every_page_is_a4_with_the_synthetic_footer(tmp_path, layout: str) -> None:
    _, path = _render(tmp_path, layout)
    reader = PdfReader(path)
    assert len(reader.pages) >= 2
    for page in reader.pages:
        assert abs(float(page.mediabox.width) - A4_W) < 1
        assert abs(float(page.mediabox.height) - A4_H) < 1
        assert FOOTER_TEXT in page.extract_text()


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_header_shows_indian_account_details(tmp_path, layout: str) -> None:
    s, path = _render(tmp_path, layout)
    first = _text(path)[0]
    for value in (s.bank_name.upper(), s.ifsc, s.micr, s.customer_id, "INR", "Savings Account"):
        assert value in first.upper() or value in first, value
    assert "Checking Account" not in first


def test_columns_layout_formats(tmp_path) -> None:
    _, path = _render(tmp_path, "columns")
    first = _text(path)[0]
    assert re.search(r"\b\d\d/\d\d/20\d\d\b", first)  # DD/MM/YYYY
    for header in (
        "Withdrawal Amt.",
        "Deposit Amt.",
        "Chq./Ref.No.",
        "Value Dt",
        "Closing Balance",
    ):
        assert header in first
    assert "STATEMENT SUMMARY" in first and "Dr Count" in first


def test_drcr_layout_formats(tmp_path) -> None:
    _, path = _render(tmp_path, "drcr")
    pages = _text(path)
    assert re.search(r"\b\d\d-[A-Z][a-z]{2}-\d\d\b", pages[0])  # DD-Mon-YY
    assert re.search(r"\d\.\d\d Dr\b", pages[0]) and re.search(r"\d\.\d\d Cr\b", pages[0])
    assert "Withdrawal" not in pages[0]
    assert "Statement Summary" in pages[-1]  # the summary comes after the last transaction


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_large_amounts_use_lakh_grouping(tmp_path, layout: str) -> None:
    # A well-paid salaried customer's balance passes 1 lakh, so 1,23,456.78 style must appear.
    for seed in range(20):
        s = generate_statement(seed, months=6)
        if s.closing_balance >= 100_000:
            break
    path = render_statement(s, tmp_path / "big.pdf", layout)
    assert re.search(r"\b\d{1,2},\d\d,\d{3}\.\d\d\b", " ".join(_text(path)))


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_same_statement_gives_byte_identical_pdfs(tmp_path, layout: str) -> None:
    s = generate_statement(seed=5, hard_cases=["emi_bounce"])
    a = render_statement(s, tmp_path / "a.pdf", layout).read_bytes()
    b = render_statement(s, tmp_path / "b.pdf", layout).read_bytes()
    assert a == b


def test_layouts_produce_different_files(tmp_path) -> None:
    s = generate_statement(seed=5)
    files = {render_statement(s, tmp_path / f"{n}.pdf", n).read_bytes() for n in LAYOUT_NAMES}
    assert len(files) == len(LAYOUT_NAMES)


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_longer_statements_get_more_pages(tmp_path, layout: str) -> None:
    _, short = _render(tmp_path, layout, months=6)
    _, long = _render(tmp_path, layout, months=12)
    assert len(PdfReader(long).pages) > len(PdfReader(short).pages)


def test_unknown_layout_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        render_statement(generate_statement(seed=1), tmp_path / "x.pdf", "fancy")
