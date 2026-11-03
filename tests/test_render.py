import pytest

from statementscreen.synth import generate_statement
from statementscreen.synth.render import LAYOUT_NAMES, render_statement


def _page_count(pdf: bytes) -> int:
    return pdf.count(b"/Type /Page\n")


@pytest.mark.parametrize("layout", LAYOUT_NAMES)
def test_render_writes_a_pdf(tmp_path, layout: str) -> None:
    path = render_statement(generate_statement(seed=5), tmp_path / "s.pdf", layout)
    assert path.read_bytes().startswith(b"%PDF")


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
    short = render_statement(generate_statement(5, months=6), tmp_path / "short.pdf", layout)
    long = render_statement(generate_statement(5, months=12), tmp_path / "long.pdf", layout)
    assert _page_count(long.read_bytes()) > _page_count(short.read_bytes()) >= 2


def test_unknown_layout_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        render_statement(generate_statement(seed=1), tmp_path / "x.pdf", "fancy")
