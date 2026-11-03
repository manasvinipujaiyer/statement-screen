from statementscreen.synth import generate_statement
from statementscreen.synth.render import render_statement


def test_render_writes_a_pdf(tmp_path) -> None:
    path = render_statement(generate_statement(seed=5), tmp_path / "s.pdf")
    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_long_statement_gets_extra_pages(tmp_path) -> None:
    s = generate_statement(seed=5)
    s.transactions = s.transactions * 3  # ~115 rows, too many for 2 pages
    data = render_statement(s, tmp_path / "long.pdf").read_bytes()
    assert data.count(b"/Type /Page\n") >= 3
