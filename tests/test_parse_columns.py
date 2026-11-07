"""Round trip for the 'columns' layout: generate -> render -> parse -> compare.

The tests may use the generator to know the truth; the parser itself never does.
"""

import shutil

import pytest

from statementscreen.parsing.pdf import parse_pdf
from statementscreen.synth import generate_statement
from statementscreen.synth.labels import paise
from statementscreen.synth.render import LAYOUTS as RENDER_LAYOUTS
from statementscreen.synth.render import _wrap, render_statement

HEADER_WORDS = ("Narration", "Chq./Ref.No.", "Value Dt", "Withdrawal", "Deposit", "Closing Balance")


def _squash(text: str) -> str:
    return "".join(text.split())


def _narration_ok(parsed: str, truth: str) -> bool:
    """Whitespace-free equal, or the printed start of the truth (cut short by the layout)."""
    a, b = _squash(parsed), _squash(truth)
    return a == b or b.startswith(a)


@pytest.fixture(scope="module")
def parsed_columns(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cols")
    out = []
    # Seed 4 uses the dash narration style without truncation, so it has 3-line narrations.
    for seed, ct, cases in ((4, "salaried", ["emi_bounce", "bonus_month"]),
                            (8, "gig", []), (13, "self_employed", ["emi_bounce"])):  # fmt: skip
        s = generate_statement(seed, ct, months=8, hard_cases=cases)
        path = render_statement(s, folder / f"stmt_{seed:04d}.pdf", "columns")
        out.append((s, path, parse_pdf(path)))
    return out


def test_round_trip_is_exact_on_every_row(parsed_columns) -> None:
    for s, _, r in parsed_columns:
        assert r.status == "ok" and r.layout == "columns"
        assert r.issues == []
        assert len(r.rows) == len(s.transactions)
        for p, t in zip(r.rows, s.transactions, strict=True):
            assert p.posted == t.posted, p.row
            assert p.amount_paise == paise(t.amount), p.row
            assert p.balance_paise == paise(t.balance), p.row
            assert _narration_ok(p.narration, t.description), (p.row, p.narration, t.description)
            assert p.value_date == t.posted


def test_header_and_summary_fields(parsed_columns) -> None:
    for s, _, r in parsed_columns:
        h = r.header
        debits = [t.amount for t in s.transactions if t.amount < 0]
        credits = [t.amount for t in s.transactions if t.amount > 0]
        assert h.account_number == s.account_number
        assert (h.period_start, h.period_end) == (s.period_start, s.period_end)
        assert h.opening_balance_paise == paise(s.opening_balance)
        assert h.closing_balance_paise == paise(s.closing_balance)
        assert h.total_debits_paise == -paise(sum(debits))
        assert h.total_credits_paise == paise(sum(credits))
        assert (h.dr_count, h.cr_count) == (len(debits), len(credits))


def test_a_three_line_narration_joins_into_one_row(parsed_columns) -> None:
    lay = RENDER_LAYOUTS["columns"]
    width = next(c.width for c in lay.columns if c.key == "narration")
    found = 0
    for s, _, r in parsed_columns:
        for p, t in zip(r.rows, s.transactions, strict=True):
            printed = _wrap(t.description, width, lay.font, lay.size, lay.narration_lines)
            if len(printed) == 3:
                found += 1
                assert _squash(p.narration) == _squash("".join(printed))
    assert found >= 1, "no 3-line narration in the sample; pick other seeds"


def test_rows_continue_across_page_breaks(parsed_columns) -> None:
    for s, _, r in parsed_columns:
        pages = [p.page for p in r.rows]
        assert pages == sorted(pages) and pages[-1] == r.pages and len(set(pages)) == r.pages
        assert [p.row for p in r.rows] == list(range(len(s.transactions)))
        for p in r.rows:
            assert "SYNTHETIC" not in p.narration and "Page " not in p.narration
            assert not any(word in p.narration for word in HEADER_WORDS), p.narration


def test_parsing_a_pdf_alone_gives_the_same_result(parsed_columns, tmp_path) -> None:
    _, path, in_batch = parsed_columns[0]
    alone = tmp_path / "empty_folder" / path.name
    alone.parent.mkdir()
    shutil.copy(path, alone)  # no labels, manifest or Parquet anywhere near it
    assert parse_pdf(alone) == in_batch


def test_unreadable_pdf_is_reported_not_raised(tmp_path) -> None:
    broken = tmp_path / "stmt_0099.pdf"
    broken.write_bytes(b"%PDF-1.4 this is not really a pdf")
    r = parse_pdf(broken)
    assert r.status == "error" and r.statement_id == "stmt_0099"
    assert [i.code for i in r.issues] == ["UNREADABLE_PDF"]
