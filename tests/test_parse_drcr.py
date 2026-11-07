"""Round trip for the 'drcr' layout: generate -> render -> parse -> compare."""

import pytest

from statementscreen.parsing.pdf import parse_pdf
from statementscreen.synth import generate_statement
from statementscreen.synth.labels import paise
from statementscreen.synth.render import render_statement

NOT_NARRATION = ("SYNTHETIC", "Page ", "Statement Summary", "computer generated", "Txn Date",
                 "Description", "Amount (INR)", "Balance (INR)", "Opening Balance")  # fmt: skip


def _squash(text: str) -> str:
    return "".join(text.split())


@pytest.fixture(scope="module")
def parsed_drcr(tmp_path_factory):
    folder = tmp_path_factory.mktemp("drcr")
    out = []
    samples = [
        (4, "salaried", 9, ["emi_bounce", "employer_rename"]),
        (8, "gig", 6, []),
        (13, "self_employed", 12, ["emi_bounce"]),
    ]
    for seed, ct, months, cases in samples:
        s = generate_statement(seed, ct, months=months, hard_cases=cases)
        path = render_statement(s, folder / f"stmt_{seed:04d}.pdf", "drcr")
        out.append((s, parse_pdf(path)))
    return out


def test_round_trip_is_exact_on_every_row(parsed_drcr) -> None:
    cut_short = 0
    for s, r in parsed_drcr:
        assert r.status == "ok" and r.layout == "drcr" and r.issues == []
        assert len(r.rows) == len(s.transactions)
        for p, t in zip(r.rows, s.transactions, strict=True):
            assert p.posted == t.posted, p.row
            assert p.amount_paise == paise(t.amount), p.row
            assert p.balance_paise == paise(t.balance), p.row
            a, b = _squash(p.narration), _squash(t.description)
            assert a == b or b.startswith(a), (p.row, p.narration, t.description)
            cut_short += a != b
            assert p.value_date is None and p.ref is None  # not printed in this layout
    assert cut_short > 0  # one-line narrations do get cut: the prefix rule is exercised


def test_summary_at_the_end_and_two_digit_period(parsed_drcr) -> None:
    for s, r in parsed_drcr:
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


def test_rows_continue_across_pages_and_summary_is_not_a_row(parsed_drcr) -> None:
    for s, r in parsed_drcr:
        pages = [p.page for p in r.rows]
        assert pages == sorted(pages)
        assert len(set(pages)) >= r.pages - 1  # the last page may hold only the summary
        # No row lost or duplicated at a page break: same count, same order of dates.
        assert len(r.rows) == len(s.transactions)
        assert [p.posted for p in r.rows] == [t.posted for t in s.transactions]
        for p in r.rows:
            assert not any(word in p.narration for word in NOT_NARRATION), p.narration


def test_drcr_pdf_is_detected_and_forcing_columns_finds_no_table(tmp_path) -> None:
    s = generate_statement(2)
    path = render_statement(s, tmp_path / "stmt_0002.pdf", "drcr")
    assert parse_pdf(path).layout == "drcr"
    forced = parse_pdf(path, force_layout="columns")  # what first_contact measured
    assert forced.rows == []
