"""Narration dialects: loading the files, and keeping the default output byte-identical."""

import hashlib
import json
from functools import cache
from pathlib import Path

import polars as pl
import pypdf
import pytest
from pypdf import PdfReader

from statementscreen.batch import BatchConfig, plan_batch
from statementscreen.batch_hash import fingerprint, fingerprint_one
from statementscreen.synth.dialect import (
    KINDS,
    DialectError,
    available_dialects,
    load_dialect,
    parse_dialect,
)
from statementscreen.synth.generator import generate_statement

FIXTURE = Path(__file__).parent / "data" / "default_batch_sha256.json"
PDF_TEXT = Path(__file__).parent / "data" / "default_batch_pdf_text.json"
KARAN = Path(__file__).parents[1] / "src/statementscreen/synth/dialects/karan_v1.txt"


# ---------------------------------------------------------------- the default stays identical


def _page_text_hashes(pdf: Path) -> tuple[int, list[str]]:
    reader = PdfReader(str(pdf))
    texts = [page.extract_text() for page in reader.pages]
    return len(reader.pages), [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]


def test_default_output_matches_before_dialects(tmp_path) -> None:
    """The fingerprint was recorded with the generator as it was before dialects existed.

    Labels, the manifest and the Parquet file are compared by their exact SHA-256. PDF bytes
    are not the same on every platform (the same generator writes different bytes on Linux
    and on macOS), so each PDF is compared by its page count and by the SHA-256 of the text
    pypdf extracts from each page. Layout and drawing operations are not compared: a change
    that moves or restyles content but keeps the text would pass.

    default_batch_pdf_text.json was recorded once, on macOS with pypdf 6.21.4, from the same
    PDFs whose bytes default_batch_sha256.json hashed. No committed command regenerates it.
    """
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pdf_text_file = json.loads(PDF_TEXT.read_text(encoding="utf-8"))
    assert pypdf.__version__ == pdf_text_file["pypdf_version"], (
        "pypdf changed since default_batch_pdf_text.json was recorded; "
        "its text extraction may differ"
    )
    pdf_text = pdf_text_file["batches"]
    now = fingerprint(tmp_path)
    assert set(now) == set(recorded) == set(pdf_text) == {"mixed", "self_employed"}
    for name in now:
        assert now[name]["plans"] == recorded[name]["plans"]
        assert set(now[name]["sha256"]) == set(recorded[name]["sha256"]), name
        for path, digest in recorded[name]["sha256"].items():
            if not path.startswith("pdfs/"):
                assert now[name]["sha256"][path] == digest, (name, path)
        recorded_pdfs = {p for p in recorded[name]["sha256"] if p.startswith("pdfs/")}
        assert set(pdf_text[name]) == recorded_pdfs, name
        for path, want in pdf_text[name].items():
            pages, hashes = _page_text_hashes(tmp_path / name / path)
            assert pages == want["pages"], (name, path, "page count")
            assert hashes == want["text_sha256"], (name, path, "page text")
    assert all("self_employed" in line for line in now["self_employed"]["plans"])


# ---------------------------------------------------------------- karan_v1


def test_every_karan_line_loads() -> None:
    d = load_dialect("karan_v1")
    lines = [ln for ln in KARAN.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(d.formats) == len(lines) == 26
    assert [f.line for f in d.formats] == list(range(1, 27))
    assert d.missing_kinds == ()  # all 16 kinds have at least one format
    assert {f.kind for f in d.formats} == set(KINDS)


def test_karan_lines_are_used_exactly_as_written() -> None:
    d = load_dialect("karan_v1")
    for f, raw in zip(d.formats, KARAN.read_text(encoding="utf-8").splitlines(), strict=True):
        assert raw.split("|", 2)[2].strip() == f.template


def test_dialect_names() -> None:
    assert load_dialect("default") is None
    assert "karan_v1" in available_dialects()
    for bad in ("nope", "../karan_v1", "KARAN_V1"):
        with pytest.raises(DialectError, match="unknown narration dialect"):
            load_dialect(bad)


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("salary | CR", "expected 'kind"),
        ("wages | CR | PAY {employer}", "unknown kind"),
        ("salary | IN | PAY {employer}", "must be CR or DR"),
        ("emi_return | DR | RTN {lender}", "generator posts"),
        ("salary | CR | PAY {employer} {amount}", "unknown placeholder"),
        ("salary | CR | PAY {employer", "unbalanced"),
        ("salary | CR | PAY {employer!r}", "unknown placeholder"),
    ],
)
def test_bad_lines_are_refused_with_their_line_number(line: str, message: str) -> None:
    with pytest.raises(DialectError, match=message) as e:
        parse_dialect(f"interest | CR | INT\n{line}\n", "test")
    assert "test line 2" in str(e.value)


# ---------------------------------------------------------------- karan_v1 statements

# 30 statements with every applicable hard case: enough for each kind to appear (EMI
# returns only come from technical bounces, self-transfers only from shortfall bounces).
PLANS = plan_batch(BatchConfig(seed=42, hard_case_rate=1.0), 30)


@cache
def _pair(number: int):
    p = PLANS[number]
    args = (p.seed, p.customer_type, p.months, p.start)
    default = generate_statement(*args, hard_cases=p.hard_cases)
    karan = generate_statement(*args, hard_cases=p.hard_cases, dialect=load_dialect("karan_v1"))
    return default, karan


def _truth(t) -> tuple:
    """Everything about a transaction except its narration."""
    return (t.posted, t.amount, t.balance, t.category, t.hard_cases, t.canonical_counterparty)


def test_karan_changes_narrations_but_not_labels_amounts_or_balances() -> None:
    changed = total = 0
    for i in range(len(PLANS)):
        default, karan = _pair(i)
        assert len(default.transactions) == len(karan.transactions)
        for a, b in zip(default.transactions, karan.transactions, strict=True):
            assert _truth(a) == _truth(b)
            changed += a.description != b.description
            total += 1
        assert default.income == karan.income
        assert default.hard_case_details == karan.hard_case_details
        assert default.opening_balance == karan.opening_balance
        assert default.declined_debits == karan.declined_debits
        assert default.narration_formats == {}
    assert changed / total > 0.95  # only advance tax and GST payments keep today's text


def test_every_kind_appears_and_every_dialect_narration_is_logged() -> None:
    kinds: set[str] = set()
    for i in range(len(PLANS)):
        _, karan = _pair(i)
        for t in karan.transactions:
            if t.description in karan.narration_formats:
                kinds.add(karan.narration_formats[t.description][0])
            else:  # the only events without a kind
                assert t.description.startswith(("TAX/ADVANCE TAX", "GST PMT/")), t.description
    assert kinds == set(KINDS)


def test_bonus_and_bounce_charges_use_the_dialect() -> None:
    for i in range(len(PLANS)):
        _, karan = _pair(i)
        for t in karan.transactions:
            if t.category == "bonus":
                assert karan.narration_formats[t.description][0] == "salary"
            if t.category in ("bounce_charge", "bounce_gst"):
                assert karan.narration_formats[t.description][0] == "emi_return_charge"


def test_karan_batch_files(tmp_path) -> None:
    """Same labels and manifest as the recorded default batch; transactions differ only in
    their narrations; narration_formats.parquet names the line behind every narration."""
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))["self_employed"]["sha256"]
    folder = tmp_path / "karan"
    now = fingerprint_one("self_employed", folder, dialect=load_dialect("karan_v1"))["sha256"]
    same = {k for k in recorded if k.startswith("labels/") or k == "manifest.csv"}
    assert {k: now[k] for k in same} == {k: recorded[k] for k in same}
    assert all(now[k] != recorded[k] for k in recorded if k.startswith("pdfs/"))

    karan = pl.read_parquet(folder / "transactions.parquet")
    formats = pl.read_parquet(folder / "narration_formats.parquet")
    assert formats.select("statement_id", "row").equals(karan.select("statement_id", "row"))
    assert set(formats["dialect"]) == {"karan_v1"}
    assert (
        formats["kind"].null_count()
        == karan.filter(
            pl.col("description").str.starts_with("TAX/ADVANCE TAX")
            | pl.col("description").str.starts_with("GST PMT/")
        ).height
    )


def test_placeholders_are_kept_in_order() -> None:
    d = parse_dialect("rent | DR | NEFT DR-{ifsc}-{payee}-HOUSE RENT {month}-{ref}\n", "t")
    assert d.formats[0].fields == ("ifsc", "payee", "month", "ref")
    assert d.missing_kinds == tuple(k for k in KINDS if k != "rent")
