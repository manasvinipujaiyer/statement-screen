"""The upload checks: what is accepted, what is refused, and that no refused file is kept."""

import asyncio
import hashlib
import io

import pytest
from pypdf import PdfWriter
from starlette.datastructures import UploadFile

from statementscreen.service import upload

REQUEST = "req1"


def _pdf(pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _store(data: bytes, folder, filename: str = "statement.pdf", **kwargs):
    source = UploadFile(file=io.BytesIO(data), filename=filename)
    return asyncio.run(upload.store(source, folder, REQUEST, **kwargs))


def _refused(data: bytes, folder, **kwargs) -> upload.Rejected:
    with pytest.raises(upload.Rejected) as caught:
        _store(data, folder, **kwargs)
    return caught.value


def test_the_limits_are_the_ones_the_spec_names():
    assert upload.MAX_BYTES == 10 * 1024 * 1024 and upload.MAX_PAGES == 50


def test_a_pdf_is_stored_with_its_hash_size_and_page_count(tmp_path):
    data = _pdf(2)
    stored = _store(data, tmp_path)
    assert stored.path == tmp_path / f"{REQUEST}.pdf" and stored.path.read_bytes() == data
    assert stored.sha256 == hashlib.sha256(data).hexdigest()
    assert stored.size == len(data) and stored.pages == 2


def test_the_clients_file_name_is_never_used(tmp_path):
    stored = _store(_pdf(), tmp_path, filename="../../etc/holder_name.pdf")
    assert stored.path.name == f"{REQUEST}.pdf" and stored.path.parent == tmp_path
    assert [p.name for p in tmp_path.iterdir()] == [f"{REQUEST}.pdf"]


def test_the_hash_and_size_are_right_when_the_file_spans_several_chunks(tmp_path):
    data = _pdf(3)
    stored = _store(data, tmp_path, chunk=7)
    assert stored.sha256 == hashlib.sha256(data).hexdigest() and stored.size == len(data)


@pytest.mark.parametrize("data", [b"", b"hello", b"<html>%PDF</html>", b"%P"])
def test_a_file_that_does_not_start_with_pdf_is_refused_and_nothing_is_kept(tmp_path, data):
    rejected = _refused(data, tmp_path)
    assert (rejected.status, rejected.code) == (400, "NOT_A_PDF")
    assert list(tmp_path.iterdir()) == []


def test_the_size_limit_is_exact_and_a_refused_file_is_not_kept(tmp_path):
    data = _pdf()
    assert _store(data, tmp_path, max_bytes=len(data)).size == len(data)  # exactly at the limit
    (tmp_path / f"{REQUEST}.pdf").unlink()
    rejected = _refused(data, tmp_path, max_bytes=len(data) - 1)
    assert (rejected.status, rejected.code) == (413, "TOO_LARGE")
    assert "MB" in rejected.message and list(tmp_path.iterdir()) == []


def test_too_many_pages_is_refused_with_what_is_known_and_nothing_is_kept(tmp_path):
    data = _pdf(3)
    assert _store(data, tmp_path, max_pages=3).pages == 3  # exactly at the limit
    (tmp_path / f"{REQUEST}.pdf").unlink()
    rejected = _refused(data, tmp_path, max_pages=2)
    assert (rejected.status, rejected.code, rejected.pages) == (422, "TOO_MANY_PAGES", 3)
    assert rejected.size == len(data) and rejected.sha256 == hashlib.sha256(data).hexdigest()
    assert list(tmp_path.iterdir()) == []


def test_a_corrupt_pdf_is_accepted_with_no_page_count(tmp_path):
    stored = _store(b"%PDF-1.7\nthis is not really a pdf", tmp_path)
    assert stored.pages is None and stored.path.is_file()


def test_a_read_that_fails_half_way_leaves_no_partial_file(tmp_path):
    class Breaks:
        def __init__(self) -> None:
            self.calls = 0

        async def read(self, size: int = -1) -> bytes:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("connection lost")
            return b"%PDF-1.7 first chunk"

    with pytest.raises(RuntimeError, match="connection lost"):
        asyncio.run(upload.store(Breaks(), tmp_path, REQUEST))
    assert list(tmp_path.iterdir()) == []
