"""Receive one uploaded PDF safely: check it, hash it, and put it in a private temp folder.

store() copies the upload into `folder/<name>.pdf` in chunks and refuses, with a Rejected that
says which HTTP status and error code to return:

    400 NOT_A_PDF         the first bytes are not "%PDF" (an empty file included)
    413 TOO_LARGE         more than MAX_BYTES (10 MB, as 10 * 1024 * 1024 bytes)
    422 TOO_MANY_PAGES    more than MAX_PAGES (50) pages

It stops reading as soon as it can say no, never keeps a partial file, and never uses the
client's file name. The caller owns the temp folder and deletes it in every case.

A PDF whose page count cannot be read (corrupt, encrypted) is NOT rejected: pages is None, the
detector reports it as unreadable, and the page limit cannot be enforced for it. The size limit
still bounds it.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader
from starlette.concurrency import run_in_threadpool

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 50
CHUNK = 64 * 1024
MAGIC = b"%PDF"


class Rejected(Exception):
    """The upload is refused. Carries what is known about it, for the response and the audit."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        size: int | None = None,
        sha256: str | None = None,
        pages: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.size, self.sha256, self.pages = size, sha256, pages


@dataclass(frozen=True)
class Stored:
    path: Path
    sha256: str
    size: int
    pages: int | None


class Source(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


def count_pages(path: Path) -> int | None:
    """The PDF's page count, or None when pypdf cannot read it (that is not a rejection)."""
    try:
        return len(PdfReader(str(path), strict=False).pages)
    except Exception:  # corrupt or encrypted: the detector reports it, the page cap can't apply
        return None


async def store(
    source: Source,
    folder: Path,
    name: str,
    max_bytes: int = MAX_BYTES,
    max_pages: int = MAX_PAGES,
    chunk: int = CHUNK,
) -> Stored:
    path = Path(folder) / f"{name}.pdf"
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("wb") as out:
            first = True
            while data := await source.read(chunk):
                if first and not data.startswith(MAGIC):
                    raise Rejected(400, "NOT_A_PDF", "the upload is not a PDF", size=len(data))
                first = False
                size += len(data)
                if size > max_bytes:
                    raise Rejected(
                        413, "TOO_LARGE", f"the upload is larger than {max_bytes // 2**20} MB",
                        size=size,
                    )  # fmt: skip
                digest.update(data)
                out.write(data)
        if size == 0:
            raise Rejected(400, "NOT_A_PDF", "the upload is empty", size=0)
        pages = await run_in_threadpool(count_pages, path)
        if pages is not None and pages > max_pages:
            raise Rejected(
                422, "TOO_MANY_PAGES", f"the PDF has {pages} pages; the limit is {max_pages}",
                size=size, sha256=digest.hexdigest(), pages=pages,
            )  # fmt: skip
    except BaseException:
        path.unlink(missing_ok=True)  # never keep a partial or refused file
        raise
    return Stored(path, digest.hexdigest(), size, pages)
