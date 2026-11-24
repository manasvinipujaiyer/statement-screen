"""File-history findings, read with pypdf. Signs that a file went through an editing tool;
never proof of editing on their own, so they are weak or info.

    INCREMENTAL_UPDATE      weak  the file holds more than one save (more than one %%EOF
                                  marker, or an update section pointing back to an
                                  earlier one with /Prev)
    MODDATE_AFTER_CREATION  weak  the modification date is later than the creation date
    EXTRA_FONTS             weak  more distinct fonts than genuine train statements of
                                  this layout ever use (config.MAX_FONTS)
    PRODUCER_CREATOR_DIFFER info  the producer and creator strings differ: listed so a
                                  reader sees it, never counted toward a verdict

No check treats any producer, creator or date as evidence that a PDF is genuine: these
only look for signs of editing. A full re-save leaves none of them except, perhaps, a
new producer string, which is info only.
"""

import io
import re
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import DictionaryObject

from statementscreen.tamper.detect import config
from statementscreen.tamper.detect.findings import Finding

EOF = re.compile(rb"%%EOF")
PREV = re.compile(rb"/Prev\s+\d+")


def _weak(code: str, reason: str) -> Finding:
    return Finding(code, "history", None, None, "weak", reason)


def font_names(reader: PdfReader) -> set[str]:
    """Distinct BaseFont names used by the pages (and form XObjects drawn on them)."""
    names: set[str] = set()

    def visit(resources, depth: int = 0) -> None:
        if not isinstance(resources, DictionaryObject) or depth > 3:
            return
        fonts = resources.get("/Font")
        if fonts is not None:
            for ref in fonts.get_object().values():
                base = ref.get_object().get("/BaseFont")
                if base is not None:
                    names.add(str(base).lstrip("/"))
        xobjects = resources.get("/XObject")
        if xobjects is not None:
            for ref in xobjects.get_object().values():
                x = ref.get_object()
                if x.get("/Subtype") == "/Form":
                    visit(x.get("/Resources", DictionaryObject()).get_object(), depth + 1)

    for page in reader.pages:
        resources = page.get("/Resources")
        visit(resources.get_object() if resources is not None else None)
    return names


def _naive(d: datetime) -> datetime:
    """Compare dates as UTC-less wall times when only one side has a time zone."""
    return d.replace(tzinfo=None)


def history_findings(pdf_path: Path, layout: str | None) -> list[Finding]:
    """File-history findings for one PDF. `layout` is the parser's layout name (or None)."""
    data = Path(pdf_path).read_bytes()
    out: list[Finding] = []

    eofs, prev = len(EOF.findall(data)), bool(PREV.search(data))
    if eofs > 1 or prev:
        signs = []
        if eofs > 1:
            signs.append(f"{eofs} end-of-file markers")
        if prev:
            signs.append("an update section pointing back to an earlier one (/Prev)")
        out.append(_weak("INCREMENTAL_UPDATE",
                         f"the file was saved more than once: {' and '.join(signs)}"))  # fmt: skip

    try:
        reader = PdfReader(io.BytesIO(data))
        meta = reader.metadata
    except Exception as e:  # unreadable by pypdf: the parse findings will say more
        return out + [Finding("CHECK_NOT_RUN", "history", None, None, "info",
                              "MODDATE_AFTER_CREATION, EXTRA_FONTS, PRODUCER_CREATOR_DIFFER: "
                              f"pypdf couldn't read the file ({e})")]  # fmt: skip

    created = modified = None
    if meta is not None:
        try:
            created, modified = meta.creation_date, meta.modification_date
        except Exception:  # a malformed date string: not evidence either way
            created = modified = None
    if created is not None and modified is not None:
        a, b = created, modified
        if (a.tzinfo is None) != (b.tzinfo is None):
            a, b = _naive(a), _naive(b)
        if b > a:
            out.append(_weak("MODDATE_AFTER_CREATION",
                             f"modified {modified.isoformat()}, after it was created "
                             f"{created.isoformat()}"))  # fmt: skip

    limit = config.MAX_FONTS.get(layout) if layout else None
    fonts = font_names(reader)
    if limit is None:
        out.append(Finding("CHECK_NOT_RUN", "history", None, None, "info",
                           f"EXTRA_FONTS: no font limit for layout {layout!r}"))  # fmt: skip
    elif len(fonts) > limit:
        out.append(_weak("EXTRA_FONTS",
                         f"{len(fonts)} fonts ({', '.join(sorted(fonts))}); genuine train "
                         f"statements of this layout use at most {limit}"))  # fmt: skip

    producer = meta.producer if meta is not None else None
    creator = meta.creator if meta is not None else None
    if producer and creator and producer != creator:
        out.append(Finding("PRODUCER_CREATOR_DIFFER", "history", None, None, "info",
                           f"producer '{producer}' differs from creator '{creator}' (listed "
                           "only, never counted)"))  # fmt: skip
    return out
