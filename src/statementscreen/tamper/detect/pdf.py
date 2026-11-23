"""One PDF in, one tamper verdict out: parse, five groups of checks, then the verdict.

    uv run python -m statementscreen.tamper.detect.pdf data/week1_n2000/pdfs/stmt_0002.pdf

The detector gets only the PDF's path: no split, manifest, label, answer key or
transactions.parquet (the Week 7 service wraps this, next to profile_pdf). The PDF is
parsed once; the layout, structure and history checks read it again with pdfplumber and
pypdf, one page at a time.

collect_findings(path) returns every finding (step 5 caches these per set, so rules can
be tuned without re-reading PDFs). tamper_pdf(path) returns the verdict alone, so its JSON
is byte-identical across runs; tamper_pdf_timed(path) also returns the seconds taken and
the process's peak memory (bytes on macOS, KB on Linux, via summary.peak_memory_bytes; it
is the peak of the whole process so far, not of this PDF alone).

A check group that crashes on a malformed PDF is listed as a CHECK_NOT_RUN info finding
with the error, never skipped silently.
"""

import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from statementscreen.parsing.models import ParsedStatement
from statementscreen.parsing.pdf import parse_pdf
from statementscreen.summary import peak_memory_bytes
from statementscreen.tamper.detect.anomaly import anomaly_findings
from statementscreen.tamper.detect.arithmetic import arithmetic_findings
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.history import history_findings
from statementscreen.tamper.detect.layout import layout_findings, read_layout
from statementscreen.tamper.detect.structure import structure_findings
from statementscreen.tamper.detect.verdict import decide, to_json


def _parse(path: Path) -> ParsedStatement:
    """parse_pdf, with a parser crash turned into an error result (as parse_one does)."""
    try:
        return parse_pdf(path)
    except Exception as e:  # a bug in the parser on a malformed file: report it
        result = ParsedStatement(path.stem, None, 0, "error", f"parser crashed: {e!r}")
        result.issue("PARSER_CRASH", None, None, str(e))
        return result


def _run(group: str, check: Callable[[], list[Finding]]) -> list[Finding]:
    try:
        return check()
    except Exception as e:  # a malformed PDF can break a check: say so, don't skip quietly
        reason = f"{group} checks didn't run: {type(e).__name__}: {e}"
        return [Finding("CHECK_NOT_RUN", group, None, None, "info", reason)]


def collect_findings(path: Path) -> list[Finding]:
    """Every finding for one PDF, from all five check groups."""
    path = Path(path)
    parsed = _parse(path)
    found = arithmetic_findings(parsed)
    found += _run("structure", lambda: structure_findings(path, parsed))
    found += _run("layout", lambda: layout_findings(read_layout(path)))
    found += _run("history", lambda: history_findings(path, parsed.layout))
    found += _run("anomaly", lambda: anomaly_findings(parsed))
    return found


def tamper_pdf(path: Path) -> dict:
    """The verdict for one PDF: label, rule version, reasons in order of strength."""
    return decide(collect_findings(Path(path)))


def tamper_pdf_timed(path: Path) -> tuple[dict, dict]:
    started = time.perf_counter()
    verdict = tamper_pdf(path)
    stats = {
        "seconds": round(time.perf_counter() - started, 3),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
    }
    return verdict, stats


def main(argv: Sequence[str] | None = None) -> int:
    paths = [Path(a) for a in (sys.argv[1:] if argv is None else argv)]
    if not paths:
        print(
            "usage: python -m statementscreen.tamper.detect.pdf FILE.pdf [FILE.pdf ...]",
            file=sys.stderr,
        )
        return 2
    for path in paths:
        if not path.is_file():
            print(f"error: {path} is not a file", file=sys.stderr)
            return 2
        verdict, stats = tamper_pdf_timed(path)
        if len(paths) == 1:
            print(to_json(verdict))
        print(
            f"{path.name}: {verdict['label']} ({verdict['summary']}); "
            f"{stats['seconds']} s, peak memory {stats['peak_memory_mb']} MB",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
