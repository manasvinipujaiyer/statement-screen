"""One PDF in, one income profile out: parse, patterns and profile, end to end.

    uv run python -m statementscreen.profile.pdf data/week1_n2000/pdfs/stmt_0002.pdf

No split, manifest or ground truth: only the PDF's own path is read (the Week 7 service
wraps this). A PDF the parser can't read still gives a profile, with a warning.

profile_pdf(path) returns the profile alone, so its JSON is byte-identical across runs.
profile_pdf_timed(path) also returns the seconds taken and the process's peak memory
(resource.getrusage's ru_maxrss via summary.peak_memory_bytes: bytes on macOS, KB on
Linux; it is the peak of the whole process so far, not of this PDF alone).
"""

import sys
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from statementscreen.parse import parse_one
from statementscreen.parsing.tables import rows_frame
from statementscreen.patterns.cli import process_statement
from statementscreen.profile.build import profile_statement, to_json
from statementscreen.profile.cli import GROUP_COLUMNS, ROW_COLUMNS, RULE_COLUMNS
from statementscreen.profile.config import DEFAULT, ProfileConfig
from statementscreen.summary import peak_memory_bytes


def _empty(columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict.fromkeys(columns, pl.Null))


def profile_pdf(path: Path, cfg: ProfileConfig = DEFAULT) -> dict:
    path = Path(path)
    parsed, _ = parse_one(path)
    header = parsed.header
    start = getattr(header, "period_start", None)
    end = getattr(header, "period_end", None)
    rows = rows_frame(parsed)
    problem = None
    if parsed.status != "ok":
        problem = f"could not read the statement: {parsed.error or 'unknown parse error'}"
    if parsed.status != "ok" or rows.height == 0:
        p = profile_statement(parsed.statement_id, _empty(ROW_COLUMNS), _empty(GROUP_COLUMNS),
                              _empty(RULE_COLUMNS), None, start, end, cfg)  # fmt: skip
    else:
        stmt = rows.select("statement_id", "row", "posted", "narration", "amount_paise")
        t = process_statement(stmt, start, end, parsed.layout)
        result = t["statement_results"].row(0, named=True)
        p = profile_statement(parsed.statement_id, t["rows"], t["group_features"],
                              t["group_rules"], result, start, end, cfg)  # fmt: skip
    if problem:
        p["warnings"].insert(0, problem)
    return p


def profile_pdf_timed(path: Path, cfg: ProfileConfig = DEFAULT) -> tuple[dict, dict]:
    started = time.perf_counter()
    p = profile_pdf(path, cfg)
    stats = {"seconds": round(time.perf_counter() - started, 3),
             "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1)}  # fmt: skip
    return p, stats


def main(argv: Sequence[str] | None = None) -> int:
    paths = [Path(a) for a in (sys.argv[1:] if argv is None else argv)]
    if not paths:
        print("usage: python -m statementscreen.profile.pdf FILE.pdf [FILE.pdf ...]",
              file=sys.stderr)  # fmt: skip
        return 2
    for path in paths:
        if not path.is_file():
            print(f"error: {path} is not a file", file=sys.stderr)
            return 2
        p, stats = profile_pdf_timed(path)
        if len(paths) == 1:
            print(to_json(p))
        f = p["findings"]
        print(f"{path.name}: income {f['estimated_monthly_income_paise']['value']} paise, "
              f"type {f['customer_type']['value']}, {len(p['warnings'])} warning(s); "
              f"{stats['seconds']} s, peak memory {stats['peak_memory_mb']} MB",
              file=sys.stderr)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
