"""run_summary.json: what a batch run produced and what it cost.

Unlike the batch outputs, this file is *meant* to differ between runs: it records
timings and memory. It contains no absolute paths and no timestamps.

Peak memory comes from ``resource.getrusage(...).ru_maxrss``. Careful: its unit depends
on the operating system. On macOS it is **bytes**; on Linux it is **kilobytes**.
"""

import platform
import resource
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from statementscreen.batch import GENERATOR_VERSION, BatchConfig, StatementPlan
from statementscreen.synth.hard_cases import HARD_CASES
from statementscreen.verify import VerifyReport


def peak_memory_bytes(maxrss: int | None = None, system: str | None = None) -> int:
    """Peak resident memory of this process, in bytes, whatever the OS."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if maxrss is None else maxrss
    system = sys.platform if system is None else system
    return maxrss if system == "darwin" else maxrss * 1024  # Linux reports kilobytes


def disk_usage(out: Path) -> dict[str, int]:
    """Bytes used by each output, plus the total."""

    def size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    names = ("pdfs", "labels", "transactions.parquet", "manifest.csv")
    parts = {name: size(out / name) for name in names if (out / name).exists()}
    return {**{f"{k}_bytes": v for k, v in parts.items()}, "total_bytes": sum(parts.values())}


def build_run_summary(
    config: BatchConfig,
    plans: Sequence[StatementPlan],
    manifest_rows: Sequence[dict],
    report: VerifyReport,
    write_seconds: float,
    verify_seconds: float,
    out: Path,
) -> dict:
    n = len(plans)
    cases = Counter(c for p in plans for c in p.hard_cases)
    return {
        "generator_version": GENERATOR_VERSION,
        "seed": config.seed,
        "n": n,
        "mix": config.mix,
        "hard_case_rate": config.hard_case_rate,
        "counts": {
            "customer_type": dict(sorted(Counter(p.customer_type for p in plans).items())),
            "layout": dict(sorted(Counter(p.layout for p in plans).items())),
            "hard_cases": {c: cases[c] for c in HARD_CASES},
            "statements_with_any_hard_case": sum(bool(p.hard_cases) for p in plans),
            "transactions": sum(int(r["transaction_count"]) for r in manifest_rows),
        },
        "verify": {
            "statements_checked": report.statements,
            "reconciled": report.reconciled,
            "reconciliation_pass_rate": report.reconciliation_rate,
            "failures": len(report.failures),
            "first_failures": report.failures[:10],
        },
        "timing": {
            "write_seconds": round(write_seconds, 2),
            "verify_seconds": round(verify_seconds, 2),
            "seconds_per_statement": round(write_seconds / n, 4),
        },
        "disk": disk_usage(out),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
        "machine": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
