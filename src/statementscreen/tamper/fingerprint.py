"""A fingerprint of everything that produces the detector's findings.

Cached findings (tamper/run.py) are only valid for the detector that made them, so a run
records this fingerprint and scoring refuses findings whose fingerprint differs from the
current one: a changed check means running the detector again, never scoring old findings.
There is no override.

Four parts, each a SHA-256, so a refusal can say which one differs:
    code     the check code: tamper/detect/*.py except the verdict rules and the config
    config   tamper/detect/config.py (tolerances, font limits)
    model    models/week6_iforest.pkl (the IsolationForest and its threshold)
    uv_lock  uv.lock (the dependency versions the checks ran with)

The verdict rules (detect/verdict.py) are left out on purpose: they turn findings into a
verdict and don't change any finding, so tuning rules never invalidates the cache. A file
that is missing hashes as "missing".

Lives outside detect/ because it hashes detect/; nothing in detect/ imports it.
"""

import hashlib
from pathlib import Path

from statementscreen.tamper import detect as _detect_package
from statementscreen.tamper.detect.anomaly import MODEL_PATH

DETECT_DIR = Path(_detect_package.__file__).parent
LOCK_PATH = Path(__file__).resolve().parents[3] / "uv.lock"
RULES_FILE = "verdict.py"
CONFIG_FILE = "config.py"
PARTS = ("code", "config", "model", "uv_lock")
WHAT = {
    "code": "the check code under tamper/detect/ (not the verdict rules)",
    "config": "tamper/detect/config.py",
    "model": "the IsolationForest model file models/week6_iforest.pkl",
    "uv_lock": "uv.lock, the dependency versions",
}


class StaleFindings(RuntimeError):
    """The findings were made by a different detector than the current one."""


def _hash(files: list[Path]) -> str:
    """SHA-256 over each file's name and bytes, in sorted order; "missing" if none exists."""
    existing = sorted(p for p in files if p.is_file())
    if not existing:
        return "missing"
    h = hashlib.sha256()
    for path in existing:
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def fingerprint(
    detect_dir: Path = DETECT_DIR, model: Path = MODEL_PATH, lock: Path = LOCK_PATH
) -> dict[str, str]:
    detect_dir = Path(detect_dir)
    code = [p for p in detect_dir.glob("*.py") if p.name not in (RULES_FILE, CONFIG_FILE)]
    return {
        "code": _hash(code),
        "config": _hash([detect_dir / CONFIG_FILE]),
        "model": _hash([Path(model)]),
        "uv_lock": _hash([Path(lock)]),
    }


def differences(recorded: dict, current: dict) -> list[str]:
    """The parts (in PARTS order) whose hash differs."""
    return [part for part in PARTS if recorded.get(part) != current.get(part)]


def check(recorded: dict | None, current: dict | None = None) -> None:
    """Raise StaleFindings, naming the differing parts, unless the fingerprints match."""
    if not recorded:
        raise StaleFindings("these findings carry no detector fingerprint, so they can't be used")
    current = fingerprint() if current is None else current
    diff = differences(recorded, current)
    if diff:
        what = "; ".join(f"{part} differs ({WHAT[part]})" for part in diff)
        raise StaleFindings(
            f"these findings were made by a different detector: {what}. "
            "Run the detector again (python -m statementscreen.tamper.run)."
        )
