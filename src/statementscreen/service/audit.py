"""The SQLite audit log: one row per request, and nothing from inside a statement.

    path     STATEMENTSCREEN_AUDIT_DB, default data/audit/audit.db (inside git-ignored data/);
             the folder is created and the file is set to mode 0600
    table    audit: the columns in COLUMNS and no others. There is no column for a PDF's
             contents, a holder's name, an account number, an employer, a counterparty or a
             narration, and no free-text column at all: verdicts, customer types and error
             codes come from fixed sets, finding codes match a strict pattern, and anything
             else is dropped (stored as NULL), never stored
    writes   record(event) only inserts. The service code issues no UPDATE or DELETE, and
             BEFORE UPDATE / BEFORE DELETE triggers abort them anyway, so a bug cannot change a
             row. Anyone who can write the file can drop a trigger: this guards against code
             mistakes, not against a person
    reads    get(request_id) uses a read-only connection

row_from_event() is the only place that decides what is kept. It reads the in-memory result's
verdict, finding codes and six profile numbers (has_salary, estimated monthly income, customer
type, EMI count, bounce count), never a reason, warning, employer, payee, lender or any text.

A refused upload is a row with status "error", its error code, declared_bytes (the
Content-Length the client declared, if any) and size_bytes (bytes actually read before it was
refused, NULL if none were read). pdf_sha256 is set only when the whole file was read.
Requests are not authenticated, so a local client can add junk rows. Rows are never deleted.

A failed write raises AuditError. What the service does then is decided in app.py, not here.
"""

import json
import os
import re
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from statementscreen.tamper.scoring import SCORED_LABELS

DB_ENV = "STATEMENTSCREEN_AUDIT_DB"
DEFAULT_DB = Path("data/audit/audit.db")
SCHEMA_VERSION = 1
COLUMNS = (
    "request_id",
    "timestamp_utc",
    "pdf_sha256",
    "declared_bytes",
    "size_bytes",
    "page_count",
    "verdict",
    "finding_codes",
    "has_salary",
    "estimated_monthly_income_paise",
    "customer_type",
    "emi_count",
    "bounce_count",
    "processing_seconds",
    "status",
    "error_code",
    "git_commit",
    "detector_fingerprint",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    request_id TEXT PRIMARY KEY CHECK (length(request_id) = 32),
    timestamp_utc TEXT NOT NULL,
    pdf_sha256 TEXT,
    declared_bytes INTEGER,
    size_bytes INTEGER,
    page_count INTEGER,
    verdict TEXT,
    finding_codes TEXT,
    has_salary INTEGER CHECK (has_salary IN (0, 1)),
    estimated_monthly_income_paise INTEGER,
    customer_type TEXT,
    emi_count INTEGER,
    bounce_count INTEGER,
    processing_seconds REAL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error_code TEXT,
    git_commit TEXT NOT NULL,
    detector_fingerprint TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
"""
INSERT = f"INSERT INTO audit ({', '.join(COLUMNS)}) VALUES ({', '.join(':' + c for c in COLUMNS)})"
ID = re.compile(r"[0-9a-f]{32}")
HEX64 = re.compile(r"[0-9a-f]{64}")
FINDING_CODE = re.compile(r"[A-Z][A-Z0-9_]*")
ERROR_CODES = re.compile(r"[A-Z][A-Z_]*(,[A-Z][A-Z_]*)*")
CUSTOMER_TYPES = frozenset({"salaried", "gig", "self_employed", "unknown"})
VERDICTS = frozenset(SCORED_LABELS)


class AuditError(Exception):
    """The audit log cannot be opened or written, or an event cannot be made into a row."""


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _finding(profile: object, name: str) -> object:
    """One profile finding's value, or None."""
    findings = profile.get("findings") if isinstance(profile, dict) else None
    found = findings.get(name) if isinstance(findings, dict) else None
    return found.get("value") if isinstance(found, dict) else None


def row_from_event(event: Mapping, now: datetime | None = None) -> dict:
    """The audit row for one request event: only the allow-listed fields, each checked."""
    request_id, status = event.get("request_id"), event.get("status")
    if not (isinstance(request_id, str) and ID.fullmatch(request_id)):
        raise AuditError("the request id is not 32 lowercase hex characters")
    if status not in ("ok", "error"):
        raise AuditError("the status must be 'ok' or 'error'")
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    profile, tamper = result.get("profile"), result.get("tamper")

    sha = event.get("pdf_sha256")
    verdict = tamper.get("verdict") if isinstance(tamper, dict) else None
    codes = None
    if isinstance(tamper, dict):
        listed = [c for c in tamper.get("finding_codes") or [] if isinstance(c, str)]
        codes = ",".join(sorted({c for c in listed if FINDING_CODE.fullmatch(c)}))
    salary = _finding(profile, "has_salary")
    customer = _finding(profile, "customer_type")
    customer = customer if isinstance(customer, str) and customer in CUSTOMER_TYPES else None
    error = event.get("error_code")
    error = error if isinstance(error, str) and ERROR_CODES.fullmatch(error) else None
    seconds = result.get("processing_seconds")
    is_number = isinstance(seconds, int | float) and not isinstance(seconds, bool)
    income = _int(_finding(profile, "estimated_monthly_income_paise"))
    version = event.get("version") or {}
    stamp = (now or datetime.now(UTC)).astimezone(UTC)
    return {
        "request_id": request_id,
        "timestamp_utc": stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "pdf_sha256": sha if isinstance(sha, str) and HEX64.fullmatch(sha) else None,
        "declared_bytes": _int(event.get("declared_bytes")),
        "size_bytes": _int(event.get("size_bytes")),
        "page_count": _int(event.get("pages")),
        "verdict": verdict if isinstance(verdict, str) and verdict in VERDICTS else None,
        "finding_codes": codes,
        "has_salary": int(salary) if isinstance(salary, bool) else None,
        "estimated_monthly_income_paise": income,
        "customer_type": customer,
        "emi_count": _int(_finding(profile, "emi_count")),
        "bounce_count": _int(_finding(profile, "bounced_emis")),
        "processing_seconds": float(seconds) if is_number else None,
        "status": status,
        "error_code": error,
        "git_commit": str(version.get("git_commit", "unknown"))[:100],
        "detector_fingerprint": json.dumps(
            version.get("detector_fingerprint"), sort_keys=True, separators=(",", ":")
        ),
    }


def _public(row: dict) -> dict:
    """A stored row as the API returns it: has_salary as a boolean, the fingerprint as an object."""
    out = dict(row)
    if out["has_salary"] is not None:
        out["has_salary"] = bool(out["has_salary"])
    out["detector_fingerprint"] = json.loads(out["detector_fingerprint"])
    return out


class Audit:
    def __init__(self, path: Path | None = None, env: Mapping[str, str] | None = None) -> None:
        chosen = path if path is not None else (os.environ if env is None else env).get(DB_ENV)
        self.path = Path(chosen) if chosen else DEFAULT_DB
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, SCHEMA_VERSION):
                    raise AuditError(f"{self.path} has schema version {version}")
                table = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'audit'"
                if db.execute(table).fetchone() is None:  # a new file: make it. An existing
                    db.executescript(SCHEMA)  # table is never altered or repaired here
                    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            os.chmod(self.path, 0o600)
        except (OSError, sqlite3.Error) as e:
            raise AuditError(f"cannot open the audit log: {type(e).__name__}") from e
        self.check()  # refuse to start on anything but a sound, writable, append-only log

    def check(self) -> dict:
        """Raise AuditError unless the log is writable, at schema version 1, has exactly the
        agreed columns and both append-only triggers. Returns a small status dict (no paths)."""
        try:
            with closing(self._existing()) as db:  # never creates a missing file
                names = tuple(r[1] for r in db.execute("PRAGMA table_info(audit)"))
                if names != COLUMNS:
                    raise AuditError(f"{self.path} has an audit table with other columns")
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version != SCHEMA_VERSION:
                    raise AuditError(f"{self.path} has schema version {version}")
                found = db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'audit'"
                )
                if {"audit_no_update", "audit_no_delete"} - {r[0] for r in found}:
                    raise AuditError(f"{self.path} is missing an append-only trigger")
                db.execute("BEGIN IMMEDIATE")  # takes the write lock: fails if not writable
                db.rollback()
        except (OSError, sqlite3.Error) as e:
            raise AuditError(f"the audit log is not usable: {type(e).__name__}") from e
        return {"status": "ok", "schema_version": SCHEMA_VERSION, "append_only": True}

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def _existing(self) -> sqlite3.Connection:
        """A read-write connection to a file that must already exist (mode=rw)."""
        return sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=rw", uri=True, timeout=5)

    def _read_connection(self) -> sqlite3.Connection:
        db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def record(self, event: Mapping) -> None:
        """Insert one row for the event. Raises AuditError if it cannot be written."""
        row = row_from_event(event)
        try:
            with closing(self._connect()) as db, db:
                db.execute(INSERT, row)
        except sqlite3.Error as e:
            raise AuditError(f"cannot write the audit log: {type(e).__name__}") from e

    def get(self, request_id: str) -> dict | None:
        """The row for a request id, or None (also for anything that is not a valid id)."""
        if not ID.fullmatch(request_id):
            return None
        try:
            with closing(self._read_connection()) as db:
                row = db.execute("SELECT * FROM audit WHERE request_id = ?", (request_id,))
                found = row.fetchone()
        except sqlite3.Error as e:
            raise AuditError(f"cannot read the audit log: {type(e).__name__}") from e
        return None if found is None else _public(dict(found))
