"""The audit log: what a row keeps, what it never keeps, and that it can only be appended to."""

import re
import sqlite3
import stat
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from statementscreen.service import audit
from statementscreen.service.app import FRAMING, create_app
from statementscreen.service.upload import MAX_BYTES

HOLDER = "Zebulon Quillfeather"
ACCOUNT = "9876543210123456"
NARRATION = "UPI/ZEPHYRMART/9988776655/groceries"
EMPLOYER = "Quokka Dynamics Ltd"
SENTINELS = (HOLDER, ACCOUNT, NARRATION, EMPLOYER, "ZEPHYRMART", "Quokka")
REQUEST = "a" * 32
VERSION = {
    "git_commit": "c" * 40,
    "detector_fingerprint": {"code": "1", "config": "2", "model": "3", "uv_lock": "4"},
    "rule_version": "v2",
}


def _finding(value, reason: str | None = None) -> dict:
    return {"value": value, "reason": reason or f"{HOLDER} {ACCOUNT} {NARRATION} {EMPLOYER}"}


def _result() -> dict:
    """A result whose every free-text field is full of the sentinels."""
    findings = {
        "has_salary": _finding(True),
        "employer": _finding(EMPLOYER),
        "estimated_monthly_income_paise": _finding(5500000),
        "customer_type": _finding("salaried"),
        "emi_count": _finding(2),
        "bounced_emis": _finding(1),
        "rent": _finding([{"payee": HOLDER, "monthly_paise": 1500000}]),
        "other_regular_income": _finding([{"counterparty": NARRATION, "channel": ACCOUNT}]),
    }
    return {
        "status": "ok",
        "error_code": None,
        "readable": True,
        "processing_seconds": 4.123,
        "profile": {"statement_id": REQUEST, "findings": findings, "warnings": [HOLDER, ACCOUNT]},
        "tamper": {
            "verdict": "suspicious",
            "summary": f"{NARRATION} {HOLDER}",
            "reasons": [{"strength": "weak", "code": "ANOMALY_SCORE", "reason": NARRATION}],
            "finding_codes": ["ANOMALY_SCORE"],
            "checks_not_run": [],
            "rule_version": "v2",
        },
    }


def _event(**changes) -> dict:
    event = {
        "request_id": REQUEST,
        "status": "ok",
        "error_code": None,
        "declared_bytes": 2048,
        "size_bytes": 1900,
        "pdf_sha256": "b" * 64,
        "pages": 3,
        "result": _result(),
        "version": VERSION,
    }
    return {**event, **changes}


@pytest.fixture
def log(tmp_path):
    return audit.Audit(tmp_path / "audit" / "audit.db")


def test_the_table_has_exactly_the_agreed_columns_and_none_for_statement_contents(log):
    with closing(sqlite3.connect(log.path)) as db:
        names = tuple(r[1] for r in db.execute("PRAGMA table_info(audit)"))
    assert names == audit.COLUMNS
    assert "declared_bytes" in names and "size_bytes" in names
    assert not [n for n in names if re.search("name|account|employer|narration|holder", n)]


def test_a_request_is_recorded_and_read_back_with_only_the_allowed_fields(log):
    log.record(_event())
    row = log.get(REQUEST)
    assert set(row) == set(audit.COLUMNS)
    assert row["verdict"] == "suspicious" and row["finding_codes"] == "ANOMALY_SCORE"
    assert row["has_salary"] is True and row["estimated_monthly_income_paise"] == 5500000
    assert row["customer_type"] == "salaried" and row["emi_count"] == 2
    assert row["bounce_count"] == 1 and row["processing_seconds"] == 4.123
    assert (row["declared_bytes"], row["size_bytes"], row["page_count"]) == (2048, 1900, 3)
    assert row["pdf_sha256"] == "b" * 64 and row["status"] == "ok" and row["error_code"] is None
    assert row["git_commit"] == "c" * 40
    assert row["detector_fingerprint"] == VERSION["detector_fingerprint"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", row["timestamp_utc"])


def test_no_name_account_employer_or_narration_reaches_the_database_file(log):
    log.record(_event())
    log.record(_event(request_id="d" * 32))
    raw = log.path.read_bytes()
    for sentinel in SENTINELS:
        assert sentinel.encode() not in raw, f"{sentinel!r} reached the audit database"
    with closing(sqlite3.connect(log.path)) as db:
        text = " ".join(str(v) for row in db.execute("SELECT * FROM audit") for v in row)
    assert not [s for s in SENTINELS if s in text]


def test_a_refused_upload_is_a_row_with_its_error_code_and_no_verdict_or_profile(log):
    event = _event(
        status="error",
        error_code="NOT_A_PDF",
        result=None,
        pdf_sha256=None,
        size_bytes=5,
        pages=None,
    )
    log.record(event)
    row = log.get(REQUEST)
    assert (row["status"], row["error_code"], row["declared_bytes"]) == ("error", "NOT_A_PDF", 2048)
    assert row["size_bytes"] == 5 and row["pdf_sha256"] is None and row["page_count"] is None
    empty = ("verdict", "finding_codes", "has_salary", "customer_type", "emi_count")
    empty += ("estimated_monthly_income_paise", "bounce_count", "processing_seconds")
    for column in empty:
        assert row[column] is None, column


def test_free_text_in_a_checked_field_is_dropped_not_stored(log):
    result = _result()
    result["tamper"]["verdict"] = HOLDER
    result["tamper"]["finding_codes"] = ["ANOMALY_SCORE", NARRATION, "lower case", 7]
    result["profile"]["findings"]["customer_type"]["value"] = EMPLOYER
    event = _event(result=result, error_code=f"{HOLDER} failed", pdf_sha256=ACCOUNT)
    log.record(event)
    row = log.get(REQUEST)
    assert row["verdict"] is None and row["customer_type"] is None
    assert row["finding_codes"] == "ANOMALY_SCORE"
    assert row["error_code"] is None and row["pdf_sha256"] is None
    assert not [s for s in SENTINELS if s.encode() in log.path.read_bytes()]


def test_an_event_that_cannot_be_a_row_is_refused(log):
    for bad in ({"request_id": "short"}, {"request_id": "A" * 32}, {"status": "fine"}):
        with pytest.raises(audit.AuditError):
            log.record(_event(**bad))
    assert log.get(REQUEST) is None


def test_the_log_is_append_only_even_for_a_direct_update_or_delete(log):
    log.record(_event())
    with closing(sqlite3.connect(log.path)) as db:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("UPDATE audit SET status = 'error' WHERE request_id = ?", (REQUEST,))
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            db.execute("DELETE FROM audit WHERE request_id = ?", (REQUEST,))
    assert log.get(REQUEST)["status"] == "ok"


def test_a_second_row_with_the_same_request_id_is_refused(log):
    log.record(_event())
    with pytest.raises(audit.AuditError, match="cannot write"):
        log.record(_event())


def test_the_path_comes_from_the_environment_the_folder_is_made_and_the_mode_is_0600(tmp_path):
    path = tmp_path / "deep" / "er" / "audit.db"
    log = audit.Audit(env={audit.DB_ENV: str(path)})
    assert log.path == path and path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert audit.DEFAULT_DB == Path("data/audit/audit.db")


def test_a_database_with_another_schema_is_refused(tmp_path):
    other = tmp_path / "other.db"
    with closing(sqlite3.connect(other)) as db:
        db.execute("CREATE TABLE audit (anything TEXT)")
    with pytest.raises(audit.AuditError, match="other columns"):
        audit.Audit(other)
    newer = tmp_path / "newer.db"
    with closing(sqlite3.connect(newer)) as db:
        db.execute("PRAGMA user_version = 7")
    with pytest.raises(audit.AuditError, match="schema version 7"):
        audit.Audit(newer)


def test_lookups_accept_only_real_ids_and_use_a_read_only_connection(log):
    log.record(_event())
    for bad in ("", "../etc/passwd", "A" * 32, "a" * 31, REQUEST + "0", "a'; DROP TABLE audit;--"):
        assert log.get(bad) is None
    assert log.get("e" * 32) is None
    with closing(log._read_connection()) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM audit")


def test_the_timestamp_is_utc_to_the_millisecond():
    stamp = datetime(2026, 11, 24, 13, 5, 41, 123456, tzinfo=UTC)
    assert audit.row_from_event(_event(), now=stamp)["timestamp_utc"] == "2026-11-24T13:05:41.123Z"


# ---------------------------------------------------------------- GET /audit/{request_id}


@pytest.fixture
def client(log):
    log.record(_event())
    return TestClient(create_app(version=VERSION, audit_log=log))


def test_the_audit_route_returns_the_row(client):
    response = client.get(f"/audit/{REQUEST}")
    assert response.status_code == 200
    assert response.json()["verdict"] == "suspicious" and response.json()["request_id"] == REQUEST


def test_the_audit_route_gives_a_json_404_and_never_echoes_the_id(client):
    for bad in ("f" * 32, "not-an-id", "<b>bold"):
        response = client.get(f"/audit/{bad}")
        assert response.status_code == 404
        assert response.json()["error_code"] == "NOT_FOUND"
        assert bad not in response.text


# ---------------------------------------------------------------- failing closed

PDF = b"%PDF-1.7\nnot really a pdf"


def _broken(events: list):
    """An audit hook that notes the event it was given and then fails to write it."""

    def hook(event: dict) -> None:
        events.append(event)
        raise audit.AuditError("cannot write the audit log: OperationalError")

    return hook


def _screen(path):
    return _result()  # a result full of the sentinels: it must never leave the service


def _crash(path):
    raise RuntimeError(HOLDER)


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    """Point the service's temp folders at an empty folder so a leftover file is visible."""
    root = tmp_path / "service-temp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _post(client, data: bytes = PDF):
    return client.post("/screen", files={"file": ("statement.pdf", data, "application/pdf")})


def _only_code_and_id(response, events: list, error_code: str | None = None) -> None:
    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "AUDIT_FAILED" and set(body) == {"request_id", "error_code"}
    assert events[-1]["request_id"] == body["request_id"]  # a row was attempted for this id
    if error_code:
        assert events[-1]["error_code"] == error_code


def test_a_failed_audit_write_withholds_the_result_and_writes_it_nowhere_else(
    temp_root, caplog, capsys
):
    events: list = []
    client = TestClient(create_app(version=VERSION, audit=_broken(events), screen=_screen))
    response = _post(client)
    _only_code_and_id(response, events)
    assert len(events) == 1 and events[0]["status"] == "ok"
    for sentinel in (*SENTINELS, "suspicious", "ANOMALY_SCORE"):
        assert sentinel not in response.text
    seen = capsys.readouterr()
    everywhere = caplog.text + seen.out + seen.err
    assert not [s for s in (*SENTINELS, "suspicious") if s in everywhere]
    assert list(temp_root.iterdir()) == []


def test_a_refusal_that_cannot_be_logged_becomes_audit_failed_too(temp_root):
    events: list = []
    client = TestClient(create_app(version=VERSION, audit=_broken(events), screen=_screen))
    _only_code_and_id(_post(client, b"hello"), events, "NOT_A_PDF")
    _only_code_and_id(client.post("/screen", data={"other": "x"}), events, "NO_FILE")
    too_big = {"content-length": str(MAX_BYTES + FRAMING + 1)}
    _only_code_and_id(client.post("/screen", content=b"x", headers=too_big), events, "TOO_LARGE")
    assert len(events) == 3 and list(temp_root.iterdir()) == []


def test_a_crash_that_cannot_be_logged_is_audit_failed_and_leaks_nothing(temp_root, caplog):
    events: list = []
    client = TestClient(create_app(version=VERSION, audit=_broken(events), screen=_crash))
    response = _post(client)
    _only_code_and_id(response, events, "INTERNAL_ERROR")
    assert HOLDER not in response.text and HOLDER not in caplog.text
    assert list(temp_root.iterdir()) == []


def test_when_the_audit_write_works_the_response_and_the_row_agree(log, temp_root):
    client = TestClient(create_app(version=VERSION, audit_log=log, screen=_screen))
    ok = _post(client)
    assert ok.status_code == 200 and ok.json()["tamper"]["verdict"] == "suspicious"
    row = log.get(ok.json()["request_id"])
    assert (row["status"], row["verdict"], row["error_code"]) == ("ok", "suspicious", None)
    refused = _post(client, b"hello")
    assert refused.status_code == 400
    row = log.get(refused.json()["request_id"])
    assert (row["status"], row["error_code"], row["verdict"]) == ("error", "NOT_A_PDF", None)
    assert list(temp_root.iterdir()) == []


# ---------------------------------------------------------------- startup and /health


def test_startup_refuses_a_log_without_both_triggers_or_not_at_version_1(tmp_path):
    path = tmp_path / "a.db"
    audit.Audit(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("DROP TRIGGER audit_no_delete")
    with pytest.raises(audit.AuditError, match="trigger"):
        audit.Audit(path)
    other = tmp_path / "b.db"
    audit.Audit(other)
    with closing(sqlite3.connect(other)) as db:
        db.execute("PRAGMA user_version = 0")
    with pytest.raises(audit.AuditError, match="schema version 0"):
        audit.Audit(other)


def test_startup_does_not_repair_an_existing_table(tmp_path):
    path = tmp_path / "a.db"
    audit.Audit(path)
    with closing(sqlite3.connect(path)) as db:
        db.execute("DROP TRIGGER audit_no_update")
    for _ in range(2):  # it refuses each time and never quietly puts the trigger back
        with pytest.raises(audit.AuditError):
            audit.Audit(path)
    with closing(sqlite3.connect(path)) as db:
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert names == {"audit_no_delete"}


def test_health_reports_the_audit_status_and_goes_503_when_the_log_stops_being_sound(log):
    client = TestClient(create_app(version=VERSION, audit_log=log))
    ok = client.get("/health")
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    assert ok.json()["audit"] == {"status": "ok", "schema_version": 1, "append_only": True}
    with closing(sqlite3.connect(log.path)) as db:
        db.execute("DROP TRIGGER audit_no_update")
    bad = client.get("/health")
    assert bad.status_code == 503
    assert bad.json()["error_code"] == "AUDIT_UNAVAILABLE" and bad.json()["audit"] == {
        "status": "error"
    }
    assert str(log.path) not in bad.text


def test_health_without_an_audit_log_says_so():
    health = TestClient(create_app(version=VERSION)).get("/health").json()
    assert health["status"] == "ok" and health["audit"] == {"status": "not_configured"}
