"""Fixes from the Week 7 review: the receive-side size cap, a required audit log, routes outside
/screen, a failing audit lookup, /health on a deleted database and the startup entry point."""

import asyncio
import json
import sqlite3
import tempfile
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from statementscreen.service import __main__ as entry
from statementscreen.service import audit
from statementscreen.service.app import FRAMING, create_app
from statementscreen.service.upload import MAX_BYTES

VERSION = {
    "git_commit": "c" * 40,
    "detector_fingerprint": {"code": "1", "config": "2", "model": "3", "uv_lock": "4"},
    "rule_version": "v2",
}


@pytest.fixture
def log(tmp_path):
    return audit.Audit(tmp_path / "audit" / "audit.db")


@pytest.fixture
def client(log):
    return TestClient(create_app(version=VERSION, audit_log=log))


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    root = tmp_path / "service-temp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _rows(log) -> int:
    with closing(sqlite3.connect(log.path)) as db:
        return db.execute("SELECT count(*) FROM audit").fetchone()[0]


def test_two_content_length_headers_are_refused_before_the_body_is_read(log):
    """Sent straight to the ASGI app, because an HTTP client or server would not normally let
    two Content-Length headers through. The body must never be read."""
    app = create_app(version=VERSION, audit_log=log)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/screen",
        "raw_path": b"/screen",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"multipart/form-data; boundary=x"),
            (b"content-length", b"10"),
            (b"content-length", b"20"),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    sent: list[dict] = []

    async def receive():
        raise AssertionError("the body was read")

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 400
    body = json.loads(b"".join(m.get("body", b"") for m in sent[1:]))
    assert body["status"] == "error" and body["error_code"] == "BAD_REQUEST"
    assert "traceback" not in json.dumps(body).lower()
    assert _rows(log) == 1
    row = log.get(body["request_id"])
    assert row["status"] == "error" and row["error_code"] == "BAD_REQUEST"
    assert row["declared_bytes"] is None and row["size_bytes"] is None
    assert row["pdf_sha256"] is None and row["verdict"] is None


def test_an_audit_log_is_required_unless_the_test_flag_is_given():
    with pytest.raises(ValueError, match="audit log is required"):
        create_app(version=VERSION)
    assert create_app(version=VERSION, audit_disabled=True)


def test_a_chunked_upload_with_no_length_is_refused_once_it_passes_the_limit(
    log, client, temp_root
):
    boundary = "xyzboundary"
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n%PDF-1.7\n"
    ).encode()
    chunk = b"0" * 65536

    def body():
        yield head
        for _ in range((MAX_BYTES + FRAMING) // len(chunk) + 16):  # a bit past the limit
            yield chunk

    headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
    response = client.post("/screen", content=body(), headers=headers)
    assert response.status_code == 413 and response.json()["error_code"] == "TOO_LARGE"
    row = log.get(response.json()["request_id"])
    assert row["declared_bytes"] is None  # no Content-Length was sent: only the byte count acted
    assert row["size_bytes"] > MAX_BYTES + FRAMING
    assert (row["status"], row["error_code"], row["pdf_sha256"]) == ("error", "TOO_LARGE", None)
    assert list(temp_root.iterdir()) == []


def test_other_paths_keep_the_frameworks_own_answers_and_get_no_audit_row(log, client):
    missing = client.get("/no-such-path")
    assert missing.status_code == 404 and missing.json() == {"detail": "Not Found"}
    wrong_method = client.get("/screen")
    assert wrong_method.status_code == 405 and "error_code" not in wrong_method.json()
    assert _rows(log) == 0


def test_a_failing_audit_lookup_is_a_json_500_not_a_plain_text_one(log, client, monkeypatch):
    def broken(request_id):
        raise audit.AuditError("cannot read the audit log: OperationalError")

    monkeypatch.setattr(log, "get", broken)
    response = client.get(f"/audit/{'a' * 32}")
    assert response.status_code == 500 and response.json()["error_code"] == "AUDIT_UNAVAILABLE"
    assert "OperationalError" not in response.text


def test_health_on_a_deleted_database_is_503_and_does_not_recreate_the_file(log, client):
    assert client.get("/health").status_code == 200
    log.path.unlink()
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["error_code"] == "AUDIT_UNAVAILABLE"
    assert not log.path.exists()


def test_the_entry_point_binds_loopback_by_default_and_opens_the_audit_log(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        entry.uvicorn, "run", lambda app, host, port: seen.update(host=host, port=port)
    )
    monkeypatch.delenv("STATEMENTSCREEN_HOST", raising=False)
    monkeypatch.delenv("STATEMENTSCREEN_PORT", raising=False)
    monkeypatch.setenv(audit.DB_ENV, str(tmp_path / "a.db"))
    entry.main()
    assert seen == {"host": "127.0.0.1", "port": 8000}
    assert (tmp_path / "a.db").is_file()


def test_the_entry_point_does_not_start_on_an_unsound_audit_log(tmp_path, monkeypatch):
    unsound = tmp_path / "newer.db"
    with closing(sqlite3.connect(unsound)) as db:
        db.execute("PRAGMA user_version = 7")

    def must_not_run(*args, **kwargs):
        raise AssertionError("the server started")

    monkeypatch.setattr(entry.uvicorn, "run", must_not_run)
    monkeypatch.setenv(audit.DB_ENV, str(unsound))
    with pytest.raises(SystemExit) as stopped:
        entry.main()
    assert "did not start" in str(stopped.value)
