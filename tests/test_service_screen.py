"""POST /screen end to end: real screening of small generated PDFs, the audit row for each
request, every refusal, and that no temp file or statement text is left behind."""

import hashlib
import io
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter

from statementscreen.service import audit, screen, upload
from statementscreen.service.app import FRAMING, create_app
from statementscreen.synth.generator import generate_statement
from statementscreen.synth.render import render_statement
from statementscreen.tamper import copy as tamper_copy
from statementscreen.tamper.detect.verdict import NONE, TAMPERED

SHAPE = {
    "request_id", "status", "error_code", "readable", "profile", "tamper", "processing_seconds",
    "version",
}  # fmt: skip
VOLATILE = {"request_id", "processing_seconds"}
VOLATILE_ROW = {"request_id", "timestamp_utc", "processing_seconds"}


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def temp_root(tmp_path_factory):
    """The service's temp folders live here, so a leftover file or folder is visible."""
    root = tmp_path_factory.mktemp("service-temp")
    patch = pytest.MonkeyPatch()
    patch.setattr(tempfile, "tempdir", str(root))
    yield root
    patch.undo()


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    """A generated genuine statement, a tampered copy of it, and the text that must never be
    stored. Written to a pytest temp folder only."""
    folder = tmp_path_factory.mktemp("pdfs")
    s = generate_statement(seed=5, customer_type="salaried", months=6, start=date(2025, 1, 1))
    genuine = render_statement(s, folder / "genuine.pdf", "columns")
    data, _ = tamper_copy.tamper_copy(genuine, "rerender_inconsistent", seed=3)
    (folder / "tampered.pdf").write_bytes(data)
    secrets = [s.account_holder, s.account_holder.title(), s.account_number]
    secrets += [t.description for t in s.transactions if len(t.description) >= 15]
    if s.income.employer:
        secrets.append(s.income.employer)
    return {
        "genuine": genuine.read_bytes(),
        "tampered": data,
        "secrets": sorted(set(secrets)),
    }


@pytest.fixture(scope="module")
def env(tmp_path_factory, temp_root):
    log = audit.Audit(tmp_path_factory.mktemp("audit") / "audit.db")
    return TestClient(create_app(audit_log=log)), log


def _post(client, temp_root, data: bytes, **kwargs):
    response = client.post("/screen", files={"file": ("statement.pdf", data, "application/pdf")})
    assert list(temp_root.iterdir()) == [], "a temp file or folder was left behind"
    return response


def _row(client, response) -> dict:
    got = client.get(f"/audit/{response.json()['request_id']}")
    assert got.status_code == 200, "no audit row for this response"
    return got.json()


def _blank_pdf(pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def genuine_run(env, temp_root, pdfs):
    client, _ = env
    return _post(client, temp_root, pdfs["genuine"])


@pytest.fixture(scope="module")
def tampered_run(env, temp_root, pdfs):
    client, _ = env
    return _post(client, temp_root, pdfs["tampered"])


# ---------------------------------------------------------------- the two verdicts


def test_a_genuine_pdf_gets_the_full_shape_a_clean_verdict_and_an_audit_row(env, pdfs, genuine_run):
    client, _ = env
    body = genuine_run.json()
    assert genuine_run.status_code == 200 and set(body) == SHAPE
    assert (body["status"], body["error_code"], body["readable"]) == ("ok", None, True)
    assert body["tamper"]["verdict"] == NONE and body["profile"]["findings"]
    assert body["processing_seconds"] > 0
    assert set(body["version"]) >= {"git_commit", "detector_fingerprint"}
    row = _row(client, genuine_run)
    data = pdfs["genuine"]
    assert row["pdf_sha256"] == hashlib.sha256(data).hexdigest()
    assert (row["size_bytes"], row["declared_bytes"] >= len(data)) == (len(data), True)
    assert row["page_count"] == len(PdfReader(io.BytesIO(data)).pages)
    assert (row["status"], row["error_code"], row["verdict"]) == ("ok", None, NONE)
    findings = body["profile"]["findings"]
    assert row["has_salary"] == findings["has_salary"]["value"]
    assert row["customer_type"] == findings["customer_type"]["value"]
    assert row["emi_count"] == findings["emi_count"]["value"]
    assert row["git_commit"] == body["version"]["git_commit"]
    assert row["detector_fingerprint"] == body["version"]["detector_fingerprint"]


def test_a_tampered_pdf_is_flagged_and_its_row_names_the_finding(env, tampered_run):
    client, _ = env
    body = tampered_run.json()
    assert tampered_run.status_code == 200 and set(body) == SHAPE
    assert body["tamper"]["verdict"] == TAMPERED
    assert "BALANCE_CHAIN" in body["tamper"]["finding_codes"]
    row = _row(client, tampered_run)
    assert row["verdict"] == TAMPERED and "BALANCE_CHAIN" in row["finding_codes"].split(",")
    assert row["status"] == "ok"


def test_two_requests_with_the_same_pdf_give_identical_results(
    env, temp_root, pdfs, genuine_run, tampered_run
):
    client, _ = env

    def same_apart_from_ids(body: dict) -> dict:
        """The response without its request id and timing. The profile's statement_id is the
        stored file's name, which the service sets to the request id, so it goes too."""
        out = {k: v for k, v in body.items() if k not in VOLATILE}
        out["profile"] = {k: v for k, v in body["profile"].items() if k != "statement_id"}
        return out

    for name, first in (("genuine", genuine_run), ("tampered", tampered_run)):
        again = _post(client, temp_root, pdfs[name])
        assert again.json()["request_id"] != first.json()["request_id"]
        for response in (first, again):  # the one id in the profile is this request's own id
            assert response.json()["profile"]["statement_id"] == response.json()["request_id"]
        assert again.json()["tamper"] == first.json()["tamper"], f"{name} verdict changed"
        assert same_apart_from_ids(again.json()) == same_apart_from_ids(first.json())
        a, b = _row(client, again), _row(client, first)
        assert {k: v for k, v in a.items() if k not in VOLATILE_ROW} == {
            k: v for k, v in b.items() if k not in VOLATILE_ROW
        }


# ---------------------------------------------------------------- refusals are logged errors


def _refused(client, response, status: int, code: str) -> dict:
    body = response.json()
    assert response.status_code == status, response.text
    assert (body["status"], body["error_code"]) == ("error", code)
    assert body["request_id"] and "Traceback" not in response.text
    row = _row(client, response)
    assert (row["status"], row["error_code"], row["verdict"]) == ("error", code, None)
    assert row["finding_codes"] is None and row["has_salary"] is None
    return row


def test_a_file_that_is_not_a_pdf_is_a_400_and_a_logged_error(env, temp_root):
    client, _ = env
    row = _refused(client, _post(client, temp_root, b"just some text"), 400, "NOT_A_PDF")
    assert row["size_bytes"] == len(b"just some text")


def test_a_file_over_10_mb_is_a_413_and_a_logged_error(env, temp_root):
    client, _ = env
    big = b"%PDF-1.7\n" + b"0" * upload.MAX_BYTES
    row = _refused(client, _post(client, temp_root, big), 413, "TOO_LARGE")
    assert row["declared_bytes"] > upload.MAX_BYTES and row["pdf_sha256"] is None


def test_a_declared_length_far_over_the_limit_is_refused_before_reading(env, temp_root):
    client, _ = env
    declared = str(upload.MAX_BYTES + FRAMING + 1)
    response = client.post("/screen", content=b"x", headers={"content-length": declared})
    row = _refused(client, response, 413, "TOO_LARGE")
    assert (row["declared_bytes"], row["size_bytes"]) == (int(declared), None)
    assert list(temp_root.iterdir()) == []


def test_a_pdf_over_50_pages_is_a_422_and_a_logged_error(env, temp_root):
    client, _ = env
    data = _blank_pdf(upload.MAX_PAGES + 1)
    row = _refused(client, _post(client, temp_root, data), 422, "TOO_MANY_PAGES")
    assert row["page_count"] == upload.MAX_PAGES + 1
    assert row["pdf_sha256"] == hashlib.sha256(data).hexdigest()


def test_no_file_and_a_wrongly_named_field_are_422_and_logged(env, temp_root):
    client, _ = env
    _refused(client, client.post("/screen", data={"other": "x"}), 422, "NO_FILE")
    wrong = client.post("/screen", files={"document": ("a.pdf", b"%PDF-1.7", "application/pdf")})
    _refused(client, wrong, 422, "NO_FILE")
    assert list(temp_root.iterdir()) == []


def test_an_unparseable_body_is_a_logged_rejection_with_a_request_id(env, temp_root):
    client, _ = env
    headers = {"content-type": "multipart/form-data; boundary=xyz"}
    response = client.post("/screen", content=b"this is not multipart", headers=headers)
    assert response.status_code in (400, 422) and "Traceback" not in response.text
    assert response.json()["error_code"] in ("BAD_REQUEST", "NO_FILE")
    assert _row(client, response)["status"] == "error"
    assert list(temp_root.iterdir()) == []


# ---------------------------------------------------------------- unreadable PDFs and crashes


@pytest.mark.parametrize("kind", ["junk", "truncated"])
def test_a_corrupt_pdf_gets_a_clear_verdict_not_a_500(env, temp_root, pdfs, kind):
    client, _ = env
    data = b"%PDF-1.7\nthis is not really a pdf" if kind == "junk" else pdfs["genuine"][:3000]
    response = _post(client, temp_root, data)
    body = response.json()
    assert response.status_code == 200 and "Traceback" not in response.text
    assert body["readable"] is False and set(body) == SHAPE
    assert body["tamper"]["verdict"] != NONE and body["tamper"]["summary"].strip()
    row = _row(client, response)
    assert row["verdict"] == body["tamper"]["verdict"] and row["has_salary"] in (None, False)


def test_a_crash_in_the_profile_part_keeps_the_verdict_and_matches_the_row(
    env, temp_root, pdfs, monkeypatch, caplog
):
    client, _ = env

    def explode(path):
        raise RuntimeError(pdfs["secrets"][0])

    monkeypatch.setattr(screen, "profile_pdf", explode)
    response = _post(client, temp_root, pdfs["genuine"])
    body = response.json()
    assert response.status_code == 200 and body["profile"] is None
    assert (body["status"], body["error_code"]) == ("error", "PROFILE_CRASHED")
    assert body["tamper"]["verdict"] == NONE
    row = _row(client, response)
    assert (row["status"], row["error_code"]) == ("error", "PROFILE_CRASHED")
    assert pdfs["secrets"][0] not in response.text + caplog.text


def test_a_crash_outside_both_blocks_is_a_json_500_and_a_logged_error(tmp_path, temp_root):
    log = audit.Audit(tmp_path / "audit.db")

    def explode(path):
        raise RuntimeError("secret statement text")

    client = TestClient(create_app(audit_log=log, screen=explode))
    response = _post(client, temp_root, b"%PDF-1.7\nx")
    _refused(client, response, 500, "INTERNAL_ERROR")
    assert "secret statement text" not in response.text


# ---------------------------------------------------------------- what the audit log never holds


def test_no_holder_account_employer_or_narration_reaches_the_audit_database(
    env, pdfs, genuine_run, tampered_run
):
    client, log = env
    assert pdfs["secrets"], "the generated statement gave nothing to look for"
    raw = log.path.read_bytes()
    leaked = [s for s in pdfs["secrets"] if s.encode() in raw]
    assert not leaked, f"reached the audit database: {leaked[:3]}"
    for response in (genuine_run, tampered_run):
        text = " ".join(str(v) for v in _row(client, response).values())
        assert not [s for s in pdfs["secrets"] if s in text]


# ---------------------------------------------------------------- health and one at a time


def test_health_reports_ok_the_audit_status_and_the_versions(env):
    client, _ = env
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["audit"]["status"] == "ok"
    assert set(body["version"]["detector_fingerprint"]) == {"code", "config", "model", "uv_lock"}
    assert body["version"]["git_commit"] and body["version"]["rule_version"]


def test_only_one_pdf_is_processed_at_a_time(tmp_path, temp_root):
    state = {"now": 0, "most": 0}
    guard = threading.Lock()

    def slow(path):
        with guard:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(0.15)
        with guard:
            state["now"] -= 1
        return {
            "status": "ok",
            "error_code": None,
            "readable": True,
            "profile": None,
            "tamper": {"verdict": NONE, "finding_codes": []},
            "processing_seconds": 0.15,
        }

    log = audit.Audit(tmp_path / "audit.db")
    files = {"file": ("a.pdf", b"%PDF-1.7\nx", "application/pdf")}  # no temp check here: other
    with TestClient(create_app(audit_log=log, screen=slow)) as client:  # requests are in flight
        with ThreadPoolExecutor(3) as pool:
            codes = list(
                pool.map(lambda _: client.post("/screen", files=files).status_code, range(3))
            )
    assert codes == [200, 200, 200] and state["most"] == 1
    assert list(temp_root.iterdir()) == []
