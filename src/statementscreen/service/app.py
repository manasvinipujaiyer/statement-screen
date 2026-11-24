"""The FastAPI app: POST /screen and GET /health. Local use only: there is no authentication.

POST /screen takes one PDF as the multipart field "file" and returns request_id, status,
error_code, readable, profile, tamper, processing_seconds and version. Rejections (not a PDF,
over 10 MB, over 50 pages, no file) are 4xx with a JSON body that has the same request_id,
status "error" and error_code, and no stack trace.

Safety, in the order a request meets it:
  1. a pure-ASGI check refuses a request whose Content-Length is over the limit (plus a margin
     for the multipart framing) before the body is read. A chunked upload with no length header
     is not caught here; upload.store() still enforces the size while copying;
  2. the upload is copied into a private temp folder (named after the request id, never the
     client's file name), checked, and hashed;
  3. an asyncio.Lock lets one PDF be processed at a time (a second request waits its turn), and
     the heavy work runs in a worker thread so the event loop stays free;
  4. the temp folder is deleted in every case, including errors.
An unexpected crash outside the profile and tamper blocks is a 500 with a JSON error code (the
exception's class name is logged, never its message).

The audit log fails closed. Every response, including every refusal, is first recorded with
`audit(event)` (the SQLite log's record method). If that write fails, the response is replaced by
a 500 whose body is only {"request_id", "error_code": "AUDIT_FAILED"}: the result is withheld and
is not written anywhere else, so every response a client sees has a matching audit row. There are
no retries beyond SQLite's own 5-second busy timeout. The event carries the in-memory result so
the log can pick the few fields it keeps; the app itself never writes a PDF's contents anywhere.
"""

import asyncio
import logging
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from statementscreen.service.audit import Audit
from statementscreen.service.screen import screen_pdf
from statementscreen.service.upload import MAX_BYTES, Rejected, store
from statementscreen.service.version import code_version

log = logging.getLogger("statementscreen.service")
FRAMING = 64 * 1024  # room for the multipart boundaries around the file in Content-Length
INTERNAL_ERROR = "INTERNAL_ERROR"
AUDIT_FAILED = "AUDIT_FAILED"
AUDIT_UNAVAILABLE = "AUDIT_UNAVAILABLE"


def _no_audit(event: dict) -> None:
    """The audit hook when none is given (tests only): records nothing."""


class Service:
    """What the routes share: the version, the lock, the audit hook and the screening function."""

    def __init__(
        self,
        version: dict,
        audit: Callable[[dict], None] | None = None,
        screen: Callable[[Path], dict] = screen_pdf,
    ) -> None:
        self.version = version
        self.audit = audit or _no_audit
        self.screen = screen
        self.lock = asyncio.Lock()

    def record(self, event: dict) -> JSONResponse | None:
        """Write the audit row. None if it was written; else the AUDIT_FAILED response to send
        instead of anything else. Only the exception's class name is logged."""
        try:
            self.audit(event)
        except Exception as e:  # any failure to log means the response is withheld
            log.error("%s: %s", AUDIT_FAILED, type(e).__name__)
            body = {"request_id": event["request_id"], "error_code": AUDIT_FAILED}
            return JSONResponse(body, status_code=500)
        return None

    def reject(
        self, request_id: str, rejected: Rejected, declared: int | None = None
    ) -> JSONResponse:
        event = {
            "request_id": request_id,
            "status": "error",
            "error_code": rejected.code,
            "declared_bytes": declared,
            "size_bytes": rejected.size,
            "pdf_sha256": rejected.sha256,
            "pages": rejected.pages,
            "result": None,
            "version": self.version,
        }
        failure = self.record(event)
        if failure is not None:
            return failure
        body = {
            "request_id": request_id,
            "status": "error",
            "error_code": rejected.code,
            "message": rejected.message,
            "version": self.version,
        }
        return JSONResponse(body, status_code=rejected.status)

    def failed(
        self, request_id: str, error: Exception, declared: int | None = None
    ) -> JSONResponse:
        log.error("%s: %s", INTERNAL_ERROR, type(error).__name__)
        event = {
            "request_id": request_id,
            "status": "error",
            "error_code": INTERNAL_ERROR,
            "declared_bytes": declared,
            "size_bytes": None,
            "pdf_sha256": None,
            "pages": None,
            "result": None,
            "version": self.version,
        }
        failure = self.record(event)
        if failure is not None:
            return failure
        body = {
            "request_id": request_id,
            "status": "error",
            "error_code": INTERNAL_ERROR,
            "message": "the service failed on this request",
            "version": self.version,
        }
        return JSONResponse(body, status_code=500)


class ContentLengthLimit:
    """Refuse an oversized POST /screen from its Content-Length header, before reading the body."""

    def __init__(self, app, service: Service, limit: int = MAX_BYTES + FRAMING) -> None:
        self.app, self.service, self.limit = app, service, limit

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/screen":
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > self.limit:
                rejected = Rejected(
                    413, "TOO_LARGE", f"the upload is larger than {MAX_BYTES // 2**20} MB"
                )  # nothing was read: size stays None, the declared length is recorded
                response = self.service.reject(uuid.uuid4().hex, rejected, int(length))
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def declared_length(request: Request) -> int | None:
    """The Content-Length the client declared, or None if it sent none."""
    raw = request.headers.get("content-length", "")
    return int(raw) if raw.isdigit() else None


def create_app(
    version: dict | None = None,
    audit: Callable[[dict], None] | None = None,
    screen: Callable[[Path], dict] = screen_pdf,
    audit_log: Audit | None = None,
) -> FastAPI:
    """Build the app. With audit_log, every request is recorded in it (and GET /audit/{id} and
    the audit part of /health exist); `audit` alone is a bare hook for tests. With neither,
    nothing is recorded: python -m statementscreen.service always passes the real log."""
    hook = audit if audit is not None else (audit_log.record if audit_log is not None else None)
    service = Service(code_version() if version is None else version, hook, screen)
    app = FastAPI(
        title="StatementScreen", description="Screen one bank-statement PDF. Local use only."
    )
    app.state.service = service
    app.add_middleware(ContentLengthLimit, service=service)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        rejected = Rejected(422, "BAD_REQUEST", "send one PDF as the multipart form field 'file'")
        return service.reject(uuid.uuid4().hex, rejected)

    @app.exception_handler(StarletteHTTPException)
    async def http_problem(request: Request, exc: StarletteHTTPException):
        # Starlette answers an unparseable multipart body with its own 400 before our route
        # runs; on /screen that becomes a normal logged rejection with a request id.
        if request.method == "POST" and request.url.path == "/screen" and exc.status_code == 400:
            rejected = Rejected(
                400, "BAD_REQUEST", "the upload could not be read as multipart form data"
            )
            return service.reject(uuid.uuid4().hex, rejected, declared_length(request))
        return await http_exception_handler(request, exc)

    @app.get("/health")
    async def health() -> JSONResponse:
        if audit_log is None:
            audit_status = {"status": "not_configured" if hook is None else "custom_hook"}
        else:
            try:
                audit_status = await run_in_threadpool(audit_log.check)
            except Exception as e:  # class name only: no path, no message
                log.error("%s: %s", AUDIT_UNAVAILABLE, type(e).__name__)
                body = {
                    "status": "error",
                    "error_code": AUDIT_UNAVAILABLE,
                    "audit": {"status": "error"},
                    "version": service.version,
                }
                return JSONResponse(body, status_code=503)
        return JSONResponse({"status": "ok", "audit": audit_status, "version": service.version})

    if audit_log is not None:

        @app.get("/audit/{request_id}")
        async def audit_row(request_id: str) -> JSONResponse:
            row = audit_log.get(request_id)
            if row is None:  # never echo the id back
                body = {
                    "status": "error",
                    "error_code": "NOT_FOUND",
                    "message": "no audit row for that request id",
                }
                return JSONResponse(body, status_code=404)
            return JSONResponse(row)

    @app.post("/screen")
    async def screen_endpoint(
        request: Request, file: Annotated[UploadFile | None, File()] = None
    ) -> JSONResponse:
        request_id = uuid.uuid4().hex
        declared = declared_length(request)
        try:
            if file is None:
                rejected = Rejected(422, "NO_FILE", "send one PDF as 'file'")
                return service.reject(request_id, rejected, declared)
            with tempfile.TemporaryDirectory(prefix="statementscreen-") as folder:
                try:
                    stored = await store(file, Path(folder), request_id)
                except Rejected as rejected:
                    return service.reject(request_id, rejected, declared)
                async with service.lock:  # one PDF at a time
                    result = await run_in_threadpool(service.screen, stored.path)
            event = {
                "request_id": request_id,
                "status": result["status"],
                "error_code": result["error_code"],
                "declared_bytes": declared,
                "size_bytes": stored.size,
                "pdf_sha256": stored.sha256,
                "pages": stored.pages,
                "result": result,
                "version": service.version,
            }
            failure = service.record(event)  # the result is sent only once its row is written
            if failure is not None:
                return failure
            return JSONResponse({"request_id": request_id, **result, "version": service.version})
        except Exception as e:  # outside the profile and tamper blocks: a JSON 500, no trace
            return service.failed(request_id, e, declared)
        finally:
            if file is not None:
                await file.close()

    return app
