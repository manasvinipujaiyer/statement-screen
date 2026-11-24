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

Every outcome goes to `audit(event)`, a hook the SQLite audit log fills in (step 6). The event
carries the in-memory result so the log can pick the few fields it keeps; the app itself never
writes a PDF's contents anywhere.
"""

import asyncio
import logging
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from statementscreen.service.screen import screen_pdf
from statementscreen.service.upload import MAX_BYTES, Rejected, store
from statementscreen.service.version import code_version

log = logging.getLogger("statementscreen.service")
FRAMING = 64 * 1024  # room for the multipart boundaries around the file in Content-Length
INTERNAL_ERROR = "INTERNAL_ERROR"


def _no_audit(event: dict) -> None:
    """The default audit hook: records nothing."""


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

    def reject(self, request_id: str, rejected: Rejected) -> JSONResponse:
        self.audit({
            "request_id": request_id, "status": "error", "error_code": rejected.code,
            "size_bytes": rejected.size, "pdf_sha256": rejected.sha256, "pages": rejected.pages,
            "result": None, "version": self.version,
        })  # fmt: skip
        body = {"request_id": request_id, "status": "error", "error_code": rejected.code,
                "message": rejected.message, "version": self.version}  # fmt: skip
        return JSONResponse(body, status_code=rejected.status)

    def failed(self, request_id: str, error: Exception) -> JSONResponse:
        log.error("%s: %s", INTERNAL_ERROR, type(error).__name__)
        self.audit({"request_id": request_id, "status": "error", "error_code": INTERNAL_ERROR,
                    "size_bytes": None, "pdf_sha256": None, "pages": None, "result": None,
                    "version": self.version})  # fmt: skip
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
                    413, "TOO_LARGE", f"the upload is larger than {MAX_BYTES // 2**20} MB",
                    size=int(length),
                )  # fmt: skip
                response = self.service.reject(uuid.uuid4().hex, rejected)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(
    version: dict | None = None,
    audit: Callable[[dict], None] | None = None,
    screen: Callable[[Path], dict] = screen_pdf,
) -> FastAPI:
    service = Service(code_version() if version is None else version, audit, screen)
    app = FastAPI(
        title="StatementScreen", description="Screen one bank-statement PDF. Local use only."
    )
    app.state.service = service
    app.add_middleware(ContentLengthLimit, service=service)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        rejected = Rejected(422, "BAD_REQUEST", "send one PDF as the multipart form field 'file'")
        return service.reject(uuid.uuid4().hex, rejected)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": service.version}

    @app.post("/screen")
    async def screen_endpoint(file: Annotated[UploadFile | None, File()] = None) -> JSONResponse:
        request_id = uuid.uuid4().hex
        try:
            if file is None:
                return service.reject(
                    request_id, Rejected(422, "NO_FILE", "send one PDF as 'file'")
                )
            with tempfile.TemporaryDirectory(prefix="statementscreen-") as folder:
                try:
                    stored = await store(file, Path(folder), request_id)
                except Rejected as rejected:
                    return service.reject(request_id, rejected)
                async with service.lock:  # one PDF at a time
                    result = await run_in_threadpool(service.screen, stored.path)
            service.audit({
                "request_id": request_id, "status": result["status"],
                "error_code": result["error_code"], "size_bytes": stored.size,
                "pdf_sha256": stored.sha256, "pages": stored.pages, "result": result,
                "version": service.version,
            })  # fmt: skip
            return JSONResponse({"request_id": request_id, **result, "version": service.version})
        except Exception as e:  # outside the profile and tamper blocks: a JSON 500, no trace
            return service.failed(request_id, e)
        finally:
            if file is not None:
                await file.close()

    return app
