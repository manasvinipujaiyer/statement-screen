"""Run the service: uv run python -m statementscreen.service

Binds 127.0.0.1:8000 by default (STATEMENTSCREEN_HOST and STATEMENTSCREEN_PORT change that; the
Docker image sets the host to 0.0.0.0 inside the container and publishes the port on the
host's loopback only). One worker, no reload. There is no authentication: local use only.

The audit log (STATEMENTSCREEN_AUDIT_DB, default data/audit/audit.db) is opened first. If it
cannot be opened for writing, is not at schema version 1, or lacks an append-only trigger, the
service does not start.
"""

import os
import sys

import uvicorn

from statementscreen.service.app import create_app
from statementscreen.service.audit import Audit, AuditError


def main() -> None:
    host = os.environ.get("STATEMENTSCREEN_HOST", "127.0.0.1")
    port = int(os.environ.get("STATEMENTSCREEN_PORT", "8000"))
    try:
        audit_log = Audit()
    except AuditError as e:
        sys.exit(f"the service did not start: {e}")
    uvicorn.run(create_app(audit_log=audit_log), host=host, port=port)


if __name__ == "__main__":
    main()
