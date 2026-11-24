"""Run the service: uv run python -m statementscreen.service

Binds 127.0.0.1:8000 by default (STATEMENTSCREEN_HOST and STATEMENTSCREEN_PORT change that; the
Docker image sets the host to 0.0.0.0 inside the container and publishes the port on the
host's loopback only). One worker, no reload. There is no authentication: local use only.
"""

import os

import uvicorn

from statementscreen.service.app import create_app


def main() -> None:
    host = os.environ.get("STATEMENTSCREEN_HOST", "127.0.0.1")
    port = int(os.environ.get("STATEMENTSCREEN_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
