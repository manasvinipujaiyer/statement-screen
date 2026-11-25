# StatementScreen service: local use only, no authentication. See "Service and blind test" in
# README.md. Build (the commit comes in as a build argument, the image has no .git):
#   docker build -t statementscreen:w7 \
#     --build-arg UV_VERSION="$(uv --version | cut -d' ' -f2)" \
#     --build-arg GIT_COMMIT="<commit>[+dirty]" .
ARG UV_VERSION
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# The detector finds uv.lock and the model by their place relative to the source tree, so the
# image keeps the repository's layout and installs the project editable: /app/src, /app/models,
# /app/uv.lock. Then the fingerprint inside the container is computed from the same files.
WORKDIR /app

# Dependencies first (cached until pyproject.toml or uv.lock changes): exactly uv.lock, no dev.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
COPY models/week6_iforest.pkl ./models/week6_iforest.pkl
RUN uv sync --frozen --no-dev

# A non-root user, and a folder for the audit database. Mount a volume on /audit to keep it.
RUN groupadd --system app \
    && useradd --system --gid app --no-create-home --home-dir /nonexistent app \
    && mkdir /audit \
    && chown app:app /audit

ARG GIT_COMMIT=unknown
ENV STATEMENTSCREEN_GIT_COMMIT=${GIT_COMMIT} \
    STATEMENTSCREEN_HOST=0.0.0.0 \
    STATEMENTSCREEN_PORT=8000 \
    STATEMENTSCREEN_AUDIT_DB=/audit/audit.db

USER app
EXPOSE 8000

# /health answers 503 when the audit log is not sound, which urlopen raises on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["python", "-m", "statementscreen.service"]
