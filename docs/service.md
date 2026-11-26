# The service

Back to the [README](../README.md).

`POST /screen` takes one PDF and returns the income profile (Week 5's `profile_pdf`), the
tamper verdict (Week 6's detector), the processing time and the code version (git commit and
detector fingerprint). `GET /health` says the service is up, whether the audit log is sound,
and the versions. `GET /audit/{request_id}` returns the audit row of one request.

> **There is no authentication. This service is for local use only.** It binds `127.0.0.1`
> by default. Anyone who can reach the port can screen PDFs, read audit rows and add rows.

**The detector is unchanged.** The service's detector code, config and model are identical
to Week 6's. What changed in the dependency files is `pyproject.toml` (the dependency lists)
and `uv.lock`, because they gained the service's dependencies: FastAPI (MIT), uvicorn
(BSD-3-Clause) and python-multipart (Apache-2.0), plus httpx and httpcore (both
BSD-3-Clause) in the dev group for the test client. The `uv.lock` diff showed no removed
line, so no package that was already locked changed version. The fingerprint's `uv_lock`
part is the only part that changed. A test checks the
code, config and model parts of the live fingerprint against Week 6's record
(`reports/week6_preregistration.json`); the `uv_lock` part cannot match any more and is
tested with a stand-in. `/health` shows the four parts:

```
code 3dd23628…  config 5b332b4b…  model d1a8aff5…  uv_lock d44f6245…
```

## Run it locally

```bash
uv sync
uv run python -m statementscreen.service          # http://127.0.0.1:8000
curl -F "file=@statement.pdf;type=application/pdf" http://127.0.0.1:8000/screen
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/audit/<request_id>
```

| Environment variable | Default | Meaning |
|---|---|---|
| `STATEMENTSCREEN_HOST` | `127.0.0.1` | Address to bind |
| `STATEMENTSCREEN_PORT` | `8000` | Port |
| `STATEMENTSCREEN_AUDIT_DB` | `data/audit/audit.db` | The SQLite audit log (inside git-ignored `data/`) |
| `STATEMENTSCREEN_GIT_COMMIT` | from `git rev-parse HEAD` | Commit to report. `+dirty` is appended to a commit read from git when the working tree has uncommitted changes. The Docker image has no `.git`, so the commit is passed in as a build argument |

## Run it in Docker (Colima, no Docker Desktop)

```bash
colima start --cpu 2 --memory 2
COMMIT="$(git rev-parse HEAD)$([ -n "$(git status --porcelain)" ] && echo +dirty)"
docker build -t statementscreen:w7 \
  --build-arg UV_VERSION="$(uv --version | cut -d' ' -f2)" \
  --build-arg GIT_COMMIT="$COMMIT" .
docker volume create statementscreen-audit
docker run -d --name statementscreen -p 127.0.0.1:8000:8000 \
  -v statementscreen-audit:/audit statementscreen:w7
curl http://127.0.0.1:8000/health
docker stop statementscreen && docker rm statementscreen && colima stop
```

The image is `python:3.12-slim` with `uv sync --frozen --no-dev` from `uv.lock`, the source,
and `models/week6_iforest.pkl`. It runs as a non-root user (`app`), exposes port 8000, has a
healthcheck on `/health` (503 when the audit log is not sound), and keeps the audit database
in `/audit`, which should be a volume. The `.dockerignore` keeps `data/`, `mlruns/`,
`.review/`, `.claude/`, `.git/` and `reports/` out of the build. No OpenMP runtime is
installed in the image (LightGBM would need one), and the container still screened PDFs, so
the service path does not need LightGBM's runtime. The image keeps the
repository's folder layout, because the fingerprint finds `uv.lock` and the model by their
place relative to the source. Inside the container the code, config and model hashes were
the same as Week 6's (the `/health` output is in
[`reports/week7_service_measurements.txt`](../reports/week7_service_measurements.txt)). No CI
job builds the image.

## Example

A generated, genuine, synthetic salaried statement, abridged (`…` marks cut text; hashes
are shortened) (from the run's own output, not committed). The commit shows `+dirty` because that image was built from a working tree
with uncommitted changes.

```json
{"request_id": "d884b9b844d64b0980f7b71cde8faebe", "status": "ok", "error_code": null,
 "readable": true,
 "profile": {"statement_id": "d884b9b844d64b0980f7b71cde8faebe", "profile_version": 1,
  "statement_months": 6, "period": ["2025-01-01", "2025-06-30"], "warnings": [],
  "findings": {
   "has_salary": {"value": true, "reason": "monthly NEFT credit in 6 of 6 months, amount always INR 84,944, paid on day 28-31"},
   "estimated_monthly_income_paise": {"value": 8494400, "reason": "INR 84,944 a month: salary INR 84,944 a month (6 credits over 6 months), over 6 months"},
   "customer_type": {"value": "salaried", "reason": "a monthly salary from 'kekeri infote'"},
   "emi_count": {"value": 1, "reason": "…"}, "bounced_emis": {"value": 0, "reason": "no returned EMI or charge"}}},
 "tamper": {"verdict": "no evidence of tampering", "summary": "no strong or weak finding",
  "reasons": [{"strength": "info", "code": "PRODUCER_CREATOR_DIFFER", "count": 1, "reason": "…(listed only, never counted) [not counted]"}],
  "finding_codes": [], "checks_not_run": [], "rule_version": "v2"},
 "processing_seconds": 3.656,
 "version": {"git_commit": "5f77e703f94832113f11f5bfa7c2261c5f053cdf+dirty",
  "detector_fingerprint": {"code": "3dd23628…", "config": "5b332b4b…", "model": "d1a8aff5…", "uv_lock": "d44f6245…"},
  "rule_version": "v2", "profile_version": 1}}
```

The profile's `statement_id` equals the `request_id`: the service names the stored file
after the request and never uses the client's file name. A tampered copy of the same
statement (a re-render with inconsistent arithmetic) came back as `"tampered"` with finding
codes `BALANCE_CHAIN` and `SUMMARY_IDENTITY`, and a synthetic gig-worker statement as
`"no evidence of tampering"`, customer type `gig`, no salary.

## What is checked, and what is refused

The upload must start with `%PDF`, be at most 10 MB and at most 50 pages. It is copied to a
private temp folder (named after the request id), hashed, screened, and the folder is
deleted in every case, including errors. One PDF is copied, checked and screened at a time
(a second request waits its turn; receiving a request's body happens before that wait), and
the screening and the audit write run off the event loop. Every refusal is a JSON body with
the request id, `"status": "error"`, an error code and no stack trace, and every refusal is
recorded in the audit log:

| Case | HTTP | `error_code` |
|---|---|---|
| Not a PDF (including empty) | 400 | `NOT_A_PDF` |
| Over 10 MB | 413 | `TOO_LARGE` |
| Over 50 pages | 422 | `TOO_MANY_PAGES` |
| No `file` field | 422 | `NO_FILE` |
| Body that is not valid multipart | 400 or 422 | `BAD_REQUEST` (or `NO_FILE`) |
| A crash outside the profile and tamper parts | 500 | `INTERNAL_ERROR` |
| The audit row could not be written | 500 | `AUDIT_FAILED` (body is only `request_id` and `error_code`) |

A PDF that parses badly is not an error: the answer is 200 with `"readable": false`, the
profile carries its own "could not read the statement" warning, and the verdict follows
Week 6's rules (a lone parse failure is `suspicious`; "could not check" means no evidence
was found but a check that applies did not run, and is never reported as clean). If the
profile part or the tamper part crashes, the response is still 200 with the part that
worked, `"status": "error"` and `PROFILE_CRASHED` or `TAMPER_CRASHED` (joined with a comma if
both), the same code as in the audit row. Only the exception's class name is logged, never
its message, because a message could contain statement text.

## The audit log

One SQLite row per request, whatever the outcome (stdlib `sqlite3`). It keeps: request id,
UTC timestamp, the PDF's SHA-256, the declared and the actual size in bytes, page count,
verdict, finding codes, has_salary, estimated monthly income (paise), customer type, EMI
count, bounce count, processing time, status (`ok` or `error`) and error code, git commit and
detector fingerprint.

**It never keeps** the PDF, its name, the account holder's name, an account number, an
employer, a counterparty or any narration. There is no column for any of them and no
free-text column at all: verdicts, customer types and error codes come from fixed sets,
finding codes must match a strict pattern, and any value that does not fit is stored as
NULL, not as text. A test builds a statement full of a made-up holder, account number,
employer and narrations and searches the raw bytes of the database file for them. Another
does the same for a generated PDF's holder name, account number, employer and narrations
after requests through the service.

The service only inserts. `BEFORE UPDATE` and `BEFORE DELETE` triggers abort anything else,
and lookups use a read-only connection. The file is mode 0600.

**It fails closed.** If the row cannot be written, the client gets `AUDIT_FAILED` and no
result, so every result a client sees has a row (the `AUDIT_FAILED` response itself has
none, because its write failed). The withheld result is not
written anywhere else, and there are no retries beyond SQLite's 5-second busy timeout.
The service refuses to start unless the database opens for writing, is at schema version 1
and has both triggers (an existing table is never repaired), and `/health` reports 503 if
that stops being true.

## Limits of the service

- **No authentication, local use only.** Anyone who can reach the port can add junk rows.
- **One PDF at a time.** Copying, page counting and screening take turns, and other
  requests queue behind them. Receiving a request's body does not wait, so several uploads
  can be in flight at once.
- **One audit row per `POST /screen` request.** Requests to other paths (an unknown path,
  `GET /screen`) get the framework's own 404 or 405 and no row.
- **Size limit and chunked uploads.** An oversized `Content-Length` is refused before the
  body is read. A request with no `Content-Length` (a chunked upload) cannot be refused that
  way, so its bytes are counted as they arrive and it is refused with a 413 once more than
  10 MB plus 64 KiB of multipart framing have arrived. The exact 10 MB limit on the file
  itself is checked afterwards, while the file is copied. Until then FastAPI and Starlette
  receive and parse the whole body and spool each file part to a temporary file, so up to
  10 MB + 64 KiB per request can be on disk before any check on the file runs. Some clients
  (curl, for one) may show a broken pipe instead of the 413.
- **Page count.** If the page count cannot be read, the 50-page limit cannot be enforced for
  that PDF (the size limit still applies); the PDF is still screened and gets the
  unreadable-PDF answer.
- **The triggers guard against code mistakes, not against a person.** Anyone who can write
  the database file can drop a trigger. Rows are never deleted by the service and there is
  no retention policy.
- **The image is large** because the project's main dependencies (MLflow, LightGBM, Faker,
  ReportLab) are installed with it; trimming them would change `pyproject.toml` and
  `uv.lock`.
- **The tests** show a Starlette deprecation warning that its `TestClient` prefers `httpx2`.
  `httpx2` is not an approved dependency, so it is left alone.
- Everything measured here is on synthetic PDFs.

## Speed and size

20 synthetic PDFs (15 genuine, 5 tampered copies), the same set both times (set hash
`23e66d4f…006a`), posted one at a time with `scripts/service_latency.py`, one run each.
Latency is as the client sees it. Docker ran in a Colima VM with 2 CPUs and 2 GB on an arm64
Mac; the local run was on the same Mac without Docker. All 20 requests returned 200 in both
runs, with 5 tampered and 15 "no evidence of tampering" verdicts.

| | Docker | Local |
|---|---|---|
| p50 | 2.834 s | 2.806 s |
| p95 (nearest rank) | 4.249 s | 4.093 s |
| Fastest / slowest | 2.015 s / 5.390 s | 1.928 s / 5.987 s |
| First request | 2.834 s | 2.657 s |

The two runs differ by about 1% at p50 and 4% at p95, well inside the spread between the
fastest and slowest request, so this does not show either is faster. Memory was not
measured. The image: 646,472,434 bytes by `docker image inspect` (arm64); Docker's listing
shows 646 MB content size and 2.91 GB disk usage. These were measured before the review
fixes (the lock now also covers copying and page counting, audit writes moved to worker
threads, and the request body is counted as it arrives); the screening code and the
detector did not change, and nothing was re-measured. The raw output of these commands is in
[`reports/week7_service_measurements.txt`](../reports/week7_service_measurements.txt).
