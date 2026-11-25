"""Time the service on N synthetic PDFs: end-to-end latency as a client sees it.

    uv run python scripts/service_latency.py --url http://127.0.0.1:8000 --label local

The PDFs are generated from fixed seeds (the same set every run; every 4th is a tampered copy),
written to a temp folder that is deleted afterwards, and posted one at a time. Prints the
status codes, the p50 and p95 of the client-side latency and of the server's own
processing_seconds, and a hash of the PDF set so two runs can be shown to have used the same
PDFs. p95 is the nearest-rank value: with 20 requests, the 19th-fastest. Synthetic data only.
"""

import argparse
import hashlib
import json
import math
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import date
from pathlib import Path

from statementscreen.synth.generator import generate_statement
from statementscreen.synth.render import render_statement
from statementscreen.tamper.copy import tamper_copy

CUSTOMER_TYPES = ("salaried", "gig", "self_employed")


def make_pdfs(folder: Path, n: int) -> list[bytes]:
    pdfs = []
    for i in range(n):
        layout = "columns" if i % 2 == 0 else "drcr"
        s = generate_statement(
            seed=1000 + i,
            customer_type=CUSTOMER_TYPES[i % 3],
            months=6,
            start=date(2025, 1, 1),
        )
        path = render_statement(s, folder / f"s{i:02d}.pdf", layout)
        if i % 4 == 0:  # these are on the columns layout
            data, _ = tamper_copy(path, "rerender_inconsistent", seed=3)
            pdfs.append(data)
        else:
            pdfs.append(path.read_bytes())
    return pdfs


def post_pdf(url: str, data: bytes) -> tuple[int, dict, float]:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="statement.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode()
    body = head + data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{url}/screen",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            status, text = response.status, response.read()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read()
    try:
        body = json.loads(text)
    except ValueError:  # a non-JSON answer (an error page, say): counted by status, not parsed
        body = {}
    return status, body, time.perf_counter() - started


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the value at rank ceil(q * n) of the sorted list."""
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)), 1) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be at least 1")

    with tempfile.TemporaryDirectory(prefix="latency-") as folder:
        pdfs = make_pdfs(Path(folder), args.n)
    set_hash = hashlib.sha256(b"".join(hashlib.sha256(p).digest() for p in pdfs)).hexdigest()

    with urllib.request.urlopen(f"{args.url}/health", timeout=30) as response:
        health = json.loads(response.read())
    seconds, server, codes, verdicts = [], [], Counter(), Counter()
    for data in pdfs:
        status, body, took = post_pdf(args.url, data)
        codes[status] += 1
        seconds.append(took)
        if status == 200:
            server.append(body["processing_seconds"])
            verdicts[body["tamper"]["verdict"]] += 1

    print(f"label            {args.label}")
    print(f"url              {args.url}")
    print(f"git_commit       {health['version']['git_commit']}")
    print(f"fingerprint      {json.dumps(health['version']['detector_fingerprint'])}")
    print(f"pdf_set_sha256   {set_hash}")
    print(f"requests         {len(pdfs)}   status codes {dict(codes)}   verdicts {dict(verdicts)}")
    print(f"first request    {seconds[0]:.3f} s (includes any warm-up)")
    c50, c95 = percentile(seconds, 0.50), percentile(seconds, 0.95)
    print(f"client  p50      {c50:.3f} s   p95 {c95:.3f} s")
    print(f"client  min/max  {min(seconds):.3f} s / {max(seconds):.3f} s")
    if server:
        p50, p95 = percentile(server, 0.50), percentile(server, 0.95)
        print(f"server  p50      {p50:.3f} s   p95 {p95:.3f} s   (processing_seconds)")


if __name__ == "__main__":
    main()
