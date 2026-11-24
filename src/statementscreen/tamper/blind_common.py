"""What the sealing, verdict and scoring steps of the Week 7 blind run share.

Nothing here opens the sealed key: the verdict step imports this module and must never be able
to read the answers. The key is read only by the unseal and scoring side.

  pdf_hashes(folder)   {opaque name: SHA-256} of every PDF in the blind folder
  pdf_set_sha256(...)  SHA-256 over the sorted (opaque name, file SHA-256) pairs, so it does not
                       depend on the order the files are listed in
  check_detector(...)  the detector fingerprint must equal the one Week 6 pre-registered
"""

import hashlib
import json
import re
from pathlib import Path

from statementscreen.tamper import fingerprint as fp

NAME = re.compile(r"blind_\d{4}")
FIXED_TIME = 1735689600  # 2025-01-01 00:00:00 UTC: every blind PDF gets this file time
WEEK6_PREREG = Path("reports/week6_preregistration.json")
KEY_NAME = "sealed_key.json"


class BlindError(Exception):
    """A blind-run precondition does not hold."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pdf_hashes(folder: Path) -> dict[str, str]:
    """{opaque name: SHA-256} of every file in `folder`; refuses any not named blind_NNNN.pdf."""
    folder = Path(folder)
    if not folder.is_dir():
        raise BlindError(f"{folder} is not a folder")
    out = {}
    for path in sorted(folder.iterdir()):
        if path.suffix != ".pdf" or not NAME.fullmatch(path.stem):
            raise BlindError(f"{path.name} is not an opaque blind PDF name (blind_NNNN.pdf)")
        out[path.stem] = sha256_bytes(path.read_bytes())
    return out


def pdf_set_sha256(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for name, digest in sorted(hashes.items()):
        h.update(f"{name}\t{digest}\n".encode())
    return h.hexdigest()


def check_detector(reference: Path = WEEK6_PREREG, current: dict | None = None) -> dict:
    """Raise unless the current detector fingerprint equals the one in Week 6's pre-registration.
    Returns the recorded fingerprint."""
    reference = Path(reference)
    if not reference.is_file():
        raise BlindError(f"{reference} not found: the Week 6 fingerprint is the reference")
    recorded = json.loads(reference.read_text(encoding="utf-8"))["detector_fingerprint"]
    fp.check(recorded, current)  # raises fp.StaleFindings, naming the part that differs
    return recorded
