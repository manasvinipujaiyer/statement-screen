"""Seal the Week 7 blind set: 200 PDFs under opaque names, and an answer key nobody reads yet.

    uv run python -m statementscreen.tamper.blind_seal --backup-dir ~/statementscreen_backup

Reads the two registered source batches (data/blind_src/default_v1 and karan_v1), and writes

    data/blind/pdfs/blind_NNNN.pdf   the 200 PDFs: the detector reads these and nothing else
    data/blind/sealed_key.json       the answers; written last, backed up to --backup-dir

A fresh random seed is drawn from the operating system and stored only inside the key (there is
no --seed option). Each source is kept genuine with probability 0.4, otherwise tampered with one
of the seven types (overlay included), the type drawn uniformly. A genuine PDF is a byte copy of
its source: it is not re-saved, because the re-renders are not written by pypdf, so no re-save
step could hide the difference from every tampered copy.

When a source cannot take its drawn type (NotApplicable, RebuildError or CopyError) the same
source gets another type, drawn from the key's seed among the types not yet tried; a redraw never
turns a tampered source into a genuine one. Every redraw is recorded inside the key. More than
MAX_BUILD_ERRORS (5) RebuildError/CopyError events is a STOP. Nothing is written until every
copy exists, so a STOP or crash leaves no output: re-run the identical command once, and a
second failure means STOP and report. page_swap donors come from the same sub-batch and layout,
never the source itself.

Names are blind_NNNN with NNNN from a seeded permutation, and every PDF (and the pdfs folder) gets
one fixed file time. This prints only: the key's SHA-256, the backup's SHA-256, the number of
PDFs, the PDF-set SHA-256 (over the sorted (name, file SHA-256) pairs), the number of build
errors, and the cost. It never prints the mix, a type, or the number of redraws.
"""

import argparse
import csv
import json
import os
import secrets
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from statementscreen.eval_batches import load_eval
from statementscreen.splits import SPLITS_DIR, SplitError, sha256_file
from statementscreen.summary import peak_memory_bytes
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.blind_common import (
    FIXED_TIME,
    KEY_NAME,
    WEEK6_PREREG,
    BlindError,
    check_detector,
    pdf_set_sha256,
    sha256_bytes,
)
from statementscreen.tamper.copy import CONTROL, TYPES, CopyError, NotApplicable, tamper_copy
from statementscreen.tamper.provenance import MAX_BUILD_ERRORS
from statementscreen.tamper.rebuild import RebuildError

ROOT = Path(__file__).resolve().parents[3]
P_GENUINE = 0.4
TAMPER_TYPES = tuple(t for t in TYPES if t != CONTROL)  # the seven, overlay included
BUILD_ERRORS = (RebuildError, CopyError)
DONOR_TRIES = 5
KEY_VERSION = 1


class SealError(BlindError):
    """The set cannot be sealed."""


@dataclass(frozen=True)
class Source:
    name: str  # the registered evaluation batch, and the sub-batch name inside the key
    folder: Path  # holds pdfs/ and manifest.csv
    main_seed: int
    dialect: str


SOURCES = (
    Source("blind_default_v1", Path("default_v1"), 7101, "default"),
    Source("blind_karan_v1", Path("karan_v1"), 7102, "karan_v1"),
)


def read_layouts(folder: Path) -> dict[str, str]:
    with (folder / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return {r["statement_id"]: r["layout"] for r in csv.DictReader(f)}


def _tamper(seed, index, source_pdf, sub, sid, layouts, folders, redraws, build_errors):
    """One tampered copy of one source: the drawn type first, then the others in drawn order."""
    rng = np.random.default_rng([seed, 2, index])
    order = [TAMPER_TYPES[int(j)] for j in rng.permutation(len(TAMPER_TYPES))]
    layout = layouts[sub][sid]
    for tamper_type in order:
        copy_seed = int(rng.integers(0, 2**31 - 1))
        donors = [None]
        if tamper_type == "page_swap":
            same = [s for s in sorted(layouts[sub]) if layouts[sub][s] == layout and s != sid]
            donors = [same[int(j)] for j in rng.permutation(len(same))[:DONOR_TRIES]]
        failure = ("not applicable", "no donor of the same layout")
        for donor in donors:
            try:
                donor_pdf = folders[sub] / "pdfs" / f"{donor}.pdf" if donor else None
                data, tamper_key = tamper_copy(
                    source_pdf, tamper_type, copy_seed, donor_pdf, allow_held_out=True
                )
                return data, tamper_type, copy_seed, donor, tamper_key
            except NotApplicable as e:
                failure = ("not applicable", str(e))
            except BUILD_ERRORS as e:  # this source cannot be built as this type: redraw
                failure = ("build error", f"{type(e).__name__}: {e}")
                break
        redraws.append({"sub_batch": sub, "source": sid, "type": tamper_type,
                        "reason": failure[0], "detail": failure[1]})  # fmt: skip
        if failure[0] == "build error":
            build_errors.append(redraws[-1])
            if len(build_errors) > MAX_BUILD_ERRORS:
                raise SealError(
                    f"more than {MAX_BUILD_ERRORS} sources failed to build: STOP. Nothing was "
                    "written; re-run the identical command once, and a second failure means "
                    "STOP and report"
                )
    raise SealError(
        f"no tamper type could be applied to {sub}/{sid}: STOP. A tampered source is never "
        "switched to genuine. Nothing was written"
    )


def seal(
    sources: Sequence[Source],
    out: Path,
    backup_dir: Path,
    seed: int | None = None,
    p_genuine: float = P_GENUINE,
) -> dict:
    """Make the 200 PDFs and the key. Returns the summary that main() prints: no mix in it."""
    out, backup_dir = Path(out), Path(backup_dir)
    if out.is_dir() and any(out.iterdir()):
        raise SealError(
            f"{out} is not empty, and a blind set is never overwritten. If an earlier run "
            "failed before writing the key, nothing was scored: delete that folder by hand and "
            "re-run the identical command once; a second failure means STOP and report"
        )
    if (backup_dir / KEY_NAME).exists():
        raise SealError(f"{backup_dir / KEY_NAME} exists; a backup is never overwritten")
    started = time.perf_counter()
    seed = secrets.randbits(63) if seed is None else seed  # from the operating system
    folders = {s.name: Path(s.folder) for s in sources}
    layouts = {s.name: read_layouts(Path(s.folder)) for s in sources}
    items = [(s.name, sid) for s in sources for sid in sorted(layouts[s.name])]
    n = len(items)
    decide = np.random.default_rng([seed, 0]).random(n)
    permutation = np.random.default_rng([seed, 1]).permutation(n)

    made: dict[str, bytes] = {}
    entries: dict[str, dict] = {}
    redraws: list[dict] = []
    build_errors: list[dict] = []
    for i, (sub, sid) in enumerate(items):
        name = f"blind_{int(permutation[i]) + 1:04d}"
        source_pdf = folders[sub] / "pdfs" / f"{sid}.pdf"
        source_bytes = source_pdf.read_bytes()
        entry = {"sub_batch": sub, "source": sid, "layout": layouts[sub][sid],
                 "source_sha256": sha256_bytes(source_bytes)}  # fmt: skip
        if decide[i] < p_genuine:
            data = source_bytes
            entry |= {"truth": "genuine", "copy_seed": None, "donor": None, "tamper_key": None}
        else:
            data, kind, copy_seed, donor, tamper_key = _tamper(
                seed, i, source_pdf, sub, sid, layouts, folders, redraws, build_errors
            )
            entry |= {"truth": kind, "copy_seed": copy_seed, "donor": donor,
                      "tamper_key": tamper_key}  # fmt: skip
        entry["pdf_sha256"] = sha256_bytes(data)
        made[name], entries[name] = data, entry

    pdfs = out / "pdfs"
    pdfs.mkdir(parents=True)
    for name in sorted(made):
        (pdfs / f"{name}.pdf").write_bytes(made[name])
    for path in (*pdfs.iterdir(), pdfs):  # one fixed file time for the files and their folder
        os.utime(path, (FIXED_TIME, FIXED_TIME))
    set_hash = pdf_set_sha256({name: entry["pdf_sha256"] for name, entry in entries.items()})
    key = {
        "version": KEY_VERSION,
        "seed": seed,
        "p_genuine": p_genuine,
        "types": list(TAMPER_TYPES),
        "sources": {
            s.name: {
                "folder": Path(s.folder).name,
                "main_seed": s.main_seed,
                "dialect": s.dialect,
                "statements": len(layouts[s.name]),
                "manifest_sha256": sha256_file(Path(s.folder) / "manifest.csv"),
            }
            for s in sources
        },  # fmt: skip
        "pdfs": entries,
        "redraws": redraws,
        "build_errors": build_errors,
        "pdf_set_sha256": set_hash,
    }
    text = json.dumps(key, indent=2, sort_keys=True) + "\n"
    temp = out / ".sealed_key.tmp"
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, out / KEY_NAME)  # the key is the last file written
    key_sha = sha256_bytes(text.encode())
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / KEY_NAME, backup_dir / KEY_NAME)
    backup_sha = sha256_file(backup_dir / KEY_NAME)
    if backup_sha != key_sha:
        raise SealError("the backup does not match the key: copy it again by hand and compare")
    return {
        "key_sha256": key_sha,
        "backup_sha256": backup_sha,
        "pdfs": len(made),
        "pdf_set_sha256": set_hash,
        "build_errors": len(build_errors),
        "seconds": round(time.perf_counter() - started, 1),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.blind_seal",
        description="Seal the Week 7 blind set. Prints the key's hash, never its contents.",
    )
    p.add_argument("--blind-src", type=Path, default=Path("data/blind_src"))
    p.add_argument("--out", type=Path, default=Path("data/blind"))
    p.add_argument("--backup-dir", type=Path, required=True,
                   help="a folder OUTSIDE the repo: the key is copied there")  # fmt: skip
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument(
        "--detector-reference",
        type=Path,
        default=ROOT / WEEK6_PREREG,
        help="the Week 6 pre-registration whose detector fingerprint must match",
    )
    args = p.parse_args(argv)
    sources = [replace(s, folder=args.blind_src / s.folder) for s in SOURCES]
    try:
        check_detector(args.detector_reference)
        for s in sources:  # the sources must be the registered ones, byte for byte
            load_eval(s.name, s.folder / "manifest.csv", args.splits_dir)
        backup = args.backup_dir.expanduser().resolve()
        if backup == ROOT or ROOT in backup.parents:
            raise SealError(f"--backup-dir {backup} is inside the repository: use another folder")
        summary = seal(sources, args.out, backup)
    except (BlindError, SplitError, fp.StaleFindings, OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"Sealed {summary['pdfs']} PDFs in {args.out / 'pdfs'}")
    print(f"  key SHA-256:     {summary['key_sha256']}  ({args.out / KEY_NAME})")
    print(f"  backup SHA-256:  {summary['backup_sha256']}  ({backup / KEY_NAME})")
    print(f"  PDF-set SHA-256: {summary['pdf_set_sha256']}")
    print(f"  build errors: {summary['build_errors']}; {summary['seconds']} s, "
          f"peak memory {summary['peak_memory_mb']} MB")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
