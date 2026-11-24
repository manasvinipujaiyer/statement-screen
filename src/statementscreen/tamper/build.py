"""Build one set of tampered copies, one PDF at a time.

    uv run python -m statementscreen.tamper data/week1_n2000/ --set dev --per-type 60

For each tamper type (overlay only with --include-overlay, see below), PER_TYPE source
statements of the split are drawn in a seeded order that alternates the two layouts, so
the sources are balanced across layouts and none is used twice within a type. When a
source can't take the type (tamper_copy raises NotApplicable, e.g. no salary for
rerender_consistent), the next source in the order is used and the replacement counted.
page_swap donors come from the same split and layout, never the source itself, drawn by
seed (up to DONOR_TRIES donors per source before the source is replaced).

Copies get opaque names: every (type, slot) is given its copy number by a seeded shuffle
before any copy is made, so a name says nothing about its type. Writes, under <out>/<set>/:

    copies/copy_NNNN.pdf   the tampered (or re-saved control) PDFs: the detector reads these
    keys/copy_NNNN.json    one answer key per copy: only the scoring module reads these
    build_run.json         counts and replacements per type, what each type applies to,
                           seeds, seconds and peak memory

A source that raises a build error (RebuildError, or CopyError: a string or row that cannot be
located) is replaced by the next source, like NotApplicable (but its other page_swap donors are
not tried), and its id, donor and reason go into build_run.json ("build_errors"). Any other
exception, RuntimeError included, fails loudly. More than MAX_BUILD_ERRORS (5) of them across
the build is a STOP: the build ends, build_run.json says "stopped" and "complete": false, and
scoring refuses it. Any other error still fails loudly.

Overlay is the held-out type: --include-overlay is refused unless the set is "test" and
STATEMENTSCREEN_OPEN_TEST=1 is set (the one-time Week 6 step 6 command). A test set is
never overwritten.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from statementscreen.generate import OutputFolderError, prepare_output_folder
from statementscreen.splits import OPEN_TEST_ENV, SPLITS_DIR, SplitError, load_split
from statementscreen.summary import peak_memory_bytes
from statementscreen.tamper.copy import (
    APPLIES_TO,
    CONTROL,
    HELD_OUT,
    TYPES,
    CopyError,
    NotApplicable,
    tamper_copy,
)
from statementscreen.tamper.provenance import MAX_BUILD_ERRORS
from statementscreen.tamper.rebuild import RebuildError

BUILD_ERRORS = (RebuildError, CopyError)  # a source that cannot be built: replace it
DEFAULT_SEED = 6
DONOR_TRIES = 5
OUTPUTS = ("copies", "keys", "build_run.json")


def layouts_of(manifest: Path, ids: Sequence[str]) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8") as f:
        wanted = set(ids)
        return {r["statement_id"]: r["layout"] for r in csv.DictReader(f)
                if r["statement_id"] in wanted}  # fmt: skip


def source_order(ids: Sequence[str], layout: dict[str, str], rng) -> list[str]:
    """A seeded order of the statements, alternating layouts (columns, drcr, ...)."""
    by_layout = {name: [ids[i] for i in rng.permutation(len(ids)) if layout[ids[i]] == name]
                 for name in sorted(set(layout.values()))}  # fmt: skip
    out: list[str] = []
    while any(by_layout.values()):
        for name in sorted(by_layout):
            if by_layout[name]:
                out.append(by_layout[name].pop(0))
    return out


def build_set(
    pdfs: Path,
    ids: Sequence[str],
    layout: dict[str, str],
    out: Path,
    per_type: int,
    seed: int = DEFAULT_SEED,
    include_overlay: bool = False,
) -> dict:
    """Make the copies; returns the build summary (also written to build_run.json)."""
    types = [t for t in TYPES if t != HELD_OUT or include_overlay]
    ids = sorted(ids)
    rng = np.random.default_rng([seed, 0])
    slots = [(t, k) for t in types for k in range(per_type)]
    numbers = {slot: n + 1 for n, slot in zip(rng.permutation(len(slots)), slots, strict=True)}
    (out / "copies").mkdir(parents=True)
    (out / "keys").mkdir()

    made: Counter = Counter()
    replaced: Counter = Counter()
    donor_retries: Counter = Counter()
    failures: dict[str, list[str]] = {}
    build_errors: list[dict] = []
    stopped = None
    started = time.perf_counter()
    for t in types:
        type_rng = np.random.default_rng([seed, 1 + TYPES.index(t)])
        order = source_order(ids, layout, type_rng)
        for source in order:
            if made[t] == per_type:
                break
            copy_seed = int(type_rng.integers(0, 2**31 - 1))
            donors = [None]
            if t == "page_swap":
                same = [s for s in ids if layout[s] == layout[source] and s != source]
                donors = [same[i] for i in type_rng.permutation(len(same))[:DONOR_TRIES]]
            result = None
            build_error = None
            for d, donor in enumerate(donors):
                try:
                    donor_pdf = pdfs / f"{donor}.pdf" if donor else None
                    result = tamper_copy(pdfs / f"{source}.pdf", t, copy_seed, donor_pdf,
                                         allow_held_out=include_overlay)  # fmt: skip
                    break
                except NotApplicable as e:
                    failures.setdefault(t, []).append(f"{source}: {e}")
                    if d < len(donors) - 1:
                        donor_retries[t] += 1
                except BUILD_ERRORS as e:  # replaced like NotApplicable, but recorded and capped
                    build_error = f"{type(e).__name__}: {e}"
                    break
            if result is None:
                replaced[t] += 1
                if build_error:
                    build_errors.append(
                        {"type": t, "source": source, "donor": donor, "error": build_error}
                    )
                    print(f"  build error on {source} ({t}): {build_error}", flush=True)
                    if len(build_errors) > MAX_BUILD_ERRORS:
                        stopped = f"more than {MAX_BUILD_ERRORS} sources failed to build: STOP"
                        break
                continue
            data, key = result
            copy_id = f"copy_{numbers[(t, made[t])]:04d}"
            (out / "copies" / f"{copy_id}.pdf").write_bytes(data)
            key |= {"copy_id": copy_id, "source_layout": layout[source]}
            text = json.dumps(key, indent=2, sort_keys=True) + "\n"
            (out / "keys" / f"{copy_id}.json").write_text(text, encoding="utf-8")
            made[t] += 1
        short = per_type - made[t]
        if stopped:
            print(f"  {stopped}", flush=True)
        print(f"  {t}: {made[t]} copies, {replaced[t]} sources replaced"
              + (f" ({short} SHORT: not enough applicable sources)" if short else ""),
              flush=True)  # fmt: skip
        if stopped:
            break
    return {
        "per_type": per_type,
        "types": types,
        "made": dict(made),
        "short": {t: per_type - made[t] for t in types if made[t] < per_type},
        "complete": all(made[t] >= per_type for t in types) and not stopped,
        "build_errors": build_errors,  # every source that failed to build, and why
        "stopped": stopped,
        "sources_replaced": dict(replaced),
        "donor_retries": dict(donor_retries),
        "not_applicable_examples": {t: v[:10] for t, v in failures.items()},
        "applies_to": {t: APPLIES_TO[t] for t in types},
        "control": CONTROL,
        "held_out_included": include_overlay,
        "seed": seed,
        "statements_in_set": len(ids),
        "seconds_total": round(time.perf_counter() - started, 1),
        "peak_memory_mb": round(peak_memory_bytes() / 1024**2, 1),
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper",
        description="Build one set of tampered copies with answer keys.",
    )
    p.add_argument("batch", type=Path, help="the reference batch (pdfs/ and manifest.csv)")
    p.add_argument("--set", required=True, help="dev or test (train gets no copies)")
    p.add_argument("--per-type", type=int, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=Path("data/tamper"))
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--overwrite", action="store_true", help="replace a dev set (never test)")
    held = "make overlay copies too: only the one-time step 6 command (test, env var set)"
    p.add_argument("--include-overlay", action="store_true", help=held)
    args = p.parse_args(argv)
    try:
        if args.set not in ("dev", "test"):
            raise ValueError(f"--set must be dev or test, not {args.set!r}: train gets no "
                             "tampered copies (train statements are for fitting the "
                             "anomaly model and measuring tolerances)")  # fmt: skip
        if args.include_overlay and (args.set != "test" or os.environ.get(OPEN_TEST_ENV) != "1"):
            raise ValueError(f"overlay is held out: --include-overlay needs --set test and "
                             f"{OPEN_TEST_ENV}=1 (the one-time step 6 command)")  # fmt: skip
        if args.set == "test" and args.overwrite:
            raise ValueError("a test set is never overwritten")
        out = args.out / args.set
        if args.set == "test" and out.is_dir() and any(out.iterdir()):
            raise ValueError(
                f"{out} is not empty, and a test set is never overwritten (--overwrite is "
                "refused). If an earlier build failed before writing build_run.json, nothing "
                "was scored: delete that folder by hand and re-run the identical command once; "
                "a second failure means STOP and report"
            )
        ids = load_split(args.set, args.batch / "manifest.csv", args.splits_dir)
        prepare_output_folder(out, args.overwrite, allowed=OUTPUTS)
        layout = layouts_of(args.batch / "manifest.csv", ids)
        summary = build_set(args.batch / "pdfs", ids, layout, out, args.per_type, args.seed,
                            args.include_overlay)  # fmt: skip
    except (ValueError, SplitError, OutputFolderError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    summary["set"] = args.set
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (out / "build_run.json").write_text(text, encoding="utf-8")
    print(f"Wrote {sum(summary['made'].values())} copies to {out} in "
          f"{summary['seconds_total']} s (peak memory {summary['peak_memory_mb']} MB)")  # fmt: skip
    print(f"  sources replaced: {summary['sources_replaced'] or 'none'}; "
          f"short: {summary['short'] or 'none'}; "
          f"build errors: {len(summary['build_errors'])}"
          + (f"; {summary['stopped']}" if summary["stopped"] else ""))  # fmt: skip
    return 0 if not summary["short"] else 1
