"""Unseal the Week 7 blind key and score the committed verdicts against it, once.

    uv run python -m statementscreen.tamper.blind_score --key data/blind/sealed_key.json \\
        --verdicts reports/week7_blind_verdicts.csv \\
        --preregistration reports/week7_blind_preregistration.json \\
        --out reports/week7_blind_report.json

The only blind module that reads the key, and it reads it last. Everything below is checked
first, with no override, and a failure leaves the key unread:
  - the pre-registration is the pushed one and the code, artefacts, rule version and detector
    fingerprint are as pre-registered (the fingerprint must also equal Week 6's);
  - data/blind/pdfs is exactly the sealed set (the PDF-set hash);
  - the verdict CSV and its .sha256 are the PUSHED copies, byte for byte, and agree with each
    other, so the verdicts were committed before anything was unsealed;
  - the verdicts cover exactly the sealed PDFs, each row carrying that PDF's SHA-256.
Then the key's SHA-256 must equal the one pre-registered (checked on the raw bytes, before they
are parsed), and the key must agree with the PDFs and the verdict rows.

The report is written once and never replaced; a path git has ever recorded cannot be written
again. A later run needs --post-hoc, writes a file with post_hoc in its name, and only after a
first report exists.

Scored from the verdict file alone (no PDF and no finding is read here). Intervals: 1,000
bootstrap resamples over PDFs, seed 2026, 2.5th to 97.5th percentile; a group that is all hits or
no hits gets a zero-width interval, which is narrower than the real uncertainty.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from statementscreen.tamper import blind_prereg as bpr
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import preregister as pre
from statementscreen.tamper.blind_common import BlindError, pdf_set_sha256, sha256_bytes
from statementscreen.tamper.blind_seal import TAMPER_TYPES
from statementscreen.tamper.blind_verdicts import COLUMNS
from statementscreen.tamper.detect.verdict import SUSPICIOUS, TAMPERED
from statementscreen.tamper.scoring import (
    BOOTSTRAP_SEED,
    COULD_NOT_CHECK,
    FLAGGED,
    RESAMPLES,
    SCORED_LABELS,
    bootstrap_ci,
)

GENUINE = "genuine"
# What Week 6 found on its test split (the same detector and tamper code, another source batch),
# as pre-registered in blind_prereg.EXPECTATIONS. Context only: nothing is tuned to it.
WEEK6_TEST = {
    "tampered": {
        "rerender_inconsistent": "100/100",
        "inplace_edit": "100/100",
        "page_swap": "100/100",
        "page_remove": "100/100",
        "overlay": "100/100",
        "inplace_edit_consistent": "97/100",
        "rerender_consistent": "0/100",
    },
    "flagged": {"rerender_consistent": "14/100"},
    "genuine_test": {"tampered": "0/300", "tampered_or_suspicious": "7/300"},
    "genuine_shift": {"tampered": "0/300", "tampered_or_suspicious": "5/300"},
}


# ---------------------------------------------------------------- the checks before the key


def _pushed(path: Path, root: Path) -> str:
    try:
        return pre.pushed_text(path, root)
    except (pre.PreregError, ValueError) as e:
        raise BlindError(str(e)) from e


def check_out(out: Path, root: Path | None = None) -> bool:
    """Refuse a report path that exists or that git has ever recorded. Returns True for a
    post-hoc report (its name says post_hoc, and a first report must already exist)."""
    out, root = Path(out), Path(bpr.ROOT if root is None else root)
    post_hoc = "post_hoc" in out.name
    if out.exists() or pre.has_git_history(out, root):
        raise BlindError(f"{out} exists or was recorded by git: a report is never overwritten")
    first = root / bpr.REPORT
    if post_hoc and not (first.exists() or pre.has_git_history(first, root)):
        raise BlindError("a post-hoc report needs a first report: there is none")
    if not post_hoc and (first.exists() or pre.has_git_history(first, root)):
        raise BlindError(
            f"a first report already exists ({bpr.REPORT}): a later run needs --post-hoc and a "
            "file with post_hoc in its name"
        )
    return post_hoc


def read_verdicts(verdicts: Path, prereg: dict, hashes: dict[str, str], root: Path) -> tuple:
    """The verdict rows, once the file is shown to be the pushed one and to cover the sealed PDFs.
    Returns (rows, SHA-256 of the CSV)."""
    verdicts = Path(verdicts)
    sha_path = verdicts.with_suffix(".sha256")
    if not verdicts.is_file() or not sha_path.is_file():
        raise BlindError(f"{verdicts} and {sha_path} must both exist")
    data = verdicts.read_bytes()
    for path, local in ((verdicts, data.decode("utf-8")), (sha_path, sha_path.read_text("utf-8"))):
        if _pushed(path, root) != local:
            raise BlindError(
                f"{path} differs from the pushed version (or was never pushed): the verdicts "
                "must be committed and pushed BEFORE the unseal step"
            )
    digest = sha256_bytes(data)
    if sha_path.read_text(encoding="utf-8").split()[0] != digest:
        raise BlindError(f"{verdicts} does not match the SHA-256 in {sha_path}")
    with verdicts.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise BlindError(f"{verdicts} does not have the columns {list(COLUMNS)}")
        rows = list(reader)
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names) or sorted(names) != sorted(hashes):
        raise BlindError("the verdict names are not exactly the sealed PDFs")
    for r in rows:
        if r["pdf_sha256"] != hashes[r["name"]]:
            raise BlindError(f"{r['name']}: the verdict row's PDF hash is not the sealed PDF's")
        if r["verdict"] not in SCORED_LABELS or r["verdict_without_history"] not in SCORED_LABELS:
            raise BlindError(f"{r['name']}: unknown verdict {r['verdict']!r}")
    return rows, digest


def unseal(key_path: Path, prereg: dict, hashes: dict[str, str], rows: list[dict]) -> dict:
    """Read the key, the only time. Its raw bytes must hash to the pre-registered SHA-256 before
    anything is parsed, and the parsed key must agree with the PDFs and the verdict rows."""
    raw = Path(key_path).read_bytes()
    sealed = prereg["sealed"]
    if sha256_bytes(raw) != sealed["key_sha256"]:
        raise BlindError(
            "the key's SHA-256 differs from the pre-registered one: it was changed, or this is "
            "not the key that was sealed. Nothing was scored"
        )
    key = json.loads(raw)
    entries = key["pdfs"]
    if sorted(entries) != sorted(hashes):
        raise BlindError("the key does not cover exactly the sealed PDFs")
    wrong = sorted(n for n, e in entries.items() if e["pdf_sha256"] != hashes[n])
    if wrong:
        raise BlindError(f"the key's PDF hashes differ from the sealed PDFs, e.g. {wrong[:3]}")
    if {n: e["pdf_sha256"] for n, e in entries.items()} != hashes:
        raise BlindError("the key and the PDFs disagree")
    if pdf_set_sha256(hashes) != key["pdf_set_sha256"]:
        raise BlindError("the key's PDF-set hash differs from the sealed set's")
    seeds = {n: s["main_seed"] for n, s in key["sources"].items()}
    if seeds != {n: s["main_seed"] for n, s in sealed["sources"].items()}:
        raise BlindError("the key's source seeds differ from the pre-registered ones")
    known = {GENUINE, *TAMPER_TYPES}
    unknown = sorted({e["truth"] for e in entries.values()} - known)
    if unknown:
        raise BlindError(f"unknown truth values in the key: {unknown}")
    if {r["name"] for r in rows} != set(entries):
        raise BlindError("the key and the verdicts do not cover the same PDFs")
    return key


# ---------------------------------------------------------------- the numbers


def _rate(hits: list[bool], sources: list[str]) -> dict:
    count, total = sum(hits), len(hits)
    return {
        "count": count,
        "total": total,
        "rate": round(count / total, 4) if total else None,
        "text": f"{count}/{total} ({100 * count / total:.1f}%)" if total else "0/0",
        "ci95": bootstrap_ci(sources, hits),
    }


def _labels(items: list[dict], column: str) -> dict[str, int]:
    count = Counter(i[column] for i in items)
    return {label: count.get(label, 0) for label in SCORED_LABELS}


def _group(items: list[dict]) -> dict:
    names = [i["name"] for i in items]

    def rate(column: str, labels: tuple[str, ...]) -> dict:
        return _rate([i[column] in labels for i in items], names)

    unrun = Counter(g for i in items for g in i["unrun"])
    codes = Counter(c for i in items for c in i["codes"])
    return {
        "n": len(items),
        "labels": _labels(items, "verdict"),
        "tampered": rate("verdict", (TAMPERED,)),
        "tampered_or_suspicious": rate("verdict", FLAGGED),
        "without_history": {
            "labels": _labels(items, "verdict_without_history"),
            "tampered": rate("verdict_without_history", (TAMPERED,)),
            "tampered_or_suspicious": rate("verdict_without_history", FLAGGED),
        },
        "could_not_check": rate("verdict", (COULD_NOT_CHECK,)),
        "checks_not_run": dict(sorted(unrun.items())),
        "codes_fired": dict(sorted(codes.items())),
    }


def _precision(items: list[dict]) -> dict:
    """Of the PDFs called 'tampered', the share that were tampered, with a bootstrap over PDFs."""
    called = np.array([i["verdict"] == TAMPERED for i in items])
    hit = called & np.array([i["truth"] != GENUINE for i in items])
    count, total = int(hit.sum()), int(called.sum())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(items), size=(RESAMPLES, len(items)))
    n_called, n_hit = called[draw].sum(axis=1), hit[draw].sum(axis=1)
    ok = n_called > 0
    ci = None
    if ok.any():
        lo, hi = np.percentile(n_hit[ok] / n_called[ok], [2.5, 97.5])
        ci = [round(float(lo), 4), round(float(hi), 4)]
    return {
        "count": count,
        "total": total,
        "rate": round(count / total, 4) if total else None,
        "text": f"{count}/{total} ({100 * count / total:.1f}%)" if total else "0/0",
        "ci95": ci,
        "resamples_with_no_tampered_call": int((~ok).sum()),
    }


def _case(i: dict, outcome: str) -> dict:
    return {
        "name": i["name"],
        "truth": i["truth"],
        "layout": i["layout"],
        "dialect": i["dialect"],
        "outcome": outcome,
        "finding_codes": sorted(i["codes"]),
        "checks_not_run": sorted(i["unrun"]),
        "reasons": i["reasons"],
    }


def _by_layout(items: list[dict]) -> dict:
    out = {}
    for layout in sorted({i["layout"] for i in items}):
        sub = [i for i in items if i["layout"] == layout]
        flagged = sum(i["verdict"] in FLAGGED for i in sub)
        called = sum(i["verdict"] == TAMPERED for i in sub)
        out[layout] = {"n": len(sub), "tampered": called, "tampered_or_suspicious": flagged}
    return out


def _compare(report: dict) -> dict:
    """Blind counts next to what Week 6's test split showed. Context, not targets."""
    det = report["detection"]
    per_type = {}
    for t in TAMPER_TYPES:
        per_type[t] = {
            "blind_tampered": det[t]["tampered"]["text"],
            "blind_tampered_or_suspicious": det[t]["tampered_or_suspicious"]["text"],
            "week6_test_tampered": WEEK6_TEST["tampered"][t],
        }
        if t in WEEK6_TEST["flagged"]:
            per_type[t]["week6_test_flagged"] = WEEK6_TEST["flagged"][t]
    genuine = report["false_alarms"]["overall"]
    return {
        "note": "Week 6 used the same detector and tamper code on another source batch with 100 "
        "copies per type; these are not targets, the counts here are small and the batches "
        "differ, so a difference is not by itself a change in the detector.",
        "per_type": per_type,
        "genuine": {
            "blind_tampered": genuine["tampered"]["text"],
            "blind_tampered_or_suspicious": genuine["tampered_or_suspicious"]["text"],
            "week6_test": WEEK6_TEST["genuine_test"],
            "week6_shift": WEEK6_TEST["genuine_shift"],
        },
    }


def build_report(key: dict, rows: list[dict]) -> dict:
    """Everything in the pre-registered metrics, from the key's truth and the verdict rows."""
    dialect = {n: s["dialect"] for n, s in key["sources"].items()}
    items = [
        {
            "name": r["name"],
            "truth": key["pdfs"][r["name"]]["truth"],
            "layout": key["pdfs"][r["name"]]["layout"],
            "dialect": dialect[key["pdfs"][r["name"]]["sub_batch"]],
            "verdict": r["verdict"],
            "verdict_without_history": r["verdict_without_history"],
            "codes": [c for c in r["finding_codes"].split(";") if c],
            "unrun": [c for c in r["checks_not_run"].split(";") if c],
            "reasons": r["reasons"],
        }
        for r in sorted(rows, key=lambda r: r["name"])
    ]
    genuine = [i for i in items if i["truth"] == GENUINE]
    tampered = [i for i in items if i["truth"] != GENUINE]
    by_type = {t: [i for i in items if i["truth"] == t] for t in TAMPER_TYPES}
    detection = {t: _group(by_type[t]) for t in TAMPER_TYPES}
    overall = _group(genuine) if genuine else None
    by_dialect = {
        d: _group([i for i in genuine if i["dialect"] == d]) for d in sorted(set(dialect.values()))
    }
    anomaly_alone = [
        i for i in genuine if i["verdict"] in FLAGGED and set(i["codes"]) == {"ANOMALY_SCORE"}
    ]
    confusion = {
        truth: _labels([i for i in items if i["truth"] == truth], "verdict")
        for truth in (GENUINE, *TAMPER_TYPES)
    }
    report = {
        "mix": {
            GENUINE: len(genuine),
            **{t: len(by_type[t]) for t in TAMPER_TYPES},
            "tampered_total": len(tampered),
            "total": len(items),
        },  # fmt: skip
        "bootstrap": {
            "resamples": RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "resampled": "PDFs",
            "interval": "2.5th to 97.5th percentile",
        },  # fmt: skip
        "detection": detection,
        "pooled_detection": _group(tampered) if tampered else None,
        "false_alarms": {
            "overall": overall,
            "by_dialect": by_dialect,
            "flagged_by_the_anomaly_check_alone": len(anomaly_alone),
            "cases": [
                _case(i, "tampered" if i["verdict"] == TAMPERED else "suspicious")
                for i in genuine
                if i["verdict"] in FLAGGED
            ],
        },
        "genuine_called_tampered": [
            _case(i, "tampered") for i in genuine if i["verdict"] == TAMPERED
        ],  # fmt: skip
        "precision_of_tampered": _precision(items),
        "recall_of_tampered": _rate(
            [i["verdict"] == TAMPERED for i in tampered], [i["name"] for i in tampered]
        ),  # fmt: skip
        "confusion_verdict_by_truth": confusion,
        "could_not_check": {
            "genuine": _group(genuine)["could_not_check"] if genuine else None,
            "tampered": _group(tampered)["could_not_check"] if tampered else None,
            "cases": [_case(i, i["verdict"]) for i in items if i["unrun"]],
        },
        "by_layout": {GENUINE: _by_layout(genuine), "tampered": _by_layout(tampered)},
        "misses": [
            _case(
                i,
                "not flagged"
                if i["verdict"] not in (SUSPICIOUS, COULD_NOT_CHECK)
                else i["verdict"],
            )  # fmt: skip
            for i in tampered
            if i["verdict"] != TAMPERED
        ],
        "redraws": key["redraws"],
        "build_errors": len(key["build_errors"]),
    }
    if overall and tampered:
        report["comparison_with_week6"] = _compare(report)
    return report


def score(prereg: dict, key: dict, rows: list[dict], digest: str, post_hoc: bool) -> dict:
    sealed = prereg["sealed"]
    return {
        "what": "Week 7 blind tamper run: the frozen Week 6 detector on 200 sealed PDFs",
        "post_hoc": post_hoc,
        "preregistration": {
            "path": bpr.DEFAULT_OUT.as_posix(),
            "frozen_commit": prereg["frozen_commit"],
        },
        "rule_version": prereg["rule_version"],
        "detector_fingerprint": prereg["detector_fingerprint"],
        "inputs": {
            "key_sha256": sealed["key_sha256"],
            "verdicts_sha256": digest,
            "pdf_set_sha256": sealed["pdf_set_sha256"],
            "pdfs": sealed["pdfs"],
        },
        **build_report(key, rows),
        "known_limits": prereg["known_limits"],
    }


def write_report(report: dict, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, out)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.blind_score",
        description="Unseal the blind key and score the committed verdicts, once.",
    )
    p.add_argument("--key", type=Path, default=Path("data/blind/sealed_key.json"))
    p.add_argument("--verdicts", type=Path, default=Path(bpr.VERDICTS))
    p.add_argument("--preregistration", type=Path, default=bpr.DEFAULT_OUT)
    p.add_argument("--pdfs", type=Path, default=Path("data/blind/pdfs"))
    p.add_argument("--out", type=Path, default=Path(bpr.REPORT))
    args = p.parse_args(argv)
    try:
        post_hoc = check_out(args.out)
        prereg = bpr.load_and_verify(args.preregistration)
        hashes = bpr.check_pdf_set(prereg, args.pdfs)
        rows, digest = read_verdicts(args.verdicts, prereg, hashes, bpr.ROOT)
        key = unseal(args.key, prereg, hashes, rows)  # the first and only read of the key
        report = score(prereg, key, rows, digest, post_hoc)
        write_report(report, args.out)
    except (
        BlindError,
        fp.StaleFindings,
        OSError,
        KeyError,
        ValueError,
        subprocess.CalledProcessError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    mix, genuine = report["mix"], report["false_alarms"]["overall"]
    print(f"Wrote {args.out}{' (post-hoc)' if post_hoc else ''}")
    print(f"  mix: {mix[GENUINE]} genuine, {mix['tampered_total']} tampered of {mix['total']}")
    if report["pooled_detection"]:
        print(f"  tampered PDFs called 'tampered': {report['recall_of_tampered']['text']}")
    if genuine:
        print(f"  genuine PDFs called 'tampered': {genuine['tampered']['text']}; "
              f"'tampered or suspicious': {genuine['tampered_or_suspicious']['text']}")  # fmt: skip
    if report["genuine_called_tampered"]:
        print("  STOP-worthy: a genuine PDF was called 'tampered'; report it with its reasons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
