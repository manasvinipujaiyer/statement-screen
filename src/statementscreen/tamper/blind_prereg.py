"""Pre-register the Week 7 blind run, after sealing and BEFORE the detector sees the PDFs.

    uv run python -m statementscreen.tamper.blind_prereg \\
        --key-sha256 <from the seal output> --backup-sha256 <from the seal output> \\
        --pdf-set-sha256 <from the seal output>

Writes reports/week7_blind_preregistration.json (never overwritten; refuses uncommitted changes,
so the recorded commit is the code that will run):

    frozen_commit         git rev-parse HEAD
    detector_fingerprint  the detector fingerprint, which must equal the one Week 6 pre-registered
    rule_version          the verdict rule version
    sealed                the key's SHA-256 and its backup's, the PDF-set SHA-256, the source seeds
    hashes                SHA-256 of every source file under src/statementscreen and of the frozen
                          artefacts (model, uv.lock, pyproject.toml, the two registries, Week 6's
                          pre-registration)
    commands, metrics, expectations, known_limits, rules   in plain text

This command never opens the sealed key: the hashes are typed from the seal output (the PDF-set
hash is also recomputed from data/blind/pdfs, which holds no answers).

The blind commands recompute everything with verify() and refuse to run if anything differs: the
commit not in this branch or not pushed, a different detector fingerprint (naming the part), a
different source hash, another rule version, or PDFs that are not the sealed set. They check the
PUSHED copy of this file (git show @{u}), which the working copy must equal. No override.
"""

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper import preregister as pre
from statementscreen.tamper.blind_common import (
    WEEK6_PREREG,
    BlindError,
    check_detector,
    pdf_hashes,
    pdf_set_sha256,
)
from statementscreen.tamper.blind_seal import P_GENUINE, SOURCES, TAMPER_TYPES
from statementscreen.tamper.detect.verdict import RULE_VERSION

ROOT = pre.ROOT
DEFAULT_OUT = Path("reports/week7_blind_preregistration.json")
VERDICTS = "reports/week7_blind_verdicts.csv"
VERDICTS_SHA = "reports/week7_blind_verdicts.sha256"
REPORT = "reports/week7_blind_report.json"
PUBLISHED_KEY = "reports/week7_blind_key.json"
EXPECTED_PDFS = 200
HEX64 = re.compile(r"[0-9a-f]{64}")
ARTEFACTS = (
    "models/week6_iforest.pkl",
    "uv.lock",
    "pyproject.toml",
    "splits/eval_blind_default_v1.json",
    "splits/eval_blind_karan_v1.json",
    "reports/week6_preregistration.json",
)
PREREG = DEFAULT_OUT.as_posix()
COMMANDS = [
    "ALREADY RUN, before this file was written: uv run python -m "
    "statementscreen.tamper.blind_seal --backup-dir ~/statementscreen_backup "
    "(printed the key, backup and PDF-set hashes recorded in 'sealed')",
    "uv run python -m statementscreen.tamper.blind_detect --pdfs data/blind/pdfs "
    f"--out data/blind/detect --preregistration {PREREG}",
    "uv run python -m statementscreen.tamper.blind_verdicts --detect data/blind/detect "
    f"--pdfs data/blind/pdfs --preregistration {PREREG} --out {VERDICTS}",
    "uv run python -m statementscreen.tamper.blind_score --key data/blind/sealed_key.json "
    f"--verdicts {VERDICTS} --preregistration {PREREG} --out {REPORT}",
    f"uv run python -m statementscreen.tamper.blind_log --report {REPORT} "
    f"--preregistration {PREREG}",
    f"AFTER the report is committed: copy data/blind/sealed_key.json to {PUBLISHED_KEY} and "
    f"run uv run python -m statementscreen.tamper.blind_verify --key {PUBLISHED_KEY} "
    f"--verdicts {VERDICTS} --blind-src data/blind_src",
]
METRICS = [
    "the true mix, counted after the fact (genuine, and each of the seven types)",
    "detection per type as counts, as 'tampered' and as 'tampered or suspicious', with 95% "
    "bootstrap intervals over PDFs (1,000 resamples, seed 2026)",
    "the same two counts with every file-history finding removed",
    "false alarms on genuine PDFs, 'tampered' and 'tampered or suspicious', by dialect "
    "(default, karan_v1) and overall",
    "precision of a 'tampered' verdict and recall of tampered PDFs (share called 'tampered'), "
    "with 95% bootstrap intervals over PDFs",
    "a confusion table of verdict by truth, 'could not check' in its own column",
    "'could not check' and the checks that did not run, per check",
    "a comparison with the Week 6 test numbers",
]
EXPECTATIONS = [
    "These come from the Week 6 test (the same detector and tamper code, a different source "
    "batch). They are not targets and nothing is tuned to them.",
    "Week 6 test, rule v2: rerender_inconsistent, inplace_edit, page_swap, page_remove and "
    "overlay 100/100 tampered each; inplace_edit_consistent 97/100; rerender_consistent 0/100 "
    "tampered and 14/100 flagged.",
    "Week 6 test, genuine PDFs: 'tampered' 0/300 on test and 0/300 on shift; 'tampered or "
    "suspicious' 7/300 and 5/300, every flag from the anomaly check alone.",
    "So on the blind set: the edit types near 100% tampered, rerender_consistent mostly "
    "missed, 'tampered' on genuine PDFs near 0 and 'suspicious' on genuine PDFs near 2%.",
    "No other numbers are pre-stated.",
    "Each source is genuine with probability 0.4, so the mix is random and per-type counts "
    "are small (on the order of 17 each, fewer for rerender_consistent, which needs a salary): "
    "their intervals will be wide.",
    "Anything far from this is reported as it is.",
]
KNOWN_LIMITS = [
    "Everything is synthetic, and every tampered PDF comes from this project's own tamper code. "
    "No outside person edited a PDF.",
    "The seal proves the key was not changed after it was sealed, not that nobody could have "
    "read it.",
    "File sizes may differ between genuine and tampered PDFs; the seal hides the mix from the "
    "scoring process, not from someone inspecting the files.",
    "data/blind_src/ stays on disk because the rebuild-and-verify check needs it; once the key "
    "is published, anyone can rebuild everything anyway.",
    "Genuine PDFs are byte copies of their sources and are not re-saved, because the two "
    "re-render types are not written by pypdf, so no re-save step could hide the difference "
    "from every tampered copy. The other tampered types are written by pypdf, so file-structure "
    "cues may separate them from genuine PDFs; the report gives the numbers without file-history "
    "findings. Re-save false alarms are not measured here (Week 6 measured them).",
    "A source that cannot take its drawn type gets another type, so the seven types are not "
    "exactly uniform: rerender_consistent, which needs a salary, is probably the rarest.",
    "Per-type counts are small, so most intervals are wide, and a group that is all hits has a "
    "zero-width bootstrap interval that is narrower than the real uncertainty.",
    "The bootstrap resamples PDFs. A page_swap donor is another blind source, so a donor's own "
    "PDF and a swapped copy are not strictly independent.",
    "A lone anomaly score gives 'suspicious', never 'tampered' (verdict rule v2), and two "
    "anomaly features depend indirectly on the Week 3 salary rule (Week 6 limits).",
    "The anomaly threshold and the layout and font baselines came from Week 6's train and dev "
    "statements. Before sealing, blind_fresh (a read-only audit that reads the earlier batches "
    "under data/ and never the key) found no seed or content shared between the blind sources "
    "and any earlier batch; its output was read by hand and is not committed, so this is "
    "stated here, not checkable from the repository.",
    "A genuine PDF is a byte copy of a file in data/blind_src/, which stays on disk: hashing "
    "a blind PDF against those files identifies the genuine ones before any scoring. Nothing "
    "in the pipeline does this, and the seal does not prevent it.",
    "log_run style limit: blind_log does not prevent a second MLflow entry for the same report; "
    "the committed report is the record of truth.",
    "Week 6's other known limits (reports/week6_preregistration.json) apply.",
]
RULES = [
    "One run only. The sealing command ran once (it never overwrites). The detector, verdict "
    "and scoring commands run once each, in the order listed.",
    "The sealing command writes the key, copies it to the backup folder and reads the copy "
    "back only to compare hashes; it prints only hashes and counts. Nothing else reads, opens "
    "or prints the key before the unseal step, and the detector and the verdict step never "
    "read it.",
    "Sealing: a source that cannot take its drawn type (NotApplicable, RebuildError or "
    "CopyError) is redrawn to another tamper type from the key's seed, never to genuine; "
    "page_swap donors come from the same sub-batch and layout, at most 5 tried. Each "
    "RebuildError or CopyError EVENT is recorded in the key and counted (a source that fails "
    "twice counts twice, so this counts events, not sources); more than 5 events is a STOP "
    "and nothing is written. This is stated for the record: sealing already ran.",
    "The verdict file is committed and pushed BEFORE the unseal step. Scoring refuses a key "
    "whose SHA-256 differs from the one recorded here, a verdict file whose SHA-256 differs from "
    "the committed one, PDFs whose set hash differs, and a detector fingerprint that differs "
    "from Week 6's.",
    "A command that fails with no output: delete its partial output by hand and re-run the "
    "identical command once. A second failure is a STOP.",
    "An interrupted detector run may be continued with --resume, which re-checks the detector "
    "fingerprint and skips finished chunks; this is not a re-run.",
    "A code change after this file is written needs a new pre-registration (delete this file, "
    "re-run the writer, commit and push it) and is disclosed in the report as a post-freeze "
    "change. Any run after a first result exists is labelled post-hoc, goes to a file with "
    "'post_hoc' in its name, and never replaces the first result.",
    "A 'tampered' verdict on a genuine PDF is reported with its reasons, never hidden or re-run.",
]
STOP_RULE = (
    "STOP and report, changing nothing: any check above refuses; a second failure of a command "
    "that failed with no output."
)


class BlindPreregError(BlindError):
    """The blind pre-registration is missing, or what is being run differs from it."""


def code_hashes(root: Path = ROOT) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): pre.sha256(p)
        for p in sorted((root / pre.CODE_DIR).rglob("*.py"))
    }


def frozen_hashes(root: Path = ROOT) -> dict:
    return {"code": code_hashes(root), "artefacts": {a: pre.sha256(root / a) for a in ARTEFACTS}}


def _hex(name: str, value: str) -> str:
    if not HEX64.fullmatch(value):
        raise ValueError(f"{name} must be 64 lowercase hex characters, got {value!r}")
    return value


def build(
    key_sha256: str,
    backup_sha256: str,
    pdf_set: str,
    root: Path = ROOT,
    head: str | None = None,
    dirty: str | None = None,
    detector: dict | None = None,
) -> dict:
    head = pre.git("rev-parse", "HEAD", root=root) if head is None else head
    dirty = pre.dirty_files(root) if dirty is None else dirty
    if dirty:
        raise ValueError(
            "uncommitted changes; commit everything first, so the recorded commit is the "
            f"code that will run:\n{dirty}"
        )
    key_sha256 = _hex("--key-sha256", key_sha256)
    backup_sha256 = _hex("--backup-sha256", backup_sha256)
    pdf_set = _hex("--pdf-set-sha256", pdf_set)
    if key_sha256 != backup_sha256:
        raise ValueError("the key and its backup have different SHA-256: fix the backup first")
    fingerprint = fp.fingerprint() if detector is None else detector
    check_detector(root / WEEK6_PREREG, fingerprint)  # must equal Week 6's, or refuse
    return {
        "frozen_commit": head,
        "detector_fingerprint": fingerprint,
        "rule_version": RULE_VERSION,
        "sealed": {
            "key_sha256": key_sha256,
            "backup_sha256": backup_sha256,
            "pdf_set_sha256": pdf_set,
            "pdfs": EXPECTED_PDFS,
            "p_genuine": P_GENUINE,
            "tamper_types": list(TAMPER_TYPES),
            "sources": {
                s.name: {"folder": s.folder.name, "main_seed": s.main_seed, "dialect": s.dialect}
                for s in SOURCES
            },
        },
        "hashes": frozen_hashes(root),
        "commands": COMMANDS,
        "metrics": METRICS,
        "expectations": EXPECTATIONS,
        "known_limits": KNOWN_LIMITS,
        "stop_rule": STOP_RULE,
        "rules": RULES,
    }


def verify(
    prereg: dict,
    root: Path = ROOT,
    ancestor: bool | None = None,
    pushed: bool | None = None,
    detector: dict | None = None,
    rule_version: str = RULE_VERSION,
) -> None:
    """Raise BlindPreregError (or fp.StaleFindings, naming the part) if anything differs."""
    commit = prereg.get("frozen_commit", "")
    if not (pre.is_ancestor(commit, root=root) if ancestor is None else ancestor):
        raise BlindPreregError(f"pre-registered commit {commit[:12]} is not in this history")
    if not (pre.is_ancestor(commit, "@{u}", root) if pushed is None else pushed):
        raise BlindPreregError(f"pre-registered commit {commit[:12]} has not been pushed")
    fp.check(prereg.get("detector_fingerprint"), detector)  # raises fp.StaleFindings
    check_detector(root / WEEK6_PREREG, detector)  # and equals Week 6's, not just its own
    if rule_version != prereg.get("rule_version"):
        raise BlindPreregError(
            f"rule version {rule_version!r} differs from the pre-registered "
            f"{prereg.get('rule_version')!r}"
        )
    problems = []
    now = frozen_hashes(root)
    for section in ("code", "artefacts"):
        old, new = prereg["hashes"][section], now[section]
        problems += [f"{name} changed" for name in sorted(old) if new.get(name) != old[name]]
        problems += [f"{name} is new" for name in sorted(set(new) - set(old))]
    if problems:
        raise BlindPreregError("changed since the pre-registration:\n  " + "\n  ".join(problems))


def load_and_verify(path: Path, root: Path = ROOT, **checks) -> dict:
    """Read the pre-registration, check it is what was pushed, and verify it against the current
    code. Returns it."""
    path = Path(path)
    if not path.is_file():
        raise BlindPreregError(f"no pre-registration at {path}; run blind_prereg first")
    local = path.read_text(encoding="utf-8")
    try:
        pushed = pre.pushed_text(path, root)
    except pre.PreregError as e:
        raise BlindPreregError(str(e)) from e
    if pushed != local:
        raise BlindPreregError(
            f"{path} differs from the pushed version: the blind commands use what was pushed, "
            "so an uncommitted edit changes nothing"
        )
    prereg = json.loads(pushed)
    verify(prereg, root, **checks)
    return prereg


def check_pdf_set(prereg: dict, folder: Path) -> dict[str, str]:
    """The folder must hold exactly the sealed PDFs: same names and bytes (the set hash)."""
    hashes = pdf_hashes(folder)
    expected = prereg["sealed"]
    if len(hashes) != expected["pdfs"]:
        raise BlindPreregError(
            f"{folder} holds {len(hashes)} PDFs, the seal made {expected['pdfs']}"
        )
    if pdf_set_sha256(hashes) != expected["pdf_set_sha256"]:
        raise BlindPreregError(f"the PDFs in {folder} are not the sealed set (set hash differs)")
    return hashes


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.blind_prereg",
        description="Pre-register the Week 7 blind run. Never opens the sealed key.",
    )
    p.add_argument("--key-sha256", required=True, help="from the seal output")
    p.add_argument("--backup-sha256", required=True, help="from the seal output")
    p.add_argument("--pdf-set-sha256", required=True, help="from the seal output")
    p.add_argument("--pdfs", type=Path, default=Path("data/blind/pdfs"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} exists; never overwritten", file=sys.stderr)
        return 2
    try:
        hashes = pdf_hashes(args.pdfs)
        if pdf_set_sha256(hashes) != args.pdf_set_sha256 or len(hashes) != EXPECTED_PDFS:
            raise BlindError(
                f"the PDFs in {args.pdfs} do not match --pdf-set-sha256 (or are not "
                f"{EXPECTED_PDFS}): is this the sealed set?"
            )
        prereg = build(args.key_sha256, args.backup_sha256, args.pdf_set_sha256)
    except (BlindError, fp.StaleFindings, ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    h = prereg["hashes"]
    print(f"Wrote {args.out}: commit {prereg['frozen_commit'][:12]}; rule {RULE_VERSION}; "
          f"hashed {len(h['code'])} source files and {len(h['artefacts'])} artefacts")  # fmt: skip
    print("  commit and push this file, then run the commands once each, in order:")
    for c in prereg["commands"]:
        print(f"    {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
