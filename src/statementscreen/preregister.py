"""Pre-register the one-time test-split and shift-batch scoring, BEFORE anything is opened.

    uv run python -m statementscreen.preregister

Writes reports/week5_preregistration.json (refuses to overwrite it, and refuses to run
with uncommitted changes, so the recorded commit really is the code that will run):

    frozen_commit   git rev-parse HEAD
    hashes          SHA-256 of every source file under src/statementscreen (rules, profile,
                    features, models, scoring), of the Week 3 PatternConfig and the
                    ProfileConfig, and of the frozen artefacts: the four models and
                    models.json, the TF-IDF vectoriser, split_v1 and the shift batch's
                    registration
    metrics         exactly what will be computed on the test split and the shift batch
    commands        exactly what will be run, once each

score_once recomputes every hash and refuses to score if any differs.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.patterns.config import DEFAULT as PATTERNS_DEFAULT
from statementscreen.profile.config import DEFAULT as PROFILE_DEFAULT

DEFAULT_OUT = Path("reports/week5_preregistration.json")
CODE_DIR = Path("src/statementscreen")
ARTEFACTS = (
    "data/models/is_salary.txt", "data/models/is_emi.txt", "data/models/is_bounce.txt",
    "data/models/is_rent.txt", "data/models/models.json", "data/features/tfidf.joblib",
    "data/features/tfidf_run.json", "splits/split_v1.csv", "splits/split_v1.json",
    "splits/eval_shift_karan_v1.json",
)  # fmt: skip
SETS = {
    "test": {"kind": "split_v1 test split of data/week1_n2000 (300 statements), never read "
                     "before", "batch": "data/week1_n2000", "parsed": "data/parsed_2000"},
    "shift_karan_v1": {"kind": "registered evaluation batch data/week4_shift_n300 (300 "
                               "statements); its label metrics repeat Week 4's frozen models, "
                               "its profiles are new", "batch": "data/week4_shift_n300",
                       "parsed": "data/parsed_shift_n300"},
}  # fmt: skip
METRICS = {
    "labels": {
        "labels": ["is_salary", "is_emi", "is_bounce", "is_rent"],
        "methods": ["rules (Week 3)", "model (Week 4, frozen)"],
        "measures": ["precision", "recall", "F1 (2tp/(2tp+fp+fn))", "F1 95% interval",
                     "model minus rules F1 and its 95% interval", "model PR-AUC"],
        "statement_level": ["has_salary precision and recall, rules and model",
                            "false salary flags on gig and self-employed statements"],
    },
    "profile": [
        "income median APE, share within 5% and within 10%: over statements with an "
        "estimate, and with a missing estimate counted as a 100% error",
        "share of statements without an income estimate",
        "has_salary precision and recall",
        "employer exact and fuzzy match rate",
        "customer-type accuracy and confusion table",
        "EMI count exact rate, total monthly EMI exact rate and mean absolute error",
        "bounce count exact rate",
        "breakdowns by layout, customer type and hard-case tag",
    ],
    "intervals": {"method": "bootstrap over statements, paired across methods",
                  "resamples": 1000, "seed": 42, "level": 0.95,
                  "income_interval_basis": "missing estimate counted as a 100% error"},
    "comparison": "test next to dev for every metric; dev overstated a metric when the dev "
                  "value is above the upper end of the test 95% interval",
}  # fmt: skip
TEST_ENV = "STATEMENTSCREEN_OPEN_TEST=1"
COMMANDS = [
    f"{TEST_ENV} uv run python -m statementscreen.patterns data/parsed_2000/ --split test "
    "--out data/patterns/",
    f"{TEST_ENV} uv run python -m statementscreen.features data/patterns/ --sets test "
    "--out data/features/ --use-saved-tfidf",
    f"{TEST_ENV} uv run python -m statementscreen.profile data/patterns/ --set test "
    "--parsed data/parsed_2000/",
    "uv run python -m statementscreen.profile data/patterns/ --set shift_karan_v1 "
    "--parsed data/parsed_shift_n300/",
    f"{TEST_ENV} uv run python -m statementscreen.score_once",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def frozen_hashes(root: Path = Path(".")) -> dict:
    """Everything that must not change between the pre-registration and the scoring."""
    code = {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted((root / CODE_DIR).rglob("*.py"))}  # fmt: skip
    return {
        "code": code,
        "configs": {"PatternConfig": _config_sha(PATTERNS_DEFAULT.as_dict()),
                    "ProfileConfig": PROFILE_DEFAULT.sha256()},
        "artefacts": {a: sha256(root / a) for a in ARTEFACTS},
    }  # fmt: skip


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def build(root: Path = Path("."), head: str | None = None, dirty: str | None = None) -> dict:
    head = git("rev-parse", "HEAD") if head is None else head
    dirty = git("status", "--porcelain") if dirty is None else dirty
    if dirty:
        raise ValueError("uncommitted changes; commit everything first, so the recorded "
                         f"commit is the code that will run:\n{dirty}")  # fmt: skip
    return {
        "frozen_commit": head,
        "hashes": frozen_hashes(root),
        "sets": SETS,
        "metrics": METRICS,
        "commands": COMMANDS,
        "rules": ["the test split and the shift batch are each scored once, after this file "
                  "is committed and pushed", "a re-run is reported next to the first result "
                  "and labelled post-hoc", "dev was used for development; test and shift "
                  "were not looked at before scoring"],
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.preregister",
        description="Pre-register the one-time test and shift scoring.",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    if args.out.exists():
        print(f"error: {args.out} exists; a pre-registration is never overwritten",
              file=sys.stderr)  # fmt: skip
        return 2
    try:
        prereg = build()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    h = prereg["hashes"]
    counts = f"{len(h['code'])} source files, {len(h['configs'])} configs, "
    counts += f"{len(h['artefacts'])} artefacts"
    print(f"Wrote {args.out}: commit {prereg['frozen_commit'][:12]}; hashed {counts}")
    print("  commands to run once each, after committing and pushing this file:")
    for c in prereg["commands"]:
        print(f"    {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
