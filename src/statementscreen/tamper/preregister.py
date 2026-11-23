"""Pre-register the one-time Week 6 test and shift scoring, BEFORE any test data is opened.

    uv run python -m statementscreen.tamper.preregister

Writes reports/week6_preregistration.json (refuses to overwrite it, and refuses to run with
uncommitted changes, so the recorded commit really is the code that will run):

    frozen_commit         git rev-parse HEAD
    detector_fingerprint  the four-part detector fingerprint (tamper/fingerprint.py)
    rule_version          the verdict rule version, frozen here
    hashes                SHA-256 of every source file under src/statementscreen (parser,
                          detector, verdict rules, scoring) and of the frozen artefacts
    seeds, commands, metrics, expectations, known_limits, stop_rule, rules   in plain text

score_tamper recomputes everything with verify() and refuses to score if anything differs:
a different commit history, a different detector fingerprint (it names the part), a
different hash or a different rule version. There is no override.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.detect.verdict import RULE_VERSION

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path("reports/week6_preregistration.json")
CODE_DIR = Path("src/statementscreen")
ARTEFACTS = (
    "models/week6_iforest.pkl",
    "uv.lock",
    "splits/split_v1.csv",
    "splits/split_v1.json",
    "splits/eval_shift_karan_v1.json",
    "reports/week6_rule_versions.json",
    "reports/week6_rule_comparison_dev.json",
)
COMPARISON = Path("reports/week6_rule_comparison_dev.json")
REGISTERED_POPULATIONS = ("test", "shift")  # scoring these needs this file

TEST_BATCH = "data/week1_n2000"
SHIFT_BATCH = "data/week4_shift_n300"
SEEDS = {
    "source_draw": "numpy default_rng([6, 0]): the seeded order of source statements, "
    "alternating layouts (--seed 6, the same seed as the dev set)",
    "tamper": "numpy default_rng([6, 1 + the type's position in TYPES]) per type draws each "
    "copy's seed; tamper_copy then uses default_rng([copy seed, type position])",
    "bootstrap": "numpy default_rng(2026), 1,000 resamples over source statements "
    "(scoring.BOOTSTRAP_SEED)",
    "anomaly_model": "IsolationForest seed 42, already fitted and frozen "
    "(models/week6_iforest.pkl)",
}
COMMANDS = [
    "STATEMENTSCREEN_OPEN_TEST=1 uv run python -m statementscreen.tamper data/week1_n2000/ "
    "--set test --per-type 100 --seed 6 --include-overlay",
    "uv run python -m statementscreen.tamper.run data/tamper/test/copies "
    "--out data/tamper/test/detect",
    "STATEMENTSCREEN_OPEN_TEST=1 uv run python -m statementscreen.tamper.run "
    "data/week1_n2000/pdfs --split test --manifest data/week1_n2000/manifest.csv "
    "--out data/tamper/genuine_test/detect",
    "uv run python -m statementscreen.tamper.run data/week4_shift_n300/pdfs "
    "--out data/tamper/genuine_shift/detect",
    "uv run python -m statementscreen.tamper.score_tamper --copies data/tamper/test "
    "--genuine test=data/tamper/genuine_test/detect "
    "--genuine shift=data/tamper/genuine_shift/detect "
    "--manifest data/week1_n2000/manifest.csv "
    "--genuine-manifest shift=data/week4_shift_n300/manifest.csv "
    "--preregistration reports/week6_preregistration.json "
    "--out reports/week6_tamper_test.json "
    "--table reports/week6_tamper_test_copies.csv",
    "uv run python -m statementscreen.tamper.log_run "
    "--report reports/week6_tamper_test.json "
    "--csv reports/week6_tamper_test_copies.csv "
    "--preregistration reports/week6_preregistration.json",
]
SETS = {
    "test_copies": "800 tampered copies built from the split_v1 test statements of "
    "data/week1_n2000: 100 per type for the six types (rerender_inconsistent, "
    "rerender_consistent, inplace_edit, inplace_edit_consistent, page_swap, page_remove), "
    "100 resaved_genuine (the re-save control) and 100 overlay (held out)",
    "genuine_test": "the split_v1 test statements of data/week1_n2000, unmodified",
    "genuine_shift": "the registered shift batch data/week4_shift_n300 (its own manifest), "
    "unmodified",
}
METRICS = [
    "detection per tamper type, counted as 'tampered' and as 'tampered or suspicious', each "
    "with a 95% bootstrap interval (1,000 resamples over source statements)",
    "the same two counts with every file-history finding removed",
    "the save-mode breakdown (incremental vs full rewrite) for the types a PDF library "
    "wrote, including resaved_genuine",
    "per-type finding-code counts: on how many copies each strong and each weak code fired",
    "the layout breakdown (columns vs drcr) as count/total and percent, no intervals; gaps "
    "over 15 points flagged, small counts noted",
    "false alarms on genuine test: 'tampered', and 'tampered or suspicious', with and "
    "without anomaly findings",
    "false alarms on genuine shift, and on the resaved_genuine copies",
    "the per-copy CSV reports/week6_tamper_test_copies.csv (copy, type, layout, save mode, "
    "verdict, first check), committed",
    "overlay: reported separately, not pooled with the six types",
]
EXPECTATIONS = [
    "These are what dev showed, written down before test is opened. They are not targets "
    "and nothing is tuned to them.",
    "dev, rule v2: rerender_inconsistent, inplace_edit, page_swap and page_remove 60/60 "
    "flagged each; inplace_edit_consistent 59/60; rerender_consistent 7/60 (11.7%, interval "
    "5-20%), tampered 0/60.",
    "dev false alarms: 'tampered' 0/300 on genuine dev and 0/60 on resaved_genuine; "
    "'tampered or suspicious' 6/300 (2.0%) on genuine dev and 1/60 on resaved_genuine; "
    "0/300 flagged on genuine dev without anomaly findings.",
    "So on test: most types expected near 100% flagged, rerender_consistent expected low, "
    "tampered false alarms expected near zero, and 'suspicious' on genuine expected near "
    "2% (the anomaly threshold's rate).",
    "overlay: no expectation. It is held out and dev has no overlay data.",
    "shift: 'tampered' expected near 0; flagged (tampered or suspicious) rate may exceed 2% "
    "because the anomaly model never saw this distribution; no number pre-stated.",
    "Anything far from this is reported as it is.",
]
KNOWN_LIMITS = [
    "The anomaly check's false-alarm rate on genuine dev and genuine train is in-sample "
    "(its threshold and model came from those statements); only test and shift are "
    "independent.",
    "v2 was adopted on dev with a thin margin on condition (a): rerender_consistent rose "
    "from 0/60 to 7/60, 11.7 points against 10 needed.",
    "The layout tolerances and the font baseline were measured on genuine train, so the "
    "zero strong findings on genuine dev and train are not independent evidence.",
    "The pooled all-six-types rate hides the rerender_consistent gap and is not a headline.",
    "Dev copies are 60 per type; test is 100 per type, so test intervals are narrower.",
]
STOP_RULE = (
    "STOP means: if any genuine PDF has a strong finding, score_tamper prints STOP and exits "
    "with code 3. Stop, report it, and change nothing. Any later re-run is labelled "
    "post-hoc and reported next to the first result."
)
RULES = [
    "test and shift are each scored once, after this file is committed and pushed",
    "nothing in src/statementscreen, the model, uv.lock or the verdict rules changes after "
    "this commit; score_tamper refuses to score if anything differs",
    "no retuning after seeing test results; a bug found is reported as a finding, not "
    "fixed silently",
    "test numbers are reported next to the dev numbers, never blended with them",
    "a re-run is reported next to the first result and labelled post-hoc",
    "If the builder cannot reach 100 copies for any type, it exits and we STOP and report "
    "the shortfall. No change to seed, count or code. Any re-run with a reduced count is "
    "labelled post-hoc.",
    "The final registered step logs the scoring run to the local MLflow store (detector "
    "fingerprint, pre-registration commit hash, all metrics, and the report and per-copy CSV "
    "as artifacts). It runs after scoring, only reads the report, and does not change any "
    "result.",
    "dev was used for development; test and shift were not looked at before scoring",
]


class PreregError(RuntimeError):
    """The pre-registration is missing, or what is being scored differs from it."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT)
    return done.stdout.strip()


def dirty_files() -> str:
    """Uncommitted changes: any change to a tracked file, and any untracked file in the code
    or frozen-artefact folders (so no code that is not in the commit can run)."""
    tracked = git("status", "--porcelain", "--untracked-files=no")
    frozen = git("status", "--porcelain", "--", "src", "models", "splits", "uv.lock")
    return "\n".join(sorted(set(tracked.splitlines()) | set(frozen.splitlines())))


def is_ancestor(commit: str) -> bool:
    done = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], capture_output=True, cwd=ROOT
    )
    return done.returncode == 0


def frozen_hashes(root: Path = ROOT) -> dict:
    """Everything that must not change between the pre-registration and the scoring."""
    code = {
        p.relative_to(root).as_posix(): sha256(p) for p in sorted((root / CODE_DIR).rglob("*.py"))
    }
    return {"code": code, "artefacts": {a: sha256(root / a) for a in ARTEFACTS}}


def dev_adoption(root: Path = ROOT) -> dict:
    """The v1 vs v2 dev comparison that met the adoption rule, copied from its report."""
    result = json.loads((root / COMPARISON).read_text(encoding="utf-8"))
    return {
        "source": COMPARISON.as_posix(),
        "decision": result["decision"],
        "conditions": result["conditions"],
        "adoption_rule": result["adoption_rule"],
    }


def build(
    root: Path = ROOT,
    head: str | None = None,
    dirty: str | None = None,
    detector: dict | None = None,
) -> dict:
    head = git("rev-parse", "HEAD") if head is None else head
    dirty = dirty_files() if dirty is None else dirty
    if dirty:
        raise ValueError(
            "uncommitted changes; commit everything first, so the recorded commit is the "
            f"code that will run:\n{dirty}"
        )
    return {
        "frozen_commit": head,
        "detector_fingerprint": fp.fingerprint() if detector is None else detector,
        "rule_version": RULE_VERSION,
        "hashes": frozen_hashes(root),
        "dev_adoption": dev_adoption(root),
        "sets": SETS,
        "seeds": SEEDS,
        "commands": COMMANDS,
        "metrics": METRICS,
        "expectations": EXPECTATIONS,
        "known_limits": KNOWN_LIMITS,
        "stop_rule": STOP_RULE,
        "rules": RULES,
    }


def needs_preregistration(copies_dir: Path | None, genuine_specs: Sequence[str]) -> bool:
    """True when scoring touches the test or shift populations (the one-time scoring)."""
    names = {spec.partition("=")[0] for spec in genuine_specs}
    return bool(names & set(REGISTERED_POPULATIONS)) or (
        copies_dir is not None and Path(copies_dir).name == "test"
    )


def verify(
    prereg: dict,
    rule_version: str = RULE_VERSION,
    root: Path = ROOT,
    ancestor: bool | None = None,
    detector: dict | None = None,
) -> None:
    """Raise PreregError (or StaleFindings, naming the part) if anything differs."""
    commit = prereg.get("frozen_commit", "")
    if not (is_ancestor(commit) if ancestor is None else ancestor):
        raise PreregError(f"pre-registered commit {commit[:12]} is not in this branch's history")
    fp.check(prereg.get("detector_fingerprint"), detector)  # raises StaleFindings
    if rule_version != prereg.get("rule_version"):
        raise PreregError(
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
        raise PreregError("changed since the pre-registration:\n  " + "\n  ".join(problems))


def load_and_verify(path: Path, rule_version: str = RULE_VERSION) -> dict:
    path = Path(path)
    if not path.is_file():
        raise PreregError(f"no pre-registration at {path}; run tamper.preregister first")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    verify(prereg, rule_version)
    return prereg


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.preregister",
        description="Pre-register the one-time Week 6 test and shift scoring.",
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
    print(f"Wrote {args.out}: commit {prereg['frozen_commit'][:12]}; rule {RULE_VERSION}; "
          f"hashed {len(h['code'])} source files and {len(h['artefacts'])} artefacts")  # fmt: skip
    print("  commands to run once each, after committing and pushing this file:")
    for c in prereg["commands"]:
        print(f"    {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
