"""Pre-register the one-time Week 6 test and shift scoring, BEFORE any test data is opened.

    uv run python -m statementscreen.tamper.preregister

Writes reports/week6_preregistration.json (refuses to overwrite it, and refuses to run with
uncommitted changes, so the recorded commit really is the code that will run):

    frozen_commit         git rev-parse HEAD
    detector_fingerprint  the four-part detector fingerprint (tamper/fingerprint.py)
    rule_version          the verdict rule version, frozen here
    hashes                SHA-256 of every source file under src/statementscreen (parser,
                          detector, verdict rules, scoring) and of the frozen artefacts,
                          including the split files and the two manifests
    expected_inputs       the counts and source folders the scoring requires
    seeds, commands, metrics, expectations, known_limits, stop_rule, rules   in plain text

score_tamper recomputes everything with verify() and refuses to score if anything differs: a
commit that is not in the branch history or not pushed, a different detector fingerprint (it
names the part), a different hash or a different rule version. The file it checks is the
PUSHED version (git show @{u}:<path>), and the working-tree copy must be identical to it, so
an uncommitted edit changes nothing. There is no override.

What cannot be hashed now, because it does not exist until step 6 (the test keys, the build
summary, the detector runs), is covered by the seeds recorded here and by its SHA-256, which
score_tamper writes into the test report's `inputs`.
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
from statementscreen.tamper.provenance import COPIES_PER_TYPE, EXPECTED_COUNTS, EXPECTED_SOURCES

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path("reports/week6_preregistration.json")
FIRST_REPORT = "reports/week6_tamper_test.json"  # written once, never replaced
FIRST_TABLE = "reports/week6_tamper_test_copies.csv"
CODE_DIR = Path("src/statementscreen")
ARTEFACTS = (
    "models/week6_iforest.pkl",
    "uv.lock",
    "pyproject.toml",
    "splits/split_v1.csv",
    "splits/split_v1.json",
    "splits/eval_shift_karan_v1.json",
    "data/week1_n2000/manifest.csv",
    "data/week4_shift_n300/manifest.csv",
    "reports/week6_rule_versions.json",
    "reports/week6_rule_comparison_dev.json",
)
COMPARISON = Path("reports/week6_rule_comparison_dev.json")

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
    "--population shift --out data/tamper/genuine_shift/detect",
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
    "'could not check' (no evidence was found and at least one applicable check did not run): "
    "its own count for every group, tampered copies and genuine alike, never counted as no "
    "evidence; and 'checks that did not run': per check, on how many PDFs it did not run",
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
    "All of v2's rerender_consistent gains come from the anomaly score alone, whose "
    "threshold is in-sample on dev.",
    "Two anomaly features (income_to_debits and salary_to_spending) depend indirectly on the "
    "Week 3 salary rule, through the profile; the anomaly check can never give 'tampered' on "
    "its own.",
    "The layout tolerances and the font baseline were measured on genuine train, so the "
    "zero strong findings on genuine dev and train are not independent evidence.",
    "The in-table character-overlap maximum on genuine PDFs was never recorded; only that no "
    "genuine dev or train PDF produced a strong finding, so the margin to the 0.30 "
    "threshold is not shown. This measurement cannot change a verdict.",
    "There is no test for a header mismatch on the drcr layout (only on columns). A missing "
    "test cannot change a verdict.",
    "Populations are identified by statement id and by the PDF hash recorded by each run, "
    "not checked against the generator; shift PDFs copied to another folder would be caught "
    "only through their overlap with the test ids.",
    "The rebuild check (reports/week6_rebuild_check_dev.json) covered only the 300 dev "
    "statements; it was never run on the test statements, so a test source may fail to "
    "build and be replaced (recorded, at most 5).",
    "Nothing at step 6 can be hashed before it exists: a deleted and rebuilt data/tamper/test "
    "is not detected before scoring, and a build that fails and is re-run once uses the same "
    "seed.",
    "The stand-ins in the test suite (the remote branch, the upstream commit) mean the full "
    "remote flow is exercised only on a throwaway repository, not end to end.",
    "log_run does not prevent a second MLflow entry for the same report; the committed report "
    "is the record of truth.",
    "The generator always prints every summary label, so a missing label here means it could "
    "not be read. Real statements may omit totals by design; that will need a per-layout "
    "list of expected labels.",
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
    "this commit; score_tamper refuses to score if anything differs, and it compares with "
    "the pushed version of this file, not the working-tree copy",
    "no retuning after seeing test results; a bug found is reported as a finding, not "
    "fixed silently",
    "test numbers are reported next to the dev numbers, never blended with them",
    "a re-run is reported next to the first result and labelled post-hoc",
    "If the builder cannot reach 100 copies for any type, it exits and we STOP and report "
    "the shortfall. No change to seed, count or code. Any re-run with a reduced count is "
    "labelled post-hoc.",
    "score_tamper identifies populations by what they contain and what each run records, "
    "not by the name typed on the command line: the recorded population must match; test "
    "must be exactly the test statements and shift exactly the shift batch from its own "
    "folder; no other population may contain a test statement (ids are unique only within a "
    "batch: the shift batch reuses ids of the main batch, so it is identified by its source "
    "folder and its manifest ids); test copies must be built only from test statements, "
    "with seed 6; a --limit run is never scored; 300 genuine test, 300 shift and 800 copies "
    "(100 of each of eight types) are required, each PDF with a recorded SHA-256; an "
    "incomplete build is refused. tamper.run also refuses to read test PDFs without "
    "--split test (or --population test) and STATEMENTSCREEN_OPEN_TEST=1. Statements are "
    "identified by id and PDF hash as recorded by the run, not checked against the "
    "generator.",
    "The test report and the per-copy CSV (required with the copies) are written once and "
    "never overwritten (--overwrite is refused, and a path that git has ever recorded "
    "cannot be written again, so deleting a committed result does not allow a second first "
    "result). Any later run uses --post-hoc, writes its own file with post_hoc in its name, "
    "and is labelled post-hoc.",
    "Inputs created at step 6 (the test keys, the build summary, the detector runs and their "
    "PDFs) cannot be hashed now: their seeds are recorded above, and their SHA-256 hashes, a "
    "combined hash of all PDFs per run, and the upstream commit hash are written into the "
    "test report.",
    "Checks that did not run: 'could not check' means no evidence was found and at least one "
    "applicable check did not run (it crashed, or it could not run: a layout not recognised, "
    "a header field not read, an unparseable PDF). Any strong finding still gives 'tampered' "
    "and a weak verdict stands; otherwise it turns 'no evidence of tampering' into 'could not "
    "check', reported in its own column for tampered copies and genuine PDFs, and the checks "
    "that did not run are counted per check. Every arithmetic 'not checked' note (a missing "
    "closing balance, total, count, period or account number) counts as a check that did not "
    "run; the only exemption is the anomaly check on a statement with no transactions to "
    "measure. A PDF the parser cannot parse keeps the earlier rule (PARSE_FAILED alone is "
    "'suspicious'), and a check that raises leaves only that check unrun, except in the "
    "arithmetic stage, where an exception marks the whole PDF as not checkable.",
    "A source that raises RebuildError or CopyError (a string or row that cannot be located) "
    "is replaced by the next source by seed, like NotApplicable, except that its other "
    "page_swap donors are not tried; each failed source id, donor and reason are recorded in "
    "the build summary and in the report. If more than 5 sources fail across the test build, "
    "the build ends, is marked incomplete, and we STOP and report. Any other error, "
    "RuntimeError included, still fails loudly.",
    "A registered command that fails with no output (a crash, not a result) is not a result: "
    "its partial output is deleted by hand and the identical command is re-run once; a second "
    "failure means STOP and report. Nothing about the seed, the counts or the code changes. "
    "An interrupted registered detector run may be continued with --resume, which re-checks "
    "the detector fingerprint and skips finished chunks; this is not a re-run.",
    "Any code change after this freeze needs a new pre-registration: delete this file, "
    "re-run tamper.preregister and commit and push the new one (both versions stay in git "
    "history and only the pushed one is used). The report discloses it as a post-freeze "
    "change; a scoring is labelled post-hoc only if a first result already exists. "
    "score_tamper refuses to score against code that differs from the pushed file.",
    "The final registered step logs the scoring run to the local MLflow store (detector "
    "fingerprint, pre-registration commit hash, all metrics, and the report and per-copy CSV "
    "as artifacts). It runs after scoring, only reads the report, and does not change any "
    "result.",
    "Test statements were opened in Week 5 for profile and label scoring, and the shift batch "
    "was scored in Weeks 4 and 5. The Week 6 detector has never been run on either, and no "
    "tampered copies of them exist before step 6.",
]


class PreregError(RuntimeError):
    """The pre-registration is missing, or what is being scored differs from it."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str, root: Path = ROOT) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=root)
    return done.stdout.strip()


def dirty_files(root: Path = ROOT) -> str:
    """Uncommitted changes: any change to a tracked file, and any untracked file in the code
    or frozen-artefact folders (so no code that is not in the commit can run). The
    pre-registration itself is excluded: replacing it (delete, regenerate, commit) must not
    block its own regeneration."""
    skip = f":(exclude){DEFAULT_OUT.as_posix()}"
    tracked = git("status", "--porcelain", "--untracked-files=no", "--", ".", skip, root=root)
    frozen = git(
        "status", "--porcelain", "--", "src", "models", "splits", "uv.lock", skip, root=root
    )
    return "\n".join(sorted(set(tracked.splitlines()) | set(frozen.splitlines())))


def is_ancestor(commit: str, ref: str = "HEAD", root: Path = ROOT) -> bool:
    done = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, ref], capture_output=True, cwd=root
    )
    return done.returncode == 0


def upstream_sha(root: Path = ROOT) -> str:
    """The commit the branch's upstream (the pushed branch) points at, as git last knew it."""
    done = subprocess.run(["git", "rev-parse", "@{u}"], capture_output=True, text=True, cwd=root)
    if done.returncode != 0:
        raise PreregError("the branch has no upstream: push the pre-registration first")
    return done.stdout.strip()


def has_git_history(path: Path, root: Path = ROOT) -> bool:
    """True if git has ever recorded this path on any ref: a result that was committed and
    then deleted is still a result that was written."""
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return False  # outside the repository
    return bool(git("log", "--all", "--oneline", "--", rel, root=root))


def pushed_text(path: Path, root: Path = ROOT) -> str:
    """The file as it is on the pushed branch (git's last-known upstream state), exactly."""
    rel = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "@{u}"], capture_output=True, text=True, cwd=root
    )
    if upstream.returncode != 0:
        raise PreregError(
            "the branch has no upstream: commit and push the pre-registration first "
            "(git push --set-upstream origin <branch>)"
        )
    shown = subprocess.run(
        ["git", "show", f"@{{u}}:{rel}"], capture_output=True, text=True, cwd=root
    )
    if shown.returncode != 0:
        raise PreregError(
            f"{rel} is not on the pushed branch {upstream.stdout.strip()}: commit and push it"
        )
    return shown.stdout


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
        "expected_inputs": {
            "counts": EXPECTED_COUNTS,
            "sources": EXPECTED_SOURCES,
            "copies_per_type": COPIES_PER_TYPE,
        },
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


def verify(
    prereg: dict,
    rule_version: str = RULE_VERSION,
    root: Path = ROOT,
    ancestor: bool | None = None,
    detector: dict | None = None,
    pushed: bool | None = None,
) -> None:
    """Raise PreregError (or StaleFindings, naming the part) if anything differs. `ancestor`:
    the frozen commit is in this branch's history; `pushed`: it is on the pushed branch."""
    commit = prereg.get("frozen_commit", "")
    if not (is_ancestor(commit, root=root) if ancestor is None else ancestor):
        raise PreregError(f"pre-registered commit {commit[:12]} is not in this branch's history")
    if not (is_ancestor(commit, "@{u}", root) if pushed is None else pushed):
        raise PreregError(f"pre-registered commit {commit[:12]} has not been pushed")
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


def load_and_verify(path: Path, rule_version: str = RULE_VERSION, root: Path = ROOT) -> dict:
    """Read the pre-registration, check that it is what was pushed, and verify it against the
    current code. Returns the pushed version."""
    path = Path(path)
    if not path.is_file():
        raise PreregError(f"no pre-registration at {path}; run tamper.preregister first")
    local = path.read_text(encoding="utf-8")
    pushed = pushed_text(path, root)
    if pushed != local:
        raise PreregError(
            f"{path} differs from the pushed version: scoring uses what was pushed, so an "
            "uncommitted edit changes nothing"
        )
    prereg = json.loads(pushed)
    verify(prereg, rule_version, root)
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
