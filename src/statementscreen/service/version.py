"""Which code is running: the git commit and the detector fingerprint, computed once at startup.

The commit comes from STATEMENTSCREEN_GIT_COMMIT when it is set (a Docker image has no .git, so
the build passes it in). Otherwise it is `git rev-parse HEAD`, with "+dirty" appended when the
working tree has uncommitted changes (untracked files that are not git-ignored count), and
"unknown" when git cannot say. The detector fingerprint is tamper/fingerprint.py's: the check
code, the config, the IsolationForest model and uv.lock, each as a SHA-256.
"""

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from statementscreen.profile.build import PROFILE_VERSION
from statementscreen.tamper import fingerprint as fp
from statementscreen.tamper.detect.verdict import RULE_VERSION

ROOT = Path(__file__).resolve().parents[3]
COMMIT_ENV = "STATEMENTSCREEN_GIT_COMMIT"


def _git(*args: str, root: Path) -> str | None:
    """git's output, or None if git is missing, fails or times out."""
    try:
        done = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=root, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_commit(root: Path = ROOT, env: Mapping[str, str] | None = None) -> str:
    given = (os.environ if env is None else env).get(COMMIT_ENV, "").strip()
    if given:
        return given
    head = _git("rev-parse", "HEAD", root=root)
    if not head:
        return "unknown"
    status = _git("status", "--porcelain", root=root)
    if status is None:
        return f"{head}+status-unknown"
    return f"{head}+dirty" if status else head


def code_version(
    root: Path = ROOT, env: Mapping[str, str] | None = None, detector: dict | None = None
) -> dict:
    return {
        "git_commit": git_commit(root, env),
        "detector_fingerprint": fp.fingerprint() if detector is None else detector,
        "rule_version": RULE_VERSION,
        "profile_version": PROFILE_VERSION,
    }
