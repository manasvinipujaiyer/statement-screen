"""One PDF path in, one screening result out: income profile and tamper verdict.

screen_pdf() never raises. The profile and the tamper checks run in separate try blocks, so one
crashing does not lose the other:

    profile crashed   profile is null, status "error", error_code PROFILE_CRASHED
    tamper crashed    verdict "could not check", status "error", error_code TAMPER_CRASHED
    both              both of the above, error codes joined with a comma

A PDF the parser cannot read is a result, not a crash: "readable" is false, the profile carries
its own "could not read the statement" warning, and the verdict is whatever Week 6's rules give
(a lone PARSE_FAILED finding is "suspicious"). "could not check" is the Week 6 label for "no
evidence was found, but at least one check that applies did not run". Only the exception's class
name is logged on a crash: its message could contain statement text.
"""

import logging
import time
from pathlib import Path

from statementscreen.profile.pdf import profile_pdf
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.pdf import collect_findings
from statementscreen.tamper.detect.verdict import RULE_VERSION
from statementscreen.tamper.scoring import COULD_NOT_CHECK, Item, score_item

log = logging.getLogger("statementscreen.service")
PROFILE_CRASHED = "PROFILE_CRASHED"
TAMPER_CRASHED = "TAMPER_CRASHED"
UNREADABLE = "could not read the statement"  # the start of profile_pdf's own warning


def tamper_result(findings: list[Finding]) -> dict:
    """The verdict for one PDF's findings, with Week 6's "could not check" rule applied."""
    scored = score_item(Item("request", "request", "unknown", "", findings=tuple(findings)),
                        RULE_VERSION)  # fmt: skip
    summary = scored.verdict["summary"]
    if scored.label == COULD_NOT_CHECK:
        summary = ("no evidence of tampering was found, but these checks did not run: "
                   f"{', '.join(scored.unrun_checks)}; so this is not a clean result")  # fmt: skip
    return {
        "verdict": scored.label,
        "summary": summary,
        "reasons": scored.verdict["reasons"],
        "finding_codes": sorted({f.code for f in findings if f.strength != "info"}),
        "checks_not_run": list(scored.unrun_checks),
        "rule_version": RULE_VERSION,
    }


def _readable(profile: dict | None, findings: list[Finding] | None) -> bool | None:
    """False when the parser could not read the PDF; None when neither part could say."""
    if findings is not None:
        return not any(f.code == "PARSE_FAILED" for f in findings)
    if profile is not None:
        return not any(w.startswith(UNREADABLE) for w in profile.get("warnings") or [])
    return None


def screen_pdf(path: Path) -> dict:
    started = time.perf_counter()
    errors: list[str] = []
    try:
        profile = profile_pdf(Path(path))
    except Exception as e:  # a bug on a malformed PDF: say so, keep the verdict
        log.error("%s: %s", PROFILE_CRASHED, type(e).__name__)
        profile = None
        errors.append(PROFILE_CRASHED)
    try:
        findings = collect_findings(Path(path))
        tamper = tamper_result(findings)
    except Exception as e:
        log.error("%s: %s", TAMPER_CRASHED, type(e).__name__)
        findings = None
        tamper = {
            "verdict": COULD_NOT_CHECK,
            "summary": "the tamper checks crashed, so nothing was checked",
            "reasons": [],
            "finding_codes": [],
            "checks_not_run": ["detector"],
            "rule_version": RULE_VERSION,
        }
        errors.append(TAMPER_CRASHED)
    return {
        "status": "error" if errors else "ok",
        "error_code": ",".join(errors) or None,
        "readable": _readable(profile, findings),
        "profile": profile,
        "tamper": tamper,
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
