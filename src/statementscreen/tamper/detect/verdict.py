"""Combine findings into one verdict, with the reasons in order of strength.

Verdicts: "no evidence of tampering", "suspicious", "tampered".

Rule version v1 (the starting rules; every version tried is logged in
reports/week6_rule_versions.json):
    any strong finding                        -> tampered
    else two or more DIFFERENT weak codes     -> suspicious
    else PARSE_FAILED (the PDF couldn't be    -> suspicious (it couldn't be checked)
    parsed at all)
    else                                      -> no evidence of tampering

Info findings (a check that couldn't run, producer differs from creator) are listed last
and never counted. Weak findings count by distinct code, not by instance: one cause can
fire on many cells, and "two weak findings" must not depend on how many cells share it.

The verdict is a pure function of the findings: the same findings in any order give the
same verdict, byte for byte. It never looks at a file name, a producer, a creator or a
date. Rule versions are frozen before the one-time scoring (step 6).
"""

import json

from statementscreen.tamper.detect.findings import GROUPS, Finding

RULE_VERSION = "v1"
NONE, SUSPICIOUS, TAMPERED = "no evidence of tampering", "suspicious", "tampered"
LABELS = (NONE, SUSPICIOUS, TAMPERED)
_STRENGTH = {"strong": 0, "weak": 1, "info": 2}
_GROUP = {g: i for i, g in enumerate(GROUPS)}


def order(findings: list[Finding]) -> list[Finding]:
    """Strongest first; then check group, page and row; then code and reason."""
    return sorted(
        findings,
        key=lambda f: (
            _STRENGTH[f.strength],
            _GROUP[f.group],
            f.page or 0,
            -1 if f.row is None else f.row,
            f.code,
            f.reason,
        ),
    )


def rules_v1(findings: list[Finding]) -> tuple[str, str]:
    """(label, why) for the findings, by the v1 rules above."""
    strong = sorted({f.code for f in findings if f.strength == "strong"})
    weak = sorted({f.code for f in findings if f.strength == "weak"})
    if strong:
        return TAMPERED, f"strong evidence: {', '.join(strong)}"
    if len(weak) >= 2:
        return SUSPICIOUS, f"{len(weak)} different weak signs: {', '.join(weak)}"
    if "PARSE_FAILED" in weak:
        return SUSPICIOUS, "the PDF couldn't be parsed, so it couldn't be checked"
    if weak:
        return NONE, f"only one weak sign ({weak[0]}), which is not enough on its own"
    return NONE, "no strong or weak finding"


RULES = {"v1": rules_v1}


def reasons(ordered: list[Finding]) -> list[dict]:
    """One entry per (strength, code), in order of strength. A code that fired many times
    shows its first reason and how many more there were."""
    grouped: dict[tuple[str, str], list[Finding]] = {}
    for f in ordered:
        grouped.setdefault((f.strength, f.code), []).append(f)
    out = []
    for (strength, code), fs in grouped.items():
        text = fs[0].reason
        if len(fs) > 1:
            text += f" (and {len(fs) - 1} more like it)"
        if strength == "info":
            text += " [not counted]"
        out.append({"strength": strength, "code": code, "count": len(fs), "reason": text})
    return out


def decide(findings: list[Finding], version: str = RULE_VERSION) -> dict:
    """The verdict for one PDF's findings: label, rule version, summary, reasons (ordered
    by strength) and every finding. Contains no timings, so it is byte-identical."""
    ordered = order(findings)
    label, why = RULES[version](ordered)
    return {
        "label": label,
        "rule_version": version,
        "summary": why,
        "reasons": reasons(ordered),
        "findings": [f.to_dict() for f in ordered],
    }


def to_json(verdict: dict) -> str:
    """One line of JSON, keys sorted: byte-identical for the same verdict."""
    return json.dumps(verdict, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
