"""Compare two verdict-rule versions on the same cached findings; apply the adoption rule.

    uv run python -m statementscreen.tamper.compare_rules --copies data/tamper/dev \\
        --genuine dev=data/tamper/genuine_dev/detect \\
        --manifest data/week1_n2000/manifest.csv \\
        --old v1 --new v2 --out reports/week6_rule_comparison_dev.json

Both versions are scored from the same cached findings (no PDF is read), so the only thing
that differs is the rules. The adoption rule was written into reports/week6_rule_versions.json
before v2 was run, and is applied here exactly as written. v2 is adopted only if ALL hold:

  (a) rerender_consistent detection, counted as tampered or suspicious, on dev rises by at
      least 10 percentage points over v1;
  (b) an anomaly score alone can never produce "tampered";
  (c) the "tampered" false-alarm rate on genuine dev and on resaved_genuine is unchanged
      from v1.

Otherwise v1 is kept. The anomaly check's false-alarm rate on genuine dev is in-sample: its
threshold was set on those statements, so about 2% of them exceed it by construction.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.verdict import RULES, TAMPERED, decide
from statementscreen.tamper.fingerprint import StaleFindings
from statementscreen.tamper.score_tamper import ScoreError, load_inputs
from statementscreen.tamper.scoring import Item, build_report, score_item

ADOPTION_RULE = (
    "(a) rerender_consistent detection, counted as tampered or suspicious, on dev rises by at "
    "least 10 percentage points over the old version",
    "(b) an anomaly score alone can never produce 'tampered'",
    "(c) the 'tampered' false-alarm rate on genuine dev and on resaved_genuine is unchanged "
    "from the old version",
)
MIN_GAIN = Fraction(1, 10)  # condition (a): 10 percentage points
IN_SAMPLE_NOTE = (
    "The anomaly check's false-alarm rate on genuine dev is in-sample: its threshold was set "
    "on these statements, so about 2% of them exceed it by construction. Under a rule that "
    "flags an anomaly score alone, that 2% is the 'suspicious' rate on genuine dev; it is not "
    "an independent estimate."
)


def _rate(metric: dict) -> dict:
    return {k: metric[k] for k in ("count", "total", "rate", "text", "ci95")}


def _suspicious(tampered: dict, flagged: dict) -> dict:
    count, total = flagged["count"] - tampered["count"], flagged["total"]
    return {
        "count": count,
        "total": total,
        "rate": round(count / total, 4) if total else None,
        "text": f"{count}/{total} ({100 * count / total:.1f}%)" if total else "0/0",
    }


def summarize(report: dict) -> dict:
    """The numbers the comparison shows, from one version's build_report output."""
    detection = {
        t: {
            "tampered": _rate(rep["with_history"]["tampered"]),
            "tampered_or_suspicious": _rate(rep["with_history"]["tampered_or_suspicious"]),
        }
        for t, rep in report["detection"].items()
    }
    alarms = {}
    groups = {
        "resaved_genuine": report["false_alarms"]["resaved_genuine"],
        "genuine_dev": report["false_alarms"]["genuine"].get("dev"),
    }
    for name, rep in groups.items():
        if rep is None:
            raise ScoreError(f"the comparison needs {name}: give resaved copies and --genuine dev=")
        tampered = _rate(rep["with_history"]["tampered"])
        flagged = _rate(rep["with_history"]["tampered_or_suspicious"])
        alarms[name] = {
            "tampered": tampered,
            "tampered_or_suspicious": flagged,
            "suspicious": _suspicious(tampered, flagged),
        }
    return {"detection": detection, "false_alarms": alarms}


def anomaly_alone_check(items: list[Item], version: str) -> dict:
    """Condition (b): under `version`, nothing is 'tampered' without a strong finding, and a
    lone anomaly score is not 'tampered'."""
    without_strong = [
        item.item
        for item in items
        if score_item(item, version).label == TAMPERED
        and not any(f.strength == "strong" for f in item.findings)
    ]
    probe = Finding("ANOMALY_SCORE", "anomaly", None, None, "weak", "probe")
    return {
        "tampered_without_a_strong_finding": without_strong,
        "lone_anomaly_score_gives": decide([probe], version)["label"],
    }


def adoption(old: dict, new: dict, anomaly: dict) -> dict:
    """Apply the adoption rule exactly as written to two summaries (see summarize)."""
    o = old["detection"]["rerender_consistent"]["tampered_or_suspicious"]
    n = new["detection"]["rerender_consistent"]["tampered_or_suspicious"]
    gain = Fraction(n["count"], n["total"]) - Fraction(o["count"], o["total"])
    a = gain >= MIN_GAIN
    no_orphans = not anomaly["tampered_without_a_strong_finding"]
    b = no_orphans and anomaly["lone_anomaly_score_gives"] != TAMPERED
    unchanged = {
        name: old["false_alarms"][name]["tampered"]["count"]
        == new["false_alarms"][name]["tampered"]["count"]
        for name in ("genuine_dev", "resaved_genuine")
    }
    c = all(unchanged.values())
    return {
        "a_rerender_consistent_gain": {
            "passed": a,
            "old": o["text"],
            "new": n["text"],
            "gain_percentage_points": round(100 * float(gain), 1),
            "needed_percentage_points": 10,
        },
        "b_anomaly_alone_never_tampered": {"passed": b, **anomaly},
        "c_tampered_false_alarms_unchanged": {
            "passed": c,
            "genuine_dev": [
                old["false_alarms"]["genuine_dev"]["tampered"]["text"],
                new["false_alarms"]["genuine_dev"]["tampered"]["text"],
            ],
            "resaved_genuine": [
                old["false_alarms"]["resaved_genuine"]["tampered"]["text"],
                new["false_alarms"]["resaved_genuine"]["tampered"]["text"],
            ],
        },
        "all_passed": a and b and c,
    }


def compare(copies: list[Item], genuine: dict[str, list[Item]], old: str, new: str) -> dict:
    for version in (old, new):
        if version not in RULES:
            raise ScoreError(f"unknown rule version {version!r}; choose from {sorted(RULES)}")
    summaries = {v: summarize(build_report(copies, genuine, v)) for v in (old, new)}
    everything = copies + [i for items in genuine.values() for i in items]
    conditions = adoption(summaries[old], summaries[new], anomaly_alone_check(everything, new))
    if conditions["all_passed"]:
        selected, decision = new, f"{new} is adopted: all three conditions hold"
    else:
        failed = [k for k, v in conditions.items() if k != "all_passed" and not v["passed"]]
        selected, decision = old, f"{old} is kept: not met: {', '.join(failed)}"
    return {
        "old": old,
        "new": new,
        "populations": "dev copies (60 per type), resaved_genuine, genuine dev",
        "side_by_side": summaries,
        "adoption_rule": list(ADOPTION_RULE),
        "conditions": conditions,
        "selected": selected,
        "decision": decision,
        "notes": [IN_SAMPLE_NOTE],
    }


def _count(metric: dict) -> str:
    """'59/60' from a metric dict."""
    return f"{metric['count']}/{metric['total']}"


def _pair(block: dict, keys: tuple[str, ...]) -> str:
    return " | ".join(f"{_count(block[k]):>6}" for k in keys)


def text(result: dict) -> str:
    old, new = result["old"], result["new"]
    s = result["side_by_side"]
    detection_keys = ("tampered", "tampered_or_suspicious")
    alarm_keys = ("tampered", "tampered_or_suspicious", "suspicious")
    lines = [
        f"{old} vs {new} on the same cached dev findings",
        "",
        f"Detection per tamper type (tampered | tampered or suspicious):  {old}  ->  {new}",
    ]
    for t in s[old]["detection"]:
        o, n = s[old]["detection"][t], s[new]["detection"][t]
        lines.append(f"  {t:<26} {_pair(o, detection_keys)}   ->   {_pair(n, detection_keys)}")
    lines += ["", f"False alarms (tampered | flagged | suspicious):  {old}  ->  {new}"]
    for name in ("resaved_genuine", "genuine_dev"):
        o, n = s[old]["false_alarms"][name], s[new]["false_alarms"][name]
        lines.append(f"  {name:<26} {_pair(o, alarm_keys)}   ->   {_pair(n, alarm_keys)}")
    lines += ["", "Adoption rule, applied as written:"]
    c = result["conditions"]
    for key, label in (
        ("a_rerender_consistent_gain", "(a) rerender_consistent gain >= 10 points"),
        ("b_anomaly_alone_never_tampered", "(b) an anomaly score alone is never 'tampered'"),
        ("c_tampered_false_alarms_unchanged", "(c) 'tampered' false alarms unchanged"),
    ):
        lines.append(f"  [{'PASS' if c[key]['passed'] else 'FAIL'}] {label}")
    a = c["a_rerender_consistent_gain"]
    lines.append(f"      (a): {a['old']} -> {a['new']}, gain {a['gain_percentage_points']} points")
    lines += ["", f"SELECTED: {result['selected']}  ({result['decision']})", "", result["notes"][0]]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.compare_rules",
        description="Compare two rule versions on cached findings; apply the adoption rule.",
    )
    p.add_argument("--copies", type=Path, required=True)
    p.add_argument("--copies-run", type=Path)
    p.add_argument("--genuine", action="append", default=[], metavar="NAME=RUN_DIR")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--old", default="v1")
    p.add_argument("--new", default="v2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    try:
        if args.out.exists() and not args.overwrite:
            raise ScoreError(f"{args.out} exists; pass --overwrite to replace it")
        copies, genuine, inputs = load_inputs(
            args.copies, args.copies_run, args.genuine, args.manifest
        )
        result = compare(copies, genuine, args.old, args.new)
        result["inputs"] = inputs
    except (ScoreError, StaleFindings) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(text(result))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
