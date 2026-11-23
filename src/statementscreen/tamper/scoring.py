"""Score the tamper detector against the answer keys. The only module that reads them.

It never opens a PDF. It reads:
  - the answer keys (<set>/keys/copy_NNNN.json): tamper type, source statement, layout,
    save mode, pages and fields changed;
  - the manifest, only for each genuine statement's layout (via splits.read_manifest);
  - the findings the detector cached (findings.parquet: item, group, code, page, row,
    strength, reason). An item with no row there had no findings.
The verdict is recomputed here from the findings with the rule version asked for, so a new
version can be scored without re-reading any PDF. The detector never imports this module.

Reported (build_report):
  detection         per tamper type: verdict counts, the "tampered" rate and the "tampered
                    or suspicious" rate, each with a 95% bootstrap interval, and the same
                    rates recomputed with every file-history finding removed
  overlay_holdout   the held-out type, in its own section, only when overlay copies exist
  false_alarms      resaved_genuine copies, and each genuine population (dev, test, shift),
                    counted the same two ways
  first_check       for each flagged copy, the first non-info finding in verdict order
                    (strong first, then check group, page, row), counted per type. A
                    ranking, not a time: whole-statement findings and earlier check groups
                    rank first, so read codes_fired for what actually caught a copy
  codes_fired       per type, on how many copies each strong and each weak finding code
                    fired at least once
  by_layout         counts as count/total and percent, no intervals; layout gaps over 15
                    points are flagged in plain words
  by_save_mode      for types a PDF library wrote, incremental save vs full rewrite
  history_dependence  types whose detection (or false alarms) fall by more than half
                    without file-history findings
  misses            every tamper copy that was not called "tampered", with the detector's
                    own reasons and the key's facts
  false_alarm_cases every genuine or re-saved item that was flagged, with reasons
  could_not_check   a check group that crashed on a PDF (a CHECK_NOT_RUN finding from any
                    group but arithmetic, or the whole detector crashing) leaves that check
                    unrun, for that check only. Any strong finding still gives "tampered" and
                    any weak verdict stands; but a verdict that would read "no evidence of
                    tampering" with a crashed check reads "could not check" instead. Those
                    PDFs are counted in every total, reported in their own column
                    (could_not_check in each group) and listed in could_not_check_cases, and
                    are never counted as flagged or as "no evidence". crashed_checks counts,
                    per check, the PDFs on which that check crashed, whatever their verdict.

Intervals: 1,000 bootstrap resamples with a fixed seed, resampling SOURCE STATEMENTS (a
statement can appear in several types; its copies stay together), 2.5th to 97.5th
percentile. A group that is all hits or no hits gets a zero-width interval, which is
narrower than the real uncertainty.
"""

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from statementscreen.splits import read_manifest
from statementscreen.tamper.copy import CONTROL, HELD_OUT, TYPES
from statementscreen.tamper.detect.findings import Finding
from statementscreen.tamper.detect.verdict import (
    LABELS,
    NONE,
    RULE_VERSION,
    SUSPICIOUS,
    TAMPERED,
    decide,
)

COULD_NOT_CHECK = "could not check"  # the detector crashed on the PDF: no verdict was possible
SCORED_LABELS = (*LABELS, COULD_NOT_CHECK)
BOOTSTRAP_SEED = 2026
RESAMPLES = 1000
LAYOUT_GAP = 0.15  # flag a type whose two layouts differ by more than this (in rate)
TAMPER_TYPES = tuple(t for t in TYPES if t not in (HELD_OUT, CONTROL))
COPY_COLUMNS = ("copy", "tamper_type", "layout", "save_mode", "verdict", "first_check")
SAVE_MODES = ("incremental", "full")
# The two ways a copy is counted: (key in the report, words for a sentence).
METRICS = (("tampered", "tampered"), ("tampered_or_suspicious", "tampered or suspicious"))


@dataclass(frozen=True)
class Item:
    """One thing the detector looked at: a tampered or re-saved copy, or a genuine PDF."""

    item: str  # copy_NNNN, or a genuine statement id
    source: str  # the source statement: what the bootstrap resamples
    kind: str  # a tamper type, "resaved_genuine", or "genuine"
    layout: str
    save_mode: str = "n/a"  # "incremental" | "full" for PDFs a library wrote, else "n/a"
    findings: tuple[Finding, ...] = ()
    facts: dict = field(default_factory=dict)  # from the key: pages, fields changed, donor
    crashed: bool = False  # the detector crashed on this PDF (its run lists it as crashed)


@dataclass(frozen=True)
class Scored:
    item: Item
    verdict: dict
    label: str
    label_without_history: str
    label_without_anomaly: str
    first: str | None  # "group:code" of the first non-info finding, if any
    crashed_checks: tuple[str, ...] = ()  # the check groups that crashed on this item


# ---------------------------------------------------------------- loading


def load_findings(path: Path) -> dict[str, list[Finding]]:
    """The detector's cached findings by item. An item with none has no entry."""
    out: dict[str, list[Finding]] = {}
    for r in pl.read_parquet(path).iter_rows(named=True):
        finding = Finding(r["code"], r["group"], r["page"], r["row"], r["strength"], r["reason"])
        out.setdefault(r["item"], []).append(finding)
    return out


def _save_mode(incremental: bool | None) -> str:
    return "n/a" if incremental is None else ("incremental" if incremental else "full")


def load_copies(
    set_dir: Path, findings: dict[str, list[Finding]], crashed: frozenset[str] = frozenset()
) -> list[Item]:
    """The set's copies, from its answer keys, with the findings cached for each. `crashed`
    are the copies the detector crashed on."""
    items = []
    for path in sorted((Path(set_dir) / "keys").glob("copy_*.json")):
        key = json.loads(path.read_text(encoding="utf-8"))
        facts = {
            "pages_changed": key["pages_changed"],
            "fields_changed": len(key["changes"]),
            "donor": key.get("donor"),
        }
        items.append(
            Item(
                item=key["copy_id"],
                source=key["source"],
                kind=key["tamper_type"],
                layout=key["source_layout"],
                save_mode=_save_mode(key["incremental_save"]),
                findings=tuple(findings.get(key["copy_id"], ())),
                facts=facts,
                crashed=key["copy_id"] in crashed,
            )
        )
    return items


def layouts_from_manifest(manifest: Path) -> dict[str, str]:
    return {r["statement_id"]: r["layout"] for r in read_manifest(Path(manifest))}


def genuine_items(
    ids: list[str],
    layout_of: dict[str, str],
    findings: dict[str, list[Finding]],
    crashed: frozenset[str] = frozenset(),
) -> list[Item]:
    return [
        Item(
            sid,
            sid,
            "genuine",
            layout_of[sid],
            findings=tuple(findings.get(sid, ())),
            crashed=sid in crashed,
        )
        for sid in sorted(ids)
    ]


# ---------------------------------------------------------------- scoring one item


def first_check(verdict: dict) -> str | None:
    """The first non-info finding in verdict order, as 'group:code'."""
    for f in verdict["findings"]:
        if f["strength"] != "info":
            return f"{f['group']}:{f['code']}"
    return None


def crashed_checks(item: Item) -> frozenset[str]:
    """The check groups that crashed on this item. A CHECK_NOT_RUN finding from any group but
    arithmetic is a check that raised (the detector's own wrapper, or tamper/run.py when the
    whole detector raised: group parse). Arithmetic's CHECK_NOT_RUN accompanies PARSE_FAILED:
    a PDF that could not be parsed, which has its own verdict rule, not a crash."""
    groups = {
        f.group for f in item.findings if f.code == "CHECK_NOT_RUN" and f.group != "arithmetic"
    }
    if item.crashed:
        groups.add("parse")
    return frozenset(groups)


def _label(verdict: dict, crashed: frozenset[str]) -> str:
    """The verdict's label, except that "no evidence" with a crashed check is "could not
    check": a strong finding or a weak verdict is evidence the crash does not remove."""
    label = verdict["label"]
    return COULD_NOT_CHECK if label == NONE and crashed else label


def score_item(item: Item, version: str = RULE_VERSION) -> Scored:
    found = list(item.findings)
    verdict = decide(found, version)
    without = decide([f for f in found if f.group != "history"], version)
    no_anomaly = decide([f for f in found if f.group != "anomaly"], version)
    crashed = crashed_checks(item)
    return Scored(
        item,
        verdict,
        _label(verdict, crashed),
        _label(without, crashed - {"history"}),  # a history crash is moot with history removed
        _label(no_anomaly, crashed - {"anomaly"}),
        first_check(verdict),
        tuple(sorted(crashed)),
    )


# ---------------------------------------------------------------- rates and intervals


def bootstrap_ci(sources: list[str], hits: list[bool]) -> list[float] | None:
    """95% interval of the hit rate, resampling source statements with replacement."""
    if not sources:
        return None
    ids = {s: i for i, s in enumerate(sorted(set(sources)))}
    idx = np.array([ids[s] for s in sources])
    n = np.bincount(idx, minlength=len(ids)).astype(float)
    h = np.bincount(idx, weights=np.array(hits, dtype=float), minlength=len(ids))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    rates = h[draw].sum(axis=1) / n[draw].sum(axis=1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _text(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.1f}%)" if total else "0/0"


def _plain(scored: list[Scored], pick: Callable[[Scored], bool]) -> dict:
    count, total = sum(bool(pick(s)) for s in scored), len(scored)
    rate = round(count / total, 4) if total else None
    return {"count": count, "total": total, "rate": rate, "text": _text(count, total)}


def _with_ci(scored: list[Scored], pick: Callable[[Scored], bool]) -> dict:
    out = _plain(scored, pick)
    hits = [bool(pick(s)) for s in scored]
    out["ci95"] = bootstrap_ci([s.item.source for s in scored], hits)
    return out


def _tampered(s: Scored) -> bool:
    return s.label == TAMPERED


FLAGGED = (SUSPICIOUS, TAMPERED)  # "could not check" is neither flagged nor "no evidence"


def _flagged(s: Scored) -> bool:
    return s.label in FLAGGED


def _could_not_check(s: Scored) -> bool:
    return s.label == COULD_NOT_CHECK


def _tampered_nh(s: Scored) -> bool:
    return s.label_without_history == TAMPERED


def _flagged_nh(s: Scored) -> bool:
    return s.label_without_history in FLAGGED


def _tampered_na(s: Scored) -> bool:
    return s.label_without_anomaly == TAMPERED


def _flagged_na(s: Scored) -> bool:
    return s.label_without_anomaly in FLAGGED


def _without_anomaly(scored: list[Scored]) -> dict:
    """The verdicts recomputed with every anomaly finding removed."""
    count = Counter(s.label_without_anomaly for s in scored)
    return {
        "labels": {label: count.get(label, 0) for label in SCORED_LABELS},
        "tampered": _with_ci(scored, _tampered_na),
        "tampered_or_suspicious": _with_ci(scored, _flagged_na),
    }


def _labels(scored: list[Scored], without_history: bool) -> dict[str, int]:
    count = Counter(s.label_without_history if without_history else s.label for s in scored)
    return {label: count.get(label, 0) for label in SCORED_LABELS}


def _pair(scored: list[Scored]) -> dict:
    return {
        "with_history": {
            "labels": _labels(scored, False),
            "tampered": _with_ci(scored, _tampered),
            "tampered_or_suspicious": _with_ci(scored, _flagged),
        },
        "without_history": {
            "labels": _labels(scored, True),
            "tampered": _with_ci(scored, _tampered_nh),
            "tampered_or_suspicious": _with_ci(scored, _flagged_nh),
        },
    }


# ---------------------------------------------------------------- breakdowns


def _by_layout(scored: list[Scored]) -> dict:
    out = {}
    for layout in sorted({s.item.layout for s in scored}):
        sub = [s for s in scored if s.item.layout == layout]
        out[layout] = {
            "n": len(sub),
            "tampered": _plain(sub, _tampered),
            "tampered_or_suspicious": _plain(sub, _flagged),
        }
    return out


def _layout_notes(name: str, by_layout: dict) -> list[str]:
    notes = []
    for key, words in METRICS:
        rates = {k: v[key] for k, v in by_layout.items() if v[key]["rate"] is not None}
        if len(rates) < 2:
            continue
        hi = max(rates, key=lambda k: rates[k]["rate"])
        lo = min(rates, key=lambda k: rates[k]["rate"])
        gap = rates[hi]["rate"] - rates[lo]["rate"]
        if gap > LAYOUT_GAP:
            notes.append(
                f"{name}: the {words} rate differs by {100 * gap:.1f} points between layouts "
                f"({hi} {rates[hi]['text']} vs {lo} {rates[lo]['text']}); the counts per "
                "layout are small, so treat the gap with caution"
            )
    return notes


def _by_save_mode(scored: list[Scored]) -> dict:
    out = {}
    for mode in SAVE_MODES:
        sub = [s for s in scored if s.item.save_mode == mode]
        if sub:
            out[mode] = {
                "n": len(sub),
                "tampered": _plain(sub, _tampered),
                "tampered_or_suspicious": _plain(sub, _flagged),
                "tampered_without_history": _plain(sub, _tampered_nh),
                "tampered_or_suspicious_without_history": _plain(sub, _flagged_nh),
            }
    return out


def _first_checks(scored: list[Scored]) -> dict:
    out: dict[str, Counter] = {"tampered": Counter(), "suspicious": Counter()}
    for s in scored:
        if s.label == TAMPERED:
            out["tampered"][s.first or "none"] += 1
        elif s.label == SUSPICIOUS:
            out["suspicious"][s.first or "none"] += 1
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def _codes_fired(scored: list[Scored]) -> dict:
    """On how many copies each strong and each weak finding code fired at least once. Unlike
    first_check (a ranking), this shows every check that caught or flagged a copy."""
    out: dict[str, Counter] = {"strong": Counter(), "weak": Counter()}
    for s in scored:
        seen = {(f["strength"], f["code"]) for f in s.verdict["findings"] if f["strength"] in out}
        for strength, code in seen:
            out[strength][code] += 1
    return {strength: dict(sorted(counts.items())) for strength, counts in out.items()}


def _history_notes(name: str, pair: dict, subject: str) -> list[str]:
    """Notes where removing file-history findings cuts a count by more than half."""
    verb = (
        "detection depends mainly on file history"
        if subject == "detection"
        else "the false alarms come mainly from file history"
    )
    notes = []
    for key, words in METRICS:
        w, wo = pair["with_history"][key], pair["without_history"][key]
        if w["count"] and wo["count"] * 2 < w["count"]:
            notes.append(
                f"{name}: the {words} rate falls from {w['text']} to {wo['text']} when "
                f"file-history findings are removed: {verb}"
            )
    return notes


def _crashed_counts(scored: list[Scored]) -> dict[str, int]:
    """Per check group, on how many items it crashed (whatever the item's verdict)."""
    count = Counter(g for s in scored for g in s.crashed_checks)
    return dict(sorted(count.items()))


def _group_report(scored: list[Scored]) -> dict:
    out = {
        "n": len(scored),
        **_pair(scored),
        "could_not_check": _plain(scored, _could_not_check),
        "crashed_checks": _crashed_counts(scored),
        "by_layout": _by_layout(scored),
        "first_check": _first_checks(scored),
        "codes_fired": _codes_fired(scored),
    }
    if any(s.item.save_mode != "n/a" for s in scored):
        out["by_save_mode"] = _by_save_mode(scored)
    return out


# ---------------------------------------------------------------- cases


def _case(s: Scored, outcome: str, population: str | None = None) -> dict:
    case = {
        "item": s.item.item,
        "type": s.item.kind,
        "layout": s.item.layout,
        "source": s.item.source,
        "save_mode": s.item.save_mode,
        "outcome": outcome,
        "summary": s.verdict["summary"],
        "reasons": [r["reason"] for r in s.verdict["reasons"]],
        **s.item.facts,
    }
    if population:
        case["population"] = population
    return case


def _misses(scored: list[Scored]) -> list[dict]:
    """Tamper copies not called "tampered": not flagged at all, or only suspicious."""
    out = []
    for s in scored:
        if s.label == NONE:
            out.append(_case(s, "not flagged"))
        elif s.label == SUSPICIOUS:
            out.append(_case(s, "only suspicious"))
        elif s.label == COULD_NOT_CHECK:
            out.append(_case(s, "could not check"))
    return out


def _false_alarms(scored: list[Scored], population: str) -> list[dict]:
    out = []
    for s in scored:
        if s.label in FLAGGED:
            out.append(_case(s, "tampered" if s.label == TAMPERED else "suspicious", population))
    return out


def _unchecked(scored: list[Scored], population: str | None = None) -> list[dict]:
    """Every item whose verdict is "could not check", and every item on which a check crashed
    but a verdict stood, with the checks that crashed and the detector's own reasons."""
    out = []
    for s in scored:
        if s.crashed_checks:
            case = _case(s, s.label, population)
            case["crashed_checks"] = list(s.crashed_checks)
            out.append(case)
    return out


# ---------------------------------------------------------------- genuine populations

# Each genuine population is reported on its own row and never combined with another. The
# label says how the population relates to the anomaly model, so nobody reads a number as
# more independent than it is.
POPULATIONS = {
    "dev": "genuine dev: the anomaly threshold was set on these statements, so the anomaly "
    "check's 2% holds here by construction",
    "train": "genuine train (in-sample for anomaly): the IsolationForest was trained on these "
    "statements",
    "test": "genuine test: seen by neither the anomaly model nor its threshold",
    "shift": "genuine shift batch (another collaborator's narration formats): seen by neither "
    "the anomaly model nor its threshold",
}


def _genuine_report(name: str, scored: list[Scored]) -> dict:
    """A genuine population: the usual rates, plus the rates with anomaly findings removed."""
    report = _group_report(scored)
    report["population"] = POPULATIONS.get(name, f"genuine {name}")
    report["without_anomaly"] = _without_anomaly(scored)
    return report


def _strong_cases(scored: list[Scored]) -> list[dict]:
    """Genuine items with any strong finding: the stop condition (never expected)."""
    out = []
    for s in scored:
        strong = [f for f in s.verdict["findings"] if f["strength"] == "strong"]
        if strong:
            out.append(
                {
                    "item": s.item.item,
                    "layout": s.item.layout,
                    "codes": sorted({f["code"] for f in strong}),
                    "reasons": [f["reason"] for f in strong],
                }
            )
    return out


# ---------------------------------------------------------------- the report


def build_report(
    copies: list[Item],
    genuine: dict[str, list[Item]],
    rule_version: str = RULE_VERSION,
) -> dict:
    """Everything in the module docstring. genuine: population name -> its genuine items."""
    unknown = sorted({i.kind for i in copies} - set(TYPES))
    if unknown:
        raise ValueError(f"unknown tamper types in the copies: {unknown}")
    scored = [score_item(i, rule_version) for i in sorted(copies, key=lambda i: i.item)]
    by_kind = {k: [s for s in scored if s.item.kind == k] for k in TYPES}
    detection = {t: _group_report(by_kind[t]) for t in TAMPER_TYPES if by_kind[t]}
    pooled = [s for t in TAMPER_TYPES for s in by_kind[t]]
    overlay = _group_report(by_kind[HELD_OUT]) if by_kind[HELD_OUT] else None
    resaved = _group_report(by_kind[CONTROL]) if by_kind[CONTROL] else None
    genuine_scored = {
        name: [score_item(i, rule_version) for i in sorted(items, key=lambda i: i.item)]
        for name, items in sorted(genuine.items())
    }
    false_alarms = {
        "resaved_genuine": resaved,
        "genuine": {name: _genuine_report(name, sc) for name, sc in genuine_scored.items()},
    }
    strong_on_genuine = [
        {"population": name, **case}
        for name, sc in genuine_scored.items()
        for case in _strong_cases(sc)
    ]

    notes = {"detection": [], "false_alarms": [], "overlay": []}
    layout_notes = []
    for t, rep in detection.items():
        notes["detection"] += _history_notes(t, rep, "detection")
        layout_notes += _layout_notes(t, rep["by_layout"])
    if overlay:
        notes["overlay"] += _history_notes(HELD_OUT, overlay, "detection")
        layout_notes += _layout_notes(HELD_OUT, overlay["by_layout"])
    if resaved:
        notes["false_alarms"] += _history_notes(CONTROL, resaved, "false_alarms")
    for name, rep in false_alarms["genuine"].items():
        notes["false_alarms"] += _history_notes(f"genuine {name}", rep, "false_alarms")

    return {
        "rule_version": rule_version,
        "bootstrap": {
            "resamples": RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "resampled": "source statements",
            "interval": "2.5th to 97.5th percentile",
        },
        "definitions": {
            "tampered": "the verdict is 'tampered'",
            "tampered_or_suspicious": "the verdict is 'tampered' or 'suspicious'",
            "without_history": "the verdict recomputed with every file-history finding removed",
            "could_not_check": "the detector crashed on the PDF: counted in the totals, never "
            "as 'no evidence of tampering', never as flagged",
            "first_check": "the first non-info finding in verdict order (strong first, then "
            "check group, page, row)",
        },
        "detection": detection,
        "pooled_detection": (
            {**_pair(pooled), "could_not_check": _plain(pooled, _could_not_check)}
            if pooled
            else None
        ),
        "overlay_holdout": overlay,
        "false_alarms": false_alarms,
        "history_dependence": notes,
        "layout_notes": layout_notes,
        "misses": _misses([s for s in scored if s.item.kind in TAMPER_TYPES]),
        "overlay_misses": _misses(by_kind[HELD_OUT]),
        "genuine_with_strong_findings": strong_on_genuine,
        "false_alarm_cases": _false_alarms(by_kind[CONTROL], CONTROL)
        + [c for name, sc in genuine_scored.items() for c in _false_alarms(sc, name)],
        "could_not_check_cases": _unchecked(scored)
        + [c for name, sc in genuine_scored.items() for c in _unchecked(sc, name)],
    }


# ---------------------------------------------------------------- the per-copy table


def copy_rows(
    items: list[Item], rule_version: str = RULE_VERSION, extended: bool = False
) -> list[dict]:
    """One row per copy: name, type, layout, save mode, verdict, first check that fired."""
    rows = []
    for s in sorted((score_item(i, rule_version) for i in items), key=lambda s: s.item.item):
        row = {
            "copy": s.item.item,
            "tamper_type": s.item.kind,
            "layout": s.item.layout,
            "save_mode": s.item.save_mode,
            "verdict": s.label,
            "first_check": s.first or "",
        }
        if extended:
            row["verdict_without_history"] = s.label_without_history
        rows.append(row)
    return rows


def write_copy_table(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else list(COPY_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
