"""Generate the README's "Results at a glance" block from the committed reports.

    uv run python scripts/results_tables.py            # print the block
    uv run python scripts/results_tables.py --write    # replace the block in README.md

Every table cell and every number in a sentence is a Cell: a format name plus the (report,
key path) of each value it shows. Nothing is typed in by hand. tests/test_results_tables.py
re-reads every source from the report files and recomputes the text with the format
functions below, and fails if the README block differs from what this script prints.

Formatting rules, defined once, here:
  * a rate prints as a percentage with two decimals, rounded half up;
  * a value below 1 never prints as 100.00% (it prints 99.99%) and a value above 0 never
    prints as 0.00% (it prints 0.01%), so a rounded number never claims perfection or zero;
  * an F1 prints with four decimals under the same two guards; a difference prints with a sign;
  * a count prints as it is, or as k/n.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
README = ROOT / "README.md"
START = "<!-- results:start -->"
END = "<!-- results:end -->"


# ------------------------------------------------------------------ formatting


def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def _round(v: Decimal, places: int) -> Decimal:
    return v.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    v = _dec(x) * 100
    r = _round(v, 2)
    if r >= 100 and v < 100:
        r = Decimal("99.99")
    if r <= 0 and v > 0:
        r = Decimal("0.01")
    return f"{r:.2f}%"


def f1(x: float | None) -> str:
    if x is None:
        return "n/a"
    v = _dec(x)
    r = _round(v, 4)
    if r >= 1 and v < 1:
        r = Decimal("0.9999")
    if r <= 0 and v > 0:
        r = Decimal("0.0001")
    return f"{r:.4f}"


def signed(x: float | None) -> str:
    if x is None:
        return "n/a"
    v = _dec(x)
    r = _round(v, 4)
    if r == 0 and v != 0:
        return ("+" if v > 0 else "-") + "<0.0001"
    sign = "+" if r > 0 else ("-" if r < 0 else "")
    return f"{sign}{abs(r):.4f}"


FORMATS = {
    "int": lambda n: f"{n:,}",
    "pct": lambda x: pct(x),
    "frac": lambda k, n: f"{k:,}/{n:,}",
    "kpct": lambda k, n, rate: f"{k:,}/{n:,} ({pct(rate)})",
    "kpci": lambda k, n, rate, lo, hi: f"{k:,}/{n:,} ({pct(rate)}) [{pct(lo)}, {pct(hi)}]",
    "vci": lambda v, lo, hi: f"{pct(v)} [{pct(lo)}, {pct(hi)}]",
    "f1": lambda x: f1(x),
    "f1diff": lambda d, lo, hi: f"{signed(d)} [{signed(lo)}, {signed(hi)}]",
}


# ------------------------------------------------------------------ reading reports


@cache
def load(report: str):
    return json.loads((REPORTS / report).read_text(encoding="utf-8"))


def get(report: str, path: str):
    node = load(report)
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


@dataclass(frozen=True)
class Cell:
    fmt: str
    sources: tuple[tuple[str, str], ...]  # (report file name, dotted key path), in order

    def values(self) -> list:
        return [get(report, path) for report, path in self.sources]

    def text(self) -> str:
        return FORMATS[self.fmt](*self.values())


def cell(fmt: str, report: str, *paths: str) -> Cell:
    return Cell(fmt, tuple((report, p) for p in paths))


def kpct(report: str, base: str) -> Cell:
    return cell("kpct", report, f"{base}.count", f"{base}.total", f"{base}.rate")


def kpci(report: str, base: str) -> Cell:
    return cell(
        "kpci", report, f"{base}.count", f"{base}.total", f"{base}.rate",
        f"{base}.ci95.0", f"{base}.ci95.1",
    )  # fmt: skip


def vci(report: str, base: str) -> Cell:
    return cell("vci", report, f"{base}.value", f"{base}.low", f"{base}.high")


@dataclass(frozen=True)
class Table:
    caption: str  # Markdown, no numbers: it names the source reports and the kind of result
    header: list[str]
    rows: list[list]  # first item of a row is a label (str); every other item is a Cell
    note: str = ""


@dataclass(frozen=True)
class Sentence:
    template: str
    cells: list[Cell]

    def text(self) -> str:
        return self.template.format(*[c.text() for c in self.cells])


# ------------------------------------------------------------------ what goes in the block

BLIND = "week7_blind_report.json"
W6 = "week6_tamper_test.json"
W5 = "week5_test_and_shift.json"
W4 = "week4_classifier.json"
W2 = "week2_parse_report.json"
W4S = "week4_shift_parse_report.json"


def link(report: str) -> str:
    return f"[`reports/{report}`](reports/{report})"


ZERO_WIDTH = (
    "An interval of zero width (every item was correct) is narrower than the real uncertainty."
)


BLIND_TYPES = ["rerender_inconsistent", "inplace_edit", "page_swap", "page_remove", "overlay",
               "inplace_edit_consistent", "rerender_consistent"]  # fmt: skip
W6_TYPES = ["rerender_inconsistent", "rerender_consistent", "inplace_edit",
            "inplace_edit_consistent", "page_swap", "page_remove"]  # fmt: skip
PROFILE_ROWS = [
    ("Income within 5%", "income_within_5pct"),
    ("Income within 10%", "income_within_10pct"),
    ("Employer exact match", "employer_exact"),
    ("Employer fuzzy match", "employer_fuzzy"),
    ("has_salary precision", "has_salary_precision"),
    ("has_salary recall", "has_salary_recall"),
    ("Customer-type accuracy", "customer_type_accuracy"),
    ("EMI count exact", "emi_count_exact"),
    ("Bounce count exact", "bounce_count_exact"),
]
LABEL_ROWS = [("is_salary", "labels.is_salary"), ("is_emi", "labels.is_emi"),
              ("is_bounce", "labels.is_bounce"), ("is_rent", "labels.is_rent"),
              ("has_salary", "has_salary")]  # fmt: skip


def tables() -> list[Table]:
    blind_headline = Table(
        f"**Tamper detection, sealed blind run (blind: scored once, key opened after the "
        f"verdicts were pushed).** Source: {link(BLIND)}. Intervals are the report's "
        f"bootstrap intervals.",
        ["Result", "Count (rate) [interval]"],
        [
            ["Tampered PDFs called tampered", kpci(BLIND, "pooled_detection.tampered")],
            ["Tampered PDFs called tampered or suspicious",
             kpci(BLIND, "pooled_detection.tampered_or_suspicious")],
            ["Precision of tampered (called tampered, and was tampered)",
             kpct(BLIND, "precision_of_tampered")],
            ["Genuine PDFs called tampered", kpct(BLIND, "false_alarms.overall.tampered")],
            ["Genuine PDFs called tampered or suspicious",
             kpct(BLIND, "false_alarms.overall.tampered_or_suspicious")],
        ],
    )  # fmt: skip
    blind_types = Table(
        f"Blind run by tamper type. Source: {link(BLIND)}.",
        ["Tamper type", "n", "Tampered", "Tampered or suspicious"],
        [[t, cell("int", BLIND, f"detection.{t}.n"), kpct(BLIND, f"detection.{t}.tampered"),
          kpct(BLIND, f"detection.{t}.tampered_or_suspicious")] for t in BLIND_TYPES],
        "A type where every PDF was caught has a zero-width bootstrap interval, narrower "
        "than the real uncertainty (see docs/blind_test.md).",
    )  # fmt: skip

    def w6_row(label: str, base: str) -> list:
        h = f"{base}.with_history"
        return [label, cell("int", W6, f"{base}.n"), kpct(W6, f"{h}.tampered"),
                kpci(W6, f"{h}.tampered_or_suspicious")]  # fmt: skip

    w6_rows = [w6_row(t, f"detection.{t}") for t in W6_TYPES]
    w6_rows += [
        w6_row("overlay (held out)", "overlay_holdout"),
        w6_row("genuine test", "false_alarms.genuine.test"),
        w6_row("genuine shift", "false_alarms.genuine.shift"),
        w6_row("resaved genuine", "false_alarms.resaved_genuine"),
    ]
    week6 = Table(
        f"**Tamper detection, Week 6 test (one-time test: scored once, copies of the test "
        f"split).** Source: {link(W6)}. Intervals are the report's bootstrap intervals.",
        ["Group", "n", "Tampered", "Tampered or suspicious [interval]"],
        w6_rows,
        ZERO_WIDTH,
    )

    cols = [("Dev (used for tuning)", "dev"), ("Test (scored once)", "test"),
            ("Shift (scored once)", "shift_karan_v1")]  # fmt: skip
    profile_rows = [["Statements"] + [cell("int", W5, f"sets.{s}.profile.overall.statements")
                                      for _, s in cols]]  # fmt: skip
    for label, key in PROFILE_ROWS:
        profile_rows.append([label] + [vci(W5, f"sets.{s}.profile.intervals.{key}")
                                       for _, s in cols])  # fmt: skip
    profile = Table(
        f"**Income profile (dev, one-time test and shift).** Source: {link(W5)}. Intervals "
        f"are the report's bootstrap intervals.",
        ["Metric"] + [c for c, _ in cols],
        profile_rows,
        "Dev was used for tuning, so the dev column is in-sample and not a held-out "
        "estimate. " + ZERO_WIDTH,
    )

    def f1_cells(sub: str) -> list:
        out = []
        for s in ("dev", "shift_karan_v1"):
            for who in ("rules", "model"):
                out.append(cell("f1", W4, f"sets.{s}.{sub}.{who}.f1"))
        base = f"sets.shift_karan_v1.{sub}.model_minus_rules"
        out.append(cell("f1diff", W4, f"{base}.f1", f"{base}.ci.low", f"{base}.ci.high"))
        return out

    rules_vs_model = Table(
        f"**Rules against the Week 4 model, F1 (dev, and the shift batch scored once).** "
        f"Source: {link(W4)}. Dev was used for tuning. The shift batch uses narration "
        f"formats written by an independent collaborator who had not seen the code.",
        ["F1", "Dev: rules", "Dev: model", "Shift: rules", "Shift: model",
         "Shift: model minus rules [interval]"],
        [[label] + f1_cells(sub) for label, sub in LABEL_ROWS],
        "n/a: the rules flag no bounce row on the shift batch, so their F1 is undefined in "
        "the first, one-time scoring; the post-hoc re-run is described in "
        "docs/classifier_vs_rules.md.",
    )  # fmt: skip

    def parse_col(report: str) -> list:
        o = "overall"
        return [
            cell("int", report, f"{o}.statements"),
            cell("frac", report, f"{o}.exact_rows", f"{o}.truth_rows"),
            cell("pct", report, f"{o}.exact_row_rate"),
            cell("frac", report, f"{o}.layout_detected_correctly", f"{o}.statements"),
            cell("int", report, f"{o}.narration.wrong"),
            cell("pct", report, f"{o}.narration.cut_short_rate"),
        ]

    names = ["Statements", "Rows parsed exactly (date, amount, balance)",
             "Exact-row rate", "Layout detected correctly", "Narrations wrong",
             "Narrations cut short by the layout (share of all narrations)"]  # fmt: skip
    ref, shift = parse_col(W2), parse_col(W4S)
    parser = Table(
        f"**Parser (the whole reference batch, and the shift batch).** Sources: {link(W2)} "
        f"and {link(W4S)}. Clean, text-based PDFs from this project's own renderer.",
        ["Parser", "Reference batch", "Shift batch"],
        [[n, a, b] for n, a, b in zip(names, ref, shift, strict=True)],
    )
    return [blind_headline, blind_types, week6, profile, rules_vs_model, parser]


def sentences() -> list[Sentence]:
    s2 = "sets.shift_karan_v1"
    return [
        Sentence(
            "On the sealed blind run the frozen detector called {0} of {1} tampered PDFs "
            "tampered (precision {2} of {3}) and called {4} of {5} genuine PDFs tampered or "
            "suspicious.",
            [cell("int", BLIND, "pooled_detection.tampered.count"),
             cell("int", BLIND, "pooled_detection.tampered.total"),
             cell("int", BLIND, "precision_of_tampered.count"),
             cell("int", BLIND, "precision_of_tampered.total"),
             cell("int", BLIND, "false_alarms.overall.tampered_or_suspicious.count"),
             cell("int", BLIND, "false_alarms.overall.tampered_or_suspicious.total")],
        ),
        Sentence(
            "A forgery with correct arithmetic is mostly missed: {0} of {1} such PDFs were "
            "called tampered on the blind run, and {2} of {3} in the Week 6 test.",
            [cell("int", BLIND, "detection.rerender_consistent.tampered.count"),
             cell("int", BLIND, "detection.rerender_consistent.tampered.total"),
             cell("int", W6, "detection.rerender_consistent.with_history.tampered.count"),
             cell("int", W6, "detection.rerender_consistent.with_history.tampered.total")],
        ),
        Sentence(
            "On the shift batch, where the narration formats were unseen, the income "
            "profile matched the employer exactly for {0} of the salaried statements (test: "
            "{1}) and got the bounce count exactly right for {2} of the statements (test: {3}).",
            [cell("pct", W5, f"{s2}.profile.intervals.employer_exact.value"),
             cell("pct", W5, "sets.test.profile.intervals.employer_exact.value"),
             cell("pct", W5, f"{s2}.profile.intervals.bounce_count_exact.value"),
             cell("pct", W5, "sets.test.profile.intervals.bounce_count_exact.value")],
        ),
        Sentence(
            "On those unseen formats the model's is_salary F1 was {0}, against {1} for the "
            "rules; the model had learned that a NEFT credit means salary (see "
            "docs/classifier_vs_rules.md).",
            [cell("f1", W4, f"{s2}.labels.is_salary.model.f1"),
             cell("f1", W4, f"{s2}.labels.is_salary.rules.f1")],
        ),
    ]  # fmt: skip


CLOSING = (
    "All of these results are on synthetic statements from this project's own generator, "
    "and no real statement has been evaluated."
)


# ------------------------------------------------------------------ rendering


def render_table(t: Table) -> str:
    lines = [t.caption, "", "| " + " | ".join(t.header) + " |",
             "|" + "---|" * len(t.header)]  # fmt: skip
    for row in t.rows:
        lines.append("| " + " | ".join(c if isinstance(c, str) else c.text() for c in row) + " |")
    if t.note:
        lines += ["", t.note]
    return "\n".join(lines)


GENERATED = "<!-- Generated by scripts/results_tables.py from reports/. Do not edit. -->"


def build_block() -> str:
    parts = [START, GENERATED, ""]
    for t in tables():
        parts += [render_table(t), ""]
    parts += ["What these show, and what they do not:", ""]
    parts += [f"- {s.text()}" for s in sentences()]
    parts += [f"- {CLOSING}", "", END]
    return "\n".join(parts)


def write_readme() -> None:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit(f"README.md must contain {START} and {END}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    README.write_text(before + build_block() + after, encoding="utf-8")
    print("README.md: results block written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="replace the block in README.md")
    args = parser.parse_args()
    if args.write:
        write_readme()
    else:
        print(build_block())


if __name__ == "__main__":
    main()
