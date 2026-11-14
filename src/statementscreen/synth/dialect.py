"""Narration dialects: whole sets of narration formats written outside this codebase.

    uv run python -m statementscreen.generate --n 300 --seed 7 --narration-dialect karan_v1 ...

A dialect file, synth/dialects/<name>.txt, has one format per line:

    kind | CR or DR | format with {placeholders}

for example ``salary | CR | SAL CREDIT {employer} {month} REF {ref}``. Files are used
exactly as written: lines are never edited, filtered or reordered. "default" means no
dialect, which gives today's formats and byte-identical output.

Each kind replaces the narration of one generator event (KINDS). A line whose direction
disagrees with how the generator posts that event is refused, so a dialect can never
make a credit read like a debit. Advance tax and GST payments have no kind and keep
today's narration (UNMAPPED).

How a dialect keeps labels, amounts and balances unchanged: the generator still calls the
default narrator for every transaction, so the statement's main random stream draws
exactly what it draws without a dialect. The dialect text then replaces it, drawing
everything it needs (references, typos, which of a kind's formats this statement uses)
from a separate stream. See Narrator.as_kind in descriptions.py.
"""

import re
import string
from dataclasses import dataclass
from pathlib import Path

DIALECTS_DIR = Path(__file__).parent / "dialects"
DEFAULT = "default"

# kind -> (direction the generator posts it with, the generator event it replaces)
KINDS: dict[str, tuple[str, str]] = {
    "salary": ("CR", "salary credits, and bonus credits (bonus_month)"),
    "emi": ("DR", "the NACH EMI debit"),
    "emi_return": ("CR", "the reversal of a technically bounced EMI, posted as a credit"),
    "emi_return_charge": ("DR", "the INR 500 bounce charge and the INR 90 GST on it"),
    "rent": ("DR", "rent to the landlord, whether paid by UPI, IMPS or NEFT"),
    "upi_spend": ("DR", "UPI spends and prepaid recharges"),
    "upi_credit": ("CR", "money from family by UPI"),
    "atm": ("DR", "ATM cash withdrawals"),
    "bill": ("DR", "electricity, postpaid and broadband bills"),
    "card": ("DR", "subscriptions paid by card"),
    "interest": ("CR", "quarterly savings interest"),
    "refund": ("CR", "merchant refunds"),
    "gig_credit": ("CR", "gig platform payouts"),
    "business_credit": ("CR", "self-employed business income, whatever the channel"),
    "transfer_in": ("CR", "money from family by IMPS, and a self-transfer coming back"),
    "transfer_out": ("DR", "the self-transfer before a shortfall EMI bounce"),
}
UNMAPPED = ("advance tax", "GST payment")  # no kind: these keep today's narration

# Every name slot is filled with the event's counterparty, with the statement's messiness.
NAME_SLOTS = frozenset({"employer", "lender", "payee", "merchant", "biller", "sender"})
PLACEHOLDERS = NAME_SLOTS | {"city", "ref", "utr", "umrn", "ifsc", "vpa", "month"}


class DialectError(ValueError):
    """A dialect name or line that can't be used. Nothing is guessed or skipped."""


@dataclass(frozen=True)
class Format:
    line: int  # 1-based line number in the dialect file
    kind: str
    direction: str  # "CR" or "DR"
    template: str
    fields: tuple[str, ...]  # placeholders, in order of appearance

    def render(self, values: dict[str, str]) -> str:
        return self.template.format(**values)


@dataclass(frozen=True)
class Dialect:
    name: str
    formats: tuple[Format, ...]

    def for_kind(self, kind: str) -> tuple[Format, ...]:
        return tuple(f for f in self.formats if f.kind == kind)

    @property
    def missing_kinds(self) -> tuple[str, ...]:
        """Kinds with no format: those events keep today's narration."""
        present = {f.kind for f in self.formats}
        return tuple(k for k in KINDS if k not in present)


def _placeholders(template: str, where: str) -> tuple[str, ...]:
    try:
        parts = list(string.Formatter().parse(template))
    except ValueError as e:
        raise DialectError(f"{where}: unbalanced braces in {template!r} ({e})") from None
    fields = []
    for _, name, spec, conversion in parts:
        if name is None:
            continue
        if name not in PLACEHOLDERS or spec or conversion:
            allowed = " ".join(f"{{{p}}}" for p in sorted(PLACEHOLDERS))
            raise DialectError(f"{where}: unknown placeholder {{{name}}}; allowed: {allowed}")
        fields.append(name)
    return tuple(fields)


def parse_dialect(text: str, name: str) -> Dialect:
    """Parse a dialect file's text. Blank lines are ignored; any other bad line is refused."""
    formats = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        where = f"{name} line {number}"
        parts = [p.strip() for p in raw.split("|", 2)]
        if len(parts) != 3 or not all(parts):
            raise DialectError(f"{where}: expected 'kind | CR or DR | format', got {raw!r}")
        kind, direction, template = parts
        if kind not in KINDS:
            raise DialectError(f"{where}: unknown kind {kind!r}; known: {', '.join(KINDS)}")
        if direction not in ("CR", "DR"):
            raise DialectError(f"{where}: direction must be CR or DR, got {direction!r}")
        posted, event = KINDS[kind]
        if direction != posted:
            raise DialectError(
                f"{where}: {kind} is marked {direction}, but the generator posts {event} as "
                f"{posted}"
            )
        formats.append(Format(number, kind, direction, template, _placeholders(template, where)))
    if not formats:
        raise DialectError(f"{name}: no formats")
    return Dialect(name, tuple(formats))


def available_dialects(folder: Path = DIALECTS_DIR) -> list[str]:
    return [DEFAULT, *sorted(p.stem for p in folder.glob("*.txt"))]


def load_dialect(name: str, folder: Path = DIALECTS_DIR) -> Dialect | None:
    """'default' -> None (today's formats); otherwise the parsed synth/dialects/<name>.txt."""
    if name == DEFAULT:
        return None
    path = folder / f"{name}.txt"
    if not re.fullmatch(r"[a-z0-9_]+", name) or not path.is_file():
        choices = ", ".join(available_dialects(folder))
        raise DialectError(f"unknown narration dialect {name!r}; choose from {choices}")
    return parse_dialect(path.read_text(encoding="utf-8"), name)
