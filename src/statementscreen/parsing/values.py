"""Turn printed text into exact values: Indian-format amounts and dates.

Money goes text -> Decimal -> integer paise. It never passes through float, because
floats can't hold most paise values exactly (0.1 + 0.2 != 0.3).

Every parser here is strict: text that doesn't match the expected format raises
ValueError instead of being guessed at.
"""

import re
from datetime import date
from decimal import Decimal

# Indian grouping: the last 3 digits, then pairs. 999.50 / 1,254.00 / 25,453.00 /
# 1,25,000.00 / 12,34,567.80. Always two decimals. An optional leading minus.
_INDIAN_AMOUNT = re.compile(r"-?(?:\d{1,2},(?:\d{2},)*\d{3}|\d{1,3})\.\d{2}")
_DMY = re.compile(r"(\d{2})/(\d{2})/(\d{4})")  # 01/01/2025
_DMY_SHORT = re.compile(r"(\d{2})/(\d{2})/(\d{2})")  # 01/01/25 (Value Dt)


def parse_amount(text: str | None) -> Decimal | None:
    """'1,25,000.00' -> Decimal('125000.00'). A blank cell gives None."""
    if text is None or not text.strip():
        return None
    text = text.strip()
    if not _INDIAN_AMOUNT.fullmatch(text):
        raise ValueError(f"not an Indian-format amount: {text!r}")
    return Decimal(text.replace(",", ""))


def is_amount(text: str) -> bool:
    return bool(_INDIAN_AMOUNT.fullmatch(text.strip()))


def to_paise(amount: Decimal) -> int:
    """Exact Decimal rupees -> integer paise. Refuses fractions of a paisa."""
    value = amount * 100
    if value != value.to_integral_value():
        raise ValueError(f"{amount} has fractions of a paisa")
    return int(value)


def parse_dmy(text: str) -> date:
    """'01/01/2025' -> date(2025, 1, 1). Rejects anything else, including 31/02/2025."""
    m = _DMY.fullmatch(text.strip())
    if not m:
        raise ValueError(f"not a DD/MM/YYYY date: {text!r}")
    day, month, year = (int(g) for g in m.groups())
    return date(year, month, day)  # raises ValueError for impossible dates


def is_dmy(text: str) -> bool:
    try:
        parse_dmy(text)
    except ValueError:
        return False
    return True


_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTHS = {name: i for i, name in enumerate(_MONTH_NAMES, start=1)}
_DMONY = re.compile(r"(\d{2})-([A-Z][a-z]{2})-(\d{2})")  # 01-Feb-24
_SIGNED = re.compile(r"(\S+) (Dr|Cr)")  # 25,453.00 Dr


def parse_dmony(text: str, period: tuple[date, date] | None = None) -> date:
    """'01-Feb-24' -> a date. Month names are matched explicitly (not via the locale).

    A two-digit year is ambiguous, so the statement period settles the century: take the
    candidate inside the period, else the one closest to it. Without a period (e.g. when
    reading the period line itself), years 00-69 are 20xx and 70-99 are 19xx.
    """
    m = _DMONY.fullmatch(text.strip())
    if not m or m.group(2) not in _MONTHS:
        raise ValueError(f"not a DD-Mon-YY date: {text!r}")
    day, month, yy = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
    # Candidate centuries: those the period spans (or 1900/2000 without a period).
    centuries = sorted({d.year // 100 * 100 for d in period}) if period else [1900, 2000]
    candidates = []
    for century in centuries:
        try:
            candidates.append(date(century + yy, month, day))
        except ValueError:
            pass  # e.g. 29-Feb in a non-leap century
    if not candidates:
        raise ValueError(f"impossible date: {text!r}")
    if period is None:
        pivot = 2000 + yy if yy < 70 else 1900 + yy
        return next((d for d in candidates if d.year == pivot), candidates[-1])
    start, end = period
    inside = [d for d in candidates if start <= d <= end]
    if inside:
        return inside[0]
    return min(candidates, key=lambda d: min(abs((d - start).days), abs((d - end).days)))


def is_dmony(text: str) -> bool:
    m = _DMONY.fullmatch(text.strip())
    return bool(m) and m.group(2) in _MONTHS


def parse_signed_amount(text: str) -> Decimal:
    """'25,453.00 Dr' -> Decimal('-25453.00'); '1,61,030.55 Cr' -> Decimal('161030.55').

    Dr (money out, or an overdrawn balance) is negative; Cr is positive. The marker is
    required: an amount without one is rejected rather than assumed to be either.
    """
    m = _SIGNED.fullmatch(text.strip())
    if not m:
        raise ValueError(f"expected an amount followed by Dr or Cr: {text!r}")
    amount = parse_amount(m.group(1))
    assert amount is not None  # the regex guarantees a non-blank amount
    return -amount if m.group(2) == "Dr" else amount


def parse_dmy_short(text: str, near: date) -> date:
    """'01/01/25' -> a date, choosing the century that lands closest to ``near``.

    Two-digit years are ambiguous on their own; a value date sits within days of its
    posting date, so the posting date settles the century.
    """
    m = _DMY_SHORT.fullmatch(text.strip())
    if not m:
        raise ValueError(f"not a DD/MM/YY date: {text!r}")
    day, month, yy = (int(g) for g in m.groups())
    candidates = []
    for century in (1900, 2000, 2100):
        try:
            candidates.append(date(century + yy, month, day))
        except ValueError:
            pass
    if not candidates:
        raise ValueError(f"impossible date: {text!r}")
    return min(candidates, key=lambda d: abs((d - near).days))
