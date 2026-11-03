"""Fictional Indian-sounding banks with fictional IFSC-style codes.

None of these names or 4-letter codes belongs to a real bank. How that was checked
(2026-11-03):
- Each 4-letter code and each name's distinctive word was compared with all 1,511
  banks in Razorpay's public IFSC dataset (src/banknames.json, MIT licence): no code
  match, no shared word, no close spelling.
- Each name was web-searched: no bank found.
- Dropped as too close to real banks: Suvarnapath (Suvarna Sahakari Bank),
  Neelkosh (Neelkanth Co-op Bank), Meghdhara (Meghna Bank, Bangladesh),
  Aranyaka (Shri Kanyaka Nagari Sahakari Bank).
- Not fully checkable: small co-operative banks without NEFT access may be missing
  from the IFSC dataset.

A real IFSC is 4 letters (bank) + "0" + 6 characters (branch); we use the same shape.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FictionalBank:
    name: str
    code: str  # 4-letter IFSC-style bank code
    upi_handle: str  # the part after "@" in a UPI ID


FICTIONAL_BANKS: tuple[FictionalBank, ...] = (
    FictionalBank("Vistarika Bank", "VSTK", "vistarika"),
    FictionalBank("Kesarvan Bank", "KSVB", "kesarvan"),
    FictionalBank("Tarangam Bank", "TRGM", "tarangam"),
    FictionalBank("Pravahini Bank", "PRVH", "pravahini"),
)
BANK_NAMES = frozenset(b.name for b in FICTIONAL_BANKS)
BANK_CODES = frozenset(b.code for b in FICTIONAL_BANKS)
UPI_HANDLES = frozenset(b.upi_handle for b in FICTIONAL_BANKS)


def random_bank(rng: np.random.Generator) -> FictionalBank:
    return FICTIONAL_BANKS[int(rng.integers(len(FICTIONAL_BANKS)))]


def random_ifsc(rng: np.random.Generator, bank: FictionalBank) -> str:
    """e.g. VSTK0004821: bank code, a literal 0, then a 6-digit branch number."""
    return f"{bank.code}0{int(rng.integers(0, 1_000_000)):06d}"
