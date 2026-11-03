"""Turn clean transaction facts into messy, bank-style description strings.

Real banks don't print "GROCERY MART"; they print things like
``POS DEBIT 4821 FRESHWAY MARKET SPRINGFIELD 09/14``. Each statement picks one
*style* (like a bank's house format) and uses it for every line, so the text is
consistent within a statement but varies across statements.

All merchant names below are invented.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

# Invented merchants per spending category.
MERCHANTS: dict[str, tuple[str, ...]] = {
    "groceries": ("FRESHWAY MARKET", "GREENLEAF GROCERS", "VALUBASKET FOODS"),
    "dining": ("CORNER CAFE", "BLUE FERN BISTRO", "NOODLE NOOK"),
    "transport": ("FUEL STOP", "QUIKGAS 112", "METRO TRANSIT"),
    "shopping": ("ONLINE SHOP", "PARCELPOINT RETAIL", "HOMEGOODS DEPOT"),
    "health": ("PHARMACY PLUS", "WELLCARE CLINIC"),
}

CITIES = ("SPRINGFIELD", "RIVERTON", "MAPLE GROVE", "LAKESIDE", "FAIRVIEW")


def _digits(rng: np.random.Generator, n: int) -> str:
    return "".join(str(int(d)) for d in rng.integers(0, 10, size=n))


@dataclass(frozen=True)
class DescriptionStyle:
    """One bank's house format. Pick with ``random_style``."""

    name: str

    def card(self, rng: np.random.Generator, merchant: str, city: str, d: date) -> str:
        if self.name == "pos":
            return f"POS DEBIT {_digits(rng, 4)} {merchant} {city} {d:%m/%d}"
        if self.name == "visa":
            return f"VISA DDA PUR {_digits(rng, 6)} {merchant} {city}"
        return f"DEBIT CARD PURCHASE - {merchant} {d:%m%d}"

    def refund(self, rng: np.random.Generator, merchant: str, d: date) -> str:
        if self.name == "pos":
            return f"POS RETURN {_digits(rng, 4)} {merchant} {d:%m/%d}"
        if self.name == "visa":
            return f"VISA DDA REFUND {merchant}"
        return f"DEBIT CARD CREDIT - {merchant} {d:%m%d}"

    def payroll(self, rng: np.random.Generator, employer: str) -> str:
        if self.name == "pos":
            return f"ACH CREDIT {employer} PAYROLL PPD ID:{_digits(rng, 10)}"
        if self.name == "visa":
            return f"DIRECT DEP {employer}"
        return f"{employer} DIR DEP {_digits(rng, 6)}"

    def bill(self, rng: np.random.Generator, payee: str) -> str:
        if self.name == "pos":
            return f"ACH DEBIT {payee} WEB ID:{_digits(rng, 10)}"
        if self.name == "visa":
            return f"AUTOPAY {payee}"
        return f"ONLINE PMT {payee} CONF#{_digits(rng, 8)}"

    def transfer_in(self, rng: np.random.Generator, sender: str) -> str:
        if self.name == "pos":
            return f"P2P CREDIT FROM {sender} REF {_digits(rng, 8)}"
        if self.name == "visa":
            return f"TRANSFER FROM {sender}"
        return f"INSTANT PMT RECVD {sender}"

    def platform_payout(self, rng: np.random.Generator, platform: str) -> str:
        """Weekly payout from a gig platform (delivery, rides, tasks)."""
        if self.name == "pos":
            return f"ACH CREDIT {platform} PAYOUT PPD ID:{_digits(rng, 10)}"
        if self.name == "visa":
            return f"DIRECT DEP {platform} PAYOUT"
        return f"{platform} INSTANT PAYOUT {_digits(rng, 6)}"

    def client_payment(self, rng: np.random.Generator, client: str) -> str:
        """A client paying a self-employed person's invoice."""
        if self.name == "pos":
            return f"ACH CREDIT {client} PMT REF {_digits(rng, 8)}"
        if self.name == "visa":
            return f"WIRE IN {client}"
        return f"ONLINE TRANSFER FROM {client} INV{_digits(rng, 5)}"

    def returned_debit(self, rng: np.random.Generator, payee: str) -> str:
        """The bank sending back a payment that couldn't be covered."""
        if self.name == "pos":
            return f"ACH RETURN {payee} R01 NSF"
        if self.name == "visa":
            return f"RETURNED ITEM {payee}"
        return f"REVERSAL - {payee} INSUFFICIENT FUNDS"

    def fee(self, name: str) -> str:
        return {"pos": f"FEE {name}", "visa": f"SERVICE CHARGE {name}"}.get(self.name, name)

    def atm(self, rng: np.random.Generator, city: str) -> str:
        if self.name == "pos":
            return f"ATM WITHDRAWAL {_digits(rng, 4)} {city}"
        if self.name == "visa":
            return f"ATM W/D {city} #{_digits(rng, 5)}"
        return f"CASH WITHDRAWAL - ATM {city}"


STYLES = ("pos", "visa", "plain")


def random_style(rng: np.random.Generator) -> DescriptionStyle:
    return DescriptionStyle(str(rng.choice(STYLES)))
