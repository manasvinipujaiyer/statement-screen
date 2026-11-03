"""Indian bank-statement narrations, with per-statement mess.

Real narrations look like ``UPI/DR/412398765012/SHARMA KIRANA STORE/TRGM/...`` or
``NEFT CR-KSVB0004821-RAVENIKA INFOTECH PVT LTD-ANANYA IYER-KSVBN25032004811``. Each
statement gets one ``Narrator`` whose quirks are drawn from the statement's seed:

- ``style``:   "slash" (``UPI/DR/...``) or "dash" (``UPI-...-...``), like a bank's house format
- ``case``:    all upper case, or names in Title Case
- ``max_len``: some banks cut narrations at a fixed width; others don't
- ``typo_rate``: how often a name is misspelt (swapped, dropped or doubled letter)
- ``sal_word``: whether salary credits say "SALARY", "SAL", or nothing at all
- ``ltd_style``: how "PVT LTD" is written (PVT LTD / PRIVATE LIMITED / P LTD / PVT. LTD.)

Only names get typos; IFSC codes, reference numbers and channel keywords stay exact.
All counterparty banks, IFSC codes and UPI handles are fictional (see ``banks.py``).
"""

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from statementscreen.synth.banks import FictionalBank, random_bank, random_ifsc

_LTD_STYLES = ("PVT LTD", "PRIVATE LIMITED", "P LTD", "PVT. LTD.")
_TECHNICAL_RETURN_REASONS = ("MANDATE NOT REGISTERED", "PAYMENT STOPPED", "A/C BLOCKED")


def _digits(rng: np.random.Generator, n: int) -> str:
    return "".join(str(int(d)) for d in rng.integers(0, 10, size=n))


def typo(rng: np.random.Generator, name: str) -> str:
    """Misspell one word of length >= 4 by swapping, dropping or doubling a letter."""
    words = name.split(" ")
    long_words = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not long_words:
        return name
    i = long_words[int(rng.integers(len(long_words)))]
    w = words[i]
    j = int(rng.integers(1, len(w) - 1))
    op = int(rng.integers(3))
    if op == 0:
        w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    elif op == 1:
        w = w[:j] + w[j + 1 :]
    else:
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def spelling_variants(name: str) -> list[str]:
    """Other ways the same organisation's name gets written (sorted, so the order is fixed)."""
    variants = {name}
    if name.endswith(" PVT LTD"):
        base = name.removesuffix(" PVT LTD")
        variants |= {base, f"{base} PRIVATE LIMITED", f"{base} P LTD"}
    for full, short in (("TECHNOLOGIES", "TECH"), ("SOLUTIONS", "SOLN"), (" & ", " AND "),
                        ("ENGINEERING", "ENGG"), ("CONSULTANCY", "CONSULTANTS")):  # fmt: skip
        variants |= {v.replace(full, short) for v in variants}
    return sorted(variants)


def pay_month_label(posted: date) -> str:
    """Salary paid early in a month is for the previous month: credited 3 Feb -> 'JAN25'."""
    year, month = posted.year, posted.month
    if posted.day <= 10:
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return f"{date(year, month, 1):%b%y}".upper()


@dataclass
class Narrator:
    rng: np.random.Generator
    own_bank: FictionalBank
    holder: str
    style: str
    case: str
    max_len: int | None
    typo_rate: float
    sal_word: str | None
    ltd_style: str
    card4: str
    _accounts: dict[str, tuple[FictionalBank, str, str]] = field(default_factory=dict)

    @classmethod
    def random(cls, rng: np.random.Generator, own_bank: FictionalBank, holder: str) -> "Narrator":
        return cls(
            rng=rng,
            own_bank=own_bank,
            holder=holder,
            style=str(rng.choice(["slash", "dash"])),
            case=str(rng.choice(["upper", "title_names"], p=[0.7, 0.3])),
            max_len=int(rng.integers(38, 61)) if rng.random() < 0.5 else None,
            typo_rate=float(rng.uniform(0.0, 0.04)),
            sal_word=[None, "SALARY", "SAL"][int(rng.integers(3))],
            ltd_style=str(rng.choice(_LTD_STYLES)),
            card4=_digits(rng, 4),
        )

    # ------------------------------------------------------------------ helpers

    def name(self, name: str) -> str:
        """Display a counterparty name with this statement's quirks applied."""
        name = name.replace(" PVT LTD", f" {self.ltd_style}")
        if self.rng.random() < self.typo_rate:
            name = typo(self.rng, name)
        return name.title() if self.case == "title_names" else name.upper()

    def _finish(self, text: str) -> str:
        if self.max_len is not None and len(text) > self.max_len:
            text = text[: self.max_len].rstrip("-/ ")
        return text

    def _rrn(self) -> str:
        return _digits(self.rng, 12)

    def _utr(self, bank: FictionalBank, d: date) -> str:
        return f"{bank.code}N{d:%y}{d.timetuple().tm_yday:03d}{_digits(self.rng, 6)}"

    def _account(self, name: str) -> tuple[FictionalBank, str, str]:
        """(bank, UPI ID, IFSC) for a counterparty. Fixed per counterparty: an employer pays
        from one account, a shop has one UPI ID."""
        if name not in self._accounts:
            bank = random_bank(self.rng)
            slug = re.sub(r"[^a-z]", "", name.lower())[:10] or "pay"
            vpa = f"{slug}{_digits(self.rng, 2)}@{bank.upi_handle}"
            self._accounts[name] = (bank, vpa, random_ifsc(self.rng, bank))
        return self._accounts[name]

    def _credit_parts(self, remitter: str, d: date) -> tuple[FictionalBank, str, str]:
        bank, _, ifsc = self._account(remitter)
        return bank, ifsc, self._utr(bank, d)

    # ------------------------------------------------------------------ credits

    def neft_credit(self, remitter: str, d: date, note: str | None = None) -> str:
        bank, ifsc, utr = self._credit_parts(remitter, d)
        who = self.name(remitter)
        tail = f" {note}" if note else ""
        if self.style == "slash":
            return self._finish(f"NEFT/CR/{utr}/{who}/{ifsc}{('/' + note) if note else ''}")
        return self._finish(f"NEFT CR-{ifsc}-{who}-{self.name(self.holder)}-{utr}{tail}")

    def rtgs_credit(self, remitter: str, d: date) -> str:
        bank, ifsc, utr = self._credit_parts(remitter, d)
        utr = utr.replace("N", "R", 1)
        if self.style == "slash":
            return self._finish(f"RTGS/CR/{utr}/{self.name(remitter)}/{ifsc}")
        return self._finish(f"RTGS CR-{ifsc}-{self.name(remitter)}-{self.name(self.holder)}-{utr}")

    def imps_credit(self, sender: str, note: str = "") -> str:
        bank = self._account(sender)[0]
        if self.style == "slash":
            return self._finish(f"IMPS/P2A/{self._rrn()}/{self.name(sender)}/{bank.code}/{note}")
        return self._finish(
            f"MMT/IMPS/{self._rrn()}/{note or 'TRANSFER'}/{self.name(sender)}/{bank.name.upper()}"
        )

    def upi_credit(self, sender: str, note: str = "UPI") -> str:
        bank, vpa, ifsc = self._account(sender)
        if self.style == "slash":
            return self._finish(
                f"UPI/CR/{self._rrn()}/{self.name(sender)}/{bank.code}/{vpa}/{note}"
            )
        return self._finish(f"UPI-{self.name(sender)}-{vpa}-{ifsc}-{self._rrn()}-{note}")

    def salary(self, employer: str, d: date, note: str | None = None) -> str:
        """Salary arrives as a NEFT credit; whether it says SALARY depends on the employer/bank."""
        word = note or self.sal_word
        return self.neft_credit(employer, d, f"{word} {pay_month_label(d)}" if word else None)

    def interest(self, start: date, end: date) -> str:
        acct = self.card4
        if self.style == "slash":
            return f"INT.PD:{acct}:{start:%d-%m-%Y} TO {end:%d-%m-%Y}"
        return f"CREDIT INTEREST CAPITALISED {start:%d%m%y}-{end:%d%m%y}"

    def refund(self, merchant: str) -> str:
        if self.style == "slash":
            return self._finish(f"UPI/REV/{self._rrn()}/{self.name(merchant)}/REFUND")
        return self._finish(f"REV-UPI-{self._rrn()}-{self.name(merchant)}")

    # ------------------------------------------------------------------ debits

    def upi_debit(self, payee: str, note: str = "UPI") -> str:
        bank, vpa, ifsc = self._account(payee)
        if self.style == "slash":
            return self._finish(f"UPI/DR/{self._rrn()}/{self.name(payee)}/{bank.code}/{vpa}/{note}")
        return self._finish(f"UPI-{self.name(payee)}-{vpa}-{ifsc}-{self._rrn()}-{note}")

    def imps_debit(self, payee: str, note: str = "") -> str:
        bank = self._account(payee)[0]
        if self.style == "slash":
            return self._finish(
                f"MMT/IMPS/{self._rrn()}/{note}/{self.name(payee)}/{bank.name.upper()}"
            )
        return self._finish(f"IMPS-{self._rrn()}-{self.name(payee)}-{bank.code}-{note}")

    def neft_debit(self, payee: str, d: date, note: str = "") -> str:
        _, _, ifsc = self._account(payee)
        utr = self._utr(self.own_bank, d)
        if self.style == "slash":
            return self._finish(f"NEFT/DR/{utr}/{self.name(payee)}/{ifsc}/{note}")
        return self._finish(f"NEFT DR-{ifsc}-{self.name(payee)}-NETBANK-{utr}-{note}")

    def card(self, merchant: str) -> str:
        if self.style == "slash":
            return self._finish(f"POS/{self.card4}/{self.name(merchant)}/{_digits(self.rng, 6)}")
        return self._finish(f"PCD/{self.card4}/{self.name(merchant)}/ONLINE")

    def bill(self, biller: str) -> str:
        if self.style == "slash":
            return self._finish(f"BBPS/{_digits(self.rng, 10)}/{self.name(biller)}")
        return self._finish(f"BIL/BPAY/{_digits(self.rng, 9)}/{self.name(biller)}")

    def atm(self, city: str) -> str:
        atm_id = f"{self.own_bank.code}{_digits(self.rng, 4)}"
        if self.style == "slash":
            return f"ATM/WDL/{self.card4}/{atm_id}/{city.upper()}"
        return f"ATW-{self.card4}-{atm_id}-{city.upper()}"

    def nach_debit(self, lender: str, umrn: str) -> str:
        if self.style == "slash":
            return self._finish(f"NACH/DR/{umrn}/{self.name(lender)}")
        return self._finish(f"ACH D- {self.name(lender)}-{umrn}")

    def nach_return(self, lender: str, umrn: str, reason: str) -> str:
        if self.style == "slash":
            return self._finish(f"NACH/RTN/{umrn}/{self.name(lender)}/{reason}")
        return self._finish(f"ACH RTN-{self.name(lender)}-{umrn}-{reason}")

    def bounce_charge(self, lender: str, reason: str) -> str:
        if self.style == "slash":
            return self._finish(f"NACH RTN CHRGS/{self.name(lender)}/{reason}")
        return self._finish(f"ACH RTN CHRG-{self.name(lender)}-{reason}")

    def bounce_gst(self) -> str:
        return (
            "GST @18% ON NACH RTN CHRGS" if self.style == "slash" else "CGST+SGST ON ACH RTN CHRG"
        )

    def advance_tax(self) -> str:
        return f"TAX/ADVANCE TAX/CHALLAN {_digits(self.rng, 11)}"

    def gst_payment(self) -> str:
        return f"GST PMT/CPIN {_digits(self.rng, 14)}"

    def technical_return_reason(self) -> str:
        return str(self.rng.choice(_TECHNICAL_RETURN_REASONS))
