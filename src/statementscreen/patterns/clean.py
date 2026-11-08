"""Clean parsed narrations into channel, direction, counterparty and purpose.

Works on the parser output only (narration and amount), one statement at a time.

Each narration is matched against known styles. A style is an opening (e.g. "UPI/DR/")
plus the roles of the fields that follow, so the counterparty is picked by position, never
by guessing. A style only counts if its fields also look right (a reference is digits,
an IFSC-style code is AAAA0XXXXXX, ...). Otherwise:

- known opening, broken fields  -> style "<name>?", channel kept, generic counterparty;
- nothing matches               -> style "unknown", channel from a keyword, generic
                                   counterparty (the longest run of plain words).

The account holder's own name never becomes a counterparty. The parser doesn't extract
it from the header, so it is inferred from the rows (resolve_self):
  1. the holder slot of "NEFT CR-" / "RTGS CR-" narrations is dropped by position, and
     its name becomes a holder candidate;
  2. rows whose purpose says "self" / "self transfer" (or a cut prefix of it) become
     counterparty "self", and their name becomes a candidate too;
  3. any other transfer row whose counterparty matches a candidate becomes "self".
"""

import re
from dataclasses import dataclass, replace

import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA

from statementscreen.patterns.config import DEFAULT, PatternConfig

CHANNELS = ("UPI", "NEFT", "RTGS", "IMPS", "NACH", "BBPS", "ATM", "CARD", "INTEREST",
            "CHARGE", "REVERSAL", "OTHER")  # fmt: skip
SELF = "self"

# ---------------------------------------------------------------- field roles

# Coded roles: full pattern, and the pattern a field cut short at the end may still match.
_CODED = {
    "num": (r"\d+", r"\d*"),
    "card": (r"\d{4}", r"\d{0,4}"),
    "ifsc": (r"[A-Z]{4}0[A-Z0-9]{6}", r"[A-Z0-9]*"),
    "utr": (r"[A-Z]{4}[NR]\d{6,}", r"[A-Z0-9]*"),
    "umrn": (r"[A-Z]{4}\d{6,}", r"[A-Z0-9]*"),
    "term": (r"[A-Z]{4}\d{4}", r"[A-Z0-9]*"),
    "bank": (r"[A-Z]{4}", r"[A-Z]{0,4}"),
    "vpa": (r"\S+@\S+", r"\S*"),
    "netbank": (r"NETBANK", r"N?E?T?B?A?N?K?"),
}
_PURPOSE_ROLES = ("note", "purpose", "reason")
# "name", "holder", "bankname", "place" and the purpose roles are free text.


@dataclass(frozen=True)
class Style:
    name: str
    channel: str
    opening: str  # regex, matched at the start, case-insensitive
    sep: str | None = None  # field separator after the opening; None = no fields used
    roles: tuple[str, ...] = ()
    fixed: str | None = None  # counterparty for styles without one (ATM, interest, ...)


STYLES = (
    # UPI and UPI refunds
    Style("upi_rev_slash", "REVERSAL", r"UPI/REV/", "/", ("num", "name", "note")),
    Style("upi_rev_dash", "REVERSAL", r"REV-UPI-", "-", ("num", "name")),
    Style("upi_slash", "UPI", r"UPI/(?:DR|CR)/", "/", ("num", "name", "bank", "vpa", "note")),
    Style("upi_dash", "UPI", r"UPI-", "-", ("name", "vpa", "ifsc", "num", "note")),
    # bank transfers
    Style("neft_cr_dash", "NEFT", r"NEFT CR-", "-", ("ifsc", "name", "holder", "utr_note")),
    Style("neft_dr_dash", "NEFT", r"NEFT DR-", "-", ("ifsc", "name", "netbank", "utr", "note")),
    Style("neft_slash", "NEFT", r"NEFT/(?:CR|DR)/", "/", ("utr", "name", "ifsc", "note")),
    Style("rtgs_cr_dash", "RTGS", r"RTGS CR-", "-", ("ifsc", "name", "holder", "utr_note")),
    Style("rtgs_slash", "RTGS", r"RTGS/(?:CR|DR)/", "/", ("utr", "name", "ifsc", "note")),
    Style("imps_mmt", "IMPS", r"MMT/IMPS/", "/", ("num", "purpose", "name", "bankname")),
    Style("imps_p2a", "IMPS", r"IMPS/P2A/", "/", ("num", "name", "bank", "note")),
    Style("imps_dash", "IMPS", r"IMPS-", "-", ("num", "name", "bank", "note")),
    # EMIs, their returns and the charges for returns
    Style("nach_slash", "NACH", r"NACH/DR/", "/", ("umrn", "name")),
    Style("ach_dash", "NACH", r"ACH D-\s*", "-", ("name", "umrn")),
    Style("ach_rtn_chrg", "CHARGE", r"ACH RTN CHRG-", "-", ("name", "reason")),
    Style("nach_rtn_chrgs", "CHARGE", r"NACH RTN CHRGS/", "/", ("name", "reason")),
    Style("gst_on_charge", "CHARGE", r"(?:CGST\+SGST|GST @\d+%) ON ", fixed=""),
    Style("nach_rtn", "REVERSAL", r"NACH/RTN/", "/", ("umrn", "name", "reason")),
    # bills, cash, cards
    Style("bbps", "BBPS", r"BBPS/", "/", ("num", "name")),
    Style("bil_bpay", "BBPS", r"BIL/BPAY/", "/", ("num", "name")),
    Style("atm_wdl", "ATM", r"ATM/WDL/", "/", ("card", "term", "place"), fixed="atm"),
    Style("atw", "ATM", r"ATW-", "-", ("card", "term", "place"), fixed="atm"),
    Style("pos", "CARD", r"POS/", "/", ("card", "name", "num")),
    Style("pcd", "CARD", r"PCD/", "/", ("card", "name", "note")),
    # interest and tax
    Style("int_capitalised", "INTEREST", r"CREDIT INTEREST CAPITALI[SZ]ED", fixed="interest"),
    Style("int_pd", "INTEREST", r"INT\.PD:", fixed="interest"),
    Style("gst_pmt", "OTHER", r"GST PMT/", fixed="gst payment"),
    Style("advance_tax", "OTHER", r"TAX/ADVANCE TAX", fixed="advance tax"),
)
_OPENINGS = [(s, re.compile(s.opening, re.IGNORECASE)) for s in STYLES]

# ---------------------------------------------------------------- name cleaning

_IFSC = re.compile(r"^[a-z]{4}0[a-z0-9]{6}$")
_UTR = re.compile(r"^[a-z]{4}[nr]\d{6,}$")
_MONTH_TAG = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'?\d{2,4}$")
_DATE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$|^\d{6,8}$")
_FILLER = {"salary", "sal", "slry", "sal.", "wages", "m/s", "ms"}
_LEGAL = {"pvt", "private", "ltd", "limited", "llp", "pvtltd"}
_CHANNEL_WORDS = {"upi", "neft", "imps", "rtgs", "nach", "ach", "mmt", "p2a", "dr", "cr", "d",
                  "rev", "inb", "ib", "trf", "transfer", "netbank", "bpay", "bil", "bbps",
                  "pos", "pcd", "atm", "atw", "wdl", "ref", "txn", "payment", "pay"}  # fmt: skip


def _is_clutter(tok: str) -> bool:
    return (
        "@" in tok
        or sum(c.isdigit() for c in tok) >= 4
        or bool(_IFSC.match(tok) or _UTR.match(tok) or _MONTH_TAG.match(tok) or _DATE.match(tok))
        or tok in _FILLER
    )


_LEGAL_LONG = ("limited", "private")


def _is_legal(tok: str) -> bool:
    """A legal-form word, or a misspelling of a long one ("limtied", "privte", "limted"):
    at least 5 letters and one edit away, where swapping two letters counts as one edit
    (OSA distance). Real words such as "limits", "limestone" or "pirate" are kept."""
    if tok in _LEGAL:
        return True
    return len(tok) >= 5 and tok.isalpha() and min(OSA.distance(tok, w) for w in _LEGAL_LONG) <= 1


def clean_name(text: str, cut: bool = False) -> str:
    """Lowercase name text without references, handles, codes, dates, filler or legal form.

    cut=True: the text was cut short, so a last word that could be the start of a legal
    word ("PVT L", "P") is dropped too.
    """
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[.,()'\"]", " ", text)
    tokens = [t for t in text.split() if not _is_clutter(t)]
    out: list[str] = []
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if _is_legal(tok) or (tok == "p" and (nxt == "ltd" or _is_legal(nxt))):
            continue
        out.append(tok)
    while cut and out and any(word.startswith(out[-1]) for word in _LEGAL):
        out.pop()
    return " ".join(out)


def _clean_purpose(text: str) -> str:
    text = " ".join(text.lower().split())
    return "" if text in ("upi", "pay", "payment") else text


def is_self_purpose(purpose: str) -> bool:
    """'self', 'self transfer', or 'self tra'-style cut prefixes of 'self transfer'."""
    return purpose.startswith("self") or (len(purpose) >= 4 and "self transfer".startswith(purpose))


# ---------------------------------------------------------------- one narration


@dataclass(frozen=True)
class Cleaned:
    channel: str
    style: str
    counterparty: str
    purpose: str = ""
    cut: bool = False  # the narration ends before all of its style's fields
    counterparty_cut: bool = False  # ... and the counterparty is the last, partial field
    holder: str = ""  # name in the holder slot, when that slot is complete


def _valid(role: str, value: str, partial: bool) -> bool:
    if role == "utr_note":
        role, value = "utr", (value.split() or [""])[0]
    if role not in _CODED:
        return True
    full, cut = _CODED[role]
    compact = value.replace(" ", "")
    return bool(re.fullmatch(cut if partial else full, compact, re.IGNORECASE))


def _match_style(style: Style, rest: str) -> Cleaned | None:
    """Fields after the opening -> Cleaned, or None if a field doesn't look right."""
    if style.sep is None:
        return Cleaned(style.channel, style.name, style.fixed or "")
    fields = [f.strip() for f in rest.split(style.sep, len(style.roles) - 1)]
    if fields and fields[-1] == "":
        fields.pop()  # cut right after a separator
    cut = len(fields) < len(style.roles)
    for i, (role, value) in enumerate(zip(style.roles, fields, strict=False)):
        if not _valid(role, value, partial=cut and i == len(fields) - 1):
            return None
    by_role = dict(zip(style.roles, fields, strict=False))
    purpose_parts = [by_role[r] for r in _PURPOSE_ROLES if r in by_role]
    if "utr_note" in by_role:
        purpose_parts.append(" ".join(by_role["utr_note"].split()[1:]))
    name_at = style.roles.index("name") if "name" in style.roles else None
    name_cut = cut and name_at is not None and name_at == len(fields) - 1
    holder = ""
    if "holder" in by_role and style.roles.index("holder") < len(fields) - 1:
        holder = clean_name(by_role["holder"])  # complete: something follows it
    if style.fixed is not None:
        counterparty = style.fixed
    else:
        counterparty = clean_name(by_role.get("name", ""), cut=name_cut)
    return Cleaned(
        channel=style.channel,
        style=style.name,
        counterparty=counterparty,
        purpose=_clean_purpose(" ".join(p for p in purpose_parts if p)),
        cut=cut,
        counterparty_cut=name_cut,
        holder=holder,
    )


_KEYWORD_CHANNELS = (
    (r"CHRG|CHARGE", "CHARGE"),
    (r"\bRTN\b|\bREV\b|RETURN|REFUND|REVERS", "REVERSAL"),
    (r"\bUPI\b", "UPI"),
    (r"\bNEFT\b", "NEFT"),
    (r"\bRTGS\b", "RTGS"),
    (r"\bIMPS\b", "IMPS"),
    (r"\bN?ACH\b", "NACH"),
    (r"\bBBPS\b|\bBPAY\b", "BBPS"),
    (r"\bATM\b|\bATW\b", "ATM"),
    (r"\bPOS\b|\bPCD\b", "CARD"),
    (r"\bINT\b|INTEREST", "INTEREST"),
)


def keyword_channel(narration: str) -> str:
    spaced = re.sub(r"[/\-:.]", " ", narration.upper())
    for pattern, channel in _KEYWORD_CHANNELS:
        if re.search(pattern, spaced):
            return channel
    return "OTHER"


def generic_counterparty(narration: str) -> str:
    """The longest run of plain words, once references, codes and channel words are gone."""
    best = ""
    for segment in re.split(r"[/\-:]+", narration):
        words = [w for w in clean_name(segment).split() if w not in _CHANNEL_WORDS]
        candidate = " ".join(w for w in words if w.isalpha())
        if len(candidate.split()) > len(best.split()):
            best = candidate
    return best


def clean_narration(narration: str) -> Cleaned:
    text = narration.strip()
    for style, opening in _OPENINGS:
        m = opening.match(text)
        if not m:
            continue
        rest = text[m.end() :]
        result = _match_style(style, rest)
        if result is not None:
            return result
        return Cleaned(style.channel, f"{style.name}?", generic_counterparty(rest))
    return Cleaned(keyword_channel(text), "unknown", generic_counterparty(text))


# ---------------------------------------------------------------- one statement

_TRANSFER_CHANNELS = {"UPI", "NEFT", "RTGS", "IMPS", "OTHER"}


def _same_person(name: str, candidate: str, name_cut: bool, cfg: PatternConfig) -> bool:
    if not name or not candidate:
        return False
    if fuzz.token_sort_ratio(name, candidate) >= cfg.holder_match:
        return True
    return name_cut and len(name) >= cfg.min_prefix and candidate.startswith(name)


def resolve_self(
    cleaned: list[Cleaned], cfg: PatternConfig = DEFAULT
) -> list[tuple[Cleaned, bool]]:
    """Replace the holder's own name with "self". Returns (cleaned, is_self) per row."""
    candidates = {c.holder for c in cleaned if c.holder}
    candidates |= {
        c.counterparty
        for c in cleaned
        if is_self_purpose(c.purpose) and c.counterparty and not c.counterparty_cut
    }
    out = []
    for c in cleaned:
        is_self = is_self_purpose(c.purpose) or (
            c.channel in _TRANSFER_CHANNELS
            and any(_same_person(c.counterparty, h, c.counterparty_cut, cfg) for h in candidates)
        )
        out.append((replace(c, counterparty=SELF) if is_self else c, is_self))
    return out


SCHEMA = {
    "statement_id": pl.String, "row": pl.Int64, "posted": pl.Date, "amount_paise": pl.Int64,
    "direction": pl.String, "channel": pl.String, "style": pl.String,
    "counterparty": pl.String, "purpose": pl.String, "is_self": pl.Boolean,
    "cut": pl.Boolean, "counterparty_cut": pl.Boolean,
}  # fmt: skip


def clean_statement(rows: pl.DataFrame, cfg: PatternConfig = DEFAULT) -> pl.DataFrame:
    """One statement's parsed rows -> one cleaned row each (same order)."""
    rows = rows.sort("row")
    resolved = resolve_self([clean_narration(n) for n in rows["narration"]], cfg)
    return pl.DataFrame({
        "statement_id": rows["statement_id"],
        "row": rows["row"],
        "posted": rows["posted"],
        "amount_paise": rows["amount_paise"],
        "direction": ["credit" if a > 0 else "debit" for a in rows["amount_paise"]],
        "channel": [c.channel for c, _ in resolved],
        "style": [c.style for c, _ in resolved],
        "counterparty": [c.counterparty for c, _ in resolved],
        "purpose": [c.purpose for c, _ in resolved],
        "is_self": [s for _, s in resolved],
        "cut": [c.cut for c, _ in resolved],
        "counterparty_cut": [c.counterparty_cut for c, _ in resolved],
    }, schema=SCHEMA)  # fmt: skip
