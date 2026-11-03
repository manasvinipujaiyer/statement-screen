"""Hard cases: deliberately tricky situations injected into a statement.

Each one is something that breaks a naive income model:

- ``salary_date_shift``: one salary credit arrives 3-10 days late.
- ``employer_rename``:   partway through, salary arrives under a new employer name
                         (a rebrand or acquisition), so "same employer?" isn't a string match.
                         The canonical counterparty stays the original employer.
- ``bonus_month``:       one large one-off bonus. Real money, but not regular income.
- ``missed_salary``:     one month with no salary credit at all.
- ``emi_bounce``:        a loan EMI collected by NACH fails, and the bank charges
                         a return fee (INR 500) plus 18% GST on it (INR 90).

The first four only make sense for salaried customers. Every transaction a hard case
creates or changes is tagged in ``Draft.hard_cases``. ``missed_salary`` removes a
transaction, so there's nothing to tag; the missed date goes in ``details`` instead.

All functions take the list of drafts and return a new list; they never edit in place.
"""

from bisect import bisect_left
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from statementscreen.synth.descriptions import Narrator
from statementscreen.synth.models import Draft, IncomeTruth
from statementscreen.synth.util import next_working_day, rupees

HARD_CASES = ("salary_date_shift", "employer_rename", "bonus_month", "missed_salary", "emi_bounce")
SALARIED_ONLY = frozenset({"salary_date_shift", "employer_rename", "bonus_month", "missed_salary"})

BOUNCE_CHARGE = Decimal("500.00")
BOUNCE_GST = Decimal("90.00")  # 18% GST on the charge

# Salary-like entries are re-described when the employer is renamed.
_PAYROLL = ("salary", "bonus")


def applicable(case: str, customer_type: str) -> bool:
    return case not in SALARIED_ONLY or customer_type == "salaried"


def validate(cases: tuple[str, ...], customer_type: str) -> None:
    for case in cases:
        if case not in HARD_CASES:
            raise ValueError(f"unknown hard case {case!r}; choose from {HARD_CASES}")
        if not applicable(case, customer_type):
            raise ValueError(f"{case!r} only applies to salaried customers, not {customer_type!r}")


def _tag(d: Draft, case: str) -> Draft:
    return d._replace(hard_cases=d.hard_cases + (case,))


def _salary_positions(drafts: list[Draft]) -> list[int]:
    """Indexes of salary drafts, in date order."""
    salary = (i for i, d in enumerate(drafts) if d.category == "salary")
    return sorted(salary, key=lambda i: drafts[i].posted)


def missed_salary(
    drafts: list[Draft], rng: np.random.Generator, details: dict[str, str]
) -> list[Draft]:
    pay = _salary_positions(drafts)
    # Drop one from the middle, so it looks like a gap rather than a short statement.
    i = pay[int(rng.integers(1, len(pay) - 1))]
    details["missed_salary"] = drafts[i].posted.isoformat()
    return drafts[:i] + drafts[i + 1 :]


def salary_date_shift(
    drafts: list[Draft], rng: np.random.Generator, period_end: date, details: dict[str, str]
) -> list[Draft]:
    delay = timedelta(days=int(rng.integers(3, 11)))
    missed = date.fromisoformat(details["missed_salary"]) if "missed_salary" in details else None

    candidates = []
    for i in _salary_positions(drafts):
        new_date = next_working_day(drafts[i].posted + delay)
        near_missed = missed is not None and abs((new_date - missed).days) <= 7
        if new_date <= period_end and not near_missed:
            candidates.append((i, new_date))
    i, new_date = candidates[int(rng.integers(len(candidates)))]

    details["salary_date_shift"] = f"{drafts[i].posted.isoformat()} -> {new_date.isoformat()}"
    out = list(drafts)
    out[i] = _tag(drafts[i]._replace(posted=new_date), "salary_date_shift")
    return out


def bonus_month(
    drafts: list[Draft],
    rng: np.random.Generator,
    narrator: Narrator,
    income: IncomeTruth,
    month_firsts: list[date],
    details: dict[str, str],
) -> list[Draft]:
    first = month_firsts[int(rng.integers(len(month_firsts)))]
    posted = next_working_day(first + timedelta(days=int(rng.integers(10, 24))))
    assert income.net_pay is not None  # salaried customers always have a fixed salary
    amount = rupees(float(income.net_pay) * rng.uniform(0.5, 1.5))

    details["bonus_month"] = f"{posted.isoformat()} {amount}"
    # Half the time the narration says BONUS; otherwise it looks exactly like salary.
    note = "BONUS" if rng.random() < 0.5 else None
    text = narrator.salary(income.employer, posted, note)
    bonus = Draft(posted, text, amount, "bonus", ("bonus_month",), income.employer)
    return [*drafts, bonus]


_LEGAL_WORDS = {"PVT", "LTD", "PRIVATE", "LIMITED", "LLP", "&", "SONS"}


def _renamed(employer: str, rng: np.random.Generator) -> str:
    """A believable new name, e.g. RAVENIKA INFOTECH PVT LTD -> RAVENIKA INDIA PVT LTD.

    The second word always changes, so the rename stays visible even when a bank cuts
    narrations short (a suffix-only change like "PVT LTD" -> "LLP" would get cut off).
    """
    words = [w for w in employer.split() if w not in _LEGAL_WORDS]
    second = words[1] if len(words) > 1 else ""
    options = [f"{words[0]} {w} PVT LTD" for w in ("INDIA", "GLOBAL", "DIGITAL") if w != second]
    return str(rng.choice(options))


def employer_rename(
    drafts: list[Draft],
    rng: np.random.Generator,
    narrator: Narrator,
    employer: str,
    details: dict[str, str],
) -> tuple[list[Draft], str]:
    pay = _salary_positions(drafts)
    cut = drafts[pay[int(rng.integers(len(pay) // 3, 2 * len(pay) // 3 + 1))]].posted
    new_name = _renamed(employer, rng)
    details["employer_rename"] = f"{employer} -> {new_name} from {cut.isoformat()}"

    out = []
    for d in drafts:
        if d.category in _PAYROLL and d.posted >= cut:
            # New narration, same canonical counterparty: it's still the same employer.
            d = _tag(d._replace(description=narrator.salary(new_name, d.posted)), "employer_rename")
        out.append(d)
    return out, new_name


def emi_bounce(
    posted: list[Draft],
    balance_before: list[Decimal],
    declined: list[Draft],
    rng: np.random.Generator,
    narrator: Narrator,
    umrn: str,
    details: dict[str, str],
) -> list[Draft]:
    """Make one EMI bounce. Runs on already-posted drafts, because it depends on balances.

    Three ways it can happen, most realistic first:

    - Natural: an EMI the account genuinely couldn't cover (declined while posting).
    - Shortfall (70% otherwise): the customer moves money out just before the EMI date
      (a self-transfer), leaving less than the EMI but enough for the charges. The EMI
      debit never posts; the transfer and the charges do. The transfer is tagged too:
      it's the cause a model should learn to spot.
    - Technical (e.g. mandate problem): the debit posts, is reversed the same day, and
      the charges follow.

    For insufficient funds the charge narration names the lender and the reason. New lines
    go where the balance is known, so the charges can never overdraw the account.
    """
    charges = BOUNCE_CHARGE + BOUNCE_GST
    closing = balance_before[-1] + posted[-1].amount if posted else Decimal(0)

    def balance_at(pos: int) -> Decimal:
        return balance_before[pos] if pos < len(posted) else closing

    # Natural bounces: the charges go at the start of that day, where the balance can pay.
    dates = [d.posted for d in posted]
    short = []
    for d in declined:
        if d.category == "loan_payment":
            pos = bisect_left(dates, d.posted)
            if balance_at(pos) >= charges:
                short.append((pos, d))
    emis = [i for i, d in enumerate(posted) if d.category == "loan_payment"]
    ample = [i for i in emis if balance_before[i] >= -posted[i].amount + charges]
    can_shortfall = [i for i in emis if balance_before[i] >= charges + 1000]
    if not short and not ample and not can_shortfall:
        raise RuntimeError("no EMI with enough balance to pay bounce charges")

    tags = ("emi_bounce",)
    reason = "INSUFFICIENT BAL"
    if short:
        pos, emi = short[int(rng.integers(len(short)))]
        lines: list[Draft] = []  # the EMI debit never posts
        head, tail = posted[:pos], posted[pos:]
    elif can_shortfall and (rng.random() < 0.7 or not ample):
        i = can_shortfall[int(rng.integers(len(can_shortfall)))]
        emi, before = posted[i], balance_before[i]
        # Leave between the charges and (just under) the EMI in the account.
        room = min(-emi.amount, before) - charges - 2
        left = charges + Decimal(str(rng.uniform(0, float(room)))).quantize(Decimal("0.01"))
        moved = Decimal(int(before - left))  # whole rupees; leaves slightly more than `left`
        text = narrator.neft_debit(narrator.holder, emi.posted, "SELF TRANSFER")
        lines = [Draft(emi.posted, text, -moved, "transfer_out", tags, narrator.holder)]
        head, tail = posted[:i], posted[i + 1 :]
    else:
        i = ample[int(rng.integers(len(ample)))]
        emi, reason = posted[i], narrator.technical_return_reason()
        reversal = Draft(emi.posted, narrator.nach_return(emi.counterparty, umrn, reason),
                         -emi.amount, "loan_reversal", tags, emi.counterparty)  # fmt: skip
        lines = [_tag(emi, "emi_bounce"), reversal]
        head, tail = posted[:i], posted[i + 1 :]

    bank = narrator.own_bank.name.upper()
    charge_text = narrator.bounce_charge(emi.counterparty, reason)
    lines += [
        Draft(emi.posted, charge_text, -BOUNCE_CHARGE, "bounce_charge", tags, bank),
        Draft(emi.posted, narrator.bounce_gst(), -BOUNCE_GST, "bounce_gst", tags, bank),
    ]
    details["emi_bounce"] = f"{emi.posted.isoformat()} {-emi.amount} {reason}"
    return head + lines + tail
