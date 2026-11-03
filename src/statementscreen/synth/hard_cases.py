"""Hard cases: deliberately tricky situations injected into a statement.

Each one is something that breaks a naive income model:

- ``salary_date_shift``: one paycheck arrives 3-10 days late.
- ``employer_rename``:   partway through, payroll arrives under a new employer name
                         (a rebrand or acquisition), so "same employer?" isn't a string match.
- ``bonus_month``:       one large one-off bonus. Real money, but not regular income.
- ``missed_salary``:     one payday with no paycheck at all.
- ``emi_bounce``:        a loan instalment (EMI) fails: the debit posts, the bank returns
                         it, and a returned-payment fee follows.

The first four only make sense for salaried customers. Every transaction a hard case
creates or changes is tagged in ``Draft.hard_cases``. ``missed_salary`` removes a
transaction, so there's nothing to tag; the missed date goes in ``details`` instead.

All functions take the list of drafts and return a new list; they never edit in place.
"""

from datetime import date, timedelta

import numpy as np

from statementscreen.synth.descriptions import DescriptionStyle
from statementscreen.synth.models import Draft, IncomeTruth
from statementscreen.synth.util import money, next_business_day

HARD_CASES = ("salary_date_shift", "employer_rename", "bonus_month", "missed_salary", "emi_bounce")
SALARIED_ONLY = frozenset({"salary_date_shift", "employer_rename", "bonus_month", "missed_salary"})

# Payroll entries are re-described when the employer is renamed.
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
    return sorted(
        (i for i, d in enumerate(drafts) if d.category == "salary"), key=lambda i: drafts[i].posted
    )


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
        new_date = next_business_day(drafts[i].posted + delay)
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
    style: DescriptionStyle,
    income: IncomeTruth,
    month_firsts: list[date],
    details: dict[str, str],
) -> list[Draft]:
    first = month_firsts[int(rng.integers(len(month_firsts)))]
    posted = next_business_day(first + timedelta(days=int(rng.integers(10, 24))))
    paychecks_per_month = 1 if income.pay_frequency == "monthly" else 26 / 12
    assert income.net_pay is not None  # salaried customers always have a fixed paycheck
    amount = money(float(income.net_pay) * paychecks_per_month * rng.uniform(0.5, 1.5))

    details["bonus_month"] = f"{posted.isoformat()} {amount}"
    bonus = Draft(posted, style.payroll(rng, income.employer), amount, "bonus", ("bonus_month",))
    return [*drafts, bonus]


_LEGAL_SUFFIXES = {"LLC", "LTD", "INC", "PLC", "GROUP"}


def _renamed(employer: str, rng: np.random.Generator) -> str:
    """A believable new name for the same employer, e.g. TURNER LTD -> TURNER HOLDINGS."""
    words = [w for w in employer.replace(",", "").split() if w not in _LEGAL_SUFFIXES]
    core = " ".join(words)
    options = [f"{core} HOLDINGS", f"{words[0]} GROUP"]
    if len(words) > 1:  # "SCOTT HOWARD AND VAUGHN" -> "SCOTT VAUGHN PARTNERS"
        options.append(f"{words[0]} {words[-1]} PARTNERS")
    options = [o for o in options if o != employer]
    return str(rng.choice(options))


def employer_rename(
    drafts: list[Draft],
    rng: np.random.Generator,
    style: DescriptionStyle,
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
            d = _tag(d._replace(description=style.payroll(rng, new_name)), "employer_rename")
        out.append(d)
    return out, new_name


def emi_bounce(
    drafts: list[Draft],
    rng: np.random.Generator,
    style: DescriptionStyle,
    lender: str,
    period_end: date,
    details: dict[str, str],
) -> list[Draft]:
    loans = [
        i
        for i, d in enumerate(drafts)
        if d.category == "loan_payment" and d.posted + timedelta(days=5) <= period_end
    ]
    i = loans[int(rng.integers(len(loans)))]
    emi = drafts[i]
    returned_on = next_business_day(emi.posted + timedelta(days=1))
    details["emi_bounce"] = f"{emi.posted.isoformat()} {-emi.amount}"

    tags = ("emi_bounce",)
    reversal = Draft(
        returned_on, style.returned_debit(rng, lender), -emi.amount, "loan_reversal", tags
    )
    fee = Draft(returned_on, style.fee("RETURNED PAYMENT"), -money(35), "fee", tags)
    out = list(drafts)
    out[i] = _tag(emi, "emi_bounce")
    return [*out, reversal, fee]
