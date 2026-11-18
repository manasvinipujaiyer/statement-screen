"""The income profile on hand-written statements, run through the real Week 3 patterns."""

from datetime import date, timedelta

import polars as pl
import pytest

from statementscreen.patterns.cli import process_statement
from statementscreen.profile.build import profile_statement, to_json
from statementscreen.profile.cli import GROUP_COLUMNS, ROW_COLUMNS, RULE_COLUMNS

SALARY = 5_000_000  # INR 50,000.00 in paise
MONTHS = [f"{m:%b}".upper() for m in (date(2025, i, 1) for i in range(1, 13))]


def salary(month: int, amount: int = SALARY, note: str | None = None, day: int = 28):
    d = date(2025, month, day)
    word = note or f"SALARY {MONTHS[month - 1]}25"
    text = (f"NEFT CR-KSVB0001234-ACME TECH PVT LTD-RAVI KUMAR-"
            f"KSVBN25{d.timetuple().tm_yday:03d}123456 {word}")  # fmt: skip
    return d, text, amount


def emi(month: int, lender: str, umrn: str, amount: int, day: int = 5):
    return date(2025, month, day), f"ACH D- {lender}-{umrn}", -amount


SHOP = "UPI/DR/412398765012/SHARMA STORE/KSVB/sharma12@ksvb/UPI"


def spend(month: int, day: int = 12, amount: int = 45_000):
    return date(2025, month, day), SHOP, -amount


def profile(entries, start=date(2025, 1, 1), end=date(2025, 12, 31)) -> dict:
    """Run hand-written rows through patterns (Week 3 rules) and the profile."""
    entries = sorted(entries, key=lambda e: e[0])
    if not entries:
        empty = {c: pl.DataFrame(schema=dict.fromkeys(cols, pl.Null))
                 for c, cols in (("rows", ROW_COLUMNS), ("groups", GROUP_COLUMNS),
                                 ("rules", RULE_COLUMNS))}  # fmt: skip
        return profile_statement("stmt_0001", empty["rows"], empty["groups"], empty["rules"],
                                 None, start, end)  # fmt: skip
    stmt = pl.DataFrame({"statement_id": ["stmt_0001"] * len(entries),
                         "row": list(range(len(entries))),
                         "posted": [e[0] for e in entries], "narration": [e[1] for e in entries],
                         "amount_paise": [e[2] for e in entries]})  # fmt: skip
    t = process_statement(stmt, start, end, "drcr")
    result = t["statement_results"].row(0, named=True)
    return profile_statement("stmt_0001", t["rows"], t["group_features"], t["group_rules"],
                             result, start, end)  # fmt: skip


def value(p: dict, name: str):
    return p["findings"][name]["value"]


def every_reason_is_given(p: dict) -> bool:
    return all(isinstance(f["reason"], str) and f["reason"].strip()
               for f in p["findings"].values())  # fmt: skip


# ---------------------------------------------------------------- income


def test_a_steady_salary_gives_the_monthly_income() -> None:
    p = profile([salary(m) for m in range(1, 13)] + [spend(m) for m in range(1, 13)])
    assert value(p, "has_salary") is True
    assert value(p, "employer") == "acme tech"
    assert value(p, "typical_monthly_salary_paise") == SALARY
    assert value(p, "estimated_monthly_income_paise") == SALARY
    assert value(p, "salary_months_paid") == {"paid": 12, "of": 12}
    assert value(p, "missed_salary_months") == []
    assert value(p, "customer_type") == "salaried"
    assert value(p, "income_confidence") == "normal"
    assert p["warnings"] == [] and every_reason_is_given(p)


def test_a_missed_month_lowers_the_average() -> None:
    p = profile([salary(m) for m in range(1, 13) if m != 6])
    # 11 salaries over the statement's 12 months, rounded half up to whole paise
    assert value(p, "estimated_monthly_income_paise") == (11 * SALARY * 2 + 12) // 24
    assert value(p, "missed_salary_months") == ["2025-06"]
    assert value(p, "salary_months_paid") == {"paid": 11, "of": 12}


def test_a_bonus_is_never_income() -> None:
    bonus = salary(7, amount=3_000_000, note="BONUS", day=15)
    p = profile([salary(m) for m in range(1, 13)] + [bonus])
    assert value(p, "estimated_monthly_income_paise") == SALARY
    assert value(p, "typical_monthly_salary_paise") == SALARY


def test_weekly_payouts_that_skip_a_week_are_gig_income() -> None:
    """Every 4th week skipped: gaps 7, 7, 14 (mean about 9.3 days), as dev's stmt_0777."""
    first = date(2025, 1, 3)
    payouts = []
    for week in range(52):
        if week % 4 == 3:
            continue
        d = first + timedelta(days=7 * week)
        text = f"IMPS/P2A/{412398765000 + week}/PARCELVYA LOGISTICS/KSVB/PAYOUT"
        payouts.append((d, text, 400_000 + 10_000 * (week % 5)))
    p = profile(payouts)
    assert value(p, "customer_type") == "gig"
    regular = value(p, "other_regular_income")
    assert regular[0]["kind"] == "weekly payouts"
    total = sum(amount for _, _, amount in payouts)
    assert value(p, "estimated_monthly_income_paise") == (2 * total + 12) // 24


# ---------------------------------------------------------------- EMIs and bounces


def test_two_emis_give_the_count_and_total() -> None:
    entries = [salary(m) for m in range(1, 13)]
    entries += [emi(m, "NIVESHA FINANCE", "KSVB7000000000000001", 1_000_000) for m in range(1, 13)]
    entries += [emi(m, "SURAKSHA LOANS", "TRGM7000000000000002", 750_000, day=10)
                for m in range(1, 13)]  # fmt: skip
    p = profile(entries)
    assert value(p, "emi_count") == 2
    assert value(p, "emi_total_monthly_paise") == 1_750_000
    assert sorted(value(p, "emi_lenders")) == ["nivesha finance", "suraksha loans"]
    assert value(p, "emi_to_income_ratio") == pytest.approx(0.35)
    assert value(p, "bounced_emis") == 0 and value(p, "bounce_charges_paise") == 0


def test_a_bounce_and_its_charge_are_counted_once() -> None:
    lender, umrn = "NIVESHA FINANCE", "KSVB7000000000000001"
    entries = [salary(m) for m in range(1, 13)]
    entries += [emi(m, lender, umrn, 1_000_000) for m in range(1, 13)]
    returned = date(2025, 4, 5)
    entries += [
        (returned, f"ACH RTN-{lender}-{umrn}-MANDATE NOT REGISTERED", 1_000_000),
        (returned, f"ACH RTN CHRG-{lender}-MANDATE NOT REGISTERED", -50_000),
        (returned, "CGST+SGST ON ACH RTN CHRG", -9_000),
    ]  # fmt: skip
    p = profile(entries)
    assert value(p, "bounced_emis") == 1
    assert value(p, "bounce_charges_paise") == 59_000  # INR 500 + INR 90 GST
    assert "1 bounced EMI" in p["warnings"]
    assert value(p, "estimated_monthly_income_paise") == SALARY  # a return is not income


def test_zero_income_gives_no_ratio_with_a_reason() -> None:
    p = profile([emi(m, "NIVESHA FINANCE", "KSVB7000000000000001", 1_000_000)
                 for m in range(1, 13)] + [spend(m) for m in range(1, 13)])  # fmt: skip
    assert value(p, "estimated_monthly_income_paise") is None
    assert value(p, "emi_to_income_ratio") is None
    assert p["findings"]["emi_to_income_ratio"]["reason"].startswith("not computed")
    assert "no recurring income found" in p["warnings"]
    assert value(p, "customer_type") == "unknown"


# ---------------------------------------------------------------- warnings and odd statements


def test_a_three_month_history_warns_and_is_not_confident() -> None:
    p = profile([salary(m) for m in (1, 2, 3)], end=date(2025, 3, 31))
    assert p["statement_months"] == 3
    assert "history shorter than 6 months (3 months)" in p["warnings"]
    assert value(p, "income_confidence") != "normal"


@pytest.mark.parametrize(
    ("entries", "end", "warning"),
    [
        ([], date(2025, 12, 31), "no transactions in the statement"),
        ([spend(m) for m in range(1, 13)], date(2025, 12, 31), "only debits"),
        ([salary(1), spend(1)], date(2025, 1, 31), "history shorter than 6 months (1 month)"),
    ],
    ids=["empty", "debits only", "single month"],
)
def test_odd_statements_return_a_profile_with_warnings(entries, end, warning) -> None:
    p = profile(entries, end=end)
    assert any(w.startswith(warning) for w in p["warnings"])
    assert "no recurring income found" in p["warnings"]
    assert value(p, "estimated_monthly_income_paise") is None
    assert every_reason_is_given(p)


def test_profile_json_is_byte_identical_across_runs() -> None:
    entries = [salary(m) for m in range(1, 13) if m != 3] + [spend(m) for m in range(1, 13)]
    assert to_json(profile(entries)) == to_json(profile(list(reversed(entries))))
