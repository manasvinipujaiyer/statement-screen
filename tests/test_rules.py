"""Salary, EMI, bounce, rent and employer rules, on small hand-made statements."""

from datetime import date, timedelta

import polars as pl

from statementscreen.patterns.cli import process_statement
from statementscreen.patterns.rules import inr


def _run(entries: list[tuple[date, str, int]]) -> dict[str, pl.DataFrame]:
    entries = sorted(entries, key=lambda e: e[0])
    first, last = entries[0][0], entries[-1][0]
    stmt = pl.DataFrame({
        "statement_id": ["stmt_0001"] * len(entries),
        "row": list(range(len(entries))),
        "posted": [e[0] for e in entries],
        "narration": [e[1] for e in entries],
        "amount_paise": [e[2] for e in entries],
    })  # fmt: skip
    start = first.replace(day=1)
    end = (last.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return process_statement(stmt, start, end, "columns")


def _flagged(tables: dict, flag: str) -> list[int]:
    rows = tables["rows"]
    return rows.filter(pl.col(flag))["row"].to_list()


def _month_day(month: int, day: int, year: int = 2025) -> date:
    return date(year, month, min(day, 28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31))


SALARY = 8_002_800


def _salary(when: date, employer: str = "NERANA TECHNOLOGIES PVT LTD", amount: int = SALARY,
            note: str = "SALARY") -> tuple[date, str, int]:  # fmt: skip
    utr = f"VSTKN25{when.toordinal():09d}"
    return when, f"NEFT/CR/{utr}/{employer}/VSTK0384513/{note}", amount


def _result(tables: dict) -> dict:
    return tables["statement_results"].row(0, named=True)


# ---------------------------------------------------------------- salary


def test_salary_fires_on_a_clean_monthly_salary() -> None:
    tables = _run([_salary(_month_day(m, 30)) for m in range(1, 13)])
    result = _result(tables)
    assert result["has_salary"] and result["employer"] == "nerana technologies"
    assert result["salary_median_paise"] == SALARY
    assert len(_flagged(tables, "is_salary")) == 12
    assert result["salary_reason"] == (
        "monthly NEFT credit in 12 of 12 months, amount always INR 80,028, paid on day 28-30"
    )


def test_salary_survives_a_shifted_a_missed_and_a_bonus_month() -> None:
    entries = [_salary(_month_day(m, 30)) for m in range(1, 13) if m not in (3, 6)]
    entries.append(_salary(date(2025, 3, 24)))  # shifted 6 days early
    # June: missed
    bonus = _salary(date(2025, 10, 30), amount=2 * SALARY, note="BONUS OCT25")
    entries.append(bonus)
    tables = _run(entries)
    assert _result(tables)["has_salary"]
    flagged = tables["rows"].filter(pl.col("is_salary"))
    assert flagged.height == 11  # 10 on time + the shifted one; not the bonus
    assert 2 * SALARY not in flagged["amount_paise"].to_list()
    reason = _result(tables)["salary_reason"]
    assert "11 of 12 months" in reason and "1 other credit(s)" in reason


def test_irregular_gig_credits_are_not_salary() -> None:
    start = date(2025, 1, 3)
    entries = []
    for week in range(26):
        when = start + timedelta(days=7 * week)
        amount = 200_000 + (week * 137_000) % 600_000
        entries.append((when, f"NEFT CR-KSVB0992206-TASKEVO SERVICES PRIVATE LIMITED-TIMOTHY "
                              f"SETTY-KSVBN24{week:09d} WEEKLY SETTLEMENT", amount))  # fmt: skip
    tables = _run(entries)
    assert not _result(tables)["has_salary"]
    assert _flagged(tables, "is_salary") == []
    assert _result(tables)["salary_reason"].startswith("no salary")


def test_a_single_large_credit_is_not_salary() -> None:
    tables = _run(
        [
            _salary(date(2025, 4, 15), employer="ANAND TRADERS", amount=50_000_000),
            (date(2025, 9, 2), "UPI/DR/357630084532/HOTEL TIYO/PRVH/x@y/UPI", -8_966),
        ]
    )
    assert not _result(tables)["has_salary"] and _flagged(tables, "is_salary") == []


def test_regular_upi_credits_are_not_salary() -> None:
    upi = "UPI/CR/1317689657{:02d}/Lakshmi Rai/KSVB/lakshmirai6@kesarvan/UPI"
    tables = _run([(_month_day(m, 5), upi.format(m), 1_500_000) for m in range(1, 13)])
    assert not _result(tables)["has_salary"]
    notes = tables["group_rules"]["salary_note"].to_list()
    assert notes == ["not salary: paid by UPI"]


def test_employer_rename_is_one_stream_named_by_the_latest_name() -> None:
    entries = [_salary(_month_day(m, 30), "KESALU SOLUTIONS PVT LTD") for m in range(1, 7)]
    entries += [_salary(_month_day(m, 30), "KESALU GLOBAL PVT LTD") for m in range(7, 13)]
    tables = _run(entries)
    result = _result(tables)
    assert result["has_salary"] and result["employer"] == "kesalu global"
    assert len(_flagged(tables, "is_salary")) == 12
    assert "name changed: 'kesalu solutions' -> 'kesalu global'" in result["salary_reason"]


# ---------------------------------------------------------------- EMI and bounce

EMI = 1_476_600
UMRN = "VSTK7475455546541154"


def _emi(when: date) -> tuple[date, str, int]:
    return when, f"NACH/DR/{UMRN}/NIVESHA HOME LOANS LTD", -EMI


def test_monthly_fixed_nach_debit_is_an_emi() -> None:
    tables = _run([_emi(_month_day(m, 5)) for m in range(1, 13)])
    assert len(_flagged(tables, "is_emi")) == 12
    reason = tables["rows"]["flag_reason"][0]
    assert reason.startswith("EMI to nivesha home loans: monthly NACH debit in 12 of 12 months")


def test_a_bounced_emi_month_is_a_bounce_and_keeps_the_emi_pattern() -> None:
    entries = [_emi(_month_day(m, 5)) for m in range(1, 13)]
    entries += [
        (date(2025, 7, 7), f"NACH/RTN/{UMRN}/NIVESHA HOME LOANS LTD/INSUFFICIENT BAL", EMI),
        (date(2025, 7, 8), "NACH RTN CHRGS/NIVESHA HOME LOANS LTD/INSUFFICIENT BAL", -50_000),
        (date(2025, 7, 8), "GST @18% ON NACH RTN CHRGS", -9_000),
    ]
    tables = _run(entries)
    rows = tables["rows"]
    assert len(_flagged(tables, "is_emi")) == 12  # the returned July debit still counts
    bounces = rows.filter(pl.col("is_bounce"))
    assert bounces["amount_paise"].to_list() == [EMI, -50_000, -9_000]
    assert "EMI group" in bounces["flag_reason"][0]
    assert "GST on the return charge" in bounces["flag_reason"][2]
    assert _result(tables)["bounces"] == 3


def test_a_charge_only_bounce_still_keeps_the_emi_pattern() -> None:
    entries = [_emi(_month_day(m, 5)) for m in range(1, 13) if m != 4]  # April debit declined
    entries += [(date(2025, 4, 6), "ACH RTN CHRG-NIVESHA HOME LOANS LTD-INSUFFICIENT BAL",
                 -50_000), (date(2025, 4, 6), "CGST+SGST ON ACH RTN CHRG", -9_000)]  # fmt: skip
    tables = _run(entries)
    assert len(_flagged(tables, "is_emi")) == 11
    assert len(_flagged(tables, "is_bounce")) == 2


def test_upi_refunds_are_not_bounces() -> None:
    tables = _run(
        [
            (date(2025, 3, 1), "UPI/REV/904735941665/MITINA FASHIONS/REFUND", 5_914),
            (date(2025, 3, 2), "REV-UPI-572523182340-Bazaario Online", 137_714),
        ]
    )
    assert _flagged(tables, "is_bounce") == []


def test_gst_line_not_after_a_return_charge_is_not_a_bounce() -> None:
    tables = _run(
        [
            (date(2025, 3, 1), "BBPS/6057069106/Jaldinet Broadband", -69_200),
            (date(2025, 3, 2), "GST @18% ON NACH RTN CHRGS", -9_000),
        ]
    )
    assert _flagged(tables, "is_bounce") == []


# ---------------------------------------------------------------- rent


def test_monthly_transfer_to_one_person_is_rent() -> None:
    rent = "IMPS-0186315822{:02d}-Zashil Loyal-TRGM-RENT"
    tables = _run([(_month_day(m, 3), rent.format(m), -3_330_000) for m in range(1, 13)])
    assert len(_flagged(tables, "is_rent")) == 12
    reason = tables["rows"]["flag_reason"][0]
    assert reason.startswith("rent to zashil loyal: monthly IMPS transfer in 12 of 12 months")
    assert reason.endswith("narration says rent")


def test_a_self_transfer_every_month_is_not_rent() -> None:
    self_tx = "NEFT DR-TRGM0143277-FALAN NAIDU-NETBANK-PRVHN24123380{:03d}-SELF"
    tables = _run([(_month_day(m, 3), self_tx.format(m), -1_000_000) for m in range(1, 13)])
    assert _flagged(tables, "is_rent") == []


def test_inr_uses_indian_grouping() -> None:
    assert inr(8_002_800) == "INR 80,028"
    assert inr(1_234_567_800) == "INR 1,23,45,678"
    assert inr(-50_000) == "INR 500"
