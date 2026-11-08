"""Grouping counterparties within one statement."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from statementscreen.patterns.clean import clean_statement
from statementscreen.patterns.config import PatternConfig
from statementscreen.patterns.group import group_ids, group_statement, name_distances

CFG = PatternConfig(group_threshold=0.30)  # fixed here; the real value comes from the dev sweep


def _cleaned(narrations: list[str], amounts: list[int]) -> pl.DataFrame:
    n = len(narrations)
    return clean_statement(pl.DataFrame({
        "statement_id": ["stmt_0001"] * n,
        "row": list(range(n)),
        "posted": [date(2025, 1, 1) + timedelta(days=i) for i in range(n)],
        "narration": narrations,
        "amount_paise": amounts,
    }))  # fmt: skip


def _groups(narrations: list[str], amounts: list[int], threshold: float = 0.30) -> list[str]:
    return group_statement(_cleaned(narrations, amounts), CFG, threshold)["group_id"].to_list()


EMPLOYER = [
    "NEFT/CR/VSTKN25031056669/NERANA TECHNOLOGIES PVT LTD/VSTK0384513/SALARY JAN25",
    "NEFT CR-PRVH0055713-NERANA TECHNOLOGIES PRIVATE LIMITED-EKTA RAO-PRVHN25038378774 "
    "SALARY FEB25",  # spelling variant, other style
    "NEFT/CR/VSTKN25059042720/Nerana Technologies Pvt Ltd/VSTK0384513/SALARY MAR25",
    "NEFT/CR/VSTKN25090737410/NERANA TECHNOLGIES PVT LTD/VSTK0384513/SALARY APR25",  # typo
    "NEFT/CR/VSTKN25120227427/NERANA TECHNO",  # cut off
]


def test_variants_typo_and_cut_off_form_one_group() -> None:
    out = group_statement(_cleaned(EMPLOYER, [8_002_800] * 5), CFG)
    assert out["group_id"].n_unique() == 1
    assert out["group_name"].unique().to_list() == ["nerana technologies"]


def test_two_employers_with_similar_names_stay_apart() -> None:
    ids = _groups(["NEFT/CR/VSTKN25031056669/TIYOKA CONSULTANCY PVT LTD/VSTK0384513/SALARY",
                   "NEFT/CR/VSTKN25059042720/TIYOKA CONSULTANCY PVT LTD/VSTK0384513/SALARY",
                   "NEFT/CR/VSTKN25090737410/TIYOKA FOODS PVT LTD/VSTK0384513/SALARY",
                   "NEFT/CR/VSTKN25120227427/TIYOKA FOODS PVT LTD/VSTK0384513/SALARY"],
                  [5_000_000] * 4)  # fmt: skip
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


def test_credit_and_debit_with_the_same_name_stay_apart() -> None:
    ids = _groups(["UPI/CR/131768965761/Lakshmi Rai/KSVB/lakshmirai6@kesarvan/UPI",
                   "UPI/DR/131768965762/LAKSHMI RAI/KSVB/lakshmirai6@kesarvan/UPI",
                   "UPI/CR/131768965763/LAKSHMI RAI/KSVB/lakshmirai6@kesarvan/UPI"],
                  [640_000, -200_000, 100_000])  # fmt: skip
    assert ids[0] == ids[2] and ids[0] != ids[1]
    assert ids[0].startswith("C") and ids[1].startswith("D")


def test_fixed_names_group_by_exact_text() -> None:
    ids = _groups(["ATW-3391-TRGM2135-ANANTAPUR", "ATM/WDL/3501/PRVH0382/AVADI",
                   "CGST+SGST ON ACH RTN CHRG", "BBPS/6057069106/Jaldinet Broadband"],
                  [-500_000, -300_000, -9_000, -69_200])  # fmt: skip
    assert ids[0] == ids[1] and len({ids[1], ids[2], ids[3]}) == 3


def test_ambiguous_cut_off_keeps_its_own_group() -> None:
    narrations = ["UPI/DR/357630084532/BAWA MEDICAL STORE/PRVH/bawamedica45@pravahini/UPI",
                  "UPI/DR/357630084533/BAWA MEDICOS/PRVH/bawamedico11@pravahini/UPI",
                  "UPI-BAWA MEDIC"]  # fmt: skip
    ids = _groups(narrations, [-13_200, -27_800, -9_900], threshold=0.10)
    assert len(set(ids)) == 3  # "bawa medic" starts both, which are different groups


def test_unambiguous_cut_off_joins_its_full_form() -> None:
    narrations = [
        "UPI/DR/357630084532/BAWA MEDICAL STORE/PRVH/bawamedica45@pravahini/UPI",
        "UPI-BAWA MEDIC",
        "UPI/DR/357630084534/HOTEL TIYO/PRVH/hoteltiyo26@pravahini/UPI",
    ]
    ids = _groups(narrations, [-13_200, -9_900, -8_966], threshold=0.10)
    assert ids[0] == ids[1] != ids[2]


def test_full_name_inside_a_longer_variant_is_still_clustered() -> None:
    # "kesalu solutions" starts "kesalu solutions limtied" at a word boundary: not a cut-off,
    # so it's clustered by similarity like any other name.
    ids = _groups(["NEFT/CR/VSTKN25031056669/KESALU SOLUTIONS PVT LTD/VSTK0384513/SALARY",
                   "NEFT/CR/VSTKN25059042720/KESALU SOLUTIONS LIMTIED/VSTK0384513/SALARY",
                   "NEFT/CR/VSTKN25090737410/KESALU SOLUTONS PVT LTD/VSTK0384513/SALARY"],
                  [4_471_500] * 3)  # fmt: skip
    assert len(set(ids)) == 1


def test_one_threshold_per_call_and_several_at_once_agree() -> None:
    cleaned = _cleaned(EMPLOYER, [8_002_800] * 5)
    many = group_ids(cleaned, [0.1, 0.3, 0.5], CFG)
    for t in (0.1, 0.3, 0.5):
        assert many[t] == group_ids(cleaned, [t], CFG)[t]


def test_distances_are_symmetric_with_zero_diagonal() -> None:
    d = name_distances(["nerana technologies", "nerana technolgies", "hotel tiyo"])
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    assert d[0, 1] < d[0, 2]
