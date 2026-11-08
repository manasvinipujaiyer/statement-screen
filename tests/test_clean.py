"""Narration cleaning: channel, counterparty, purpose, cut-offs and the holder's own name.

Examples are synthetic narrations of the styles seen in the train split, from both
layouts (columns prints full narrations, drcr cuts them to one line).
"""

from datetime import date

import polars as pl
import pytest

from statementscreen.patterns.clean import (
    CHANNELS,
    SELF,
    clean_name,
    clean_narration,
    clean_statement,
    is_self_purpose,
    resolve_self,
)

# (narration, channel, style, counterparty, purpose)
STYLES = [
    ("UPI/DR/357630084532/HOTEL TIYO/PRVH/hoteltiyo26@pravahini/UPI",
     "UPI", "upi_slash", "hotel tiyo", ""),
    ("UPI/CR/131768965761/Lakshmi Rai/KSVB/lakshmirai6",
     "UPI", "upi_slash", "lakshmi rai", ""),
    ("UPI-NENADRA CLINIC-nenadracli65@tarangam-TRGM0527985-692422128843-UPI",
     "UPI", "upi_dash", "nenadra clinic", ""),
    ("UPI/DR/713547121041/TARANG TELECOM PREPAID/VSTK/tarangtele49@vistarika/RECHARGE",
     "UPI", "upi_slash", "tarang telecom prepaid", "recharge"),
    ("NEFT CR-KSVB0992206-TASKEVO SERVICES PRIVATE LIMITED-TIMOTHY SETTY-KSVBN24262781561 "
     "WEEKLY SETTLEMENT", "NEFT", "neft_cr_dash", "taskevo services", "weekly settlement"),
    ("NEFT/CR/VSTKN25093705947/SHAMOVI RETAIL PVT LTD/VSTK0384513/SALARY MAR25",
     "NEFT", "neft_slash", "shamovi retail", "salary mar25"),
    ("NEFT DR-PRVH0925915-NEELIMA CHAKRABARTI-NETBANK-VSTKN25306848248-RENT",
     "NEFT", "neft_dr_dash", "neelima chakrabarti", "rent"),
    ("NEFT/DR/VSTKN24284994658/ABEER BAVA/PRVH0401593/RENT",
     "NEFT", "neft_slash", "abeer bava", "rent"),
    ("RTGS CR-PRVH0796731-KANATA FOODS P LTD-ISHWAR BALA-PRVHR24100018890",
     "RTGS", "rtgs_cr_dash", "kanata foods", ""),
    ("RTGS/CR/PRVHR24169745567/VALURA FOODS PVT. LTD./PRVH0282861",
     "RTGS", "rtgs_slash", "valura foods", ""),
    ("MMT/IMPS/177192013521/RENT/Idika Mani/VISTARIKA BAN",
     "IMPS", "imps_mmt", "idika mani", "rent"),
    ("IMPS/P2A/789609942028/TASKEVO SERVICES PRIVATE LIMITED/KSVB/PAYOUT",
     "IMPS", "imps_p2a", "taskevo services", "payout"),
    ("IMPS-018631582263-Zashil Loyal-TRGM-RENT", "IMPS", "imps_dash", "zashil loyal", "rent"),
    ("NACH/DR/VSTK7525661505255730/NIVESHA HOME LOANS LTD",
     "NACH", "nach_slash", "nivesha home loans", ""),
    ("ACH D- KRAMIKA CAPITAL LTD-KSVB7851104798000106",
     "NACH", "ach_dash", "kramika capital", ""),
    ("NACH/RTN/TRGM7242629495023172/Nivesha Home Loans Ltd/MANDATE NOT REGISTERED",
     "REVERSAL", "nach_rtn", "nivesha home loans", "mandate not registered"),
    ("ACH RTN CHRG-DHANVRIKSH FINANCE LTD-INSUFFICIENT BAL",
     "CHARGE", "ach_rtn_chrg", "dhanvriksh finance", "insufficient bal"),
    ("NACH RTN CHRGS/KRAMIKA CAPITAL LTD/A/C BLOCKED",
     "CHARGE", "nach_rtn_chrgs", "kramika capital", "a/c blocked"),
    ("CGST+SGST ON ACH RTN CHRG", "CHARGE", "gst_on_charge", "", ""),
    ("GST @18% ON NACH RTN CHRGS", "CHARGE", "gst_on_charge", "", ""),
    ("BBPS/6057069106/Jaldinet Broadband", "BBPS", "bbps", "jaldinet broadband", ""),
    ("BIL/BPAY/816307042/VIDYUTA POWER DISTRIBUTION LTD",
     "BBPS", "bil_bpay", "vidyuta power distribution", ""),
    ("ATM/WDL/3501/PRVH0382/AVADI", "ATM", "atm_wdl", "atm", ""),
    ("ATW-3391-TRGM2135-ANANTAPUR", "ATM", "atw", "atm", ""),
    ("POS/0442/Streamverse/514124", "CARD", "pos", "streamverse", ""),
    ("PCD/1301/STREAMVERSE/ONLINE", "CARD", "pcd", "streamverse", "online"),
    ("CREDIT INTEREST CAPITALISED 010425-300625", "INTEREST", "int_capitalised", "interest", ""),
    ("INT.PD:7968:01-07-2024 TO 30-09-2024", "INTEREST", "int_pd", "interest", ""),
    ("UPI/REV/904735941665/MITINA FASHIONS/REFUND",
     "REVERSAL", "upi_rev_slash", "mitina fashions", "refund"),
    ("REV-UPI-572523182340-Bazaario Online", "REVERSAL", "upi_rev_dash", "bazaario online", ""),
    ("GST PMT/CPIN 46285392937437", "OTHER", "gst_pmt", "gst payment", ""),
    ("TAX/ADVANCE TAX/CHALLAN 20655197464", "OTHER", "advance_tax", "advance tax", ""),
]  # fmt: skip


@pytest.mark.parametrize(("narration", "channel", "style", "counterparty", "purpose"), STYLES)
def test_each_style(narration, channel, style, counterparty, purpose) -> None:
    c = clean_narration(narration)
    assert (c.channel, c.style, c.counterparty, c.purpose) == (
        channel, style, counterparty, purpose)  # fmt: skip
    assert c.channel in CHANNELS


@pytest.mark.parametrize("narration", [n for n, *_ in STYLES])
def test_no_references_handles_or_codes_survive(narration) -> None:
    cp = clean_narration(narration).counterparty
    assert "@" not in cp and not any(ch.isdigit() for ch in cp)
    assert cp == cp.lower()


def test_clean_name_rules() -> None:
    assert clean_name("KESALU SOLUTIONS PVT. LTD.") == "kesalu solutions"
    assert clean_name("SAGAR & SONS PRIVATE LIMTIED") == "sagar and sons"  # misspelt legal word
    assert clean_name("SALARY MAR25") == ""
    assert clean_name("hoteltiyo26@pravahini VSTK0384513 VSTKN25093705947 01-07-2024") == ""
    assert clean_name("NERANA TECHNOLOGIES PVT L", cut=True) == "nerana technologies"
    assert clean_name("Lakshmi P Rao") == "lakshmi p rao"  # an initial isn't a legal word


def test_misspelt_legal_words_go_but_real_words_stay() -> None:
    for typo in ("LIMTIED", "LIMTED", "LIMITD", "PRIVTE", "PVIVATE"):
        assert clean_name(f"KESALU SOLUTIONS {typo}") == "kesalu solutions"
    assert clean_name("KESALU SOLUTIONS P LIMTIED") == "kesalu solutions"
    for word in ("LIMITS", "LIMESTONE", "PIRATE", "PRIVY"):
        assert clean_name(f"ANAND {word}") == f"anand {word.lower()}"


# --------------------------------------------------------------- cut-off narrations


def test_cut_counterparty_columns_style() -> None:
    c = clean_narration("NEFT/CR/VSTKN25031056669/NERANA TECHNOLOGIES PVT L")
    assert (c.channel, c.counterparty, c.cut, c.counterparty_cut) == (
        "NEFT", "nerana technologies", True, True)  # fmt: skip


def test_cut_after_the_name_drcr_style() -> None:
    c = clean_narration("UPI/DR/183335025630/Bazaario Online/VSTK/bazaarioon79@v")
    assert (c.style, c.counterparty, c.cut, c.counterparty_cut) == (
        "upi_slash", "bazaario online", True, False)  # fmt: skip


def test_cut_inside_the_name() -> None:
    c = clean_narration("MMT/IMPS/949961987923/PAYMENT/BRINDA KHATR")
    assert (c.counterparty, c.purpose, c.counterparty_cut) == ("brinda khatr", "", True)
    c = clean_narration("NEFT CR-PRVH0299244-Kaketi Retail Pvt. Ltd")
    assert (c.counterparty, c.counterparty_cut, c.holder) == ("kaketi retail", True, "")


def test_cut_holder_slot_is_not_a_candidate() -> None:
    c = clean_narration("NEFT CR-TRGM0934706-Kaketi Retail-Kashvi M")
    assert (c.counterparty, c.holder, c.counterparty_cut) == ("kaketi retail", "", False)


# --------------------------------------------------------------- unknown narrations


def test_known_opening_with_broken_fields_is_a_variant() -> None:
    c = clean_narration("UPI/DR/ABC123/HOTEL TIYO/PRVH")
    assert (c.channel, c.style, c.counterparty) == ("UPI", "upi_slash?", "hotel tiyo")


def test_unseen_format_falls_back() -> None:
    c = clean_narration("XFER/883412/ANAND TRADERS PVT LTD/KOLKATA")
    assert (c.channel, c.style, c.counterparty) == ("OTHER", "unknown", "anand traders")


def test_unseen_format_takes_its_channel_from_a_keyword() -> None:
    c = clean_narration("ACH RTN-NIVESHA HOME LOANS LTD-VSTK7475455546541154-INSUFFICIENT BAL")
    assert (c.channel, c.style, c.counterparty) == ("REVERSAL", "unknown", "nivesha home loans")


def test_no_name_gives_empty_counterparty() -> None:
    for text in ("12345678/0042", "", "   "):
        c = clean_narration(text)
        assert (c.channel, c.style, c.counterparty) == ("OTHER", "unknown", "")


# --------------------------------------------------------------- the holder's own name


def _resolve(narrations: list[str]) -> list[tuple[str, bool]]:
    return [(c.counterparty, s) for c, s in resolve_self([clean_narration(n) for n in narrations])]


def test_self_purpose_words() -> None:
    assert is_self_purpose("self") and is_self_purpose("self transfer")
    assert is_self_purpose("self tra")  # cut
    assert not is_self_purpose("se") and not is_self_purpose("rent")


def test_self_note_marks_the_same_name_elsewhere() -> None:
    assert _resolve([
        "IMPS/P2A/564895559864/EKTA CHAUDHARY/KSVB/SELF TRA",
        "NEFT/DR/TRGMN25006559622/EKTA CHAUDHARY/KSVB075293",
        "NEFT/DR/TRGMN25006559622/EKTA CHAU",
        "UPI/CR/131768965761/Lakshmi Rai/KSVB/lakshmirai6",
    ]) == [(SELF, True), (SELF, True), (SELF, True), ("lakshmi rai", False)]  # fmt: skip


def test_holder_slot_name_is_never_the_counterparty() -> None:
    out = _resolve([
        "NEFT CR-PRVH0055713-KESALU SOLUTIONS PVT. LTD.-INDRAJIT CHOKSHI-PRVHN25038378774 "
        "SALARY JAN25",
        "MMT/IMPS/717591269449/TRANSFER/INDRAJIT CHOKSHI/VISTARIKA BANK",
        "UPI/CR/131768965761/INDRAJIT CHKOSHI/KSVB/indrajit",  # typo
        "UPI/CR/131768965761/INDIRA CHOPRA/KSVB/indira",  # someone else
    ])  # fmt: skip
    assert out == [("kesalu solutions", False), (SELF, True), (SELF, True),
                   ("indira chopra", False)]  # fmt: skip
    assert all("indrajit" not in cp for cp, _ in out)


def test_self_note_in_dash_style() -> None:
    out = _resolve(["NEFT DR-TRGM0143277-FALAN NAIDU-NETBANK-PRVHN24123380661-SELF"])
    assert out == [(SELF, True)]


def test_lender_named_like_holder_is_not_made_self() -> None:
    # Only transfer channels are matched against the holder: EMIs and charges keep the lender.
    out = _resolve([
        "MMT/IMPS/495836586764/SELF TRANSFER/KRAMIKA CAPITAL/VISTARIKA",
        "NACH/DR/VSTK7525661505255730/KRAMIKA CAPITAL LTD",
    ])  # fmt: skip
    assert out == [(SELF, True), ("kramika capital", False)]


# --------------------------------------------------------------- one statement


def test_clean_statement_keeps_order_and_sets_direction() -> None:
    rows = pl.DataFrame({
        "statement_id": ["stmt_0001"] * 3,
        "row": [2, 0, 1],
        "posted": [date(2025, 1, 3), date(2025, 1, 1), date(2025, 1, 2)],
        "narration": ["ATW-3391-TRGM2135-ANANTAPUR",
                      "IMPS/P2A/564895559864/EKTA CHAUDHARY/KSVB/SELF TRA",
                      "BBPS/6057069106/Jaldinet Broadband"],
        "amount_paise": [-500_000, 834_300, -69_200],
    })  # fmt: skip
    out = clean_statement(rows)
    assert out["row"].to_list() == [0, 1, 2]
    assert out["direction"].to_list() == ["credit", "debit", "debit"]
    assert out["channel"].to_list() == ["IMPS", "BBPS", "ATM"]
    assert out["counterparty"].to_list() == [SELF, "jaldinet broadband", "atm"]
    assert out["is_self"].to_list() == [True, False, False]
