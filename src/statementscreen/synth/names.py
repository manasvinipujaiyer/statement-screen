"""Invented counterparty names: employers, gig platforms, lenders, billers, merchants.

Faker's en_IN company names carry US suffixes ("LLC", "Inc"), so employers and shops
are built from made-up syllables instead ("RAVENIKA INFOTECH PVT LTD"). Personal names
(landlords, family, shop owners) come from Faker en_IN, which is fine: they are random
fictional people. Fixed brand-like names below are invented; unlike the bank names they
were not checked against company registers, so treat any resemblance as accidental.
"""

import numpy as np
from faker import Faker

_SYLLABLES = ("ka", "ra", "vi", "na", "sha", "to", "mi", "lu", "dra", "va", "ne", "ri",
              "sa", "ta", "yo", "ke", "ti", "mo", "ja", "pu")  # fmt: skip

INDUSTRIES = ("TECHNOLOGIES", "INFOTECH", "SOLUTIONS", "ENGINEERING", "PHARMA", "LOGISTICS",
              "TEXTILES", "RETAIL", "FOODS", "CONSULTANCY", "SYSTEMS", "AUTOMOTIVE")  # fmt: skip

GIG_PLATFORMS = (
    "ZOOVIK DELIVERY PVT LTD",
    "RIDORA MOBILITY PVT LTD",
    "TASKEVO SERVICES PVT LTD",
    "PARCELVYA LOGISTICS PVT LTD",
)
LENDERS = ("DHANVRIKSH FINANCE LTD", "KRAMIKA CAPITAL LTD", "NIVESHA HOME LOANS LTD")

ELECTRICITY = "VIDYUTA POWER DISTRIBUTION LTD"
POSTPAID = "TARANG TELECOM POSTPAID"
PREPAID = "TARANG TELECOM PREPAID"
BROADBAND = "JALDINET BROADBAND"
SUBSCRIPTIONS = {  # name -> monthly plan prices in INR
    "STREAMVERSE": (149, 299, 499, 649),
    "CINEMOX": (199, 399),
    "SURTAAL MUSIC": (149,),
}


def coined(rng: np.random.Generator, syllables: int = 3) -> str:
    return "".join(str(s) for s in rng.choice(_SYLLABLES, size=syllables)).upper()


def employer_name(rng: np.random.Generator, fake: Faker) -> str:
    r = rng.random()
    if r < 0.75:
        return f"{coined(rng)} {rng.choice(INDUSTRIES)} PVT LTD"
    if r < 0.90:
        return f"{coined(rng)} {rng.choice(INDUSTRIES)} LLP"
    return f"{fake.last_name().upper()} & SONS PVT LTD"


def business_client(rng: np.random.Generator) -> str:
    return f"{coined(rng)} {rng.choice(INDUSTRIES)} PVT LTD"


# Spending categories: (relative frequency, typical amount in INR before scaling).
SPEND_CATEGORIES = {
    "groceries": (0.28, 450),
    "dining": (0.22, 250),
    "transport": (0.20, 120),
    "fuel": (0.08, 900),
    "shopping": (0.12, 900),
    "health": (0.10, 300),
}


def favourite_merchants(rng: np.random.Generator, fake: Faker) -> dict[str, list[str]]:
    """The handful of places one person keeps paying, per category (people are regulars)."""

    def surname() -> str:
        return fake.last_name().upper()

    return {
        "groceries": [f"{surname()} KIRANA STORE", f"{coined(rng)} SUPERMART",
                      f"{surname()} GENERAL STORES"],
        "dining": [f"{coined(rng)} CAFE", f"HOTEL {coined(rng, 2)}", "KHAANAGO FOODS"],
        "transport": [f"AUTO {fake.first_name().upper()}", f"{coined(rng)} CABS"],
        "fuel": [f"{coined(rng)} FUEL STATION"],
        "shopping": ["BAZAARIO ONLINE", f"{coined(rng)} FASHIONS"],
        "health": [f"{surname()} MEDICAL STORE", f"{coined(rng)} CLINIC"],
    }  # fmt: skip
