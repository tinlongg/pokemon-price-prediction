"""Nostalgia: era, card age, and how much buying power the generation that grew up with it has now."""

from __future__ import annotations

import math
from datetime import date

ERAS = [  # (name, first year, era index)
    ("WOTC", 1999, 0), ("EX", 2003, 1), ("DP/HGSS", 2007, 2), ("BW", 2011, 3),
    ("XY", 2014, 4), ("SM", 2017, 5), ("SWSH", 2020, 6), ("SV", 2023, 7), ("Mega Evolution", 2025, 8),
]


def era_of(release: date | None) -> tuple[str, int]:
    if release is None:
        return "Unknown", 7
    name, idx = ERAS[0][0], ERAS[0][2]
    for n, year, i in ERAS:
        if release.year >= year:
            name, idx = n, i
    if name == "Mega Evolution" and release < date(2025, 9, 26):
        name, idx = "SV", 7
    return name, idx


def age_years(release: date | None, as_of: date) -> float:
    if release is None:
        return 0.0
    return max(0.0, (as_of - release).days / 365.25)


def cohort_score(release: date | None, as_of: date) -> float:
    """People who were about 10 when a set came out are its nostalgia buyers. Spending on
    childhood collectibles peaks once they are around 30-35 with disposable income.
    Returns 0-1: how close that cohort is to peak buying age right now."""
    if release is None:
        return 0.0
    cohort_age = 10 + age_years(release, as_of)
    return math.exp(-((cohort_age - 33) ** 2) / (2 * 8 ** 2))


def nostalgia_score(release: date | None, as_of: date, gen: int | None, popularity: float) -> float:
    """Single 0-1 blend used by the simulator and shown in the UI."""
    era_name, _ = era_of(release)
    wotc = 1.0 if era_name == "WOTC" else 0.0
    gen12 = 1.0 if gen in (1, 2) else 0.0
    age = min(age_years(release, as_of) / 25, 1.0)
    return min(1.0, 0.35 * cohort_score(release, as_of) + 0.25 * wotc + 0.15 * gen12
               + 0.15 * age + 0.10 * popularity)
