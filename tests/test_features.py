from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pokeprice.features.events import build_calendar, event_features
from pokeprice.features.market import add_market_features, daily_panel
from pokeprice.features.nostalgia import cohort_score, era_of
from pokeprice.features.popularity import generation_of, popularity_score, species_of
from pokeprice.features.rarity import artist_premium, is_special_art, rarity_tier
from pokeprice.features.supply import reprint_risk


@pytest.mark.parametrize("name,supertype,expected", [
    ("Charizard", "Pokémon", "Charizard"),
    ("Blaine's Charizard", "Pokémon", "Charizard"),
    ("Umbreon VMAX", "Pokémon", "Umbreon"),
    ("Charizard ★ δ", "Pokémon", "Charizard"),
    ("Shining Mewtwo", "Pokémon", "Mewtwo"),
    ("Pikachu & Zekrom-GX", "Pokémon", "Pikachu"),
    ("Team Rocket's Mewtwo ex", "Pokémon", "Mewtwo"),
    ("Dark Charizard", "Pokémon", "Charizard"),
    ("M Rayquaza-EX", "Pokémon", "Rayquaza"),
    ("Lillie's Determination", "Trainer", "Lillie"),
])
def test_species_of(name, supertype, expected):
    assert species_of(name, supertype) == expected


def test_popularity_ranks_charizard_first():
    assert popularity_score("Charizard") == 1.0
    assert popularity_score("Pikachu") > popularity_score("Psyduck") > popularity_score("Bidoof", dex_number=399)
    # Unknown Gen 1 Pokémon get a nostalgia bump over unknown later ones.
    assert popularity_score("Paras", dex_number=46) > popularity_score("Bidoof", dex_number=399)
    assert generation_of(25) == 1 and generation_of(197) == 2 and generation_of(906) == 9


def test_rarity_and_art():
    assert rarity_tier("Special Illustration Rare") > rarity_tier("Illustration Rare") > rarity_tier("Rare Holo")
    assert rarity_tier(None) == 1
    assert is_special_art("Special Illustration Rare") == 1 and is_special_art("Rare Holo") == 0
    assert artist_premium("Mitsuhiro Arita") == 1.0 and artist_premium(None) == 0.0


def test_nostalgia_cohort_peaks_for_wotc_now():
    today = date(2026, 11, 21)
    assert era_of(date(1999, 1, 9))[0] == "WOTC"
    assert era_of(date(2023, 9, 22))[0] == "SV"
    # A 1999 card's cohort (~37 today) is near peak buying age; a 2025 card's (~11) is not.
    assert cohort_score(date(1999, 1, 9), today) > 0.8 > cohort_score(date(2025, 1, 1), today)


def test_reprint_risk_higher_for_new_popular_cards():
    assert reprint_risk(0.5, 1.0, 0) > reprint_risk(0.5, 0.2, 0) > reprint_risk(5.0, 1.0, 0)
    assert reprint_risk(0.5, 1.0, 1) < reprint_risk(0.5, 1.0, 0)


def test_event_features_count_down_to_pokemon_day():
    dates = pd.Series(pd.to_datetime(["2027-02-20", "2027-03-01"]))
    ev = event_features(dates, build_calendar(2020, 2028))
    assert ev["days_to_event"].iloc[0] == 7
    assert ev["days_since_event"].iloc[1] == 2


def test_market_features_returns():
    prices = pd.DataFrame({
        "card_id": "x", "date": pd.date_range("2026-01-01", periods=100, freq="D"),
        "market": np.linspace(10, 20, 100), "is_synthetic": 0})
    p = add_market_features(daily_panel(prices))
    last = p.iloc[-1]
    assert np.isclose(last["ret_30"], np.log(20 / p.iloc[-31]["market"]))
    assert last["drawdown_90"] == pytest.approx(0.0)


def test_daily_panel_forward_fills_gaps():
    prices = pd.DataFrame({"card_id": "x", "date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
                           "market": [10.0, 12.0], "is_synthetic": [0, 0]})
    p = daily_panel(prices)
    assert len(p) == 5 and p["market"].tolist() == [10, 10, 10, 10, 12]
