from __future__ import annotations

import pytest

from pokeprice import db
from pokeprice.config import settings

FIXTURE_CARDS = [
    dict(card_id="base1-4", name="Charizard", pokemon="Charizard", supertype="Pokémon", subtypes=["Stage 2"],
         set_id="base1", set_name="Base", series="Base", release_date="1999-01-09", number="4",
         rarity="Rare Holo", artist="Mitsuhiro Arita", dex_number=6),
    dict(card_id="base1-58", name="Pikachu", pokemon="Pikachu", supertype="Pokémon", subtypes=["Basic"],
         set_id="base1", set_name="Base", series="Base", release_date="1999-01-09", number="58",
         rarity="Common", artist="Mitsuhiro Arita", dex_number=25),
    dict(card_id="swsh7-215", name="Umbreon VMAX", pokemon="Umbreon", supertype="Pokémon", subtypes=["VMAX"],
         set_id="swsh7", set_name="Evolving Skies", series="Sword & Shield", release_date="2021-08-27",
         number="215", rarity="Rare Secret", artist="KEIICHIRO ITO", dex_number=197),
    dict(card_id="sv3pt5-199", name="Charizard ex", pokemon="Charizard", supertype="Pokémon", subtypes=["ex"],
         set_id="sv3pt5", set_name="151", series="Scarlet & Violet", release_date="2023-09-22",
         number="199", rarity="Special Illustration Rare", artist="mingo", dex_number=6),
    dict(card_id="sv4-100", name="Bidoof", pokemon="Bidoof", supertype="Pokémon", subtypes=["Basic"],
         set_id="sv4", set_name="Paradox Rift", series="Scarlet & Violet", release_date="2023-11-03",
         number="100", rarity="Common", artist=None, dex_number=399),
    dict(card_id="sm12-1", name="Lillie's Full Force", pokemon="Lillie", supertype="Trainer", subtypes=["Supporter"],
         set_id="sm12", set_name="Cosmic Eclipse", series="Sun & Moon", release_date="2019-11-01",
         number="230", rarity="Rare Ultra", artist="Hitoshi Ariga", dex_number=None),
]
FIXTURE_PRICES = {"base1-4": 950.0, "base1-58": 6.0, "swsh7-215": 1700.0, "sv3pt5-199": 220.0,
                  "sv4-100": 0.25, "sm12-1": 80.0}


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(settings, "db_path", path)
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "synthetic_years", 3)
    return path


@pytest.fixture
def seeded_db(tmp_db):
    db.upsert_cards(FIXTURE_CARDS)
    db.insert_snapshots({"card_id": cid, "date": "2026-11-20", "source": "pokemontcg", "market": p,
                         "is_synthetic": 0} for cid, p in FIXTURE_PRICES.items())
    return tmp_db
