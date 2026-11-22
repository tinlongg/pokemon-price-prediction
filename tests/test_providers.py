from __future__ import annotations

import httpx
import respx

from pokeprice.providers.pokemontcg import PokemonTCGProvider, pick_variant
from pokeprice.providers.pokewallet import PokeWalletProvider

POKEMONTCG_CARD = {
    "id": "base1-4", "name": "Charizard", "supertype": "Pokémon", "subtypes": ["Stage 2"],
    "set": {"id": "base1", "name": "Base", "series": "Base", "releaseDate": "1999/01/09"},
    "number": "4", "artist": "Mitsuhiro Arita", "rarity": "Rare Holo", "nationalPokedexNumbers": [6],
    "images": {"small": "s.png", "large": "l.png"},
    "tcgplayer": {"updatedAt": "2026/10/15", "prices": {
        "holofoil": {"low": 499.99, "mid": 587.98, "high": 3499.1, "market": 954.65}}},
}

POKEWALLET_CARD = {
    "id": "pk_123",
    "card_info": {"name": "Charizard", "set_name": "Base Set", "card_number": "4/102"},
    "tcgplayer": {"prices": [
        {"sub_type_name": "Normal", "market_price": None, "mid_price": None},
        {"sub_type_name": "Holofoil", "low_price": 500, "mid_price": 600, "high_price": 3000,
         "market_price": 960.5},
        {"sub_type_name": "1st Edition Holofoil", "market_price": 9000},
    ]},
    "cardmarket": None,
}


def test_pokemontcg_conversion():
    row = PokemonTCGProvider.to_card_row(POKEMONTCG_CARD)
    assert row["release_date"] == "1999-01-09" and row["pokemon"] == "Charizard" and row["dex_number"] == 6
    q = PokemonTCGProvider.to_quote(POKEMONTCG_CARD)
    assert q.market == 954.65 and q.variant == "holofoil"
    assert q.date == "2026-10-15"  # dated by the price's update time, not fetch time


def test_pick_variant_prefers_holofoil_over_first_edition():
    v, _ = pick_variant({"1stEditionHolofoil": {"market": 9000}, "unlimitedHolofoil": {"market": 500}})
    assert v == "unlimitedHolofoil"
    assert pick_variant({}) is None


@respx.mock
def test_pokemontcg_search_paginates():
    route = respx.get("https://api.pokemontcg.io/v2/cards").mock(side_effect=[
        httpx.Response(200, json={"data": [POKEMONTCG_CARD], "totalCount": 2}),
        httpx.Response(200, json={"data": [{**POKEMONTCG_CARD, "id": "base1-5"}], "totalCount": 2}),
    ])
    out = list(PokemonTCGProvider(api_key="").search("name:Charizard", page_size=1))
    assert [c["id"] for c in out] == ["base1-4", "base1-5"] and route.call_count == 2


def test_pokewallet_parse_quote_uses_preferred_variant():
    q = PokeWalletProvider.parse_quote("base1-4", POKEWALLET_CARD)
    assert q.market == 960.5 and q.variant == "Holofoil" and q.source == "pokewallet"
    q1 = PokeWalletProvider.parse_quote("base1-4", POKEWALLET_CARD, preferred_variant="1stEditionHolofoil")
    assert q1.market == 9000


def _pw(id_, name, set_name, number, prices):
    return {"id": id_, "card_info": {"name": name, "set_name": set_name, "card_number": number,
                                     "product_type": "card"},
            "tcgplayer": {"prices": [{"sub_type_name": k, "market_price": v, "updated_at": "2026-11-21T13:30:27"}
                                     for k, v in prices.items()]}}


PW_RESULTS = [  # shapes from the live /search response
    _pw("pk_bs2", "Charizard", "Base Set 2", "004/130", {"Holofoil": 513.67}),
    _pw("pk_shadowless", "Charizard", "Base Set (Shadowless)", "004/102",
        {"Unlimited Holofoil": 2172.24, "1st Edition Holofoil": None}),
    _pw("pk_base", "Charizard", "Base Set", "004/102", {"Holofoil": 1001.05}),
    _pw("pk_stamp", "Charmander - 18/147 (#1 Charizard Stamped)", "Battle Academy", "018/147", {"Normal": 0.68}),
]


def test_pokewallet_match_by_number_and_set():
    card = {"card_id": "base1-4", "name": "Charizard", "set_name": "Base", "number": "4"}
    raw = PokeWalletProvider.match(card, PW_RESULTS)
    assert raw["id"] == "pk_base"
    q = PokeWalletProvider.parse_quote("base1-4", raw, "holofoil")
    assert q.market == 1001.05 and q.date == "2026-11-21"   # dated by PokéWallet's price update
    # A saved id wins over matching.
    assert PokeWalletProvider.match({**card, "pokewallet_id": "pk_bs2"}, PW_RESULTS)["id"] == "pk_bs2"


@respx.mock
def test_pokewallet_search_all_paginates():
    respx.get("https://api.pokewallet.io/search").mock(side_effect=[
        httpx.Response(200, json={"results": PW_RESULTS[:2], "pagination": {"page": 1, "total_pages": 2}}),
        httpx.Response(200, json={"results": PW_RESULTS[2:], "pagination": {"page": 2, "total_pages": 2}}),
    ])
    pw = PokeWalletProvider(api_key="test")
    rows, complete = pw.search_all("Charizard")
    assert len(rows) == 4 and complete and pw.requests_made == 2


# --- TCGapi (shapes copied from live responses) -------------------------------------
from pokeprice.providers.tcgapi import TCGApiProvider, best_match, number_core  # noqa: E402

TCGAPI_ROWS = [
    {"id": 33831, "name": "Charizard", "number": "004/102", "set_name": "Base Set (Shadowless)",
     "printing": "Holofoil", "product_type": "Cards", "market_price": 1000, "market_price_as_of": "2026-11-21"},
    {"id": 34273, "name": "Charizard", "number": "004/102", "set_name": "Base Set",
     "printing": "Holofoil", "product_type": "Cards", "market_price": 1001.05, "market_price_as_of": "2026-11-21"},
    {"id": 30057, "name": "Charizard", "number": "004/130", "set_name": "Base Set 2",
     "printing": "Holofoil", "product_type": "Cards", "market_price": 513.67, "market_price_as_of": "2026-11-21"},
    {"id": 26300, "name": "Charizard", "number": "11/108", "set_name": "XY - Evolutions",
     "printing": "Holofoil", "product_type": "Cards", "market_price": 90.9, "market_price_as_of": "2026-11-21"},
    {"id": 4287, "name": "Charizard (Celebrations Metal Card)", "number": "004/102",
     "set_name": "Miscellaneous Cards & Products", "product_type": "Cards", "market_price": None},
]


def test_number_core():
    assert number_core("004/102") == number_core("4") == "4"
    assert number_core("TG03/TG30") == number_core("TG03") == "tg3"


def test_tcgapi_matching_prefers_exact_set():
    base = {"name": "Charizard", "number": "4", "set_name": "Base"}
    assert best_match(base, TCGAPI_ROWS)["id"] == 34273           # not Shadowless, not Base Set 2
    evo = {"name": "Charizard", "number": "11", "set_name": "Evolutions"}
    assert best_match(evo, TCGAPI_ROWS)["id"] == 26300
    assert best_match({"name": "Charizard", "number": "4", "set_name": "Team Rocket"}, TCGAPI_ROWS) is None


def test_tcgapi_history_keeps_one_printing():
    points = [
        {"date": "2026-11-15", "printing": "Holofoil", "market_price": 987.13, "low_price": 169.99},
        {"date": "2026-11-15", "printing": "Reverse Holofoil", "market_price": 20.0},
        {"date": "2026-11-16", "printing": "Holofoil", "market_price": 1001.05, "low_price": None},
    ]
    qs = TCGApiProvider.parse_history("base1-4", points, printing="Holofoil")
    assert [q.market for q in qs] == [987.13, 1001.05] and all(q.source == "tcgapi" for q in qs)
    qs = TCGApiProvider.parse_history("x", points, variant="reverseHolofoil")
    assert [q.market for q in qs] == [20.0]


@respx.mock
def test_tcgapi_search_paginates_and_tracks_quota():
    respx.get("https://api.tcgapi.dev/v1/search").mock(side_effect=[
        httpx.Response(200, json={"data": TCGAPI_ROWS[:2], "meta": {"has_more": True},
                                  "rate_limit": {"daily_remaining": 50}}),
        httpx.Response(200, json={"data": TCGAPI_ROWS[2:], "meta": {"has_more": False},
                                  "rate_limit": {"daily_remaining": 49}}),
    ])
    api = TCGApiProvider(api_key="test")
    rows = api.search_all("Charizard")
    assert len(rows) == 5 and api.requests_made == 2 and api.daily_remaining == 49


def test_tcgapi_history_uses_reference_price_to_pick_printing():
    points = [
        {"date": "2026-11-15", "printing": "Holofoil", "market_price": 150.0},
        {"date": "2026-11-15", "printing": "1st Edition Holofoil", "market_price": 390.0},
        {"date": "2026-11-16", "printing": "1st Edition Holofoil", "market_price": 385.0},
    ]
    qs = TCGApiProvider.parse_history("base5-4", points, printing="Holofoil", reference=385.11)
    assert {q.variant for q in qs} == {"1st Edition Holofoil"}
    # Nothing near the reference: treat as a wrong match.
    assert TCGApiProvider.parse_history("neo4-107", points, reference=4799.99) == []
