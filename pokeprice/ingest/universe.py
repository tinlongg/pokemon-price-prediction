"""Choose which cards to track.

1. Every Illustration Rare and Special Illustration Rare (Scarlet & Violet onwards, about
   1,000 cards across 25 sets). These are always kept, even without a pokemontcg.io price:
   PokéWallet prices them, and newer sets often have no pokemontcg.io prices at all.
2. Up to `max_cards` (600) other cards that matter to collectors: vintage WOTC holos, gold
   stars and shinings, other modern chase cards, and popular Pokémon across eras. These are
   ranked by a priority score, with a quarter of the slots spread across eras.

PokéWallet prices all of a Pokémon's cards from one search, so a full price refresh is about
700 requests, within its free 1,000/day.
"""

from __future__ import annotations

import math
import random
import time
from datetime import date
from typing import Callable

from ..config import settings
from ..features.nostalgia import era_of
from ..features.popularity import POPULAR_NAMES, popularity_score
from ..features.rarity import artist_premium, rarity_tier
from ..providers.base import PriceQuote
from ..providers.pokemontcg import PokemonTCGProvider

LOW = '-rarity:Common -rarity:Uncommon'
VINTAGE_SETS = ["base1", "base2", "base3", "base4", "base5", "base6", "gym1", "gym2",
                "neo1", "neo2", "neo3", "neo4", "ecard1", "ecard2", "ecard3"]
ICONIC_MODERN_SETS = ["xy12", "sm115", "sma", "swsh45sv", "cel25", "cel25c", "swsh7",
                      "swsh12pt5gg", "sv3pt5", "sv4pt5", "sv8pt5"]

# Always tracked, every card of these rarities. They only exist in these series.
REQUIRED_RARITIES = ["Illustration Rare", "Special Illustration Rare"]
REQUIRED_SERIES = {"Scarlet & Violet", "Mega Evolution"}
_RARITY_CLAUSE = "(" + " OR ".join(f'rarity:"{r}"' for r in REQUIRED_RARITIES) + ")"


def required_queries(provider: PokemonTCGProvider, log: Callable[[str], None] = print) -> list[str]:
    """One small query per Scarlet & Violet / Mega Evolution set: pokemontcg.io keeps failing
    (HTTP 500) on a single query for every Illustration Rare, but per-set queries of ~45 cards
    are reliable. Falls back to the big per-rarity queries if the set list can't be fetched."""
    try:
        sets = provider.http.get("/sets", {"orderBy": "-releaseDate", "pageSize": 100,
                                           "select": "id,series,releaseDate"}).get("data", [])
        ids = [s["id"] for s in sets if s.get("series") in REQUIRED_SERIES]
    except Exception as e:
        log(f"  couldn't list sets ({type(e).__name__}); querying IRs/SIRs by rarity instead")
        ids = []
    if not ids:
        return [f'rarity:"{r}"' for r in REQUIRED_RARITIES]
    return [f"set.id:{sid} {_RARITY_CLAUSE}" for sid in ids]

CHASE_RARITIES = ["Rare Holo Star", "Rare Shining", "Rare Holo LV.X", "Rare Secret",
                  "Hyper Rare", "Rare Rainbow", "Trainer Gallery Rare Holo", "Shiny Ultra Rare"]

# pokemontcg.io struggles with big OR queries (HTTP 500), so each query is kept simple.
QUERIES = (
    [f"set.id:{s} {LOW}" for s in VINTAGE_SETS]
    + [f'rarity:"{r}"' for r in CHASE_RARITIES]
    + [f"set.id:{s} {LOW}" for s in ICONIC_MODERN_SETS]
    + [f'name:"{n}" {LOW}' for n in POPULAR_NAMES[:20]]
)


def priority(card: dict, price: float | None) -> float:
    release = date.fromisoformat(card["release_date"]) if card.get("release_date") else None
    _, era_idx = era_of(release)
    pop = popularity_score(card.get("pokemon"), card.get("supertype"), card.get("dex_number"))
    value = math.log10(price) if price and price > 1 else 0.0
    return (3.0 * pop + 1.5 * rarity_tier(card.get("rarity")) / 8 + 1.0 * (era_idx == 0)
            + 0.3 * artist_premium(card.get("artist")) + 0.5 * value)


def build_universe(max_cards: int | None = None, provider: PokemonTCGProvider | None = None,
                   log: Callable[[str], None] = print) -> tuple[list[dict], list[PriceQuote]]:
    """Fetch candidates from pokemontcg.io. Returns (card rows, pokemontcg.io price quotes);
    every IR/SIR is included whether or not it has a price, plus the top `max_cards` others."""
    max_cards = max_cards or settings.max_cards
    provider = provider or PokemonTCGProvider()
    required: dict[str, tuple[dict, PriceQuote | None]] = {}
    others: dict[str, tuple[dict, PriceQuote]] = {}

    def run(q: str, page_size: int, max_pages: int, is_required: bool) -> bool:
        n = 0
        try:
            for raw in provider.search(q, page_size=page_size, max_pages=max_pages):
                if raw["id"] in required or (raw["id"] in others and not is_required):
                    continue
                quote = provider.to_quote(raw)
                if is_required:
                    required[raw["id"]] = (provider.to_card_row(raw), quote)
                    others.pop(raw["id"], None)
                elif quote:
                    others[raw["id"]] = (provider.to_card_row(raw), quote)
                else:
                    continue
                n += 1
        except Exception as e:  # one failing query shouldn't sink the whole universe
            log(f"  query failed ({type(e).__name__}): {q}")
            return False
        log(f"  +{n:>4} cards  {q}")
        return True

    jobs = [(q, True) for q in required_queries(provider, log)] + [(q, False) for q in QUERIES]
    failed = [(q, r) for q, r in jobs if not run(q, 250, 4, r)]
    for attempt in (1, 2):   # pokemontcg.io errors are often brief: pause, then retry smaller
        if not failed:
            break
        log(f"Retrying {len(failed)} failed queries with smaller pages (pass {attempt})…")
        time.sleep(15 * attempt)
        failed = [(q, r) for q, r in failed if not run(q, 100, 10, r)]
    if failed:
        log(f"Still failing after retries: {[q for q, _ in failed]}")

    ranked_all = sorted(others.values(), key=lambda cq: priority(cq[0], cq[1].market), reverse=True)
    # 75% of slots go to the top-priority cards; the rest are spread across eras so the model
    # also sees ordinary cards and can learn what popularity and nostalgia add.
    n_top = int(max_cards * 0.75)
    ranked, rest = ranked_all[:n_top], ranked_all[n_top:]
    by_era: dict[int, list] = {}
    for cq in rest:
        rd = cq[0].get("release_date")
        by_era.setdefault(era_of(date.fromisoformat(rd) if rd else None)[1], []).append(cq)
    rng = random.Random(42)
    for lst in by_era.values():
        rng.shuffle(lst)
    while len(ranked) < max_cards and any(by_era.values()):
        for lst in by_era.values():
            if lst and len(ranked) < max_cards:
                ranked.append(lst.pop())
    chosen = list(required.values()) + ranked
    unpriced = sum(1 for _, q in required.values() if q is None)
    log(f"Universe: {len(required)} IR/SIR cards ({unpriced} without a pokemontcg.io price) "
        f"+ {len(ranked)} of {len(others)} other candidates")
    return [c for c, _ in chosen], [q for _, q in chosen if q is not None]
