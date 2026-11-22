"""Choose which cards to track.

PokéWallet's free tier allows 1,000 requests/day, so we track a few hundred cards that matter
most to collectors: vintage WOTC holos, gold stars and shinings, modern chase cards, and
popular Pokémon across eras. Cards are ranked by a priority score and the top N are kept.
"""

from __future__ import annotations

import math
import random
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

CHASE_RARITIES = ["Rare Holo Star", "Rare Shining", "Rare Holo LV.X", "Rare Secret",
                  "Special Illustration Rare", "Hyper Rare", "Rare Rainbow", "Illustration Rare",
                  "Trainer Gallery Rare Holo", "Shiny Ultra Rare"]

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
    """Fetch candidate cards from pokemontcg.io and keep the highest-priority ones that have a price."""
    max_cards = max_cards or settings.max_cards
    provider = provider or PokemonTCGProvider()
    seen: dict[str, tuple[dict, PriceQuote]] = {}

    def run(q: str, page_size: int, max_pages: int) -> bool:
        n = 0
        try:
            for raw in provider.search(q, page_size=page_size, max_pages=max_pages):
                if raw["id"] in seen:
                    continue
                quote = provider.to_quote(raw)
                if quote:
                    seen[raw["id"]] = (provider.to_card_row(raw), quote)
                    n += 1
        except Exception as e:  # one failing query shouldn't sink the whole universe
            log(f"  query failed ({type(e).__name__}): {q}")
            return False
        log(f"  +{n:>4} cards  {q}")
        return True

    failed = [q for q in QUERIES if not run(q, 250, 4)]
    if failed:
        log(f"Retrying {len(failed)} failed queries with smaller pages…")
        for q in failed:
            run(q, 100, 10)

    ranked_all = sorted(seen.values(), key=lambda cq: priority(cq[0], cq[1].market), reverse=True)
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
    log(f"Universe: kept {len(ranked)} of {len(seen)} candidates")
    return [c for c, _ in ranked], [q for _, q in ranked]
