"""Daily price snapshot: the job that builds up real price history over time.

1. pokemontcg.io (free, no key): batched requests of 25 cards. Its prices update irregularly.
2. PokéWallet (if POKEWALLET_API_KEY is set): searches once per Pokémon (100 results/page);
   each result carries today's TCGplayer price, so one search prices every tracked card of
   that Pokémon. A full sweep is a few hundred requests; each run uses at most
   `max_requests` (95) to stay under the free tier's 100/hour, so run it hourly and the
   rest of the cards are covered over the day. Cards already priced today are skipped.
"""

from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd

from .. import db
from ..config import settings
from ..providers.base import RateLimitExceeded
from ..providers.pokemontcg import PokemonTCGProvider


def run_snapshot(limit: int | None = None, max_requests: int = 95, use_pokewallet: bool = True,
                 log: Callable[[str], None] = print) -> dict:
    cards = db.load_cards()
    if cards.empty:
        log("No cards yet. Run `python scripts/refresh.py --universe` first.")
        return {"pokemontcg": 0, "pokewallet": 0}
    if limit:
        cards = cards.head(limit)
    counts = {"pokemontcg": 0, "pokewallet": 0}

    # 1. pokemontcg.io batch quotes
    try:
        quotes = PokemonTCGProvider().quotes_for(cards["card_id"].tolist())
        counts["pokemontcg"] = db.insert_snapshots(q.as_row() for q in quotes)
        log(f"pokemontcg.io: saved {counts['pokemontcg']} prices")
    except Exception as e:  # keep going with the other source
        log(f"pokemontcg.io failed: {e}")

    # 2. PokéWallet: one search per Pokémon prices all of its cards
    if use_pokewallet and settings.pokewallet_api_key:
        counts["pokewallet"] = _pokewallet_snapshot(cards, max_requests, log)
    elif use_pokewallet:
        log("PokéWallet skipped: set POKEWALLET_API_KEY in .env to use it.")
    return counts


NO_MATCH = "-"


def _pokewallet_snapshot(cards: pd.DataFrame, max_requests: int, log: Callable[[str], None]) -> int:
    from ..features.popularity import popularity_score
    from ..providers.matching import search_term
    from ..providers.pokewallet import PokeWalletProvider

    pw = PokeWalletProvider()
    have = db.load_prices(include_synthetic=False)
    recent = pd.Timestamp(date.today()) - pd.Timedelta(days=1)
    done = set(have.loc[(have["source"] == "pokewallet") & (have["date"] >= recent), "card_id"]) \
        if not have.empty else set()
    variants = (have[have["source"] == "pokemontcg"].sort_values("date").groupby("card_id")["variant"].last()
                if not have.empty else pd.Series(dtype=str))
    todo = cards[~cards["card_id"].isin(done) & (cards["pokewallet_id"].fillna("") != NO_MATCH)].copy()
    if todo.empty:
        log("PokéWallet: every card already has a price from today.")
        return 0
    todo["_term"] = [search_term(c) for c in todo.to_dict("records")]
    todo["_pop"] = [popularity_score(p, s, d) for p, s, d in zip(todo["pokemon"], todo["supertype"], todo["dex_number"])]
    order = todo.groupby("_term")["_pop"].max().sort_values(ascending=False).index

    saved = matched = unmatched = 0
    try:
        for term in order:
            pages_left = max_requests - pw.requests_made
            if not term or pages_left <= 0:
                break
            results, complete = pw.search_all(term, max_pages=min(4, pages_left))
            rows = []
            for card in todo[todo["_term"] == term].to_dict("records"):
                raw = pw.match(card, results)
                if raw is None:
                    if complete:  # searched everything for this name: don't try again
                        db.set_provider_id(card["card_id"], "pokewallet_id", NO_MATCH)
                        unmatched += 1
                    continue
                if card.get("pokewallet_id") != raw["id"]:
                    db.set_provider_id(card["card_id"], "pokewallet_id", raw["id"])
                    matched += 1
                q = pw.parse_quote(card["card_id"], raw, variants.get(card["card_id"]))
                if q:
                    rows.append(q.as_row())
            saved += db.insert_snapshots(rows)
    except RateLimitExceeded:
        log("PokéWallet rate limit reached; the next run will continue.")
    left = len(todo) - saved - unmatched
    log(f"PokéWallet: saved {saved} prices ({matched} newly matched, {unmatched} without a match) "
        f"using {pw.requests_made} requests (remaining {pw.remaining()}); about {max(left, 0)} cards left "
        f"for later runs")
    return saved
