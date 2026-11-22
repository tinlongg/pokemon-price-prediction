"""Real price history from TCGapi's free tier (100 requests/day, 7 days of history per call).

Each run:
  1. Matching: for cards without a TCGapi id, search once per Pokémon (most popular first),
     match our cards by number and set name, and save each match's current price.
     Cards with no match are marked '-' so they aren't searched again.
  2. History: for matched cards, fetch the last 7 days of daily prices, starting with the
     cards whose TCGapi history is oldest. At ~90 cards a day, every tracked card is
     revisited about once a week, which keeps its real history continuous.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .. import db
from ..config import settings
from ..features.popularity import popularity_score
from ..providers.base import RateLimitExceeded
from ..providers.tcgapi import TCGApiProvider, best_match, consistent, search_term

NO_MATCH = "-"


def reference_prices() -> pd.Series:
    """Latest price per card from the other real sources (PokéWallet first, then pokemontcg.io)."""
    real = db.load_prices(include_synthetic=False)
    if real.empty:
        return pd.Series(dtype=float)
    other = real[real["source"] != "tcgapi"].copy()
    other["_p"] = other["source"].map({"pokewallet": 0}).fillna(1)
    return other.sort_values(["card_id", "_p", "date"], ascending=[True, True, False]).groupby("card_id")["market"].first()


def drop_inconsistent(log: Callable[[str], None] = print) -> int:
    """Remove TCGapi rows for cards whose TCGapi price disagrees with the reference price,
    and stop using TCGapi for those cards (they matched a different printing or product)."""
    ref = reference_prices()
    with db.connect() as conn:
        tcg = pd.read_sql("SELECT card_id, date, market FROM price_snapshots WHERE source='tcgapi'", conn)
        if tcg.empty or ref.empty:
            return 0
        latest = tcg.sort_values("date").groupby("card_id")["market"].last()
        bad = [cid for cid, m in latest.items() if cid in ref.index and not consistent(m, ref[cid])]
        for cid in bad:
            conn.execute("DELETE FROM price_snapshots WHERE source='tcgapi' AND card_id=?", (cid,))
            conn.execute("UPDATE cards SET tcgapi_id=?, tcgapi_printing=NULL WHERE card_id=?", (NO_MATCH, cid))
    if bad:
        log(f"TCGapi: dropped {len(bad)} cards whose TCGapi price disagreed with other sources")
    return len(bad)


def run_backfill(max_requests: int = 90, match_share: float = 0.5,
                 log: Callable[[str], None] = print, api: TCGApiProvider | None = None) -> dict:
    if api is None and not settings.tcgapi_key:
        log("TCGapi backfill skipped: set TCGAPI_KEY in .env to use it.")
        return {"matched": 0, "history_rows": 0, "requests": 0}
    api = api or TCGApiProvider()
    cards = db.load_cards()
    if cards.empty:
        return {"matched": 0, "history_rows": 0, "requests": 0}
    cards["_pop"] = [popularity_score(p, s, d) for p, s, d in
                     zip(cards["pokemon"], cards["supertype"], cards["dex_number"])]
    stats = {"matched": 0, "unmatched": 0, "search_prices": 0, "history_rows": 0}
    drop_inconsistent(log)
    cards = db.load_cards()
    cards["_pop"] = [popularity_score(p, s, d) for p, s, d in
                     zip(cards["pokemon"], cards["supertype"], cards["dex_number"])]
    ref = reference_prices()

    # --- 1. match unmatched cards, one search per Pokémon -------------------------------
    unmatched = cards[cards["tcgapi_id"].isna()].copy()
    unmatched["_term"] = [search_term(c) for c in unmatched.to_dict("records")]
    budget = int(max_requests * match_share) if (cards["tcgapi_id"].notna() & (cards["tcgapi_id"] != NO_MATCH)).any() \
        else max_requests
    try:
        order = unmatched.groupby("_term")["_pop"].max().sort_values(ascending=False).index
        for term in order:
            if not term or api.requests_made + 2 > budget:
                break
            rows = api.search_all(term, max_pages=2)
            quotes = []
            for card in unmatched[unmatched["_term"] == term].to_dict("records"):
                row = best_match(card, rows)
                if row is None:
                    db.set_tcgapi_match(card["card_id"], NO_MATCH, None)
                    stats["unmatched"] += 1
                    continue
                db.set_tcgapi_match(card["card_id"], str(row["id"]), row.get("printing"))
                stats["matched"] += 1
                q = api.quote_from_search_row(card["card_id"], row)
                if q and consistent(q.market, ref.get(card["card_id"])):
                    quotes.append(q.as_row())
            stats["search_prices"] += db.insert_snapshots(quotes)
    except RateLimitExceeded:
        log("TCGapi daily limit reached while matching.")

    # --- 2. history for matched cards, stalest first ------------------------------------
    cards = db.load_cards()
    matched = cards[cards["tcgapi_id"].notna() & (cards["tcgapi_id"] != NO_MATCH)].copy()
    if not matched.empty and api.requests_made < max_requests:
        real = db.load_prices(include_synthetic=False)
        tcg = real[real["source"] == "tcgapi"] if not real.empty else real
        last_hist = tcg.groupby("card_id")["date"].max() if not tcg.empty else pd.Series(dtype="datetime64[ns]")
        variants = (real[real["source"] == "pokemontcg"].sort_values("date").groupby("card_id")["variant"].last()
                    if not real.empty else pd.Series(dtype=str))
        matched["_last"] = matched["card_id"].map(last_hist)
        # A search price gives one day; treat cards with fewer than 3 TCGapi days as never fetched.
        n_days = tcg.groupby("card_id")["date"].nunique() if not tcg.empty else pd.Series(dtype=int)
        matched.loc[matched["card_id"].map(n_days).fillna(0) < 3, "_last"] = pd.NaT
        matched["_pop"] = [popularity_score(p, s, d) for p, s, d in
                           zip(matched["pokemon"], matched["supertype"], matched["dex_number"])]
        matched = matched.sort_values(["_last", "_pop"], ascending=[True, False], na_position="first")
        try:
            for card in matched.to_dict("records"):
                if api.requests_made >= max_requests:
                    break
                quotes = api.history(card["card_id"], card["tcgapi_id"], card.get("tcgapi_printing"),
                                     variants.get(card["card_id"]), reference=ref.get(card["card_id"]))
                if not quotes:  # no printing close to the reference price: wrong match
                    db.set_tcgapi_match(card["card_id"], NO_MATCH, None)
                    stats["rejected"] = stats.get("rejected", 0) + 1
                    continue
                stats["history_rows"] += db.insert_snapshots(q.as_row() for q in quotes)
        except RateLimitExceeded:
            log("TCGapi daily limit reached while fetching history.")

    stats["requests"] = api.requests_made
    log(f"TCGapi: matched {stats['matched']} cards ({stats['unmatched']} without a match), "
        f"saved {stats['search_prices']} current prices and {stats['history_rows']} daily history prices "
        f"using {api.requests_made} requests ({api.daily_remaining} left today)")
    return stats
