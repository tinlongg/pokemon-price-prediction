"""TCGapi (https://tcgapi.dev). Free tier: 100 requests/day, about 7 days of price history.

Endpoints (checked against the live API, November 2026):
  GET /v1/search?q=<name>&game=pokemon&per_page=200&page=N
      -> {"data": [{id, name, number, rarity, set_name, printing, market_price,
                    market_price_as_of, product_type, ...}], "meta": {total, has_more},
          "rate_limit": {daily_remaining, ...}}
      Search matches card names only (not set or number), so we search once per Pokémon
      and match our cards locally by number and set name. Each result carries a fresh price.
  GET /v1/cards/<id>/history?range=7d
      -> {"data": [{date, printing, market_price, low_price, avg_sales_price, sales_volume}]}
Auth: header X-API-Key.
"""

from __future__ import annotations

import numpy as np

from ..config import settings
from .base import HttpClient, PriceQuote, RateLimitExceeded, to_float
from .matching import best_match, match_score, number_core, search_term  # noqa: F401

# pokemontcg.io variant key -> TCGapi printing names, in preference order.
VARIANT_TO_PRINTING = {
    "holofoil": ["Holofoil"],
    "unlimitedHolofoil": ["Holofoil", "Unlimited Holofoil"],
    "normal": ["Normal"],
    "reverseHolofoil": ["Reverse Holofoil"],
    "1stEditionHolofoil": ["1st Edition Holofoil", "1st Edition"],
    "1stEditionNormal": ["1st Edition", "1st Edition Normal"],
}

MAX_LOG_GAP = 0.5   # ~65% apart: treat as a different printing or a wrong match


def consistent(price: float | None, reference: float | None) -> bool:
    if not price or not reference:
        return True
    return abs(np.log(price / reference)) <= MAX_LOG_GAP


class TCGApiProvider:
    name = "tcgapi"

    def __init__(self, api_key: str | None = None):
        key = api_key if api_key is not None else settings.tcgapi_key
        if not key:
            raise ValueError("TCGAPI_KEY is not set")
        self.http = HttpClient("https://api.tcgapi.dev/v1", headers={"X-API-Key": key}, min_interval=0.5)
        self.requests_made = 0
        self.daily_remaining: int | None = None

    def _get(self, path: str, params: dict) -> dict:
        if self.daily_remaining is not None and self.daily_remaining <= 0:
            raise RateLimitExceeded("TCGapi daily quota used up")
        data = self.http.get(path, params)
        self.requests_made += 1
        rl = data.get("rate_limit") or {}
        if "daily_remaining" in rl:
            self.daily_remaining = int(rl["daily_remaining"])
        return data

    def search_all(self, term: str, max_pages: int = 3) -> list[dict]:
        rows: list[dict] = []
        for page in range(1, max_pages + 1):
            data = self._get("/search", {"q": term, "game": "pokemon", "per_page": 200, "page": page})
            rows += data.get("data") or []
            if not (data.get("meta") or {}).get("has_more"):
                break
        return rows

    @staticmethod
    def quote_from_search_row(card_id: str, row: dict) -> PriceQuote | None:
        market = to_float(row.get("market_price"))
        day = str(row.get("market_price_as_of") or "")[:10]
        if not market or not day:
            return None
        return PriceQuote(card_id=card_id, source="tcgapi", market=market, low=to_float(row.get("low_price")),
                          variant=row.get("printing"), date=day)

    def history(self, card_id: str, tcgapi_id: str, printing: str | None = None,
                variant: str | None = None, reference: float | None = None,
                range_: str = "7d") -> list[PriceQuote]:
        data = self._get(f"/cards/{tcgapi_id}/history", {"range": range_})
        points = data.get("data") or []
        return self.parse_history(card_id, points, printing, variant, reference)

    @staticmethod
    def parse_history(card_id: str, points: list[dict], printing: str | None = None,
                      variant: str | None = None, reference: float | None = None) -> list[PriceQuote]:
        """Keep one printing so the series is consistent. With a reference price (from another
        source), pick the printing whose prices are closest to it and return nothing if none
        is within the tolerance (likely a wrong match). Otherwise prefer our variant's printing,
        then the printing the product was matched with, then the most common one."""
        available = [p.get("printing") for p in points if to_float(p.get("market_price"))]
        if not available:
            return []
        if reference:
            medians = {}
            for pr in set(available):
                vals = [to_float(p.get("market_price")) for p in points if p.get("printing") == pr]
                medians[pr] = float(np.median([v for v in vals if v]))
            chosen = min(medians, key=lambda pr: abs(np.log(medians[pr] / reference)))
            if not consistent(medians[chosen], reference):
                return []
        else:
            wanted = VARIANT_TO_PRINTING.get(variant or "", []) + ([printing] if printing else [])
            chosen = next((w for w in wanted if w in available), max(set(available), key=available.count))
        out = []
        for p in points:
            market = to_float(p.get("market_price"))
            if p.get("printing") == chosen and market and p.get("date"):
                out.append(PriceQuote(card_id=card_id, source="tcgapi", market=market,
                                      low=to_float(p.get("low_price")), variant=chosen,
                                      date=str(p["date"])[:10]))
        return out
