"""PokéWallet API (https://www.pokewallet.io/api-docs). Free tier: 1,000 requests/day, 100/hour.

Checked against the live API (November 2026):
  GET /search?q=<name>&limit=100&page=N
      -> {"results": [{id, card_info: {name, set_name, set_code, card_number, product_type, ...},
                       tcgplayer: {prices: [{sub_type_name, market_price, low_price, mid_price,
                                             high_price, updated_at}]}, cardmarket: {...}}],
          "pagination": {page, limit, total, total_pages}}
  GET /cards/<id> -> one result in the same shape.
Search results already carry today's TCGplayer prices, so the daily snapshot searches once
per Pokémon and prices all of that Pokémon's cards from the results.
"""

from __future__ import annotations

from ..config import settings
from .base import HttpClient, PriceQuote, RateLimitExceeded, to_float
from .matching import best_match

SUBTYPE_ORDER = ["Holofoil", "Unlimited Holofoil", "Normal", "Reverse Holofoil", "Unlimited",
                 "1st Edition Holofoil", "1st Edition", "Shadowless"]
VARIANT_TO_SUBTYPE = {
    "holofoil": "Holofoil", "unlimitedHolofoil": "Unlimited Holofoil", "normal": "Normal",
    "reverseHolofoil": "Reverse Holofoil", "1stEditionHolofoil": "1st Edition Holofoil",
    "1stEditionNormal": "1st Edition", "unlimited": "Unlimited", "1stEdition": "1st Edition",
}


def row_view(r: dict) -> dict:
    """Normalise a PokéWallet result for the shared matcher."""
    info = r.get("card_info") or {}
    return {"id": r.get("id"), "name": info.get("name"), "number": info.get("card_number"),
            "set_name": info.get("set_name"), "product_type": info.get("product_type") or "card", "_raw": r}


class PokeWalletProvider:
    name = "pokewallet"

    def __init__(self, api_key: str | None = None):
        key = api_key if api_key is not None else settings.pokewallet_api_key
        if not key:
            raise ValueError("POKEWALLET_API_KEY is not set")
        self.http = HttpClient("https://api.pokewallet.io", headers={"X-API-Key": key}, min_interval=0.3)
        self.requests_made = 0

    def remaining(self) -> dict:
        h = self.http.last_headers or {}
        return {"hour": h.get("X-RateLimit-Remaining-Hour"), "day": h.get("X-RateLimit-Remaining-Day")}

    def _get(self, path: str, params: dict | None = None) -> dict:
        rem = self.remaining()
        if rem["hour"] is not None and int(rem["hour"]) <= 0:
            raise RateLimitExceeded("PokéWallet hourly quota used up")
        data = self.http.get(path, params)
        self.requests_made += 1
        return data

    def search_all(self, term: str, max_pages: int = 4) -> tuple[list[dict], bool]:
        """All results for a name (up to max_pages of 100). Returns (rows, complete)."""
        rows: list[dict] = []
        for page in range(1, max_pages + 1):
            data = self._get("/search", {"q": term, "limit": 100, "page": page})
            rows += data.get("results") or []
            total_pages = (data.get("pagination") or {}).get("total_pages") or 1
            if page >= total_pages:
                return rows, True
        return rows, False

    @staticmethod
    def parse_quote(card_id: str, raw: dict, preferred_variant: str | None = None) -> PriceQuote | None:
        prices = {p.get("sub_type_name"): p for p in ((raw.get("tcgplayer") or {}).get("prices") or []) if p}
        order = list(SUBTYPE_ORDER)
        if preferred_variant and VARIANT_TO_SUBTYPE.get(preferred_variant):
            order.insert(0, VARIANT_TO_SUBTYPE[preferred_variant])
        for sub in order + list(prices):
            p = prices.get(sub)
            if not p:
                continue
            market = to_float(p.get("market_price")) or to_float(p.get("mid_price"))
            if market:
                day = str(p.get("updated_at") or "")[:10] or None
                q = PriceQuote(card_id=card_id, source="pokewallet", market=market,
                               low=to_float(p.get("low_price")), mid=to_float(p.get("mid_price")),
                               high=to_float(p.get("high_price")), variant=sub)
                if day:
                    q.date = day
                return q
        return None

    @staticmethod
    def match(card: dict, rows: list[dict]) -> dict | None:
        """Find this card among search results: by saved id if we have one, else by number + set."""
        if card.get("pokewallet_id"):
            for r in rows:
                if r.get("id") == card["pokewallet_id"]:
                    return r
        hit = best_match(card, [row_view(r) for r in rows])
        return hit["_raw"] if hit else None

    def quote(self, card: dict, preferred_variant: str | None = None) -> PriceQuote | None:
        """Single-card lookup (used when a card's id is known)."""
        if not card.get("pokewallet_id"):
            return None
        raw = self._get(f"/cards/{card['pokewallet_id']}")
        return self.parse_quote(card["card_id"], raw, preferred_variant)
