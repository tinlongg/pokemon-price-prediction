"""Pokémon TCG API (pokemontcg.io): card metadata plus TCGplayer prices. Works without a key."""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterator

from ..config import settings
from .base import HttpClient, PriceQuote, to_float

# Order used to pick a card's "headline" TCGplayer variant. Keeping it fixed means the
# series for a card always tracks the same printing.
VARIANT_ORDER = ["holofoil", "unlimitedHolofoil", "normal", "reverseHolofoil",
                 "1stEditionHolofoil", "1stEditionNormal", "unlimited", "1stEdition"]

SELECT = ("id,name,supertype,subtypes,set,number,rarity,artist,nationalPokedexNumbers,"
          "images,tcgplayer,cardmarket")


def _iso(d: str | None) -> str | None:
    if not d:
        return None
    try:
        return datetime.strptime(d, "%Y/%m/%d").date().isoformat()
    except ValueError:
        return d


def pick_variant(prices: dict | None) -> tuple[str, dict] | None:
    if not prices:
        return None
    for v in VARIANT_ORDER:
        p = prices.get(v)
        if p and to_float(p.get("market") or p.get("mid")):
            return v, p
    for v, p in prices.items():  # anything else with a price
        if p and to_float(p.get("market") or p.get("mid")):
            return v, p
    return None


class PokemonTCGProvider:
    name = "pokemontcg"

    def __init__(self, api_key: str | None = None):
        key = api_key if api_key is not None else settings.pokemontcg_api_key
        headers = {"X-Api-Key": key} if key else {}
        # Keyless use is limited, so space requests out a little.
        self.http = HttpClient("https://api.pokemontcg.io/v2", headers=headers,
                               min_interval=0.2 if key else 1.0, timeout=60)

    def search(self, q: str, page_size: int = 250, max_pages: int = 4) -> Iterator[dict]:
        for page in range(1, max_pages + 1):
            # A fixed sort order is required for paging: without it, pokemontcg.io can return
            # the same card on two pages and skip others.
            data = self.http.get("/cards", {"q": q, "pageSize": page_size, "page": page,
                                            "orderBy": "id", "select": SELECT})
            yield from data.get("data", [])
            if page * page_size >= data.get("totalCount", 0):
                break

    def get(self, card_id: str) -> dict | None:
        data = self.http.get(f"/cards/{card_id}", {"select": SELECT})
        return data.get("data")

    # --- conversion -------------------------------------------------------

    @staticmethod
    def to_card_row(raw: dict) -> dict:
        from ..features.popularity import species_of

        s = raw.get("set") or {}
        dex = raw.get("nationalPokedexNumbers") or [None]
        return {
            "card_id": raw["id"],
            "name": raw.get("name"),
            "pokemon": species_of(raw.get("name", ""), raw.get("supertype")),
            "supertype": raw.get("supertype"),
            "subtypes": raw.get("subtypes") or [],
            "set_id": s.get("id"),
            "set_name": s.get("name"),
            "series": s.get("series"),
            "release_date": _iso(s.get("releaseDate")),
            "number": raw.get("number"),
            "rarity": raw.get("rarity"),
            "artist": raw.get("artist"),
            "dex_number": dex[0],
            "language": "en",
            "image_url": (raw.get("images") or {}).get("large") or (raw.get("images") or {}).get("small"),
        }

    @staticmethod
    def to_quote(raw: dict) -> PriceQuote | None:
        tcg = raw.get("tcgplayer") or {}
        picked = pick_variant(tcg.get("prices"))
        if not picked:
            return None
        variant, p = picked
        market = to_float(p.get("market")) or to_float(p.get("mid"))
        # pokemontcg.io refreshes prices irregularly, so date the quote by when the
        # price was last updated, not by when we fetched it.
        updated = _iso(tcg.get("updatedAt")) or date.today().isoformat()
        return PriceQuote(card_id=raw["id"], source="pokemontcg", market=market,
                          low=to_float(p.get("low")), mid=to_float(p.get("mid")),
                          high=to_float(p.get("high")), variant=variant, date=updated)

    def quote(self, card: dict) -> PriceQuote | None:
        raw = self.get(card["card_id"])
        return self.to_quote(raw) if raw else None

    def quotes_for(self, card_ids: list[str], batch: int = 25) -> list[PriceQuote]:
        """Batch price lookup: `batch` ids per request via an OR query (kept small because
        pokemontcg.io returns HTTP 500 on large OR queries). Failed batches are skipped."""
        out = []
        for i in range(0, len(card_ids), batch):
            chunk = card_ids[i:i + batch]
            q = " OR ".join(f"id:{cid}" for cid in chunk)
            try:
                for raw in self.search(f"({q})", page_size=250, max_pages=1):
                    qt = self.to_quote(raw)
                    if qt:
                        out.append(qt)
            except Exception:
                continue
        return out
