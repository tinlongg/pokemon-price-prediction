"""Match our cards (from pokemontcg.io) to another API's search results.

Search APIs match names only, so we search once per Pokémon and pick, for each of our cards,
the result with the same card number and the most similar set name
("Base" ~ "Base Set", "Evolutions" ~ "XY - Evolutions", "151" ~ "SV: Scarlet & Violet 151").
Rows must be normalised to dicts with name, number, set_name and product_type.
"""

from __future__ import annotations

import re

_SET_STOPWORDS = {"set", "the", "pokemon", "tcg", "and"}


def _tokens(s: str | None) -> set[str]:
    words = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split()
    return {w for w in words if w not in _SET_STOPWORDS}


def number_core(n: str | None) -> str:
    """'004/102' -> '4', 'TG03/TG30' -> 'tg3', 'SM230' -> 'sm230'."""
    n = (n or "").split("/")[0].lower().strip()
    m = re.fullmatch(r"([a-z]*)0*(\d+)([a-z]*)", n)
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else n


def search_term(card: dict) -> str:
    """Search by species/character when we have it (one search covers all its cards)."""
    term = card.get("pokemon") or card.get("name") or ""
    return re.sub(r"[^\w\s'-]", " ", term).strip()


def match_score(card: dict, row: dict) -> float:
    """0 if not a match; otherwise higher is a better match."""
    if (row.get("product_type") or "cards").lower() not in ("cards", "card"):
        return 0.0
    if number_core(row.get("number")) != number_core(card.get("number")):
        return 0.0
    ours, theirs = _tokens(card.get("set_name")), _tokens(row.get("set_name"))
    if not ours or not theirs:
        return 0.0
    containment = len(ours & theirs) / len(ours)
    if containment < 0.5:
        return 0.0
    jaccard = len(ours & theirs) / len(ours | theirs)
    name_ours = re.sub(r"[^a-z0-9]", "", (card.get("name") or "").lower())
    name_theirs = re.sub(r"[^a-z0-9]", "", (row.get("name") or "").lower())
    name_bonus = 1.0 if name_theirs == name_ours else 0.5 if name_theirs.startswith(name_ours[:6]) else 0.0
    return 2 * containment + jaccard + name_bonus


def best_match(card: dict, rows: list[dict]) -> dict | None:
    scored = [(match_score(card, r), r) for r in rows]
    scored = [sr for sr in scored if sr[0] > 0]
    return max(scored, key=lambda sr: sr[0])[1] if scored else None
