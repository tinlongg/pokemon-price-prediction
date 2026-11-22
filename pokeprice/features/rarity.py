"""Rarity, special-art and artist factors."""

from __future__ import annotations

# Ordinal rarity tier, roughly how hard the card is to pull (0 = common, 9 = top chase card).
RARITY_TIER: dict[str, int] = {
    "Common": 0, "Uncommon": 1, "Rare": 2, "Promo": 2, "Classic Collection": 5,
    "Rare Holo": 3, "Rare Holo EX": 4, "Rare Holo GX": 4, "Rare Holo V": 4, "Double Rare": 4,
    "Rare Holo VMAX": 4, "Rare Holo VSTAR": 4, "Rare BREAK": 4, "Rare Prime": 4, "Radiant Rare": 4,
    "ACE SPEC Rare": 4, "Rare ACE": 4, "Rare Prism Star": 4,
    "Rare Ultra": 5, "Ultra Rare": 5, "Rare Holo LV.X": 5, "LEGEND": 5, "Amazing Rare": 5,
    "Trainer Gallery Rare Holo": 5, "Rare Shiny": 5, "Shiny Rare": 5,
    "Illustration Rare": 6, "Rare Rainbow": 6, "Rare Shiny GX": 6, "Shiny Ultra Rare": 6,
    "Rare Shining": 7, "Rare Secret": 7, "Hyper Rare": 7,
    "Rare Holo Star": 8, "Special Illustration Rare": 8,
}

SPECIAL_ART_RARITIES = {
    "Illustration Rare", "Special Illustration Rare", "Trainer Gallery Rare Holo",
    "Rare Holo Star", "Rare Shining", "Rare Rainbow", "Rare Secret", "Hyper Rare",
    "Amazing Rare", "Shiny Rare", "Shiny Ultra Rare", "Rare Shiny", "Rare Shiny GX",
}

# Illustrators whose cards carry a premium with collectors.
ARTIST_PREMIUM: dict[str, float] = {
    "Mitsuhiro Arita": 1.0, "Atsuko Nishida": 0.8, "Ken Sugimori": 0.8,
    "Kagemaru Himeno": 0.7, "Akira Egawa": 0.7, "Naoki Saito": 0.65, "Yuu Nishida": 0.6,
    "Tomokazu Komiya": 0.6, "Keiji Kinebuchi": 0.55, "HYOGONOSUKE": 0.6, "sowsow": 0.6,
    "Shinji Kanda": 0.5, "Teeziro": 0.5, "Ryota Murayama": 0.5, "Kouki Saitou": 0.55,
    "Kawayoo": 0.5, "Mina Nakai": 0.45, "Tetsu Kayama": 0.45, "Shibuzoh.": 0.5,
    "Ayaka Yoshida": 0.5, "Jiro Sasumo": 0.45, "Oswaldo KATO": 0.45,
}


def rarity_tier(rarity: str | None) -> int:
    if not rarity:
        return 1
    if rarity in RARITY_TIER:
        return RARITY_TIER[rarity]
    r = rarity.lower()
    if "special illustration" in r:
        return 8
    if "secret" in r or "hyper" in r:
        return 7
    if "illustration" in r or "rainbow" in r:
        return 6
    if "ultra" in r or "gallery" in r:
        return 5
    if "holo" in r:
        return 3
    return 2


def is_special_art(rarity: str | None) -> int:
    return int(bool(rarity) and (rarity in SPECIAL_ART_RARITIES or "illustration" in rarity.lower()))


def is_full_art(rarity: str | None, subtypes: list[str] | None) -> int:
    subs = {s.lower() for s in (subtypes or [])}
    return int(rarity_tier(rarity) >= 5 or "vmax" in subs or "vstar" in subs)


def artist_premium(artist: str | None) -> float:
    if not artist:
        return 0.0
    for name, val in ARTIST_PREMIUM.items():
        if name.lower() == artist.lower():
            return val
    return 0.0
