"""How much collectors want a Pokémon (or trainer): the "brand name" factor.

Scores are 0-1 and reflect long-run collector demand (Charizard is the benchmark). They are
hand-set starting points; the Google Trends features adjust for what is hot right now.
"""

from __future__ import annotations

import re

POKEMON_POPULARITY: dict[str, float] = {
    "Charizard": 1.00, "Pikachu": 0.95, "Umbreon": 0.92, "Mewtwo": 0.86, "Mew": 0.86,
    "Rayquaza": 0.85, "Lugia": 0.83, "Gengar": 0.80, "Eevee": 0.78, "Espeon": 0.76,
    "Blastoise": 0.75, "Sylveon": 0.73, "Charmander": 0.70, "Gyarados": 0.70, "Dragonite": 0.70,
    "Greninja": 0.68, "Venusaur": 0.66, "Snorlax": 0.66, "Ho-Oh": 0.66, "Giratina": 0.66,
    "Glaceon": 0.65, "Lucario": 0.65, "Vaporeon": 0.63, "Mimikyu": 0.62, "Jolteon": 0.61,
    "Flareon": 0.61, "Leafeon": 0.61, "Suicune": 0.60, "Tyranitar": 0.60, "Garchomp": 0.60,
    "Gardevoir": 0.60, "Bulbasaur": 0.60, "Squirtle": 0.60, "Arceus": 0.56, "Celebi": 0.56,
    "Darkrai": 0.56, "Latias": 0.55, "Latios": 0.55, "Lapras": 0.55, "Blaziken": 0.55,
    "Psyduck": 0.52, "Magikarp": 0.50, "Ditto": 0.50, "Ninetales": 0.50,
    "Arcanine": 0.50, "Alakazam": 0.50, "Moltres": 0.50, "Zapdos": 0.50, "Articuno": 0.50,
    "Entei": 0.50, "Raikou": 0.50, "Zekrom": 0.50, "Reshiram": 0.50, "Kyogre": 0.50,
    "Groudon": 0.50, "Haunter": 0.42, "Gastly": 0.42, "Dragonair": 0.45, "Dratini": 0.45,
    "Cubone": 0.45, "Jigglypuff": 0.45, "Scizor": 0.45, "Absol": 0.45, "Metagross": 0.45,
    "Zoroark": 0.45, "Togepi": 0.40, "Machamp": 0.40, "Sceptile": 0.42, "Infernape": 0.42,
    "Pichu": 0.45, "Raichu": 0.45, "Clefairy": 0.40, "Pidgeot": 0.35,
    "Typhlosion": 0.45, "Feraligatr": 0.40, "Meganium": 0.38, "Gallade": 0.40, "Zacian": 0.42,
    "Zamazenta": 0.38, "Miraidon": 0.40, "Koraidon": 0.40, "Terapagos": 0.40, "Pecharunt": 0.30,
}

TRAINER_POPULARITY: dict[str, float] = {
    "Lillie": 0.85, "Marnie": 0.72, "Iono": 0.72, "Cynthia": 0.62, "Misty": 0.60,
    "Serena": 0.56, "Erika": 0.52, "N": 0.50, "Rosa": 0.50, "Acerola": 0.46,
    "Sabrina": 0.45, "Penny": 0.45, "Lusamine": 0.40, "Blaine": 0.35, "Giovanni": 0.40,
    "Professor's Research": 0.15, "Boss's Orders": 0.22,
}

# Last National Pokédex number of each generation.
GEN_BOUNDS = [151, 251, 386, 493, 649, 721, 809, 905, 1025]

_PREFIXES = re.compile(
    r"^(Dark|Light|Shining|Radiant|Mega|M|Alolan|Galarian|Hisuian|Paldean|Detective|"
    r"Shadow|Rapid Strike|Single Strike|Ancient|Future|Iron)\s+", re.I)
_SUFFIXES = re.compile(
    r"([\s-](ex|EX|GX|V|VMAX|VSTAR|V-UNION|LV\.X|Lv\.X|BREAK|Prime|LEGEND|δ|Star|★|◇|"
    r"Delta Species|C|G|GL|FB|4|SP))+$")


def species_of(name: str, supertype: str | None = None) -> str:
    """'Blaine's Charizard' -> 'Charizard', 'Umbreon VMAX' -> 'Umbreon',
    'Pikachu & Zekrom-GX' -> 'Pikachu', "Lillie's Determination" -> 'Lillie'."""
    name = (name or "").strip()
    if supertype and supertype.lower().startswith("trainer"):
        for t in sorted(TRAINER_POPULARITY, key=len, reverse=True):
            if name.lower().startswith(t.lower()):
                return t
        return name
    s = re.split(r"\s*&\s*", name)[0]
    s = re.sub(r"^.*?'s\s+", "", s)          # owner: "Team Rocket's Mewtwo" -> "Mewtwo"
    for _ in range(2):
        s = _PREFIXES.sub("", s)
        s = _SUFFIXES.sub("", s).strip()
    return s or name


def generation_of(dex_number: int | float | None) -> int | None:
    if dex_number is None or dex_number != dex_number:  # NaN check
        return None
    for gen, last in enumerate(GEN_BOUNDS, start=1):
        if dex_number <= last:
            return gen
    return len(GEN_BOUNDS) + 1


def popularity_score(pokemon: str | None, supertype: str | None = None,
                     dex_number: int | float | None = None) -> float:
    if not pokemon:
        return 0.1
    for table in (POKEMON_POPULARITY, TRAINER_POPULARITY):
        for key, val in table.items():
            if key.strip().lower() == pokemon.lower():
                return val
    if supertype and supertype.lower().startswith("trainer"):
        return 0.15
    gen = generation_of(dex_number)
    # Unknown Pokémon: Gen 1 still carries a nostalgia bump.
    return 0.30 if gen == 1 else 0.22 if gen == 2 else 0.15


POPULAR_NAMES = [k.strip() for k, v in POKEMON_POPULARITY.items() if v >= 0.6]
