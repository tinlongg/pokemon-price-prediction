"""Is it worth grading this card? Expected value of grading vs selling it raw.

Grade multipliers and gem rates are heuristics by era, rarity and popularity, not live
PSA data. Vintage cards are rarely gem mint, so a PSA 10 is worth many times the raw price;
modern pack-fresh cards grade 10 often, so the premium is smaller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Probability of each grade for a card the owner believes is Near Mint, by era index.
GRADE_ODDS_BY_ERA = {
    0: {"10": 0.06, "9": 0.30, "<=8": 0.64},   # WOTC
    1: {"10": 0.15, "9": 0.40, "<=8": 0.45},   # EX
    2: {"10": 0.22, "9": 0.45, "<=8": 0.33},
    3: {"10": 0.30, "9": 0.45, "<=8": 0.25},
    4: {"10": 0.35, "9": 0.45, "<=8": 0.20},
    5: {"10": 0.40, "9": 0.42, "<=8": 0.18},
    6: {"10": 0.45, "9": 0.40, "<=8": 0.15},
    7: {"10": 0.50, "9": 0.38, "<=8": 0.12},
    8: {"10": 0.50, "9": 0.38, "<=8": 0.12},
}

CONDITION_SHIFT = {  # moves probability mass between grades
    "Pack fresh / mint": 1.3,
    "Near Mint": 1.0,
    "Lightly played": 0.35,
}


@dataclass
class GradingResult:
    raw_price: float
    psa10_price: float
    psa9_price: float
    p10: float
    p9: float
    p_low: float
    expected_graded_value: float
    total_cost: float
    expected_profit_vs_raw: float
    worth_it: bool
    breakeven_p10: float

    def dict(self) -> dict:
        return asdict(self)


def multipliers(era_idx: int, rarity_tier: int, popularity: float) -> tuple[float, float]:
    """PSA 10 and PSA 9 price as a multiple of the raw Near Mint price."""
    vintage = max(0.0, (6 - era_idx) / 6)               # 1.0 for WOTC, 0 for modern
    psa10 = 1.6 + 7.0 * vintage ** 1.5 + 1.5 * popularity + 0.15 * rarity_tier
    psa9 = 1.0 + 1.0 * vintage + 0.3 * popularity
    return round(psa10, 2), round(psa9, 2)


def grading_ev(raw_price: float, era_idx: int, rarity_tier: int, popularity: float,
               condition: str = "Near Mint", grading_fee: float = 25.0, shipping: float = 10.0,
               selling_fee_pct: float = 0.13) -> GradingResult:
    odds = dict(GRADE_ODDS_BY_ERA.get(era_idx, GRADE_ODDS_BY_ERA[7]))
    k = CONDITION_SHIFT.get(condition, 1.0)
    p10 = min(0.95, odds["10"] * k)
    p9 = min(0.95 - p10, odds["9"] * (1.0 if k >= 1 else 0.7))
    p_low = max(0.0, 1 - p10 - p9)
    m10, m9 = multipliers(era_idx, rarity_tier, popularity)
    psa10, psa9 = raw_price * m10, raw_price * m9
    low_value = raw_price * 0.85                          # PSA 8 and below ~ a bit under raw NM
    keep = 1 - selling_fee_pct
    ev_graded = (p10 * psa10 + p9 * psa9 + p_low * low_value) * keep
    cost = grading_fee + shipping
    raw_net = raw_price * keep
    profit = ev_graded - cost - raw_net
    # P(10) needed to break even, holding the 9/low split proportional.
    denom = (psa10 - (p9 * psa9 + p_low * low_value) / max(p9 + p_low, 1e-9)) * keep
    rest = ((p9 * psa9 + p_low * low_value) / max(p9 + p_low, 1e-9)) * keep
    breakeven = (raw_net + cost - rest) / denom if denom > 0 else 1.0
    return GradingResult(raw_price, psa10, psa9, p10, p9, p_low, ev_graded, cost, profit,
                         profit > 0, max(0.0, min(1.0, breakeven)))
