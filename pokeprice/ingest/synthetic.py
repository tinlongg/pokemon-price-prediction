"""Simulated price history, used until enough real daily snapshots have built up.

Each card gets a daily price path built from the same factors the model uses:
  * a shared market cycle (2021-22 boom-and-cooldown, 2024-26 recovery), weighted per card
  * a slow drift driven by popularity, nostalgia, rarity, artist and supply
  * hype from a per-Pokémon search-interest series plus event bumps (anniversaries etc.)
  * a post-release hype premium that fades for new sets
  * occasional reprint shocks that dent the price and slowly recover
  * noise that is larger for cheaper cards
The path is shifted so it ends at the card's real current price. Every row is flagged
is_synthetic=1, shown dashed in the app, and down-weighted in training.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from .. import db
from ..config import settings
from ..features.build import add_time_card_features, card_static_features
from ..features.events import build_calendar

# (start date, annualised log drift of the overall collector market)
MARKET_REGIMES = [
    ("2015-01-01", 0.04), ("2020-06-01", 0.90), ("2021-03-01", 0.15), ("2021-11-01", -0.35),
    ("2023-01-01", -0.05), ("2023-09-01", 0.25), ("2025-01-01", 0.35), ("2026-03-01", 0.05),
]


def _market_path(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    starts = pd.to_datetime([d for d, _ in MARKET_REGIMES])
    drifts = np.array([m for _, m in MARKET_REGIMES])
    idx = np.searchsorted(starts.values, dates.values, side="right") - 1
    mu = drifts[np.clip(idx, 0, len(drifts) - 1)] / 365
    return mu + rng.normal(0, 0.12 / np.sqrt(365), len(dates))


def _event_signal(dates: pd.DatetimeIndex, cal: pd.DataFrame) -> np.ndarray:
    """0-1 bump that builds over 90 days before each event and fades 60 days after it."""
    d = dates.values.astype("datetime64[D]").astype(int)
    out = np.zeros(len(d))
    for ev, s in zip(cal["date"].values.astype("datetime64[D]").astype(int), cal["strength"].values):
        diff = ev - d
        before = np.where((diff >= 0) & (diff <= 90), 1 - diff / 90, 0)
        after = np.where((diff < 0) & (diff >= -60), 1 + diff / 60, 0)
        out = np.maximum(out, s * np.maximum(before, after))
    return out


def base_price_estimate(row: pd.Series) -> float:
    """Rough fallback price when a card has no real price yet."""
    return float(np.exp(0.8 + 3.0 * row["popularity"] + 0.5 * row["rarity_tier"]
                        + 1.5 * (row["era_idx"] == 0) + 0.5 * row["artist_premium"]))


def generate(cards: pd.DataFrame, anchors: pd.DataFrame, years: int | None = None,
             end: date | None = None, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (price rows, trend rows). `anchors` has card_id, date, market: the first real price
    for each card. Simulated history runs up to the day before that date."""
    rng = np.random.default_rng(seed)
    years = years or settings.synthetic_years
    end = end or date.today()
    start = pd.Timestamp(end - timedelta(days=int(365.25 * years)))
    all_dates = pd.date_range(start, pd.Timestamp(end), freq="D")
    cal = build_calendar(all_dates[0].year, all_dates[-1].year + 1)
    market = np.cumsum(_market_path(all_dates, rng))
    events = _event_signal(all_dates, cal)

    static = card_static_features(cards).set_index("card_id")
    anchors = anchors.set_index("card_id") if not anchors.empty else pd.DataFrame(columns=["date", "market"])

    # Per-Pokémon hype series (AR(1) + events), shared by all cards of that Pokémon.
    hype: dict[str, np.ndarray] = {}
    trend_rows = []
    for pkmn, pop in static.groupby("pokemon")["popularity"].first().items():
        shocks = rng.normal(0, 0.035, len(all_dates))
        h = np.zeros(len(all_dates))
        for t in range(1, len(h)):
            h[t] = 0.993 * h[t - 1] + shocks[t]
        hype[pkmn] = h
        interest = np.clip(15 + 55 * pop + 12 * h + 25 * events * (0.5 + pop), 1, 100)
        weekly = slice(None, None, 7)
        trend_rows.append(pd.DataFrame({"pokemon": pkmn, "date": all_dates[weekly].strftime("%Y-%m-%d"),
                                        "interest": interest[weekly].round(1), "is_synthetic": 1}))

    price_frames = []
    for cid, c in static.iterrows():
        if cid in anchors.index:
            a_date = pd.Timestamp(anchors.loc[cid, "date"])
            a_price = float(anchors.loc[cid, "market"])
        else:
            a_date, a_price = pd.Timestamp(end), base_price_estimate(c)
        first = max(all_dates[0], c["release_date"]) if pd.notna(c["release_date"]) else all_dates[0]
        mask = (all_dates >= first) & (all_dates < a_date)
        if mask.sum() < 30:
            continue
        dates = all_dates[mask]
        n = len(dates)

        # Time-varying factor values for this card.
        tf = pd.DataFrame({"date": dates, "release_date": c["release_date"], **{
            k: c[k] for k in ("popularity", "special_art", "era_idx", "is_gen1", "is_gen2")}})
        tf = add_time_card_features(tf)

        card_alpha = rng.normal(0, 0.03)
        mu = (0.10 * c["popularity"] + 0.10 * tf["nostalgia"].values + 0.04 * c["rarity_tier"] / 8
              + 0.03 * c["artist_premium"] + 0.04 * c["special_art"] - 0.15 * tf["reprint_risk"].values
              - 0.05 * c["print_run_index"] + 0.04 * (1 - tf["in_print"].values) - 0.02 + card_alpha) / 365
        beta = 0.4 + 0.6 * c["popularity"] + 0.3 * (c["era_idx"] == 0)
        mkt = np.diff(market[mask], prepend=market[mask][0])
        annual_vol = 0.18 + 0.30 / (1 + a_price / 25)
        eps = rng.normal(0, annual_vol / np.sqrt(365), n)
        walk = np.cumsum(mu + beta * mkt + eps)

        # Level effects (not cumulative): hype, events, release premium, reprint shocks.
        h = hype.get(c["pokemon"], np.zeros(len(all_dates)))[mask]
        level = 0.10 * c["popularity"] * h + 0.10 * (0.3 + c["popularity"]) * events[mask]
        days_since_release = tf["age_years"].values * 365.25
        level += 0.35 * (c["rarity_tier"] / 8) * np.exp(-days_since_release / 45)
        reprints = rng.random(n) < (tf["reprint_risk"].values * 0.4 / 365)
        for t in np.flatnonzero(reprints):
            level[t:] -= 0.15 * np.exp(-np.arange(n - t) / 260)

        logp = walk + level
        logp += np.log(a_price) - logp[-1] - (mu[-1] if n else 0)
        price_frames.append(pd.DataFrame({
            "card_id": cid, "date": dates.strftime("%Y-%m-%d"), "source": "synthetic",
            "market": np.round(np.exp(logp), 2), "low": None, "mid": None, "high": None,
            "variant": None, "is_synthetic": 1,
        }))
    prices = pd.concat(price_frames, ignore_index=True) if price_frames else pd.DataFrame(columns=db.PRICE_COLUMNS)
    trends = pd.concat(trend_rows, ignore_index=True) if trend_rows else pd.DataFrame()
    return prices, trends


def ensure_synthetic(force: bool = False, log: Callable[[str], None] = print) -> int:
    """(Re)generate simulated history for every card, anchored to its first real price."""
    cards = db.load_cards()
    if cards.empty:
        return 0
    if force:
        db.delete_synthetic_prices()
    else:
        mix = db.data_mix()
        if mix["synthetic"]["cards"] >= len(cards) * 0.9:
            log("Simulated history already exists (use --resimulate to rebuild it).")
            return 0
        db.delete_synthetic_prices()
    real = db.load_prices(include_synthetic=False)
    anchors = (real.sort_values("date").groupby("card_id").first().reset_index()[["card_id", "date", "market"]]
               if not real.empty else pd.DataFrame(columns=["card_id", "date", "market"]))
    prices, trends = generate(cards, anchors)
    n = db.insert_snapshots(prices.to_dict("records"))
    if not trends.empty:
        db.insert_trends(trends.to_dict("records"))
    log(f"Simulated history: {n:,} daily prices for {prices['card_id'].nunique() if n else 0} cards")
    return n
