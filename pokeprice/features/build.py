"""Assemble the full feature matrix: one row per card per date."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HORIZONS
from .events import build_calendar, event_features
from .market import add_market_features, add_targets, daily_panel
from .nostalgia import era_of
from .popularity import generation_of, popularity_score
from .rarity import artist_premium, is_full_art, is_special_art, rarity_tier
from .supply import print_run_index

# Feature name -> factor group (used to explain forecasts in plain English).
FEATURE_GROUPS: dict[str, str] = {
    "popularity": "Pokémon popularity",
    "trend_level": "Hype & search interest", "trend_change_90": "Hype & search interest",
    "trend_rel": "Hype & search interest",
    "days_to_event": "Hype & search interest", "next_event_strength": "Hype & search interest",
    "days_since_event": "Hype & search interest", "last_event_strength": "Hype & search interest",
    "month": "Hype & search interest",
    "era_idx": "Nostalgia", "age_years": "Nostalgia", "cohort_score": "Nostalgia",
    "is_gen1": "Nostalgia", "is_gen2": "Nostalgia", "nostalgia": "Nostalgia",
    "rarity_tier": "Rarity & art", "special_art": "Rarity & art", "full_art": "Rarity & art",
    "artist_premium": "Rarity & art", "is_trainer": "Rarity & art",
    "print_run_index": "Supply & reprints", "in_print": "Supply & reprints",
    "reprint_risk": "Supply & reprints",
    "log_price": "Market momentum", "ret_7": "Market momentum", "ret_30": "Market momentum",
    "ret_90": "Market momentum", "ret_365": "Market momentum", "vol_30": "Market momentum",
    "drawdown_90": "Market momentum", "rel_ret_90": "Market momentum",
    "mkt_ret_90": "Overall market", "mkt_ret_365": "Overall market",
}
FEATURES = list(FEATURE_GROUPS)

FEATURE_LABELS: dict[str, str] = {
    "popularity": "Pokémon popularity", "trend_level": "Google search interest",
    "trend_change_90": "Search interest change (90d)", "trend_rel": "Search interest vs its yearly average",
    "days_to_event": "Days until next hype event", "next_event_strength": "Size of next hype event",
    "days_since_event": "Days since last hype event", "last_event_strength": "Size of last hype event",
    "month": "Time of year", "era_idx": "Era", "age_years": "Card age", "cohort_score": "Childhood-cohort buying power",
    "is_gen1": "Gen 1 Pokémon", "is_gen2": "Gen 2 Pokémon", "nostalgia": "Nostalgia score",
    "rarity_tier": "Rarity", "special_art": "Special / alt art", "full_art": "Full art",
    "artist_premium": "Artist premium", "is_trainer": "Trainer card", "print_run_index": "Print run size",
    "in_print": "Set still in print", "reprint_risk": "Reprint risk", "log_price": "Price level",
    "ret_7": "Momentum (7d)", "ret_30": "Momentum (30d)", "ret_90": "Momentum (90d)",
    "vol_30": "Volatility (30d)", "drawdown_90": "Distance from 90-day high",
    "ret_365": "Momentum (1y)", "rel_ret_90": "Momentum vs the market (90d)",
    "mkt_ret_90": "Whole market momentum (90d)", "mkt_ret_365": "Whole market momentum (1y)",
}


def _str(v) -> str | None:
    return v if isinstance(v, str) and v else None


def card_static_features(cards: pd.DataFrame) -> pd.DataFrame:
    """Features that don't change over time, one row per card."""
    rows = []
    for c in cards.itertuples(index=False):
        c = c._replace(**{f: _str(getattr(c, f)) for f in ("pokemon", "supertype", "rarity", "artist", "release_date")})
        release = pd.to_datetime(c.release_date).date() if c.release_date else None
        _, era_idx = era_of(release)
        gen = generation_of(c.dex_number)
        sub = c.subtypes if isinstance(c.subtypes, list) else []
        rows.append({
            "card_id": c.card_id,
            "release_date": pd.to_datetime(c.release_date) if c.release_date else pd.NaT,
            "popularity": popularity_score(c.pokemon, c.supertype, c.dex_number),
            "is_gen1": int(gen == 1), "is_gen2": int(gen == 2),
            "is_trainer": int(str(c.supertype or "").lower().startswith("trainer")),
            "era_idx": era_idx,
            "rarity_tier": rarity_tier(c.rarity),
            "special_art": is_special_art(c.rarity),
            "full_art": is_full_art(c.rarity, sub),
            "artist_premium": artist_premium(c.artist),
            "print_run_index": print_run_index(era_idx),
            "pokemon": c.pokemon,
        })
    return pd.DataFrame(rows)


def add_time_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that depend on both the card and the date (vectorised versions of the
    nostalgia and supply helpers)."""
    age = ((df["date"] - df["release_date"]).dt.days / 365.25).clip(lower=0).fillna(0)
    df["age_years"] = age
    cohort_age = 10 + age
    df["cohort_score"] = np.exp(-((cohort_age - 33) ** 2) / (2 * 8 ** 2))
    df["in_print"] = (age < 1.5).astype(int)
    recency = (1 - age / 4).clip(lower=0)
    df["reprint_risk"] = recency * (0.4 + 0.6 * df["popularity"]) * np.where(df["special_art"] == 1, 0.5, 1.0)
    wotc = (df["era_idx"] == 0).astype(float)
    gen12 = ((df["is_gen1"] + df["is_gen2"]) > 0).astype(float)
    df["nostalgia"] = (0.35 * df["cohort_score"] + 0.25 * wotc + 0.15 * gen12
                       + 0.15 * (age / 25).clip(upper=1) + 0.10 * df["popularity"]).clip(upper=1)
    return df


def add_trend_features(df: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    if trends is None or trends.empty:
        for col in ("trend_level", "trend_change_90", "trend_rel"):
            df[col] = np.nan
        return df
    t = []
    for pkmn, g in trends.groupby("pokemon"):
        s = g.set_index("date")["interest"].sort_index()
        s = s[~s.index.duplicated(keep="last")].asfreq("D").interpolate(limit_area="inside").ffill()
        frame = pd.DataFrame({"date": s.index, "pokemon": pkmn, "trend_level": s.values})
        frame["trend_change_90"] = frame["trend_level"] - frame["trend_level"].shift(90)
        mean365 = frame["trend_level"].rolling(365, min_periods=30).mean()
        frame["trend_rel"] = frame["trend_level"] / mean365.replace(0, np.nan)
        t.append(frame)
    tdf = pd.concat(t, ignore_index=True)
    return df.merge(tdf, on=["pokemon", "date"], how="left")


def build_features(cards: pd.DataFrame, prices: pd.DataFrame, trends: pd.DataFrame | None = None,
                   horizons: tuple[int, ...] = HORIZONS, with_targets: bool = True,
                   stride: int | None = None, latest_only: bool = False) -> pd.DataFrame:
    """Feature matrix, one row per card per date (plus forward-return targets if requested).
    `stride` keeps one row every N days (and always each card's latest row); `latest_only`
    keeps just the latest row per card. Both are applied before the heavier joins."""
    panel = daily_panel(prices)
    if panel.empty:
        return panel
    panel = add_market_features(panel)
    if with_targets:
        panel = add_targets(panel, horizons)
    is_last = panel["date"] == panel.groupby("card_id")["date"].transform("max")
    if latest_only:
        panel = panel[is_last]
    elif stride:
        panel = panel[(panel["date"].dt.day_of_year % stride == 0) | is_last]
    panel = panel.reset_index(drop=True)
    static = card_static_features(cards)
    df = panel.merge(static, on="card_id", how="inner")
    df = add_time_card_features(df)
    df = add_trend_features(df, trends)
    ev = event_features(df["date"], build_calendar())
    df = pd.concat([df, ev], axis=1)
    return df.sort_values(["card_id", "date"]).reset_index(drop=True)


def latest_rows(features: pd.DataFrame) -> pd.DataFrame:
    """The most recent row per card: what we forecast from."""
    return features.sort_values("date").groupby("card_id").tail(1).set_index("card_id")
