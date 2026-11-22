"""Market momentum features computed from a card's daily price series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_panel(prices: pd.DataFrame, ffill_limit: int = 45) -> pd.DataFrame:
    """Turn sparse snapshots into one row per card per day (forward-filling gaps)."""
    if prices.empty:
        return pd.DataFrame(columns=["card_id", "date", "market", "is_synthetic"])
    out = []
    for cid, g in prices.groupby("card_id", sort=False):
        s = g.set_index("date")[["market", "is_synthetic"]].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.asfreq("D")
        s["market"] = s["market"].ffill(limit=ffill_limit)
        s["is_synthetic"] = s["is_synthetic"].ffill(limit=ffill_limit)
        s = s.dropna(subset=["market"])
        s["card_id"] = cid
        out.append(s.reset_index())
    panel = pd.concat(out, ignore_index=True)
    panel["is_synthetic"] = panel["is_synthetic"].astype(int)
    return panel


def add_market_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Expects one row per card per consecutive day, sorted by card then date."""
    p = panel.sort_values(["card_id", "date"]).reset_index(drop=True)
    logp = np.log(p["market"].clip(lower=0.01))
    g = logp.groupby(p["card_id"])
    p["log_price"] = logp
    for d in (7, 30, 90, 365):
        p[f"ret_{d}"] = logp - g.shift(d)
    daily_ret = logp - g.shift(1)
    p["vol_30"] = daily_ret.groupby(p["card_id"]).transform(lambda x: x.rolling(30, min_periods=10).std())
    roll_max = g.transform(lambda x: x.rolling(90, min_periods=5).max())
    p["drawdown_90"] = logp - roll_max
    # Overall market: the median card's momentum on each date, and each card relative to it.
    # Lets the model tell a market-wide rally or cooldown apart from card-specific moves.
    by_date = p.groupby("date")
    p["mkt_ret_90"] = by_date["ret_90"].transform("median")
    p["mkt_ret_365"] = by_date["ret_365"].transform("median")
    p["rel_ret_90"] = p["ret_90"] - p["mkt_ret_90"]
    return p


def add_targets(panel: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    p = panel
    g = p.groupby("card_id")["log_price"]
    for h in horizons:
        p[f"target_{h}"] = g.shift(-h) - p["log_price"]
        # Was the future price real? Used to score the model on real outcomes only.
        p[f"target_real_{h}"] = (1 - p.groupby("card_id")["is_synthetic"].shift(-h)).fillna(0)
    return p
