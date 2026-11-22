"""Watchlist evaluation: portfolio value, forecasts and alerts."""

from __future__ import annotations

import pandas as pd

UNDERVALUED_MIN_UPSIDE = 0.15   # p50 at least 15% above today's price...
UNDERVALUED_MAX_DOWNSIDE = 0.05  # ...and p10 no more than 5% below it


def evaluate(watch: pd.DataFrame, forecasts: pd.DataFrame, cards: pd.DataFrame,
             horizon: int = 90) -> pd.DataFrame:
    """Join the watchlist with current prices and forecasts and attach alert flags."""
    if watch.empty:
        return watch
    f = forecasts[forecasts["horizon"] == horizon].set_index("card_id")
    out = watch.merge(cards[["card_id", "name", "set_name", "rarity", "image_url"]], on="card_id", how="left")
    out["current"] = out["card_id"].map(f["current"])
    out["forecast_p10"] = out["card_id"].map(f["p10"])
    out["forecast_p50"] = out["card_id"].map(f["p50"])
    out["forecast_p90"] = out["card_id"].map(f["p90"])
    out["upside"] = out["forecast_p50"] / out["current"] - 1
    out["value"] = out["current"] * out["quantity"].fillna(0)
    out["forecast_value"] = out["forecast_p50"] * out["quantity"].fillna(0)
    out["gain"] = out["value"] - out["cost_basis"]

    def alerts(r) -> str:
        flags = []
        if pd.notna(r.target_price) and pd.notna(r.current) and r.current <= r.target_price:
            flags.append("At or below target")
        if (pd.notna(r.upside) and r.upside >= UNDERVALUED_MIN_UPSIDE
                and r.forecast_p10 >= r.current * (1 - UNDERVALUED_MAX_DOWNSIDE)):
            flags.append("Looks undervalued")
        if pd.notna(r.upside) and r.upside <= -0.15:
            flags.append("Forecast to fall")
        return " · ".join(flags)

    out["alerts"] = out.apply(alerts, axis=1)
    return out
