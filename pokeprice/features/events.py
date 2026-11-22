"""Calendar of hype events: Pokémon Day, anniversaries, big game releases and landmark sets.

Edit EXTRA_EVENTS to add upcoming releases as they are announced.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# (date, name, strength 0-1)
EXTRA_EVENTS: list[tuple[str, str, float]] = [
    ("2021-02-27", "25th anniversary", 1.0),
    ("2021-10-08", "Celebrations set", 0.8),
    ("2022-01-28", "Pokémon Legends: Arceus", 0.5),
    ("2022-11-18", "Pokémon Scarlet & Violet", 0.6),
    ("2023-09-22", "Scarlet & Violet 151 set", 0.8),
    ("2025-01-17", "Prismatic Evolutions set", 0.8),
    ("2025-10-16", "Pokémon Legends: Z-A", 0.5),
    ("2026-02-27", "30th anniversary", 1.0),
]


def build_calendar(start_year: int = 1999, end_year: int | None = None) -> pd.DataFrame:
    end_year = end_year or date.today().year + 2
    rows = [(f"{y}-02-27", "Pokémon Day", 0.3) for y in range(start_year, end_year + 1)]
    rows += EXTRA_EVENTS
    df = pd.DataFrame(rows, columns=["date", "name", "strength"])
    df["date"] = pd.to_datetime(df["date"])
    # Keep the stronger event when two fall on the same day.
    return df.sort_values(["date", "strength"]).drop_duplicates("date", keep="last").reset_index(drop=True)


def event_features(dates: pd.Series, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Days to the next event, days since the last one, and that event's strength."""
    cal = calendar if calendar is not None else build_calendar()
    ev = cal["date"].values.astype("datetime64[D]")
    d = pd.to_datetime(dates).values.astype("datetime64[D]")
    nxt = np.searchsorted(ev, d, side="left")
    prv = nxt - 1
    nxt_c, prv_c = np.clip(nxt, 0, len(ev) - 1), np.clip(prv, 0, len(ev) - 1)
    days_to = (ev[nxt_c] - d).astype(int).astype(float)
    days_since = (d - ev[prv_c]).astype(int).astype(float)
    days_to[nxt >= len(ev)] = 365.0
    days_since[prv < 0] = 365.0
    strength = cal["strength"].values
    return pd.DataFrame({
        "days_to_event": np.minimum(days_to, 365),
        "next_event_strength": strength[nxt_c],
        "days_since_event": np.minimum(days_since, 365),
        "last_event_strength": strength[prv_c],
        "month": pd.to_datetime(dates).dt.month.values,
    }, index=dates.index)


def anniversary_boost(d: pd.Timestamp, calendar: pd.DataFrame) -> float:
    """Simulator helper: a hype bump that builds before an event and fades after it."""
    diffs = (calendar["date"] - d).dt.days.values
    s = calendar["strength"].values
    before = np.where((diffs >= 0) & (diffs <= 90), (1 - diffs / 90) * s, 0)
    after = np.where((diffs < 0) & (diffs >= -60), (1 + diffs / 60) * s, 0)
    return float(max(before.max(initial=0), after.max(initial=0)))
