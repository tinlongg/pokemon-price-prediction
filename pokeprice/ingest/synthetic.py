"""Simulated price history, used until enough real daily snapshots have built up.

Each card gets a daily price path built from the same factors the model uses:
  * a shared market cycle (2021-22 boom-and-cooldown, 2024-26 recovery), weighted per card
  * a slow drift driven by popularity, nostalgia, rarity, artist and supply
  * hype from a per-Pokémon search-interest series plus event bumps (anniversaries etc.)
  * a post-release hype premium that fades for new sets
  * occasional reprint shocks that dent the price and slowly recover
  * noise that is larger for cheaper cards
The path is shifted so it ends at the card's first real price. Every row is flagged
is_synthetic=1, shown dashed in the app, and down-weighted in training.

Randomness is seeded per series (the market, each Pokémon, each card) on a fixed calendar
starting at ORIGIN, so a Pokémon's hype series and a card's path come out the same no matter
which other cards are generated alongside them or on which day. That lets new cards be added,
or one card be re-anchored, without regenerating everything. Cards with no real price yet get
no simulated history; they join once PokéWallet or TCGapi has priced them.
"""

from __future__ import annotations

import zlib
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from .. import db
from ..config import settings
from ..features.build import add_time_card_features, card_static_features
from ..features.events import build_calendar

ORIGIN = pd.Timestamp("2015-01-01")   # every random series starts here, so they line up

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


def _rng(seed: int, key: str) -> np.random.Generator:
    """Independent, reproducible random stream for one named series."""
    return np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])


def _ar1(shocks: np.ndarray, phi: float = 0.993) -> np.ndarray:
    try:
        from scipy.signal import lfilter
        out = lfilter([1.0], [1.0, -phi], shocks)
        out -= shocks[0] * phi ** np.arange(len(shocks))   # start the series at 0
        return out
    except ImportError:  # scipy ships with scikit-learn, but keep a fallback
        h = np.zeros(len(shocks))
        for t in range(1, len(h)):
            h[t] = phi * h[t - 1] + shocks[t]
        return h


def generate(cards: pd.DataFrame, anchors: pd.DataFrame, years: int | None = None,
             end: date | None = None, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (price rows, trend rows) for the given cards. `anchors` has card_id, date, market:
    each card's first real price. A card's history runs up to the day before that date; cards
    without an anchor are skipped."""
    years = years or settings.synthetic_years
    end = end or date.today()
    all_dates = pd.date_range(ORIGIN, pd.Timestamp(end), freq="D")
    window_start = pd.Timestamp(end - timedelta(days=int(365.25 * years)))
    in_window = all_dates >= window_start
    cal = build_calendar(ORIGIN.year, all_dates[-1].year + 1)
    market = np.cumsum(_market_path(all_dates, _rng(seed, "market")))
    events = _event_signal(all_dates, cal)

    static = card_static_features(cards).set_index("card_id")
    anchors = anchors.set_index("card_id") if not anchors.empty else pd.DataFrame(columns=["date", "market"])
    static = static[static.index.isin(anchors.index)]

    # Per-Pokémon hype series (AR(1) + events), shared by all cards of that Pokémon.
    hype: dict[str, np.ndarray] = {}
    trend_rows = []
    weekly = np.flatnonzero(in_window)[::7]
    for pkmn, pop in static.groupby("pokemon")["popularity"].first().items():
        h = _ar1(_rng(seed, f"hype:{pkmn}").normal(0, 0.035, len(all_dates)))
        hype[pkmn] = h
        interest = np.clip(15 + 55 * pop + 12 * h + 25 * events * (0.5 + pop), 1, 100)
        trend_rows.append(pd.DataFrame({"pokemon": pkmn, "date": all_dates[weekly].strftime("%Y-%m-%d"),
                                        "interest": interest[weekly].round(1), "is_synthetic": 1}))

    price_frames = []
    for cid, c in static.iterrows():
        rng = _rng(seed, f"card:{cid}")
        a_date = pd.Timestamp(anchors.loc[cid, "date"])
        a_price = float(anchors.loc[cid, "market"])
        first = max(window_start, c["release_date"]) if pd.notna(c["release_date"]) else window_start
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
    """Make sure every priced card has simulated history that ends the day before its first
    real price. Only cards that need it are (re)generated: new cards, cards that just got their
    first price, and cards whose first real price moved (e.g. a TCGapi backfill added older
    days, or the card switched to a daily price source). `force` regenerates every card."""
    cards = db.load_cards()
    if cards.empty:
        return 0
    real = db.load_prices(include_synthetic=False)
    if real.empty:
        log("Simulated history: no real prices yet, nothing to anchor to.")
        return 0
    anchors = real.sort_values("date").groupby("card_id").first().reset_index()[["card_id", "date", "market"]]
    with db.connect() as conn:
        syn_last = pd.read_sql("SELECT card_id, MAX(date) AS last FROM price_snapshots "
                               "WHERE is_synthetic=1 GROUP BY card_id", conn).set_index("card_id")["last"]
    expected = (anchors.set_index("card_id")["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    if force:
        todo = list(expected.index)
    else:
        todo = [cid for cid, exp in expected.items() if syn_last.get(cid) != exp]
    if not todo:
        log("Simulated history is up to date.")
        return 0
    db.delete_synthetic_prices(todo)
    prices, trends = generate(cards[cards["card_id"].isin(todo)], anchors[anchors["card_id"].isin(todo)])
    n = db.insert_snapshots(prices.to_dict("records"))
    if not trends.empty:
        db.insert_trends(trends.to_dict("records"))
    log(f"Simulated history: {n:,} daily prices for {prices['card_id'].nunique() if n else 0} cards "
        f"({len(todo)} needed it)")
    return n
