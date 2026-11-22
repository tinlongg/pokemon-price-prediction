"""End-to-end: simulated history -> features -> training -> forecast -> explanation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pokeprice import db
from pokeprice.config import HORIZONS
from pokeprice.features.build import FEATURES, build_features, latest_rows
from pokeprice.ingest.synthetic import ensure_synthetic, generate
from pokeprice.model import registry
from pokeprice.model.explain import explain_row, group_contributions, reasons
from pokeprice.model.grading import grading_ev
from pokeprice.model.predict import forecast
from pokeprice.model.train import train_all
from pokeprice.watchlist import evaluate


def test_synthetic_history_ends_at_real_price(seeded_db):
    cards = db.load_cards()
    anchors = pd.DataFrame({"card_id": ["base1-4"], "date": ["2026-11-20"], "market": [950.0]})
    prices, trends = generate(cards, anchors, years=2, seed=1)
    char = prices[prices["card_id"] == "base1-4"]
    assert (char["is_synthetic"] == 1).all()
    assert char["date"].max() < "2026-11-20"
    assert abs(char["market"].iloc[-1] / 950 - 1) < 0.05
    assert not trends.empty and trends["interest"].between(1, 100).all()
    # Same seed, same history.
    again, _ = generate(cards, anchors, years=2, seed=1)
    assert again["market"].equals(prices["market"])


def test_end_to_end(seeded_db):
    ensure_synthetic(force=True, log=lambda *_: None)
    mix = db.data_mix()
    assert mix["synthetic"]["rows"] > 1000 and mix["real"]["rows"] == 6

    result = train_all(backtest=True, log=lambda *_: None)
    bundle = registry.load()
    assert set(bundle["models"]) == set(HORIZONS)
    m = result["metrics"][365]
    assert m["mae"]["model"] > 0
    cal = m["calibration"]
    assert 0 <= cal["shrink"] <= 1 and cal["scale"] > 0
    assert m["calibrated"]["mae"] <= m["mae"]["model"] + 1e-9   # shrinking can only help in-sample

    feats = build_features(db.load_cards(), db.load_prices(), db.load_trends(), with_targets=False,
                           latest_only=True)
    latest = latest_rows(feats)
    assert latest.loc["base1-4", "market"] == 950.0  # forecasts start from the real price
    fc = forecast(bundle, latest)
    assert len(fc) == 6 * len(HORIZONS)
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()

    contrib, base = explain_row(bundle, latest.loc["base1-4"], 365)
    assert set(contrib["feature"]) == set(FEATURES)
    # SHAP contributions add up to the model's prediction.
    shrink = bundle["calibration"].get(365, {}).get("shrink", 1.0)
    pred = shrink * bundle["models"][365][0.5].predict(latest.loc[["base1-4"], FEATURES].astype(float))[0]
    assert np.isclose(base + contrib["shap"].sum(), pred, atol=1e-4)
    assert len(reasons(contrib, "Charizard", "Charizard")) == 5
    assert not group_contributions(contrib).empty

    db.watchlist_add("base1-4", target_price=2000, quantity=1, cost_basis=500)
    w = evaluate(db.load_watchlist(), fc, db.load_cards(), horizon=90)
    assert "At or below target" in w.iloc[0]["alerts"]
    assert w.iloc[0]["gain"] == 450


def test_grading_vintage_vs_modern():
    vintage = grading_ev(100, era_idx=0, rarity_tier=3, popularity=1.0)
    modern = grading_ev(100, era_idx=7, rarity_tier=3, popularity=1.0)
    assert vintage.psa10_price > modern.psa10_price      # bigger gem premium for vintage
    assert vintage.p10 < modern.p10                      # but much harder to get a 10
    cheap = grading_ev(3, era_idx=7, rarity_tier=1, popularity=0.1)
    assert not cheap.worth_it                            # fees swamp a $3 card
    assert 0 <= vintage.breakeven_p10 <= 1


def test_series_stays_on_one_source_family(seeded_db):
    # pokemontcg.io's stale $79 must not mix with PokéWallet's daily $329 (a fake 4x jump).
    db.insert_snapshots([
        {"card_id": "base1-58", "date": "2026-10-15", "source": "pokemontcg", "market": 79.0, "is_synthetic": 0},
        {"card_id": "base1-58", "date": "2026-11-21", "source": "pokewallet", "market": 329.0, "is_synthetic": 0},
        {"card_id": "base1-58", "date": "2026-11-21", "source": "tcgapi", "market": 330.0, "is_synthetic": 0},
    ])
    p = db.load_prices(["base1-58"])
    assert set(p["source"]) == {"pokewallet"}                 # pokemontcg dropped, PokéWallet wins the day
    # Cards with only pokemontcg.io data keep it.
    assert set(db.load_prices(["base1-4"])["source"]) == {"pokemontcg"}
