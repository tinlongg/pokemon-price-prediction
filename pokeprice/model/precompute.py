"""Precompute every card's latest features and forecasts.

Building features needs the whole price table (1M+ rows), which is too heavy for a small
hosted app. The refresh job runs this after each update and ships the result
(models/latest.joblib, ~1 MB); the app uses it whenever it matches the current data and model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .. import db
from ..config import HORIZONS, settings
from ..features.build import build_features, latest_rows
from . import registry
from .predict import forecast

LATEST = "latest.joblib"


def compute(bundle: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = build_features(db.load_cards(), db.load_prices(), db.load_trends(), HORIZONS,
                           with_targets=False, latest_only=True)
    if feats.empty:
        return pd.DataFrame(), pd.DataFrame()
    latest = latest_rows(feats)
    return latest, (forecast(bundle, latest) if bundle else pd.DataFrame())


def save(models_dir: Path | None = None) -> Path | None:
    bundle = registry.load(models_dir)
    if bundle is None:
        return None
    latest, fc = compute(bundle)
    path = Path(models_dir or settings.models_dir) / LATEST
    joblib.dump({"latest": latest, "forecasts": fc, "data_stamp": db.data_stamp(),
                 "trained_at": bundle["trained_at"]}, path)
    return path


def load_if_current(bundle: dict | None, models_dir: Path | None = None):
    """Return (latest, forecasts) if the saved file matches the current data and model."""
    path = Path(models_dir or settings.models_dir) / LATEST
    if bundle is None or not path.exists():
        return None
    saved = joblib.load(path)
    if saved.get("trained_at") != bundle["trained_at"] or tuple(saved.get("data_stamp", ())) != db.data_stamp():
        return None
    return saved["latest"], saved["forecasts"]
