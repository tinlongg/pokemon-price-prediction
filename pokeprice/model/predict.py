"""Turn the latest feature rows into price forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import QUANTILES
from .train import apply_calibration


def forecast(bundle: dict, latest: pd.DataFrame) -> pd.DataFrame:
    """latest: one row per card (index card_id) with feature columns and `market`.
    Returns long format: card_id, horizon, current, p10, p50, p90 (prices) and pct change at p50."""
    feats = bundle["features"]
    X = latest[feats]
    out = []
    for h, qmodels in bundle["models"].items():
        preds = np.column_stack([qmodels[q].predict(X) for q in QUANTILES])
        preds.sort(axis=1)  # keep p10 <= p50 <= p90
        # Backtest calibration: shrink toward "no change" and widen the band (see train.calibrate).
        cal = bundle.get("calibration", {}).get(h)
        if cal:
            preds = np.column_stack(apply_calibration(preds[:, 0], preds[:, 1], preds[:, 2], cal))
        cur = latest["market"].values
        out.append(pd.DataFrame({
            "card_id": latest.index, "horizon": int(h), "current": cur,
            "p10": cur * np.exp(preds[:, 0]), "p50": cur * np.exp(preds[:, 1]),
            "p90": cur * np.exp(preds[:, 2]), "pct_p50": np.exp(preds[:, 1]) - 1,
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
