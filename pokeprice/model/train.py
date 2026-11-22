"""Train one global LightGBM model per horizon that predicts the log return of a card's price.

Three quantile models per horizon (p10, p50, p90) give a forecast range. Evaluation uses
time-ordered folds with a gap equal to the horizon, so no future prices leak into training.
Simulated rows are down-weighted; metrics are also reported on real outcomes once enough exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import db
from ..config import HORIZONS, QUANTILES, settings
from ..features.build import FEATURES, build_features
from . import registry
from .baselines import NoChange, ridge

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)


def quantile_model(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="quantile", alpha=alpha, **LGB_PARAMS)


def training_frame(db_path: Path | None = None, stride: int | None = None) -> pd.DataFrame:
    cards = db.load_cards(db_path)
    prices = db.load_prices(db_path=db_path)
    trends = db.load_trends(db_path)
    return build_features(cards, prices, trends, HORIZONS, with_targets=True,
                          stride=stride or settings.train_stride_days)


def _weights(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["is_synthetic"] == 1, settings.synthetic_weight, 1.0)


def _scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"mae": float(np.mean(np.abs(y - pred))),
            "direction_acc": float(np.mean(np.sign(pred) == np.sign(y))) if len(y) else float("nan")}


def calibrate(y: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict:
    """Fit two numbers on out-of-sample backtest predictions:
    shrink: scale on the median forecast (0 = always predict "no change") minimising MAE;
    scale:  how much to stretch the p10–p90 band around it to cover ~80% of outcomes."""
    grid = np.linspace(0, 1, 21)
    maes = [np.mean(np.abs(y - lam * p50)) for lam in grid]
    lam = float(grid[int(np.argmin(maes))])
    centre = lam * p50
    lo_w, hi_w = np.maximum(p50 - p10, 1e-6), np.maximum(p90 - p50, 1e-6)
    scale = 6.0
    for s in np.arange(0.5, 6.01, 0.1):
        if np.mean((y >= centre - s * lo_w) & (y <= centre + s * hi_w)) >= 0.8:
            scale = float(round(s, 2))
            break
    return {"shrink": lam, "scale": scale}


def apply_calibration(p10, p50, p90, cal: dict | None):
    if not cal:
        return p10, p50, p90
    lam, s = cal["shrink"], cal["scale"]
    centre = lam * p50
    return (centre - s * np.maximum(p50 - p10, 0), centre, centre + s * np.maximum(p90 - p50, 0))


def evaluate_horizon(df: pd.DataFrame, h: int, n_folds: int = 3) -> dict:
    """Walk-forward backtest: train on the past, test on the next block of dates."""
    d = df.dropna(subset=[f"target_{h}"])
    dates = np.sort(d["date"].unique())
    if len(dates) < 20:
        return {"rows": int(len(d)), "note": "not enough history to backtest"}
    results = {"model": [], "no_change": [], "ridge": [], "coverage": [], "real_rows": 0}
    real_preds = {"y": [], "model": [], "no_change": [], "ridge": []}
    pooled = {"y": [], 0.1: [], 0.5: [], 0.9: []}
    cut_points = np.linspace(0.6, 0.9, n_folds)
    block = max(1, int(len(dates) * 0.1))
    for frac in cut_points:
        cut = dates[int(len(dates) * frac)]
        train = d[d["date"] <= cut - np.timedelta64(h, "D")]   # gap: train targets end before the test starts
        test_dates = dates[(dates > cut)][:block]
        test = d[d["date"].isin(test_dates)]
        if len(train) < 200 or len(test) < 20:
            continue
        X, y, w = train[FEATURES], train[f"target_{h}"], _weights(train)
        preds = {}
        for q in QUANTILES:
            m = quantile_model(q).fit(X, y, sample_weight=w)
            preds[q] = m.predict(test[FEATURES])
        lin = ridge().fit(X, y, ridge__sample_weight=w)
        yt = test[f"target_{h}"].values
        p50, plin = preds[0.5], lin.predict(test[FEATURES])
        results["model"].append(_scores(yt, p50))
        results["no_change"].append(_scores(yt, NoChange().predict(test)))
        results["ridge"].append(_scores(yt, plin))
        results["coverage"].append(float(np.mean((yt >= preds[0.1]) & (yt <= preds[0.9]))))
        pooled["y"].append(yt)
        for q in QUANTILES:
            pooled[q].append(preds[q])
        real = test[f"target_real_{h}"].values == 1
        results["real_rows"] += int(real.sum())
        real_preds["y"].append(yt[real]); real_preds["model"].append(p50[real])
        real_preds["no_change"].append(np.zeros(real.sum())); real_preds["ridge"].append(plin[real])

    def avg(lst, key):
        return float(np.mean([r[key] for r in lst])) if lst else float("nan")

    out = {
        "rows": int(len(d)),
        "mae": {k: avg(results[k], "mae") for k in ("model", "no_change", "ridge")},
        "direction_acc": {k: avg(results[k], "direction_acc") for k in ("model", "ridge")},
        "coverage_p10_p90": float(np.mean(results["coverage"])) if results["coverage"] else float("nan"),
        "real_rows": results["real_rows"],
    }
    if pooled["y"]:
        y = np.concatenate(pooled["y"])
        q10, q50, q90 = (np.concatenate(pooled[q]) for q in QUANTILES)
        cal = calibrate(y, q10, q50, q90)
        c10, c50, c90 = apply_calibration(q10, q50, q90, cal)
        out["calibration"] = cal
        out["calibrated"] = {"mae": float(np.mean(np.abs(y - c50))),
                             "coverage_p10_p90": float(np.mean((y >= c10) & (y <= c90)))}
    if results["real_rows"] >= 100:
        y = np.concatenate(real_preds["y"])
        out["real_mae"] = {k: float(np.mean(np.abs(y - np.concatenate(real_preds[k]))))
                           for k in ("model", "no_change", "ridge")}
    return out


def train_all(db_path: Path | None = None, models_dir: Path | None = None, backtest: bool = True,
              log: Callable[[str], None] = print) -> dict:
    df = training_frame(db_path)
    if df.empty:
        raise RuntimeError("No price data. Run scripts/refresh.py first.")
    log(f"Training rows: {len(df):,} ({int((df['is_synthetic'] == 0).sum()):,} real) "
        f"across {df['card_id'].nunique()} cards")
    models, metrics, importance = {}, {}, {}
    for h in HORIZONS:
        d = df.dropna(subset=[f"target_{h}"])
        if len(d) < 200:
            log(f"  {h}d: only {len(d)} rows, skipping")
            continue
        if backtest:
            metrics[h] = evaluate_horizon(df, h)
            m = metrics[h]
            if "mae" in m:
                log(f"  {h:>3}d backtest MAE (log return): model {m['mae']['model']:.3f} | "
                    f"no-change {m['mae']['no_change']:.3f} | ridge {m['mae']['ridge']:.3f} | "
                    f"coverage {m['coverage_p10_p90']:.0%}")
                if "calibration" in m:
                    log(f"        calibrated: shrink {m['calibration']['shrink']:.2f}, band ×{m['calibration']['scale']:.1f} "
                        f"-> MAE {m['calibrated']['mae']:.3f}, coverage {m['calibrated']['coverage_p10_p90']:.0%}")
        X, y, w = d[FEATURES], d[f"target_{h}"], _weights(d)
        models[h] = {q: quantile_model(q).fit(X, y, sample_weight=w) for q in QUANTILES}
        importance[h] = dict(zip(FEATURES, models[h][0.5].booster_.feature_importance("gain").tolist()))
    mix = db.data_mix(db_path)
    calibration = {h: m["calibration"] for h, m in metrics.items() if "calibration" in m}
    path = registry.save(models, metrics, FEATURES,
                         {"importance": importance, "data_mix": mix, "calibration": calibration},
                         models_dir=models_dir)
    log(f"Saved models to {path}")
    return {"metrics": metrics, "path": str(path)}
