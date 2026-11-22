"""Explain a forecast with SHAP: how much each factor pushed the predicted price up or down."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from ..features.build import FEATURE_GROUPS, FEATURE_LABELS

_explainers: dict[int, shap.TreeExplainer] = {}


def explainer(bundle: dict, horizon: int) -> shap.TreeExplainer:
    key = id(bundle["models"][horizon][0.5])
    if key not in _explainers:
        _explainers[key] = shap.TreeExplainer(bundle["models"][horizon][0.5])
    return _explainers[key]


def explain_row(bundle: dict, row: pd.Series, horizon: int) -> tuple[pd.DataFrame, float]:
    """Return per-feature contributions (log-return units and approx. %) and the base value."""
    feats = bundle["features"]
    X = row[feats].to_frame().T.astype(float)
    ex = explainer(bundle, horizon)
    # The served forecast is shrink × model output, so contributions scale the same way.
    shrink = bundle.get("calibration", {}).get(horizon, {}).get("shrink", 1.0)
    sv = np.asarray(ex.shap_values(X))[0] * shrink
    base = float(np.ravel(ex.expected_value)[0]) * shrink
    df = pd.DataFrame({
        "feature": feats,
        "label": [FEATURE_LABELS.get(f, f) for f in feats],
        "group": [FEATURE_GROUPS.get(f, "Other") for f in feats],
        "value": X.iloc[0].values,
        "shap": sv,
    })
    df["pct"] = np.exp(df["shap"]) - 1
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True), base


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    g = contrib.groupby("group", as_index=False)["shap"].sum()
    g["pct"] = np.exp(g["shap"]) - 1
    return g.sort_values("shap", ascending=False).reset_index(drop=True)


def reasons(contrib: pd.DataFrame, card_name: str, pokemon: str | None, top: int = 5) -> list[str]:
    """Plain-English bullets for the biggest contributions."""
    out = []
    for r in contrib.head(top).itertuples():
        sign = "+" if r.shap >= 0 else "−"
        pct = abs(r.pct) * 100
        label = r.label
        if r.feature == "popularity" and pokemon:
            label = f"{pokemon} popularity"
        out.append(f"{sign}{pct:.1f}% from {label} (value: {_fmt(r.feature, r.value)})")
    return out


def _fmt(feature: str, v: float) -> str:
    if v != v:  # NaN
        return "no data"
    if "ret_" in feature or feature == "drawdown_90":
        return f"{(np.exp(v) - 1) * 100:+.1f}%"
    if feature == "log_price":
        return f"${np.exp(v):,.2f}"
    if feature in ("age_years",):
        return f"{v:.1f} years"
    if feature.startswith("days_"):
        return f"{v:.0f} days"
    if feature in ("is_gen1", "is_gen2", "special_art", "full_art", "in_print", "is_trainer"):
        return "yes" if v >= 0.5 else "no"
    return f"{v:.2f}"
