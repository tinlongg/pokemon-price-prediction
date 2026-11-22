"""Model Performance: backtest results vs baselines, data mix and factor importance."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import data_mix, FORECAST_COLOR, HORIZON_LABELS, REAL_COLOR, SIM_COLOR, bundle, require_data
from pokeprice import db
from pokeprice.features.build import FEATURE_GROUPS

st.set_page_config(page_title="Model Performance · PokéPrice", page_icon=":material/monitoring:", layout="wide")
st.title("Model performance")
require_data()
b = bundle()
mix = data_mix()

st.caption(f"Trained {b['trained_at']}. One LightGBM model per horizon, trained across all cards, "
           "predicting the log change in price with p10/p50/p90 quantiles.")

k1, k2, k3 = st.columns(3)
k1.metric("Real price rows", f"{mix['real']['rows']:,}", help=f"{mix['real_days']} distinct days")
k2.metric("Simulated rows", f"{mix['synthetic']['rows']:,}")
share = mix["real"]["rows"] / max(1, mix["real"]["rows"] + mix["synthetic"]["rows"])
k3.metric("Share of real data", f"{share:.2%}")
if mix["real_days"] < 60:
    st.info("Backtest scores below are measured mostly on simulated history, so they show that the "
            "model learns the factor structure, not that it predicts the real market. Real-data scores "
            "appear once real snapshots cover more than a forecast horizon.", icon=":material/info:")

rows = []
for h, m in b["metrics"].items():
    if "mae" not in m:
        continue
    rows.append({"Horizon": HORIZON_LABELS[int(h)], "Model MAE": m["mae"]["model"],
                 "No-change MAE": m["mae"]["no_change"], "Ridge MAE": m["mae"]["ridge"],
                 "Calibrated MAE": m.get("calibrated", {}).get("mae"),
                 "Shrink": m.get("calibration", {}).get("shrink"),
                 "Direction accuracy": m["direction_acc"]["model"],
                 "p10–p90 coverage": m.get("calibrated", {}).get("coverage_p10_p90", m["coverage_p10_p90"]),
                 "Real-outcome rows": m.get("real_rows", 0),
                 "Real MAE (model / no-change)": (f"{m['real_mae']['model']:.3f} / {m['real_mae']['no_change']:.3f}"
                                                  if "real_mae" in m else "not enough yet")})
if rows:
    t = pd.DataFrame(rows)
    st.subheader("Walk-forward backtest")
    st.dataframe(t, hide_index=True, column_config={
        c: st.column_config.NumberColumn(c, format="%.3f")
        for c in ("Model MAE", "Calibrated MAE", "No-change MAE", "Ridge MAE")} | {
        "Shrink": st.column_config.NumberColumn(format="%.2f", help="How much of the raw model forecast is kept "
                                                "after backtest calibration (0 = always 'no change')"),
        "Direction accuracy": st.column_config.NumberColumn(format="percent"),
        "p10–p90 coverage": st.column_config.NumberColumn(format="percent")})
    fig = go.Figure()
    for col, color in (("Calibrated MAE", FORECAST_COLOR), ("Ridge MAE", REAL_COLOR), ("No-change MAE", SIM_COLOR)):
        fig.add_bar(x=t["Horizon"], y=t[col], name=col.replace(" MAE", ""), marker_color=color)
    fig.update_layout(barmode="group", height=340, yaxis_title="Mean abs. error (log return)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Lower is better. Calibration scales each horizon's forecast toward 'no change' by how much "
               "skill the model showed in the backtest, and widens its range until p10–p90 covers about 80% "
               "of outcomes. A low shrink (e.g. at 12 months) means long-range moves aren't predictable yet.")
else:
    st.info("Train without --fast to see backtest results.")

st.subheader("Which factors matter most")
h = st.segmented_control("Horizon", [int(k) for k in b["importance"]], default=365,
                         format_func=HORIZON_LABELS.get) or 365
imp = pd.Series(b["importance"][h])
groups = imp.groupby(imp.index.map(lambda f: FEATURE_GROUPS.get(f, "Other"))).sum()
groups = (groups / groups.sum()).sort_values()
fig = go.Figure(go.Bar(x=groups.values, y=groups.index, orientation="h", marker_color=FORECAST_COLOR,
                       text=[f"{v:.0%}" for v in groups.values], textposition="outside"))
fig.update_layout(height=320, xaxis_tickformat=".0%", margin=dict(l=10, r=40, t=10, b=10),
                  xaxis_title="Share of model gain")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
