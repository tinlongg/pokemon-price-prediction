"""Card Forecast: price history (real and simulated) with the forecast range for each horizon."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (FORECAST_COLOR, HORIZON_LABELS, REAL_COLOR, SIM_COLOR, card_picker,
                    latest_and_forecasts, money, prices, require_data, synthetic_note)
from pokeprice import db

st.set_page_config(page_title="Card Forecast · PokéPrice", page_icon=":material/trending_up:", layout="wide")
st.title("Card Forecast")
require_data()

card = card_picker()
card_id = card.name
latest, fc = latest_and_forecasts()
f = fc[fc["card_id"] == card_id].sort_values("horizon")
hist = prices(card_id)

if f.empty or hist.empty:
    st.warning("No price data for this card yet.")
    st.stop()

img, info = st.columns([1, 3])
with img:
    if card["image_url"]:
        st.image(card["image_url"], width=240)
with info:
    st.subheader(card["name"])
    st.caption(f"{card['set_name']} · #{card['number']} · {card['rarity'] or 'Unknown rarity'} · "
               f"illustrated by {card['artist'] or 'unknown'} · released {card['release_date']}")
    current = float(f["current"].iloc[0])
    st.metric("Price now", money(current))
    last_real = hist[hist["is_synthetic"] == 0]
    if not last_real.empty:
        lr = last_real.iloc[-1]
        st.caption(f"Latest real price {money(lr['market'])} from {lr['source']} on {lr['date']:%b %d, %Y}"
                   + (f" ({lr['variant']})" if lr["variant"] else ""))

cols = [c for _ in range(2) for c in st.columns(2)]   # 2x2 grid fits narrow windows
for col, r in zip(cols, f.itertuples()):
    col.metric(f"In {HORIZON_LABELS[r.horizon]}", money(r.p50), f"{r.pct_p50:+.1%}", border=True,
               help=f"80% range: {money(r.p10)} to {money(r.p90)}")

synthetic_note()

# --- chart --------------------------------------------------------------
fig = go.Figure()
sim = hist[hist["is_synthetic"] == 1]
real = hist[hist["is_synthetic"] == 0]
if not sim.empty:
    fig.add_scatter(x=sim["date"], y=sim["market"], name="Simulated history",
                    line=dict(color=SIM_COLOR, dash="dot", width=1.5))
if not real.empty:
    fig.add_scatter(x=real["date"], y=real["market"], name="Real prices", mode="lines+markers",
                    line=dict(color=REAL_COLOR, width=2.5), marker=dict(size=5))

start = hist["date"].max()
fx = [start] + [start + pd.Timedelta(days=int(h)) for h in f["horizon"]]
fig.add_scatter(x=fx, y=[current] + f["p90"].tolist(), mode="lines", line=dict(width=0), showlegend=False,
                hoverinfo="skip")
fig.add_scatter(x=fx, y=[current] + f["p10"].tolist(), mode="lines", fill="tonexty", line=dict(width=0),
                fillcolor="rgba(227,53,13,0.15)", name="80% forecast range")
fig.add_scatter(x=fx, y=[current] + f["p50"].tolist(), name="Forecast (median)", mode="lines+markers",
                line=dict(color=FORECAST_COLOR, width=2.5))
fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  yaxis_title="Price (USD)", legend=dict(orientation="h", y=1.08))
log_scale = st.toggle("Log scale", value=False)
if log_scale:
    fig.update_yaxes(type="log")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
st.caption("Each horizon is calibrated on its own backtest: where the model showed little skill (so far the "
           "12-month horizon), its forecast is pulled toward today's price and its range is widened. "
           "That's why a longer horizon can show a smaller move than a shorter one.")

table = f[["horizon", "p10", "p50", "p90", "pct_p50"]].copy()
table["horizon"] = table["horizon"].map(HORIZON_LABELS)
st.dataframe(table, hide_index=True, column_config={
    "horizon": "Horizon",
    "p10": st.column_config.NumberColumn("Low (p10)", format="$%.2f"),
    "p50": st.column_config.NumberColumn("Median", format="$%.2f"),
    "p90": st.column_config.NumberColumn("High (p90)", format="$%.2f"),
    "pct_p50": st.column_config.NumberColumn("Median change", format="percent"),
})

# --- watchlist --------------------------------------------------------------
with st.expander("Add to watchlist"):
    with st.form("watch"):
        a, b, c = st.columns(3)
        target = a.number_input("Alert me at or below ($)", min_value=0.0, value=round(current * 0.9, 2))
        qty = b.number_input("Copies owned", min_value=0, value=0, step=1)
        cost = c.number_input("Total paid ($)", min_value=0.0, value=0.0)
        if st.form_submit_button("Save to watchlist", type="primary"):
            db.watchlist_add(card_id, target or None, int(qty), cost or None)
            st.success("Saved. See the Watchlist page.")

st.page_link("pages/2_Why_This_Price.py", label="Why this forecast?", icon=":material/insights:")
