"""PokéPrice: home page with market overview and the biggest forecast movers."""

from __future__ import annotations

import streamlit as st

from common import (data_mix, HORIZON_LABELS, bundle, cards, latest_and_forecasts, require_data,
                    synthetic_note)
from pokeprice import db

st.set_page_config(page_title="PokéPrice", page_icon=":material/trending_up:", layout="wide")

st.title("PokéPrice")
st.caption("Forecasts for Pokémon card prices, driven by popularity, nostalgia, rarity, hype and market momentum.")

require_data()
synthetic_note()

c = cards()
b = bundle()
latest, fc = latest_and_forecasts()
mix = data_mix()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cards tracked", f"{len(c):,}")
k2.metric("Days of real prices", f"{mix['real_days']:,}")
k3.metric("Simulated price rows", f"{mix['synthetic']['rows'] / 1e6:.1f}M")
k4.metric("Model trained", f"{b['trained_at'][5:10]}", help=b["trained_at"])

horizon = st.segmented_control("Forecast horizon", list(HORIZON_LABELS), default=365,
                               format_func=HORIZON_LABELS.get)
horizon = horizon or 365
f = fc[fc["horizon"] == horizon].merge(c[["card_id", "name", "set_name", "rarity", "image_url"]], on="card_id")
f = f[f["current"] >= 2]  # ignore bulk-priced cards

cols = {
    "image_url": st.column_config.ImageColumn("", width="small"),
    "name": "Card", "set_name": "Set", "rarity": "Rarity",
    "current": st.column_config.NumberColumn("Now", format="$%.2f"),
    "p50": st.column_config.NumberColumn("Forecast", format="$%.2f"),
    "pct_p50": st.column_config.NumberColumn("Change", format="percent"),
    "p10": st.column_config.NumberColumn("Low (p10)", format="$%.2f"),
    "p90": st.column_config.NumberColumn("High (p90)", format="$%.2f"),
}
show = ["image_url", "name", "set_name", "rarity", "current", "p50", "pct_p50", "p10", "p90"]

left, right = st.columns(2)
with left:
    st.subheader(f"Forecast to rise most ({HORIZON_LABELS[horizon]})")
    st.dataframe(f.nlargest(15, "pct_p50")[show], column_config=cols, hide_index=True, row_height=60)
with right:
    st.subheader(f"Forecast to fall most ({HORIZON_LABELS[horizon]})")
    st.dataframe(f.nsmallest(15, "pct_p50")[show], column_config=cols, hide_index=True, row_height=60)

st.subheader("Most valuable cards tracked")
st.dataframe(f.nlargest(15, "current")[show], column_config=cols, hide_index=True, row_height=60)
st.caption("Pick any card on the **Card Forecast** page to see its full chart and the reasons behind its forecast.")
