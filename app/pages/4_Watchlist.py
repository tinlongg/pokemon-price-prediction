"""Watchlist: tracked cards, portfolio value and alerts."""

from __future__ import annotations

import streamlit as st

from common import HORIZON_LABELS, card_picker, cards, latest_and_forecasts, money, require_data
from pokeprice import db
from pokeprice.watchlist import evaluate

st.set_page_config(page_title="Watchlist · PokéPrice", page_icon=":material/bookmark:", layout="wide")
st.title("Watchlist")
require_data()

latest, fc = latest_and_forecasts()
horizon = st.segmented_control("Forecast horizon", list(HORIZON_LABELS), default=90,
                               format_func=HORIZON_LABELS.get) or 90
w = evaluate(db.load_watchlist(), fc, cards(), horizon=horizon)

if w.empty:
    st.info("Your watchlist is empty. Add a card below or from the Card Forecast page.")
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cards watched", len(w))
    k2.metric("Collection value", money(w["value"].sum()))
    k3.metric(f"Forecast value ({HORIZON_LABELS[horizon]})", money(w["forecast_value"].sum()),
              f"{(w['forecast_value'].sum() / w['value'].sum() - 1):+.1%}" if w["value"].sum() > 0 else None)
    k4.metric("Alerts", int((w["alerts"] != "").sum()))

    alerts = w[w["alerts"] != ""]
    for r in alerts.itertuples():
        st.warning(f"**{r.name}** ({r.set_name}): {r.alerts}. Now {money(r.current)}, "
                   f"forecast {money(r.forecast_p50)}.", icon=":material/notifications:")

    st.dataframe(w[["image_url", "name", "set_name", "current", "target_price", "forecast_p50", "upside",
                    "quantity", "value", "gain", "alerts"]], hide_index=True, row_height=60, column_config={
        "image_url": st.column_config.ImageColumn("", width="small"), "name": "Card", "set_name": "Set",
        "current": st.column_config.NumberColumn("Now", format="$%.2f"),
        "target_price": st.column_config.NumberColumn("Target", format="$%.2f"),
        "forecast_p50": st.column_config.NumberColumn("Forecast", format="$%.2f"),
        "upside": st.column_config.NumberColumn("Upside", format="percent"),
        "quantity": "Owned", "value": st.column_config.NumberColumn("Value", format="$%.2f"),
        "gain": st.column_config.NumberColumn("Gain vs paid", format="$%.2f"), "alerts": "Alerts"})

    rm = st.selectbox("Remove a card", ["—"] + w["card_id"].tolist(),
                      format_func=lambda i: i if i == "—" else w.set_index("card_id").loc[i, "name"])
    if rm != "—" and st.button("Remove"):
        db.watchlist_remove(rm)
        st.rerun()

st.subheader("Add a card")
card = card_picker("watch_picker")
cur = float(latest.loc[card.name, "market"]) if card.name in latest.index else 0.0
with st.form("add"):
    a, b, c = st.columns(3)
    target = a.number_input("Alert at or below ($)", min_value=0.0, value=round(cur * 0.9, 2))
    qty = b.number_input("Copies owned", min_value=0, value=0, step=1)
    cost = c.number_input("Total paid ($)", min_value=0.0, value=0.0)
    if st.form_submit_button("Add to watchlist", type="primary"):
        db.watchlist_add(card.name, target or None, int(qty), cost or None)
        st.rerun()
st.caption("A card is flagged **undervalued** when its median forecast is at least 15% above today's "
           "price and even the pessimistic (p10) forecast is no more than 5% below it.")
