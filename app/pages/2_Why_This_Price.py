"""Why This Price: SHAP breakdown of what drives a card's forecast."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import (DOWN_COLOR, HORIZON_LABELS, UP_COLOR, bundle, card_picker,
                    latest_and_forecasts, money, require_data, synthetic_note)
from pokeprice.model.explain import explain_row, group_contributions, reasons

st.set_page_config(page_title="Why This Price · PokéPrice", page_icon=":material/insights:", layout="wide")
st.title("Why this forecast?")
require_data()

card = card_picker()
horizon = st.segmented_control("Horizon", list(HORIZON_LABELS), default=365, format_func=HORIZON_LABELS.get) or 365
latest, fc = latest_and_forecasts()
b = bundle()
if card.name not in latest.index or horizon not in b["models"]:
    st.warning("No forecast for this card yet.")
    st.stop()

row = latest.loc[card.name]
contrib, base = explain_row(b, row, horizon)
groups = group_contributions(contrib)
f = fc[(fc["card_id"] == card.name) & (fc["horizon"] == horizon)].iloc[0]

st.markdown(f"**{card['name']}** ({card['set_name']}): {money(f['current'])} now, "
            f"**{money(f['p50'])}** forecast in {HORIZON_LABELS[horizon]} (**{f['pct_p50']:+.1%}**).")
st.caption(f"The average card is forecast to move {np.exp(base) - 1:+.1%} over this horizon. "
           "The factors below explain how this card differs from that average.")
synthetic_note()

left, right = st.columns([2, 3])
with left:
    st.subheader("By factor")
    fig = go.Figure(go.Bar(
        x=groups["pct"] * 100, y=groups["group"], orientation="h",
        marker_color=[UP_COLOR if v >= 0 else DOWN_COLOR for v in groups["shap"]],
        text=[f"{v:+.1f}%" for v in groups["pct"] * 100], textposition="outside"))
    fig.update_layout(height=340, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="Effect on forecast (%)",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("In plain English")
    for r in reasons(contrib, card["name"], card["pokemon"], top=6):
        st.markdown(f"- {r}")

with right:
    st.subheader("Top drivers")
    top = contrib.head(12).iloc[::-1]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=top["label"], x=top["shap"] * 100, base=0,
        increasing=dict(marker_color=UP_COLOR), decreasing=dict(marker_color=DOWN_COLOR),
        text=[f"{v:+.1f}" for v in top["shap"] * 100], textposition="outside"))
    fig.update_layout(height=520, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis_title="Contribution (log-return points ×100)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with st.expander("All factor values for this card"):
    st.dataframe(contrib[["group", "label", "value", "pct"]], hide_index=True, column_config={
        "group": "Factor", "label": "Feature", "value": st.column_config.NumberColumn("Value", format="%.3f"),
        "pct": st.column_config.NumberColumn("Effect", format="percent")})
