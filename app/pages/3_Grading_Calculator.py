"""Grading Calculator: is it worth sending this card to PSA?"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from common import (DOWN_COLOR, UP_COLOR, card_picker, latest_and_forecasts, money, require_data)
from pokeprice.model.grading import CONDITION_SHIFT, grading_ev

st.set_page_config(page_title="Grading Calculator · PokéPrice", page_icon=":material/verified:", layout="wide")
st.title("Grading calculator")
st.caption("Expected value of grading a card vs selling it raw. Grade odds and PSA premiums are "
           "estimates by era, rarity and popularity, not live PSA population data.")
require_data()

card = card_picker()
latest, fc = latest_and_forecasts()
row = latest.loc[card.name]
f12 = fc[(fc["card_id"] == card.name) & (fc["horizon"] == 365)]

a, b, c, d = st.columns(4)
raw = a.number_input("Raw Near Mint price ($)", min_value=0.01, value=float(round(row["market"], 2)))
condition = b.selectbox("Condition", list(CONDITION_SHIFT), index=1)
fee = c.number_input("Grading fee ($)", min_value=0.0, value=25.0, step=5.0)
ship = d.number_input("Shipping + insurance ($)", min_value=0.0, value=10.0, step=1.0)
sell_fee = st.slider("Selling fees (eBay/TCGplayer)", 0, 20, 13, format="%d%%") / 100

args = dict(era_idx=int(row["era_idx"]), rarity_tier=int(row["rarity_tier"]), popularity=float(row["popularity"]),
            condition=condition, grading_fee=fee, shipping=ship, selling_fee_pct=sell_fee)
res = grading_ev(raw, **args)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Estimated PSA 10", money(res.psa10_price))
m2.metric("Estimated PSA 9", money(res.psa9_price))
m3.metric("Expected value if graded", money(res.expected_graded_value), help="After selling fees")
m4.metric("Profit vs selling raw", money(res.expected_profit_vs_raw),
          "Worth grading" if res.worth_it else "Not worth it", delta_color="normal" if res.worth_it else "inverse")

left, right = st.columns(2)
with left:
    fig = go.Figure(go.Bar(x=["PSA 10", "PSA 9", "PSA 8 or lower"], y=[res.p10, res.p9, res.p_low],
                           marker_color=[UP_COLOR, "#60A5FA", "#9CA3AF"],
                           text=[f"{v:.0%}" for v in (res.p10, res.p9, res.p_low)], textposition="outside"))
    fig.update_layout(title="Estimated grade odds", height=320, yaxis_tickformat=".0%",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
with right:
    st.markdown("**How to read this**")
    if res.breakeven_p10 <= 0.005:
        odds_line = ("- Grading pays off **even without a PSA 10**: graded 9s of this card sell for well above raw "
                     f"(estimated chance of a 10: **{res.p10:.0%}**).")
    elif res.breakeven_p10 >= 0.995:
        odds_line = "- Even a guaranteed PSA 10 wouldn't cover the fees for this card."
    else:
        odds_line = (f"- You need roughly a **{res.breakeven_p10:.0%}** chance of a PSA 10 to break even; "
                     f"the estimate for this card is **{res.p10:.0%}**.")
    st.markdown(
        f"{odds_line}\n"
        f"- Total cost to grade: {money(res.total_cost)}.\n"
        f"- Vintage cards rarely come back a 10, but when they do the premium is large; modern cards "
        f"grade 10 more often with smaller premiums.")
    if not f12.empty:
        future_raw = float(f12["p50"].iloc[0])
        fut = grading_ev(future_raw, **args)
        st.markdown(f"**In 12 months** (raw forecast {money(future_raw)}): grading would be worth "
                    f"{money(fut.expected_graded_value)} expected, "
                    f"{'a' if fut.worth_it else 'not a'} profit of {money(fut.expected_profit_vs_raw)} vs raw.")
color = UP_COLOR if res.worth_it else DOWN_COLOR
st.markdown(f"<div style='padding:12px;border-left:4px solid {color}'>"
            f"{'Grading looks worth it for this card.' if res.worth_it else 'Selling raw looks better for this card.'}"
            f"</div>", unsafe_allow_html=True)
