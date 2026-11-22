"""Shared loaders and widgets for the Streamlit pages."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pokeprice import db  # noqa: E402
from pokeprice.config import settings  # noqa: E402
from pokeprice.features.popularity import popularity_score  # noqa: E402
from pokeprice import remote  # noqa: E402
from pokeprice.model import precompute, registry  # noqa: E402

REAL_COLOR = "#2563EB"
SIM_COLOR = "#9CA3AF"
FORECAST_COLOR = "#E3350D"
UP_COLOR = "#16A34A"
DOWN_COLOR = "#DC2626"

HORIZON_LABELS = {30: "1 month", 90: "3 months", 180: "6 months", 365: "12 months"}


@st.cache_data(ttl=1800, show_spinner="Downloading the latest card data…")
def _sync() -> str | None:
    """On a hosted app, pull the data the refresh job published (at most every 30 minutes)."""
    try:
        return remote.sync()
    except Exception as e:  # keep serving whatever data we already have
        st.toast(f"Couldn't fetch the latest data: {e}")
        return remote.local_version()


def hosted() -> bool:
    return remote.configured()


def _db_stamp() -> tuple:
    return db.data_stamp()


def _model_stamp() -> float:
    p = Path(settings.models_dir) / registry.BUNDLE
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _cards(stamp: tuple) -> pd.DataFrame:
    cards = db.load_cards()
    if cards.empty:
        return cards
    cards["popularity"] = [popularity_score(p, s, d) for p, s, d in
                           zip(cards["pokemon"], cards["supertype"], cards["dex_number"])]
    cards["label"] = (cards["name"] + " · " + cards["set_name"].fillna("?") + " #" + cards["number"].fillna("?")
                      + " · " + cards["rarity"].fillna(""))
    return cards.sort_values(["popularity", "release_date"], ascending=[False, True]).reset_index(drop=True)


def cards() -> pd.DataFrame:
    return _cards(_db_stamp())


@st.cache_data(show_spinner=False)
def _prices(card_id: str, stamp: tuple) -> pd.DataFrame:
    return db.load_prices([card_id])


def prices(card_id: str) -> pd.DataFrame:
    return _prices(card_id, _db_stamp())


@st.cache_resource(show_spinner=False)
def _bundle(stamp: float):
    return registry.load()


def bundle():
    return _bundle(_model_stamp())


@st.cache_data(show_spinner="Computing forecasts for every card…")
def _latest_and_forecasts(db_stamp: tuple, model_stamp: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = _bundle(model_stamp)
    saved = precompute.load_if_current(b)   # shipped by the refresh job; skips the heavy rebuild
    return saved if saved is not None else precompute.compute(b)


def latest_and_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _latest_and_forecasts(_db_stamp(), _model_stamp())


@st.cache_data(show_spinner=False)
def _mix(stamp: tuple) -> dict:
    return db.data_mix()


def data_mix() -> dict:
    return _mix(_db_stamp())


def require_data() -> bool:
    """Pull fresh published data (hosted app), then show setup steps and stop the page if
    there is no data or model yet. Every page calls this first."""
    if hosted():
        _sync()
    if cards().empty:
        st.warning("No cards in the database yet.")
        st.code("python scripts/refresh.py --universe\npython scripts/train.py", language="bash")
        st.stop()
    if bundle() is None:
        st.warning("No trained model yet.")
        st.code("python scripts/train.py", language="bash")
        st.stop()
    return True


def card_picker(key: str = "card_picker") -> pd.Series:
    """Searchable card selector shared across pages via session state."""
    c = cards()
    ids = c["card_id"].tolist()
    current = st.session_state.get("card_id", ids[0])
    idx = ids.index(current) if current in ids else 0
    labels = dict(zip(c["card_id"], c["label"]))
    choice = st.selectbox("Card", ids, index=idx, format_func=lambda i: labels.get(i, i), key=key,
                          help="Start typing a name, e.g. Charizard, Umbreon, Lillie")
    st.session_state["card_id"] = choice
    return c.set_index("card_id").loc[choice]


def money(v: float | None) -> str:
    if v is None or v != v:
        return "–"
    return f"${v:,.2f}" if v < 100 else f"${v:,.0f}"


def synthetic_note() -> None:
    mix = data_mix()
    real_days = mix.get("real_days", 0)
    if real_days < 60:
        st.info(f"Real price history so far: {real_days} day(s). Until more builds up, forecasts lean on "
                "simulated history (dashed lines), so treat them as illustrations of the factor model "
                "rather than investment advice.", icon=":material/info:")
