"""Shared loaders and widgets for the Streamlit pages."""

from __future__ import annotations

import os
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


def _secrets_to_env() -> None:
    """Copy POKEPRICE_* Streamlit secrets into environment variables, where the pokeprice
    package reads its settings. Streamlit Cloud doesn't always do this before the app's
    code runs."""
    try:
        for key, value in st.secrets.items():
            if key.startswith("POKEPRICE_") and isinstance(value, str) and value.strip():
                os.environ[key] = value.strip()
    except Exception:  # no secrets file (e.g. running locally): nothing to copy
        pass


@st.cache_data(ttl=1800, show_spinner="Downloading the latest card data…")
def _sync_cached() -> str | None:
    # Exceptions aren't cached, so a failed download is retried on the next page load.
    return remote.sync()


def _sync() -> Exception | None:
    """On a hosted app, pull the data the refresh job published (at most every 30 minutes).
    Returns the error if the download failed; whatever data we already have keeps serving."""
    try:
        _sync_cached()
        return None
    except Exception as e:
        return e


def hosted() -> bool:
    _secrets_to_env()
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


def _sync_problem(err: Exception, blocking: bool) -> None:
    repo = os.environ.get("POKEPRICE_DATA_REPO", "")
    hint = ""
    status = getattr(getattr(err, "response", None), "status_code", None)
    if status in (401, 403):
        hint = ("The token was rejected. Check that `POKEPRICE_GITHUB_TOKEN` is pasted correctly, hasn't "
                "expired, and has *Contents: Read-only* on the data repo.")
    elif status == 404:
        hint = (f"GitHub couldn't find `{repo}` or its `data` release. Check the repo name, that the token "
                "is limited to that repo, and that the *Refresh data* workflow has published at least once.")
    msg = f"Couldn't download the published data from `{repo}`: {err}"
    if blocking:
        st.error(msg + ("\n\n" + hint if hint else ""))
        st.stop()
    st.caption(f"Showing the last downloaded data. {msg}")


def credits() -> None:
    """Data credits. PokéWallet's terms (section 4.4) require a visible, linked credit when its
    data is shown publicly; TCGapi suggests one."""
    line = ("Data by [PokéWallet](https://www.pokewallet.io) · Pricing data via [TCG API](https://tcgapi.dev) "
            "· Card data and images from the [Pokémon TCG API](https://pokemontcg.io)")
    st.caption(line)   # in the page body, so it stays visible when the sidebar is collapsed
    st.sidebar.caption("Not affiliated with Nintendo, Game Freak or The Pokémon Company. Forecasts are "
                       "estimates, not financial advice.")


def require_data() -> bool:
    """Pull fresh published data (hosted app), then show setup steps and stop the page if
    there is no data or model yet. Every page calls this first."""
    credits()
    err = _sync() if hosted() else None
    if err is not None:
        _sync_problem(err, blocking=cards().empty)
    if cards().empty:
        st.warning("No cards in the database yet.")
        st.markdown("**Running locally?** Build the data first:")
        st.code("python scripts/refresh.py --universe\npython scripts/train.py", language="bash")
        st.markdown("**On Streamlit Cloud?** Add `POKEPRICE_DATA_REPO` and `POKEPRICE_GITHUB_TOKEN` under "
                    "the app's *Settings → Secrets* (see the README's Deploy section).")
        st.stop()
    if bundle() is None:
        st.warning("No trained model yet.")
        st.code("python scripts/train.py", language="bash")
        st.stop()
    return True


def card_picker(key: str = "card_picker") -> pd.Series:
    """Searchable card selector shared across pages via session state."""
    c = cards()
    # Only cards with a forecast. Newly added cards appear once they have a real price
    # (usually within a day, as the hourly refresh works through them).
    latest, _ = latest_and_forecasts()
    if not latest.empty:
        c = c[c["card_id"].isin(latest.index)]
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
