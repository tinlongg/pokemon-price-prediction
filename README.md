# pokemon-price-prediction

**PokéPrice** forecasts Pokémon card prices 1, 3, 6 and 12 months out. It uses a LightGBM
model trained on many factors: how popular the Pokémon is (Charizard, Pikachu, Umbreon…),
nostalgia, rarity and art, supply and reprint risk, hype (Google Trends plus anniversaries and
game releases), and market momentum. A Streamlit app shows the forecasts and explains them.

## What's in the app

| Page | What it does |
|---|---|
| **Home** | Cards forecast to rise and fall the most, and the most valuable cards tracked |
| **Card Forecast** | Price history and the forecast range (p10–p90) for each horizon; add a card to your watchlist |
| **Why This Price** | A SHAP breakdown by factor, e.g. "+18% from Charizard popularity, −6% from reprint risk" |
| **Grading Calculator** | Expected value of grading with PSA vs selling raw, and the PSA 10 odds you need to break even |
| **Watchlist** | Your collection's value and forecast, with alerts for target prices and undervalued cards |
| **Model Performance** | Walk-forward backtest against "no change" and linear baselines, and which factors matter most |

## Setup

Needs Python 3.11–3.14 (LightGBM and SHAP don't publish wheels for 3.15 yet).

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env        # add POKEWALLET_API_KEY and TCGAPI_KEY (both free)
```

## Run it

```bash
.venv/bin/python scripts/refresh.py --universe   # first run: choose ~600 cards, fetch prices, simulate history (~8 min)
.venv/bin/python scripts/train.py                # train and backtest the models (~1–2 min)
.venv/bin/streamlit run app/Home.py
```

After that, run `scripts/refresh.py` daily (or hourly; see below) to record real prices, and
retrain now and then. Add `--trends` about once a week to pull Google Trends.

## Data sources

| Source | Key? | Used for |
|---|---|---|
| [PokéWallet](https://www.pokewallet.io) | Free key, no credit card | **Main daily price source** (TCGplayer market price). Searches once per Pokémon (100 results/page) and prices every tracked card of that Pokémon from the results, so a full sweep is a few hundred requests. Each run stays under the free 100/hour, so run hourly ([launchd template](scripts/com.pokeprice.refresh.plist)) to cover every card each day. |
| [TCGapi](https://tcgapi.dev) | Free key | **Real daily history**: 7 days per request, 100 requests/day. Runs automatically when `TCGAPI_KEY` is set. About 90 cards are refreshed each day (stalest first), so every card gets a continuous history. |
| [Pokémon TCG API](https://pokemontcg.io) | Not needed | Card details (set, rarity, artist, release date, images) and a fallback price. Its prices can be weeks old, so a card switches to PokéWallet/TCGapi as soon as it has daily data. It returns HTTP 500 on heavy queries, so queries are kept small and retried. |
| Google Trends (`pytrends`) | Not needed | Search interest per Pokémon (`--trends`). Unofficial and rate-limited; failures are skipped. |

Search APIs match names only, so cards are matched to PokéWallet and TCGapi by card number
plus set-name similarity ([matching.py](pokeprice/providers/matching.py)). TCGapi prices
are checked against PokéWallet's. A card whose TCGapi price is more than ~65% off (usually a
different printing) stops using TCGapi.

### About the simulated history

None of the free APIs offer long price history, so the app records its own daily snapshots
from the first run on. Until enough real history builds up, the model also trains on a
**simulated** multi-year history for each card, built from the same factors and ending at the
card's real current price (see [synthetic.py](pokeprice/ingest/synthetic.py)).

Simulated rows are flagged throughout, drawn dashed in charts and down-weighted in training,
and the Model Performance page shows how much data is real. Early forecasts therefore show
how the factor model behaves, not a proven read of the real market. Not financial advice.

## Project layout

```
pokeprice/
  providers/   pokemontcg.io, PokéWallet, TCGapi clients
  ingest/      card universe, daily snapshot, backfill, Google Trends, simulated history
  features/    popularity, nostalgia, rarity, supply, events, market momentum, feature builder
  model/       training + backtest, baselines, predictions, SHAP explanations, grading EV
  watchlist.py alerts and portfolio value
app/           Streamlit pages
scripts/       refresh.py, train.py, launchd job
tests/         pytest (HTTP mocked, no network needed)
```

Run the tests with `.venv/bin/python -m pytest`.

## Ideas for later

- Sealed product: is a booster box worth more opened or kept sealed?
- Price gaps between Japanese and English printings, and between TCGplayer and Cardmarket
- Discord or email alerts for watchlist hits
- Reddit and YouTube sentiment as extra hype signals
- Estimating a card's grade from a photo
- eBay sold prices, or PokéWallet Pro for real price history
- Deploying to Streamlit Community Cloud
