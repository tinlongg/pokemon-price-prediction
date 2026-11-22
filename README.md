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
.venv/bin/python scripts/refresh.py --universe   # first run: choose ~1,600 cards, fetch prices, simulate history (~10 min)
.venv/bin/python scripts/train.py                # train and backtest the models (~1–2 min)
.venv/bin/streamlit run app/Home.py
```

After that, run `scripts/refresh.py` daily (or hourly; see below) to record real prices, and
retrain now and then. Add `--trends` about once a week to pull Google Trends.

## Which cards are tracked

- **Every Illustration Rare and Special Illustration Rare**: about 1,000 cards across the 25
  sets from Scarlet & Violet (March 2023) onwards. These are always tracked, even when
  pokemontcg.io has no price for them (common for the newest sets); PokéWallet prices them.
- **Up to 600 other collector cards**: vintage WOTC holos, gold stars and shinings, other
  modern chase cards, and popular Pokémon across eras. A quarter of these slots are spread
  across eras so the model also sees ordinary cards.

The list is rebuilt on the 1st of each month by the hosted workflow (or with
`refresh.py --universe`), so new sets' IRs and SIRs are added automatically. A newly added
card shows up in the app once it has its first real price, usually within a day.

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

A card gets simulated history only once it has a real price to end at, and only cards that
need it are regenerated (new cards, or a card whose first real price changed). Each card's and
each Pokémon's simulated series is seeded on its own, so it stays the same however the rest
of the card list changes.

Simulated rows are flagged throughout, drawn dashed in charts and down-weighted in training,
and the Model Performance page shows how much data is real. Early forecasts therefore show
how the factor model behaves, not a proven read of the real market. Not financial advice.

## Deploy (Streamlit Community Cloud)

The code is public; the data is not. TCG API's terms forbid public downloads of its records,
so the database lives in a separate **private** data repo.

A GitHub Actions workflow ([refresh.yml](.github/workflows/refresh.yml)) runs every hour. It
downloads the last published data from the data repo, fetches new prices, precomputes every
card's forecast and publishes the results to that repo's `data` release. Once a day it also
pulls TCGapi history and retrains. The hosted app downloads the release on startup and checks
for a newer version every 30 minutes ([remote.py](pokeprice/remote.py)). The workflow and
the seeding script both refuse to publish to a repo that isn't private.

1. **Create the private data repo**, e.g. `pokeprice-data`. Tick *Add a README*: a release
   needs at least one commit.
2. **Create two fine-grained tokens** (GitHub → Settings → Developer settings → Fine-grained
   tokens), each limited to the data repo only:
   - for the workflow: *Contents: Read and write*
   - for the app: *Contents: Read-only*
3. **Configure this repo** (Settings → Secrets and variables → Actions):
   - Variables tab: `POKEPRICE_DATA_REPO` = `tinlongg/pokeprice-data`
   - Secrets tab: `DATA_REPO_TOKEN` (the read-and-write token), `POKEWALLET_API_KEY`, `TCGAPI_KEY`
4. **Publish the first data**, either way:
   - Actions → *Refresh data* → *Run workflow*. Builds everything from scratch (~15–20 min).
   - Or from this machine, keeping the prices and card matches you already have:
     `brew install gh && gh auth login`, then `scripts/publish_data.sh tinlongg/pokeprice-data`.
5. **Create the app** at [share.streamlit.io](https://share.streamlit.io): pick this repo,
   branch `main`, main file `app/Home.py`. Under *Advanced settings*, choose Python 3.12 and
   paste these secrets:

   ```toml
   POKEPRICE_DATA_REPO = "tinlongg/pokeprice-data"
   POKEPRICE_GITHUB_TOKEN = "github_pat_..."   # the read-only token
   ```

Notes:
- **Actions minutes:** public repos have no limit, so the workflow runs hourly. That uses
  PokéWallet's free 100 requests/hour, which covers every card several times a day.
- **Inactive repos:** GitHub pauses scheduled workflows in public repos after 60 days without
  commits. Re-enable it from the Actions tab if that happens.
- **Watchlist:** on the hosted app it's shared by all visitors and resets on restart. Your
  local watchlist is never published (the workflow and `publish_data.sh` clear it).
- **Running locally:** leave `POKEPRICE_DATA_REPO` unset and the app uses your own `data/`
  and `models/`.

## Project layout

```
pokeprice/
  providers/   pokemontcg.io, PokéWallet, TCGapi clients
  ingest/      card universe, daily snapshot, backfill, Google Trends, simulated history
  features/    popularity, nostalgia, rarity, supply, events, market momentum, feature builder
  model/       training + backtest, baselines, predictions, SHAP explanations, grading EV
  watchlist.py alerts and portfolio value
  remote.py    downloads the published data for a hosted app
app/           Streamlit pages
scripts/       refresh.py, train.py, precompute.py, publish_data.sh, launchd job
.github/       workflow that refreshes and publishes data for the hosted app
tests/         pytest (HTTP mocked, no network needed)
```

Run the tests with `.venv/bin/python -m pytest`.

## Data credits and terms

- **PokéWallet:** its terms require a visible "Data by PokéWallet" credit linking to
  pokewallet.io wherever its data is shown publicly. The app shows this in the sidebar.
- **TCG API:** the free plan is for non-commercial use only. Its licensing page forbids
  "public, unrestricted downloads of our records", so the published database must stay in
  a private repo, never a public release (see Deploy).
- **Pokémon TCG API:** deprecated and now part of Scrydex. Keys work until March 1, 2027.
- Pokémon and card artwork belong to Nintendo, Game Freak and The Pokémon Company. This
  project isn't affiliated with them.

## Ideas for later

- Sealed product: is a booster box worth more opened or kept sealed?
- Price gaps between Japanese and English printings, and between TCGplayer and Cardmarket
- Discord or email alerts for watchlist hits
- Reddit and YouTube sentiment as extra hype signals
- Estimating a card's grade from a photo
- eBay sold prices, or PokéWallet Pro for real price history
