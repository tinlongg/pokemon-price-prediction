"""Fetch data: card universe (first run), today's prices, Google Trends, TCGapi backfill, and
simulated history for cards that need it.

    python scripts/refresh.py --universe        # first run: pick cards + first prices + simulated history
    python scripts/refresh.py                   # daily: new prices + TCGapi history (run hourly to use all PokéWallet quota)
    python scripts/refresh.py --trends          # also pull Google Trends (weekly is plenty)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokeprice import db  # noqa: E402
from pokeprice.features.popularity import POKEMON_POPULARITY, TRAINER_POPULARITY  # noqa: E402
from pokeprice.ingest.backfill import run_backfill  # noqa: E402
from pokeprice.ingest.snapshot import run_snapshot  # noqa: E402
from pokeprice.ingest.synthetic import ensure_synthetic  # noqa: E402
from pokeprice.ingest.trends import fetch_trends  # noqa: E402
from pokeprice.ingest.universe import build_universe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", action="store_true", help="(re)build the list of tracked cards")
    ap.add_argument("--max-cards", type=int, default=None, help="cards to track (default 600)")
    ap.add_argument("--limit", type=int, default=None, help="only snapshot the first N cards")
    ap.add_argument("--max-requests", type=int, default=95, help="PokéWallet requests this run")
    ap.add_argument("--trends", action="store_true", help="pull Google Trends")
    ap.add_argument("--no-backfill", action="store_true",
                    help="skip TCGapi (by default it runs when TCGAPI_KEY is set)")
    ap.add_argument("--tcgapi-requests", type=int, default=90, help="TCGapi requests this run (100/day free)")
    ap.add_argument("--resimulate", action="store_true", help="rebuild simulated history")
    ap.add_argument("--no-pokewallet", action="store_true")
    args = ap.parse_args()

    if args.universe or db.load_cards().empty:
        print("Building card universe from pokemontcg.io …")
        cards, quotes = build_universe(args.max_cards)
        db.upsert_cards(cards)
        db.insert_snapshots(q.as_row() for q in quotes)

    run_snapshot(limit=args.limit, max_requests=args.max_requests, use_pokewallet=not args.no_pokewallet)
    if not args.no_backfill:
        run_backfill(max_requests=args.tcgapi_requests)
    if args.trends:
        cards = db.load_cards()
        names = [n for n in cards["pokemon"].value_counts().index
                 if n in POKEMON_POPULARITY or n in TRAINER_POPULARITY]
        fetch_trends(names[:40])
    ensure_synthetic(force=args.resimulate)
    mix = db.data_mix()
    print(f"Done. Real prices: {mix['real']['rows']:,} rows over {mix['real_days']} days; "
          f"simulated: {mix['synthetic']['rows']:,} rows.")


if __name__ == "__main__":
    main()
