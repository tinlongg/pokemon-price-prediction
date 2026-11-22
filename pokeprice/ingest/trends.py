"""Google Trends search interest per Pokémon (the "hype" signal), via the unofficial pytrends.

Google only compares up to 5 terms at once and scales each request 0-100, so every batch
includes an anchor term ("Pikachu") and batches are rescaled to a common anchor level.
Google rate-limits aggressively; failures are logged and skipped.
"""

from __future__ import annotations

import time
from typing import Callable

from .. import db

ANCHOR = "Pikachu"


def fetch_trends(names: list[str], timeframe: str = "today 5-y", pause: float = 8.0,
                 log: Callable[[str], None] = print) -> int:
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log("pytrends is not installed; skipping Google Trends.")
        return 0
    names = [n for n in dict.fromkeys(names) if n and n != ANCHOR]
    py = TrendReq(hl="en-US", tz=360)
    anchor_ref: float | None = None
    rows: list[dict] = []
    for i in range(0, len(names), 4):
        batch = names[i:i + 4]
        try:
            py.build_payload([ANCHOR] + batch, timeframe=timeframe)
            df = py.interest_over_time()
        except Exception as e:
            log(f"Google Trends failed for {batch}: {e}. Stopping; already-fetched batches are kept.")
            break
        if df.empty or ANCHOR not in df:
            continue
        anchor_mean = float(df[ANCHOR].mean()) or 1.0
        if anchor_ref is None:
            anchor_ref = anchor_mean
            rows += [{"pokemon": ANCHOR, "date": d.date().isoformat(), "interest": v}
                     for d, v in df[ANCHOR].items()]
        scale = anchor_ref / anchor_mean
        for name in batch:
            if name in df:
                rows += [{"pokemon": name, "date": d.date().isoformat(), "interest": min(100.0, v * scale)}
                         for d, v in df[name].items()]
        time.sleep(pause)
    n = db.insert_trends(rows)
    log(f"Google Trends: saved {n} weekly points for {len({r['pokemon'] for r in rows})} Pokémon")
    return n
