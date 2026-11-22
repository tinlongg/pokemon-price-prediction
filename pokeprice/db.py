"""SQLite storage for cards, daily price snapshots, Google Trends and the watchlist."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    card_id        TEXT PRIMARY KEY,   -- pokemontcg.io id, e.g. base1-4
    name           TEXT NOT NULL,
    pokemon        TEXT,               -- base species / character, e.g. Charizard
    supertype      TEXT,
    subtypes       TEXT,               -- JSON list
    set_id         TEXT,
    set_name       TEXT,
    series         TEXT,
    release_date   TEXT,               -- ISO date
    number         TEXT,
    rarity         TEXT,
    artist         TEXT,
    dex_number     INTEGER,
    language       TEXT DEFAULT 'en',
    image_url      TEXT,
    pokewallet_id  TEXT,
    tcgapi_id      TEXT,               -- TCGapi product id, or '-' if no match was found
    tcgapi_printing TEXT,
    updated_at     TEXT
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    card_id       TEXT NOT NULL,
    date          TEXT NOT NULL,       -- ISO date
    source        TEXT NOT NULL,       -- pokewallet | pokemontcg | tcgapi | synthetic
    market        REAL NOT NULL,       -- USD market price of the card's headline variant
    low           REAL,
    mid           REAL,
    high          REAL,
    variant       TEXT,
    is_synthetic  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (card_id, date, source)
);
CREATE INDEX IF NOT EXISTS ix_prices_card_date ON price_snapshots(card_id, date);

CREATE TABLE IF NOT EXISTS trends (
    pokemon       TEXT NOT NULL,
    date          TEXT NOT NULL,
    interest      REAL NOT NULL,       -- 0-100, Google Trends scale
    is_synthetic  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (pokemon, date, is_synthetic)
);

CREATE TABLE IF NOT EXISTS watchlist (
    card_id       TEXT PRIMARY KEY,
    target_price  REAL,
    quantity      INTEGER NOT NULL DEFAULT 0,
    cost_basis    REAL,                -- total paid for all copies
    notes         TEXT,
    added_at      TEXT NOT NULL
);
"""

CARD_COLUMNS = [
    "card_id", "name", "pokemon", "supertype", "subtypes", "set_id", "set_name", "series",
    "release_date", "number", "rarity", "artist", "dex_number", "language", "image_url",
    "pokewallet_id", "tcgapi_id",
]
DAILY_SOURCES = ("pokewallet", "tcgapi")
PRICE_COLUMNS = ["card_id", "date", "source", "market", "low", "mid", "high", "variant", "is_synthetic"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cards)")}
    if "tcgapi_printing" not in cols:
        conn.execute("ALTER TABLE cards ADD COLUMN tcgapi_printing TEXT")


# --- cards -----------------------------------------------------------------

def upsert_cards(cards: Iterable[dict], db_path: Path | None = None) -> int:
    rows = []
    for c in cards:
        c = dict(c)
        if isinstance(c.get("subtypes"), (list, tuple)):
            c["subtypes"] = json.dumps(list(c["subtypes"]))
        rows.append([c.get(col) for col in CARD_COLUMNS] + [_now()])
    if not rows:
        return 0
    cols = CARD_COLUMNS + ["updated_at"]
    placeholders = ",".join("?" * len(cols))
    # Keep provider ids that were matched earlier if the new row lacks them.
    updates = ",".join(
        f"{c}=COALESCE(excluded.{c}, cards.{c})" if c in ("pokewallet_id", "tcgapi_id") else f"{c}=excluded.{c}"
        for c in cols if c != "card_id"
    )
    with connect(db_path) as conn:
        conn.executemany(
            f"INSERT INTO cards ({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(card_id) DO UPDATE SET {updates}",
            rows,
        )
    return len(rows)


def set_provider_id(card_id: str, column: str, value: str, db_path: Path | None = None) -> None:
    assert column in ("pokewallet_id", "tcgapi_id")
    with connect(db_path) as conn:
        conn.execute(f"UPDATE cards SET {column}=? WHERE card_id=?", (value, card_id))


def set_tcgapi_match(card_id: str, tcgapi_id: str, printing: str | None, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE cards SET tcgapi_id=?, tcgapi_printing=? WHERE card_id=?",
                     (tcgapi_id, printing, card_id))


def load_cards(db_path: Path | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM cards", conn)
    df["subtypes"] = df["subtypes"].apply(lambda s: json.loads(s) if s else [])
    return df


# --- prices ----------------------------------------------------------------

def insert_snapshots(rows: Iterable[dict], db_path: Path | None = None) -> int:
    data = [[r.get(c) for c in PRICE_COLUMNS] for r in rows if r.get("market") is not None]
    if not data:
        return 0
    with connect(db_path) as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO price_snapshots ({','.join(PRICE_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(PRICE_COLUMNS))})",
            data,
        )
    return len(data)


def delete_synthetic_prices(db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM price_snapshots WHERE is_synthetic=1")
        conn.execute("DELETE FROM trends WHERE is_synthetic=1")


def load_prices(card_ids: list[str] | None = None, include_synthetic: bool = True,
                db_path: Path | None = None) -> pd.DataFrame:
    """One row per (card, date): real sources win over simulated ones on the same day."""
    sql = "SELECT * FROM price_snapshots"
    clauses, params = [], []
    if card_ids is not None:
        clauses.append(f"card_id IN ({','.join('?' * len(card_ids))})")
        params.extend(card_ids)
    if not include_synthetic:
        clauses.append("is_synthetic=0")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    with connect(db_path) as conn:
        df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    # Keep each card's series on one price source family. PokéWallet and TCGapi both report
    # TCGplayer's daily market price and agree closely; pokemontcg.io can be weeks stale and
    # sometimes tracks another printing, so mixing it in creates fake price jumps.
    daily = df["source"].isin(DAILY_SOURCES) & (df["is_synthetic"] == 0)
    has_daily = set(df.loc[daily, "card_id"])
    df = df[~(df["card_id"].isin(has_daily) & (df["source"] == "pokemontcg"))]
    # PokéWallet and TCGapi refresh daily; pokemontcg.io prices can be weeks old.
    priority = {"pokewallet": 0, "tcgapi": 1, "pokemontcg": 2, "synthetic": 9}
    df["_p"] = df["source"].map(priority).fillna(5)
    df = df.sort_values(["card_id", "date", "_p"]).drop_duplicates(["card_id", "date"]).drop(columns="_p")
    return df.reset_index(drop=True)


def data_mix(db_path: Path | None = None) -> dict:
    with connect(db_path) as conn:
        cur = conn.execute(
            "SELECT is_synthetic, COUNT(*), COUNT(DISTINCT card_id), MIN(date), MAX(date) "
            "FROM price_snapshots GROUP BY is_synthetic"
        )
        out = {"real": {"rows": 0, "cards": 0, "first": None, "last": None},
               "synthetic": {"rows": 0, "cards": 0, "first": None, "last": None}}
        for syn, rows, cards, first, last in cur.fetchall():
            out["synthetic" if syn else "real"] = {"rows": rows, "cards": cards, "first": first, "last": last}
        out["real_days"] = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM price_snapshots WHERE is_synthetic=0").fetchone()[0]
    return out


# --- trends ----------------------------------------------------------------

def insert_trends(rows: Iterable[dict], db_path: Path | None = None) -> int:
    data = [(r["pokemon"], r["date"], float(r["interest"]), int(r.get("is_synthetic", 0))) for r in rows]
    if not data:
        return 0
    with connect(db_path) as conn:
        conn.executemany("INSERT OR REPLACE INTO trends VALUES (?,?,?,?)", data)
    return len(data)


def load_trends(db_path: Path | None = None) -> pd.DataFrame:
    """One row per (pokemon, date); real Google Trends data wins over simulated."""
    with connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM trends", conn)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return (df.sort_values(["pokemon", "date", "is_synthetic"])
              .drop_duplicates(["pokemon", "date"]).reset_index(drop=True))


# --- watchlist -------------------------------------------------------------

def watchlist_add(card_id: str, target_price: float | None = None, quantity: int = 0,
                  cost_basis: float | None = None, notes: str | None = None,
                  db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO watchlist VALUES (?,?,?,?,?,?) ON CONFLICT(card_id) DO UPDATE SET "
            "target_price=excluded.target_price, quantity=excluded.quantity, "
            "cost_basis=excluded.cost_basis, notes=excluded.notes",
            (card_id, target_price, quantity, cost_basis, notes, _now()),
        )


def watchlist_remove(card_id: str, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM watchlist WHERE card_id=?", (card_id,))


def load_watchlist(db_path: Path | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM watchlist ORDER BY added_at", conn)
