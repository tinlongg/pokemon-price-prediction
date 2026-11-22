"""The hosted app's download path, with GitHub's API mocked."""

from __future__ import annotations

import gzip
import json
import sqlite3

import httpx
import respx

from pokeprice import db, remote
from pokeprice.config import settings

API = "https://api.github.com/repos/me/repo/releases/tags/data"


def _published_db(tmp_path) -> bytes:
    path = tmp_path / "published.db"
    with db.connect(path):
        pass
    con = sqlite3.connect(path)
    con.execute("INSERT INTO cards (card_id, name) VALUES ('base1-4', 'Charizard')")
    con.commit()
    con.close()
    return gzip.compress(path.read_bytes())


def _mock_release(tmp_path, version: str):
    files = {"manifest.json": json.dumps({"version": version}).encode(),
             "pokeprice.db.gz": _published_db(tmp_path), "bundle.joblib": b"model",
             "latest.joblib": b"latest", "metrics.json": b"{}"}
    respx.get(API).mock(return_value=httpx.Response(200, json={"assets": [
        {"name": n, "url": f"https://api.github.com/assets/{n}"} for n in files]}))
    for n, body in files.items():
        respx.get(f"https://api.github.com/assets/{n}").mock(return_value=httpx.Response(200, content=body))


@respx.mock
def test_sync_downloads_new_version_and_keeps_watchlist(tmp_db, tmp_path):
    db.watchlist_add("base1-4", target_price=900)          # a visitor's watchlist entry
    _mock_release(tmp_path, "v1")
    assert remote.sync(repo="me/repo", token="t") == "v1"
    assert db.load_cards()["card_id"].tolist() == ["base1-4"]
    assert (settings.models_dir / "bundle.joblib").read_bytes() == b"model"
    assert db.load_watchlist()["target_price"].tolist() == [900]
    assert remote.local_version() == "v1"


@respx.mock
def test_sync_skips_download_when_version_unchanged(tmp_db, tmp_path):
    _mock_release(tmp_path, "v1")
    remote.sync(repo="me/repo")
    db_route = respx.routes[2]  # pokeprice.db.gz
    calls = db_route.call_count
    assert remote.sync(repo="me/repo") == "v1"
    assert db_route.call_count == calls                     # only the manifest was fetched


def test_sync_is_off_without_config(tmp_db, monkeypatch):
    monkeypatch.delenv("POKEPRICE_DATA_REPO", raising=False)
    assert remote.sync() is None and not remote.configured()
