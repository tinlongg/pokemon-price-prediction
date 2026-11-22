"""Fetch the published data (database, model, precomputed forecasts) for a hosted app.

The GitHub Actions workflow (.github/workflows/refresh.yml) refreshes prices, retrains and
uploads these files to the `data` release of a separate private repo (TCG API's terms forbid
public downloads of its records). A hosted app calls `sync()` on
startup and every so often; it downloads only when the release's manifest version changes.

Configure with environment variables (or Streamlit secrets, which are exposed as env vars):
  POKEPRICE_DATA_REPO     owner/repo of the private data repo (required to enable syncing)
  POKEPRICE_GITHUB_TOKEN  read-only token for that repo
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

import httpx

from . import db
from .config import settings

TAG = "data"
DB_ASSET = "pokeprice.db.gz"
MODEL_ASSETS = ("bundle.joblib", "latest.joblib", "metrics.json")
MANIFEST = "manifest.json"


def configured() -> bool:
    return bool(os.environ.get("POKEPRICE_DATA_REPO"))


def _headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _asset_urls(client: httpx.Client, repo: str) -> dict[str, str]:
    r = client.get(f"https://api.github.com/repos/{repo}/releases/tags/{TAG}")
    r.raise_for_status()
    return {a["name"]: a["url"] for a in r.json().get("assets", [])}


def _download(client: httpx.Client, url: str, dest: Path) -> None:
    # The asset API URL redirects to storage; httpx drops the Authorization header on the
    # cross-origin redirect, which is what the storage host expects.
    with client.stream("GET", url, headers={"Accept": "application/octet-stream"},
                       follow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)


def local_version() -> str | None:
    p = Path(settings.models_dir) / MANIFEST
    return json.loads(p.read_text()).get("version") if p.exists() else None


def sync(repo: str | None = None, token: str | None = None, force: bool = False) -> str | None:
    """Download the latest published data if it is newer than what we have. Returns the
    version now in place (None if syncing isn't configured). The local watchlist is kept."""
    repo = repo or os.environ.get("POKEPRICE_DATA_REPO")
    if not repo:
        return None
    token = token or os.environ.get("POKEPRICE_GITHUB_TOKEN") or None
    with httpx.Client(headers=_headers(token), timeout=60) as client:
        assets = _asset_urls(client, repo)
        if MANIFEST not in assets:
            return local_version()
        # Download next to the data, not in the system temp folder: os.replace can't move
        # files across disks, and on Streamlit Cloud /tmp is a different disk from the app.
        data_dir = Path(settings.db_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data_dir, prefix=".download-") as tmp:
            tmp = Path(tmp)
            _download(client, assets[MANIFEST], tmp / MANIFEST)
            remote = json.loads((tmp / MANIFEST).read_text())
            have_files = Path(settings.db_path).exists() and (Path(settings.models_dir) / MODEL_ASSETS[0]).exists()
            if not force and have_files and remote.get("version") == local_version():
                return local_version()

            for name in (DB_ASSET, *MODEL_ASSETS):
                if name in assets:
                    _download(client, assets[name], tmp / name)
            db_tmp = tmp / "pokeprice.db"
            with gzip.open(tmp / DB_ASSET, "rb") as src, open(db_tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)

            watch = db.load_watchlist() if Path(settings.db_path).exists() else None
            models_dir = Path(settings.models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            os.replace(db_tmp, settings.db_path)
            for name in MODEL_ASSETS:
                if (tmp / name).exists():
                    os.replace(tmp / name, models_dir / name)
            if watch is not None:
                for r in watch.to_dict("records"):
                    db.watchlist_add(r["card_id"], r["target_price"], int(r["quantity"] or 0),
                                     r["cost_basis"], r["notes"])
            os.replace(tmp / MANIFEST, models_dir / MANIFEST)
    return remote.get("version")
