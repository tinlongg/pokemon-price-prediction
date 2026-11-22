#!/usr/bin/env bash
# Publish your local database, model and forecasts to the `data` GitHub release that the
# hosted app downloads. Optional: the "Refresh data" workflow does this on its own, but
# seeding from your machine keeps the real prices and card matches you already have.
# Needs the GitHub CLI, logged in: brew install gh && gh auth login
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

$PY scripts/precompute.py
out=$(mktemp -d)
cp data/pokeprice.db "$out/pokeprice.db"
$PY - "$out" <<'PY'
import json, sqlite3, sys, datetime
out = sys.argv[1]
con = sqlite3.connect(f"{out}/pokeprice.db")
con.execute("DELETE FROM watchlist")          # your watchlist stays private
con.commit(); con.execute("VACUUM"); con.close()
json.dump({"version": datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")},
          open(f"{out}/manifest.json", "w"))
PY
gzip -c "$out/pokeprice.db" > "$out/pokeprice.db.gz"
cp models/bundle.joblib models/latest.joblib models/metrics.json "$out/"
gh release view data >/dev/null 2>&1 || \
  gh release create data --prerelease --title "Published data" \
    --notes "Database, model and forecasts for the hosted app. Updated by the Refresh data workflow."
gh release upload data "$out/pokeprice.db.gz" "$out/bundle.joblib" "$out/latest.joblib" "$out/metrics.json" --clobber
gh release upload data "$out/manifest.json" --clobber
rm -rf "$out"
echo "Published."
