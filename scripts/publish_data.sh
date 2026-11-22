#!/usr/bin/env bash
# Publish your local database, model and forecasts to the `data` release of your PRIVATE
# data repo, which the hosted app downloads. Optional: the "Refresh data" workflow does this
# on its own, but seeding from your machine keeps the real prices and card matches you have.
# Needs the GitHub CLI, logged in: brew install gh && gh auth login
#   scripts/publish_data.sh tinlongg/pokeprice-data
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
REPO=${1:-${POKEPRICE_DATA_REPO:-}}
if [ -z "$REPO" ]; then
  echo "Usage: scripts/publish_data.sh owner/private-data-repo" >&2; exit 1
fi
if [ "$(gh api "repos/$REPO" --jq .private)" != "true" ]; then
  echo "$REPO is not private. The data can't be published publicly (TCG API terms)." >&2; exit 1
fi

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
gh release view data -R "$REPO" >/dev/null 2>&1 || \
  gh release create data -R "$REPO" --title "Published data" \
    --notes "Database, model and forecasts for the PokéPrice app. Updated by the Refresh data workflow."
gh release upload data -R "$REPO" "$out/pokeprice.db.gz" "$out/bundle.joblib" "$out/latest.joblib" "$out/metrics.json" --clobber
gh release upload data -R "$REPO" "$out/manifest.json" --clobber
rm -rf "$out"
echo "Published to $REPO."
