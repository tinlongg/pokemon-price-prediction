"""Recompute every card's forecast with the current model (after a refresh, without retraining)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokeprice.model.precompute import save  # noqa: E402

if __name__ == "__main__":
    path = save()
    print(f"Precomputed forecasts to {path}" if path else "No trained model yet; run scripts/train.py first.")
