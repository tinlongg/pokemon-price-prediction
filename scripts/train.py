"""Rebuild features and retrain the forecast models for every horizon.

    python scripts/train.py            # with walk-forward backtest (slower)
    python scripts/train.py --fast     # skip the backtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokeprice.model.train import train_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the walk-forward backtest")
    args = ap.parse_args()
    train_all(backtest=not args.fast)
