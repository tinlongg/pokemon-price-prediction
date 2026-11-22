"""Save and load the trained model bundle."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ..config import settings

BUNDLE = "bundle.joblib"


def save(models: dict, metrics: dict, features: list[str], extra: dict | None = None,
         models_dir: Path | None = None) -> Path:
    d = Path(models_dir or settings.models_dir)
    d.mkdir(parents=True, exist_ok=True)
    bundle = {"models": models, "metrics": metrics, "features": features,
              "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}
    path = d / BUNDLE
    joblib.dump(bundle, path)
    (d / "metrics.json").write_text(json.dumps({"trained_at": bundle["trained_at"], "metrics": metrics,
                                                **(extra or {})}, indent=2, default=str))
    return path


def load(models_dir: Path | None = None) -> dict | None:
    path = Path(models_dir or settings.models_dir) / BUNDLE
    return joblib.load(path) if path.exists() else None
