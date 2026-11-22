"""App settings, loaded from environment variables and an optional .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

HORIZONS = (30, 90, 180, 365)  # forecast horizons in days
QUANTILES = (0.1, 0.5, 0.9)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    pokewallet_api_key: str = ""
    tcgapi_key: str = ""
    pokemontcg_api_key: str = ""

    db_path: Path = ROOT / "data" / "pokeprice.db"
    models_dir: Path = ROOT / "models"

    # How many cards to track. PokéWallet's free tier allows 1,000 requests/day.
    max_cards: int = 600
    # Years of simulated history generated before real snapshots begin.
    synthetic_years: int = 5
    # Training weight of a simulated row relative to a real one.
    synthetic_weight: float = 0.3
    # Sample one training row per card every N days.
    train_stride_days: int = 7


settings = Settings()
