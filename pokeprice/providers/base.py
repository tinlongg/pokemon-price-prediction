"""Shared provider plumbing: price quote type, HTTP client with retries, and a simple rate limiter."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


@dataclass
class PriceQuote:
    card_id: str
    source: str
    market: float
    low: float | None = None
    mid: float | None = None
    high: float | None = None
    variant: str | None = None
    date: str = field(default_factory=lambda: date.today().isoformat())

    def as_row(self) -> dict:
        return {"card_id": self.card_id, "date": self.date, "source": self.source,
                "market": self.market, "low": self.low, "mid": self.mid, "high": self.high,
                "variant": self.variant, "is_synthetic": 0}


class PriceProvider(Protocol):
    name: str

    def quote(self, card: dict) -> PriceQuote | None:
        """Current price for a card row from the `cards` table, or None if not found."""


class RateLimitExceeded(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (500, 502, 503, 504)
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


class HttpClient:
    """httpx wrapper with retries on transient errors and a minimum gap between requests."""

    def __init__(self, base_url: str, headers: dict[str, str] | None = None,
                 min_interval: float = 0.0, timeout: float = 30.0):
        self._client = httpx.Client(base_url=base_url, headers=headers or {}, timeout=timeout)
        self._min_interval = min_interval
        self._last = 0.0
        self.last_headers: httpx.Headers | None = None

    def _wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self._min_interval:
            time.sleep(self._min_interval - gap)
        self._last = time.monotonic()

    @retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, max=20), reraise=True)
    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self._wait()
        resp = self._client.get(path, params=params)
        self.last_headers = resp.headers
        if resp.status_code == 429:
            raise RateLimitExceeded(resp.text[:300])
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()


def to_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None
