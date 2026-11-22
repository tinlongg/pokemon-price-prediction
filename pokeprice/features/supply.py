"""Supply side: print-run size by era, whether the set is still being printed, and reprint risk."""

from __future__ import annotations

# Relative print-run size by era index (WOTC smallest; 2020-2022 was a massive print wave).
PRINT_RUN_BY_ERA = {0: 0.25, 1: 0.15, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.55, 6: 1.0, 7: 0.85, 8: 0.8}


def print_run_index(era_idx: int) -> float:
    return PRINT_RUN_BY_ERA.get(era_idx, 0.8)


def in_print(age_years: float) -> int:
    """Modern English sets are typically printed for 12-18 months."""
    return int(age_years < 1.5)


def reprint_risk(age_years: float, popularity: float, special_art: int) -> float:
    """0-1 chance-weighted pressure from reprints or new versions diluting demand.
    New, popular, non-special cards are the most exposed; special-art and vintage cards the least."""
    recency = max(0.0, 1.0 - age_years / 4.0)
    return round(recency * (0.4 + 0.6 * popularity) * (0.5 if special_art else 1.0), 4)
